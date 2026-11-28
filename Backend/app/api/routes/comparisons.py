from contextlib import contextmanager
import logging
import re
import traceback
from urllib.parse import unquote_plus, urlsplit

from fastapi import APIRouter, HTTPException, Query, Response
from pymongo.errors import PyMongoError

from app.core.config import get_settings
from app.db.mongodb import get_database
from app.modules.claim_extraction.errors import ClaimStorageError, PaperNotFoundError
from app.modules.contradiction_detection.evaluator import PairEvaluator
from app.modules.contradiction_detection.repository import ComparisonRepository
from app.modules.contradiction_detection.service import ComparisonService
from app.modules.embeddings.errors import EmbeddingError
from app.modules.embeddings.service import get_embedding_service
from app.modules.structural_similarity.service import StructuralSimilarityService
from app.schemas.comparisons import (
    ComparisonList, DetectionRequest, DetectionResponse, DiscoveryRequest, DiscoveryResponse,
    Mismatch, ProcessingStatus, Relationship,
)


router = APIRouter(prefix="/comparisons", tags=["cross-paper comparisons"])
logger = logging.getLogger("uvicorn.error")

# Driver exceptions can echo a connection string, an auth header or a settings
# dictionary. Remove these from the rendered trace before it reaches any handler.
_AUTH_TEXT = re.compile(
    r"""mongodb(?:\+srv)?://[^\s'"><]+|\b(?:basic|bearer)[ \t]+\S+|\b(?:gsk_|sk-)[\w-]+"""
    r"""|\b(?:authorization|[\w-]*api[_-]?key|[\w-]*password|[\w-]*token)['"]?\s*[:=][^\r\n]*""",
    re.IGNORECASE,
)


def _storage_traceback(error: Exception) -> str:
    try:
        settings = get_settings()
        uri = settings.mongodb_uri
        address = urlsplit(uri)
        secrets = {uri, settings.llm_api_key.get_secret_value()}
        for credential in (address.username, address.password):
            if credential:
                secrets.update((credential, unquote_plus(credential)))
    except (ValueError, RuntimeError, OSError):
        # Configuration may itself be the failure. Keep frame locations and
        # exception types without rendering uninspectable credential values.
        chain = []
        entry = traceback.TracebackException.from_exception(error, lookup_lines=False)
        while entry is not None:
            frames = "".join(f'  File "{f.filename}", line {f.lineno}, in {f.name}\n' for f in entry.stack)
            chain.append(f"Traceback (most recent call last):\n{frames}{entry.exc_type.__name__} (message withheld)")
            entry = entry.__cause__ or (None if entry.__suppress_context__ else entry.__context__)
        return "\n\n".join(reversed(chain))

    trace = "".join(traceback.format_exception(error))
    for secret in sorted(filter(None, secrets), key=len, reverse=True):
        trace = trace.replace(secret, "[redacted]")
    return _AUTH_TEXT.sub("[redacted]", trace).rstrip()


@contextmanager
def _api_errors():
    try:
        yield
    except PaperNotFoundError as error:
        raise HTTPException(404, "One of the selected papers does not exist.") from error
    except EmbeddingError as error:
        # EmbeddingError subclasses wrap driver/model errors in safe messages.
        raise HTTPException(503, str(error)) from error
    except (ClaimStorageError, PyMongoError, RuntimeError) as error:
        # Passing exc_info here would let logging append the raw unredacted exception.
        logger.error("Comparison storage or source provenance failure\n%s", _storage_traceback(error))
        raise HTTPException(503, "Comparison storage or source provenance is unavailable.") from error
    except ValueError as error:
        raise HTTPException(422, "Select two independent publications with valid source data.") from error


@router.post("/similar", response_model=DiscoveryResponse)
def find_similar(request: DiscoveryRequest):
    """Shortlist and queue cross-paper pairs using local embeddings and Atlas. No LLM call."""
    with _api_errors():
        settings = get_settings()
        if request.limit is not None and request.limit > settings.comparison_discovery_limit:
            raise HTTPException(422, f"limit exceeds COMPARISON_DISCOVERY_LIMIT ({settings.comparison_discovery_limit}).")
        database = get_database()
        finder = StructuralSimilarityService(database, get_embedding_service(), settings)
        pairs, cursor, scanned = finder.discover(
            request.source_paper_id, request.target_paper_id, request.after_claim_id,
            request.limit or settings.comparison_discovery_limit,
            settings.comparison_min_similarity if request.min_similarity is None else request.min_similarity,
        )
        repository = ComparisonRepository(database)
        comparisons = []
        queued = 0
        for candidate in pairs:
            record, changed = repository.enqueue(candidate, settings)
            queued += int(changed)
            comparisons.append(record)
        return DiscoveryResponse(scanned=scanned, next_cursor=cursor, queued=queued, comparisons=repository.present(comparisons))


@router.post("/detect", response_model=DetectionResponse, responses={
    code: {"model": DetectionResponse, "description": description} for code, description in (
        (429, "Rate limit interrupted the batch"), (502, "Invalid response or provider failure"),
        (504, "Model request timed out"),
    )
})
def detect(request: DetectionRequest, response: Response):
    """Evaluate queued pairs. Retry failed pairs explicitly to resume saved progress."""
    with _api_errors():
        settings = get_settings()
        if request.limit is not None and request.limit > settings.comparison_detection_limit:
            raise HTTPException(422, f"limit exceeds COMPARISON_DETECTION_LIMIT ({settings.comparison_detection_limit}).")
        service = ComparisonService(ComparisonRepository(get_database()), PairEvaluator(settings), settings)
        report, response.status_code = service.run(request, request.limit or settings.comparison_detection_limit)
        return report


@router.get("", response_model=ComparisonList)
def list_comparisons(
    paper_id: str | None = Query(default=None, min_length=1, max_length=128),
    relationship: Relationship | None = None, mismatch_type: Mismatch | None = None,
    processing_status: ProcessingStatus | None = None,
    skip: int = Query(default=0, ge=0), limit: int = Query(default=20, ge=1, le=100),
):
    """List stored assessments and recheck whether both source snapshots are currently eligible."""
    with _api_errors():
        repository = ComparisonRepository(get_database())
        rows, total = repository.list_results(paper_id, relationship, mismatch_type, processing_status, skip, limit)
        return ComparisonList(total=total, skip=skip, limit=limit, comparisons=repository.present(rows))
