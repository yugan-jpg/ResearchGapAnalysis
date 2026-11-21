import logging

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from pymongo.errors import PyMongoError

from app.core.config import get_settings
from app.db.mongodb import get_database
from app.modules.claim_extraction.errors import (
    ClaimExtractionError,
    ClaimResponseValidationError,
    ClaimStorageError,
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    PaperNotFoundError,
)
from app.modules.claim_extraction.llm_client import QwenClaimClient
from app.modules.claim_extraction.gemini_verifier import GeminiClaimVerifier
from app.modules.claim_extraction.repository import ClaimRepository
from app.modules.claim_extraction.service import ClaimExtractionService
from app.schemas.claims import (
    ClaimExtractionRequest,
    ClaimExtractionResponse,
    ClaimListResponse,
    ClaimResponse,
    ClaimVerificationRequest,
    ClaimVerificationResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/claims", tags=["claims"])


@router.post(
    "/extract/{paper_id}",
    response_model=ClaimExtractionResponse,
    responses={
        404: {"description": "Paper not found"},
        429: {"description": "Hosted LLM rate limit"},
        502: {"description": "Hosted LLM response or provider failure"},
        503: {"description": "Missing LLM configuration or database unavailable"},
        504: {"description": "Hosted LLM timeout"},
    },
)
async def extract_claims(paper_id: str, request: ClaimExtractionRequest) -> ClaimExtractionResponse:
    """Extract evidence-grounded claims from a bounded number of paper chunks."""
    try:
        settings = get_settings()
        service = ClaimExtractionService(
            repository=ClaimRepository(get_database()),
            llm_client=QwenClaimClient(settings),
            model_name=settings.llm_model_name,
            verifier=GeminiClaimVerifier(settings),
        )
        result = await run_in_threadpool(service.extract, paper_id, request.max_chunks, request.force)
        return ClaimExtractionResponse(**result.__dict__)
    except PaperNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except LLMConfigurationError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    except LLMRateLimitError as error:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(error)) from error
    except LLMTimeoutError as error:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(error)) from error
    except (LLMProviderError, ClaimResponseValidationError) as error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    except (ClaimStorageError, PyMongoError, RuntimeError) as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="The database is unavailable for claim extraction. Please retry.") from error
    except ClaimExtractionError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    except Exception as error:
        logger.exception("Unexpected scientific claim extraction failure")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Claims could not be extracted safely. Please try again.") from error


@router.post(
    "/verify/{paper_id}",
    response_model=ClaimVerificationResponse,
    responses={
        404: {"description": "Paper not found"},
        503: {"description": "Database unavailable"},
    },
)
async def verify_existing_claims(
    paper_id: str, request: ClaimVerificationRequest
) -> ClaimVerificationResponse:
    """Add Gemini verification metadata to stored claims without re-extracting."""
    try:
        settings = get_settings()
        service = ClaimExtractionService(
            repository=ClaimRepository(get_database()),
            llm_client=QwenClaimClient(settings),
            model_name=settings.llm_model_name,
            verifier=GeminiClaimVerifier(settings),
        )
        result = await run_in_threadpool(service.verify_existing, paper_id, request.limit, request.force)
        return ClaimVerificationResponse(**result.__dict__)
    except PaperNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except (ClaimStorageError, PyMongoError, RuntimeError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The database is unavailable for claim verification. Please retry.",
        ) from error
    except Exception as error:
        logger.exception("Claim verification failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Claims could not be verified safely. Please try again.",
        ) from error


@router.get(
    "/{paper_id}",
    response_model=ClaimListResponse,
    responses={404: {"description": "Paper not found"}, 503: {"description": "Database unavailable"}},
)
def list_claims(paper_id: str, skip: int = Query(default=0, ge=0), limit: int = Query(default=50, ge=1, le=200)) -> ClaimListResponse:
    """Retrieve previously verified claims for one paper without invoking the LLM."""
    try:
        repository = ClaimRepository(get_database())
        repository.ensure_paper(paper_id)
        claims, total = repository.list_claims(paper_id, skip, limit)
        return ClaimListResponse(
            paper_id=paper_id,
            claims=[ClaimResponse(**claim) for claim in claims],
            total=total,
            skip=skip,
            limit=limit,
        )
    except PaperNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except (ClaimStorageError, PyMongoError, RuntimeError) as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="The database is unavailable for claim retrieval. Please retry.") from error
    except Exception as error:
        logger.exception("Claim retrieval failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Claims could not be retrieved safely. Please try again.") from error
