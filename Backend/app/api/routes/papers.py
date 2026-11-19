import logging

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from pymongo.errors import PyMongoError

from app.core.config import get_settings
from app.db.mongodb import get_database
from app.modules.pdf_processing.chunking import PageAwareChunker
from app.modules.pdf_processing.errors import (
    ChunkingError,
    DuplicatePaperError,
    EmptyPdfError,
    FileTooLargeError,
    InvalidPdfError,
    PasswordProtectedPdfError,
    PdfProcessingError,
    ScannedPdfError,
    StorageError,
    UploadValidationError,
)
from app.modules.pdf_processing.extraction import extract_pdf
from app.modules.pdf_processing.repository import PaperRepository
from app.modules.pdf_processing.upload import stream_upload_to_temporary_file, validate_upload_metadata
from app.schemas.papers import PaperListResponse, PaperMetadataResponse, PaperUploadResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/papers", tags=["papers"])


def _metadata_response(paper: dict) -> PaperMetadataResponse:
    """Preserve the stored field and provide the existing frontend's filename."""
    return PaperMetadataResponse(filename=paper["original_filename"], **paper)


@router.post("/upload", response_model=PaperUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_paper(file: UploadFile | None = File(default=None)) -> PaperUploadResponse:
    """Ingest one text-based scientific PDF without retaining its source file."""
    temporary_path = None
    try:
        settings = get_settings()
        filename = validate_upload_metadata(file)
        assert file is not None
        saved_upload = await stream_upload_to_temporary_file(file, filename, settings.pdf_max_upload_bytes)
        temporary_path = saved_upload.path
        extraction = await run_in_threadpool(extract_pdf, saved_upload.path)
        chunker = PageAwareChunker(settings.pdf_chunk_size, settings.pdf_chunk_overlap)
        chunks = await run_in_threadpool(chunker.chunk_pages, extraction.pages)
        stored = await run_in_threadpool(PaperRepository(get_database()).create_paper, saved_upload, extraction, chunks)
        return PaperUploadResponse(
            paper_id=stored["paper_id"],
            filename=stored["original_filename"],
            total_pages=stored["total_pages"],
            pages_with_text=stored["pages_with_text"],
            total_chunks=stored["total_chunks"],
            status=stored["status"],
        )
    except FileTooLargeError as error:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(error)) from error
    except UploadValidationError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    except (InvalidPdfError, EmptyPdfError, PasswordProtectedPdfError, ScannedPdfError, ChunkingError) as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    except DuplicatePaperError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"message": str(error), "paper_id": error.paper_id}) from error
    except (StorageError, PyMongoError, RuntimeError) as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    except Exception as error:
        logger.exception("Unexpected PDF ingestion failure")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="The PDF could not be processed safely. Please try again.") from error
    finally:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
        if file is not None:
            await file.close()


@router.get("/{paper_id}", response_model=PaperMetadataResponse)
def get_paper(paper_id: str) -> PaperMetadataResponse:
    try:
        paper = PaperRepository(get_database()).get_paper(paper_id)
    except Exception as error:
        logger.exception("Paper metadata lookup failed")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="The database is unavailable. Please try again.") from error
    if paper is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Paper not found.")
    return _metadata_response(paper)


@router.get("", response_model=PaperListResponse)
def list_papers(skip: int = Query(default=0, ge=0), limit: int = Query(default=20, ge=1, le=100)) -> PaperListResponse:
    try:
        papers, total = PaperRepository(get_database()).list_papers(skip, limit)
    except Exception as error:
        logger.exception("Paper listing failed")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="The database is unavailable. Please try again.") from error
    return PaperListResponse(papers=[_metadata_response(paper) for paper in papers], total=total, skip=skip, limit=limit)
