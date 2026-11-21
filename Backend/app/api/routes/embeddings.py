import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pymongo.errors import PyMongoError

from app.core.config import get_settings
from app.db.mongodb import get_database
from app.modules.embeddings.backfill import EmbeddingBackfillService
from app.modules.embeddings.errors import (
    EmbeddingDimensionError,
    EmbeddingModelError,
    EmbeddingStorageError,
)
from app.modules.embeddings.repository import EmbeddingRepository
from app.modules.embeddings.service import get_embedding_service
from app.schemas.embeddings import BackfillRequest, BackfillResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post("/backfill", response_model=BackfillResponse)
async def backfill_embeddings(request: BackfillRequest) -> BackfillResponse:
    """Generate local embeddings for a bounded set of missing/stale chunks."""
    try:
        result = await run_in_threadpool(
            EmbeddingBackfillService(EmbeddingRepository(get_database()), get_embedding_service()).backfill,
            request.paper_id,
            request.limit,
            request.force,
        )
        return BackfillResponse(**result.__dict__)
    except EmbeddingDimensionError as error:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="The configured embedding dimensions do not match the loaded model.") from error
    except EmbeddingModelError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    except (EmbeddingStorageError, PyMongoError, RuntimeError) as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="The database is unavailable for embedding backfill. Please retry.") from error
    except Exception as error:
        logger.exception("Unexpected embedding backfill failure")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Embeddings could not be generated safely. Please try again.") from error
