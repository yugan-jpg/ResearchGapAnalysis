import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pymongo.errors import PyMongoError

from app.core.config import get_settings
from app.db.mongodb import get_database
from app.modules.embeddings.errors import EmbeddingDimensionError, EmbeddingModelError, VectorSearchError
from app.modules.embeddings.repository import EmbeddingRepository
from app.modules.embeddings.service import get_embedding_service
from app.schemas.embeddings import SemanticSearchRequest, SemanticSearchResponse, SemanticSearchResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


@router.post("/semantic", response_model=SemanticSearchResponse)
async def semantic_search(request: SemanticSearchRequest) -> SemanticSearchResponse:
    """Retrieve related paper chunks with MongoDB Atlas Vector Search."""
    try:
        settings = get_settings()
        embedder = get_embedding_service()
        query_vector = await run_in_threadpool(embedder.embed_texts, [request.query])
        results = await run_in_threadpool(
            EmbeddingRepository(get_database()).vector_search,
            query_vector[0],
            settings.vector_search_index_name,
            request.top_k,
            max(request.top_k * settings.vector_search_candidate_multiplier, request.top_k),
        )
        return SemanticSearchResponse(
            query=request.query,
            results=[SemanticSearchResult(**result) for result in results],
        )
    except EmbeddingDimensionError as error:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="The configured embedding dimensions do not match the loaded model.") from error
    except EmbeddingModelError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    except (VectorSearchError, PyMongoError, RuntimeError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Semantic search is unavailable. Confirm that the Atlas Vector Search index is active and correctly configured.",
        ) from error
    except Exception as error:
        logger.exception("Unexpected semantic search failure")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Semantic search could not be completed safely. Please try again.") from error
