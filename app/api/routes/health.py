from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.db.mongodb import mongo_health

router = APIRouter(tags=["system"])


@router.get("/health", summary="Check API and MongoDB availability")
def health_check() -> JSONResponse:
    """Lightweight health endpoint; it never returns connection credentials."""
    settings = get_settings()
    database = mongo_health()
    payload = {
        "status": "ok" if database["connected"] else "degraded",
        "service": settings.app_name,
        "database": database,
    }
    response_status = status.HTTP_200_OK if database["connected"] else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=response_status, content=payload)
