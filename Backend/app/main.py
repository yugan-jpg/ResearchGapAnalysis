from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.papers import router as papers_router
from app.core.config import get_settings
from app.db.indexes import initialize_database
from app.db.mongodb import close_mongo_client, get_database


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    if settings.auto_initialize_database:
        initialize_database(get_database())
    yield
    close_mongo_client()


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

# The React frontend reads VITE_API_BASE_URL and makes HTTP requests. Restrict
# browser access to its configured origin rather than allowing all origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(papers_router, prefix=settings.api_prefix)


@app.get("/", tags=["system"], include_in_schema=False)
def root() -> dict[str, str]:
    return {"message": "Scientific Literature Gap Discovery API. See /docs for Phase 1 endpoints."}
