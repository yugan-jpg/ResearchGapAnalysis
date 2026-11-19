from functools import lru_cache
from typing import Any

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError

from app.core.config import get_settings


@lru_cache
def get_client() -> MongoClient:
    """Create one lazy PyMongo client for this process."""
    settings = get_settings()
    if not settings.mongodb_uri:
        raise RuntimeError("MONGODB_URI is not configured.")
    return MongoClient(
        settings.mongodb_uri,
        appname="literature-gap-discovery-api",
        serverSelectionTimeoutMS=settings.mongodb_timeout_ms,
        connectTimeoutMS=settings.mongodb_timeout_ms,
        socketTimeoutMS=settings.mongodb_timeout_ms,
    )


def get_database() -> Database:
    return get_client()[get_settings().mongodb_database]


def mongo_health() -> dict[str, Any]:
    """Return a safe connection state without exposing URI credentials."""
    settings = get_settings()
    if not settings.mongodb_uri:
        return {"configured": False, "connected": False, "message": "MONGODB_URI is not configured."}
    try:
        get_client().admin.command("ping")
        return {"configured": True, "connected": True, "message": "MongoDB Atlas connection succeeded."}
    except (PyMongoError, RuntimeError):
        # Avoid returning driver details that could disclose connection context.
        return {"configured": True, "connected": False, "message": "MongoDB is unreachable or credentials are invalid."}


def close_mongo_client() -> None:
    """Close and clear the cached client during FastAPI shutdown."""
    if get_client.cache_info().currsize:
        get_client().close()
    get_client.cache_clear()
