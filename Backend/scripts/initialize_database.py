"""Manual Phase 1 MongoDB collection/index setup.

Run from Backend: python scripts/initialize_database.py
"""
import sys
from pathlib import Path

# Direct execution makes `scripts/` the import root, so explicitly expose the
# Backend project directory before importing the application package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.db.indexes import initialize_database
from app.db.mongodb import close_mongo_client, get_database


def main() -> None:
    if not get_settings().mongodb_uri:
        raise SystemExit("MONGODB_URI is missing. Copy .env.example to .env and configure MongoDB Atlas first.")
    try:
        result = initialize_database(get_database())
        print(f"Initialized MongoDB database: {result['database']}")
        print("Collections:", ", ".join(result["collections"]))
    finally:
        close_mongo_client()


if __name__ == "__main__":
    main()
