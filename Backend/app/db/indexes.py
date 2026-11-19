from datetime import datetime, timezone

from pymongo import ASCENDING, DESCENDING
from pymongo.database import Database
from pymongo.errors import CollectionInvalid


# These collections are storage preparation only. No research analysis is
# performed in Phase 1.
COLLECTIONS = (
    "papers",
    "document_chunks",
    "paper_chunks",
    "processing_jobs",
    "claims",
    "contradictions",
    "research_gaps",
    "analysis_runs",
)


def _ensure_collection(database: Database, name: str) -> None:
    try:
        database.create_collection(name)
    except CollectionInvalid:
        pass


def initialize_database(database: Database) -> dict[str, object]:
    """Create Phase 1 collections and standard indexes.

    Atlas Vector Search indexes are intentionally deferred until an embedding
    schema exists in a later phase.
    """
    for name in COLLECTIONS:
        _ensure_collection(database, name)

    database.papers.create_index([("status", ASCENDING), ("created_at", DESCENDING)], name="paper_status_created")
    database.papers.create_index([("metadata.doi", ASCENDING)], name="paper_doi", sparse=True)
    database.papers.create_index([("content_sha256", ASCENDING)], name="paper_content_hash", unique=True, sparse=True)
    database.document_chunks.create_index([("paper_id", ASCENDING), ("chunk_index", ASCENDING)], name="chunk_paper_position", unique=True)
    database.paper_chunks.create_index([("chunk_id", ASCENDING)], name="paper_chunk_id", unique=True)
    database.paper_chunks.create_index([("paper_id", ASCENDING), ("chunk_index", ASCENDING)], name="paper_chunk_position", unique=True)
    database.processing_jobs.create_index([("paper_id", ASCENDING), ("created_at", DESCENDING)], name="job_paper_created")
    database.processing_jobs.create_index([("status", ASCENDING), ("updated_at", DESCENDING)], name="job_status_updated")
    database.claims.create_index([("paper_id", ASCENDING), ("created_at", DESCENDING)], name="claim_paper_created")
    database.contradictions.create_index([("paper_ids", ASCENDING)], name="contradiction_papers")
    database.research_gaps.create_index([("status", ASCENDING), ("created_at", DESCENDING)], name="gap_status_created")
    database.analysis_runs.create_index([("created_at", DESCENDING)], name="analysis_created")

    return {
        "database": database.name,
        "collections": list(COLLECTIONS),
        "initialized_at": datetime.now(timezone.utc).isoformat(),
    }
