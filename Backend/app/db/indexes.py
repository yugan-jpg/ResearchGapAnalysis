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
    "research_claims",
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
    """Create standard collection indexes. Configure the vector index separately in Atlas."""
    for name in COLLECTIONS:
        _ensure_collection(database, name)

    database.papers.create_index([("status", ASCENDING), ("created_at", DESCENDING)], name="paper_status_created")
    database.papers.create_index([("metadata.doi", ASCENDING)], name="paper_doi", sparse=True)
    database.papers.create_index([("content_sha256", ASCENDING)], name="paper_content_hash", unique=True, sparse=True)
    database.document_chunks.create_index([("paper_id", ASCENDING), ("chunk_index", ASCENDING)], name="chunk_paper_position", unique=True)
    database.paper_chunks.create_index([("chunk_id", ASCENDING)], name="paper_chunk_id", unique=True)
    database.paper_chunks.create_index([("paper_id", ASCENDING), ("chunk_index", ASCENDING)], name="paper_chunk_position", unique=True)
    database.paper_chunks.create_index([("embedding_status", ASCENDING), ("paper_id", ASCENDING)], name="paper_chunk_embedding_status")
    database.paper_chunks.create_index([("claim_extraction_status", ASCENDING), ("paper_id", ASCENDING)], name="paper_chunk_claim_status")
    database.research_claims.create_index([("claim_id", ASCENDING)], name="research_claim_id", unique=True)
    database.research_claims.create_index([("claim_hash", ASCENDING)], name="research_claim_hash", unique=True)
    database.research_claims.create_index([("normalized_claim_hash", ASCENDING)], name="research_claim_normalized_unique", unique=True, sparse=True)
    database.research_claims.create_index([("claim_content_hash", ASCENDING), ("paper_id", ASCENDING)], name="research_claim_content_hash")
    database.research_claims.create_index([("verification_cache_key", ASCENDING)], name="research_claim_verification_cache")
    database.research_claims.create_index([("verification_status", ASCENDING), ("paper_id", ASCENDING)], name="research_claim_verification_status")
    database.research_claims.create_index([("paper_id", ASCENDING), ("chunk_id", ASCENDING), ("created_at", DESCENDING)], name="research_claim_paper_chunk")
    database.processing_jobs.create_index([("paper_id", ASCENDING), ("created_at", DESCENDING)], name="job_paper_created")
    database.processing_jobs.create_index([("status", ASCENDING), ("updated_at", DESCENDING)], name="job_status_updated")
    database.claims.create_index([("paper_id", ASCENDING), ("created_at", DESCENDING)], name="claim_paper_created")
    database.contradictions.create_index([("paper_ids", ASCENDING)], name="contradiction_papers")
    database.contradictions.create_index(
        [("paper_ids", ASCENDING), ("processing_status", ASCENDING), ("_id", ASCENDING)],
        name="comparison_queue",
    )
    database.contradictions.create_index(
        [("relationship", ASCENDING), ("mismatch_types", ASCENDING)], name="comparison_classification",
    )
    database.research_claims.create_index(
        [("paper_id", ASCENDING), ("verification_status", ASCENDING), ("claim_id", ASCENDING)],
        name="comparison_source_cursor",
    )
    database.research_gaps.create_index([("status", ASCENDING), ("created_at", DESCENDING)], name="gap_status_created")
    database.analysis_runs.create_index([("created_at", DESCENDING)], name="analysis_created")

    return {
        "database": database.name,
        "collections": list(COLLECTIONS),
        "initialized_at": datetime.now(timezone.utc).isoformat(),
    }
