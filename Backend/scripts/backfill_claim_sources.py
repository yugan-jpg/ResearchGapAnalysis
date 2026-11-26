"""Backfill verifiable source references; dry-run unless --apply."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pymongo.errors import PyMongoError
from app.db.mongodb import close_mongo_client, get_database


def backfill(database, paper_id, apply=False):
    counts = {"examined": 0, "eligible": 0, "updated": 0, "skipped": 0}
    query = {"paper_id": paper_id, "$or": [{"source_references": {"$exists": False}}, {"source_references": []}]}
    for claim in database.research_claims.find(query):
        counts["examined"] += 1
        chunk_id, page, evidence = claim.get("chunk_id"), claim.get("page_number"), claim.get("supporting_evidence")
        if not chunk_id or not isinstance(page, int) or page < 1 or not isinstance(evidence, str) or not evidence:
            counts["skipped"] += 1
            continue
        chunk = database.paper_chunks.find_one({"paper_id": paper_id, "chunk_id": chunk_id, "page_number": page}, {"text": 1})
        if not chunk or evidence not in chunk.get("text", ""):
            counts["skipped"] += 1
            continue
        counts["eligible"] += 1
        if apply:
            ref = {"chunk_id": chunk_id, "page_number": page, "supporting_evidence": evidence}
            result = database.research_claims.update_one(
                {"_id": claim["_id"], "$or": [{"source_references": {"$exists": False}}, {"source_references": []}]},
                {"$addToSet": {"source_references": ref}},
            )
            counts["updated"] += result.modified_count
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        print(backfill(get_database(), args.paper_id, args.apply))
    except (PyMongoError, RuntimeError):
        raise SystemExit("Database operation failed; check the connection configuration.")
    finally:
        close_mongo_client()


if __name__ == "__main__":
    main()
