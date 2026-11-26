"""Queue explicitly chosen historical review claims, without calling an LLM."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pymongo.errors import PyMongoError
from app.db.mongodb import close_mongo_client, get_database


def queue_claims(database, paper_id, claim_ids, *, apply=False):
    result = {"requested": len(set(claim_ids)), "eligible": [], "skipped": [], "updated": 0}
    for claim_id in sorted(set(claim_ids)):
        record = database.research_claims.find_one({"paper_id": paper_id, "claim_id": claim_id})
        if (not record or record.get("verification_status") != "review"
                or record.get("reverification_required") or not record.get("verified_at")):
            result["skipped"].append(claim_id)
            continue
        evidence = record.get("supporting_evidence")
        chunk = database.paper_chunks.find_one(
            {"paper_id": paper_id, "chunk_id": record.get("chunk_id"), "page_number": record.get("page_number")},
            {"text": 1},
        )
        if not isinstance(evidence, str) or not evidence or not chunk or evidence not in chunk.get("text", ""):
            result["skipped"].append(claim_id)
            continue
        result["eligible"].append(claim_id)
        if not apply:
            continue
        fields = ("verification_status", "verification_verdict", "factual_verdict", "evidence_supported",
                  "numerical_consistency", "completeness", "category_correct", "verification_explanation",
                  "verifier_model", "verified_at", "verification_attempted_at", "duplicate_assessment",
                  "duplicate_candidate_id", "duplicate_type", "duplicate_of_claim_id")
        snapshot = {field: record[field] for field in fields if field in record}
        updated = database.research_claims.update_one(
            {"_id": record["_id"], "verification_status": "review",
             "reverification_required": {"$ne": True}, "verified_at": record["verified_at"]},
            {"$set": {"reverification_required": True, "reverification_token": str(uuid4())},
             "$push": {"verification_history": {"at": datetime.now(timezone.utc),
                      "reason": "explicitly queued for factual and duplicate re-verification", "previous": snapshot}}},
        )
        result["updated"] += updated.modified_count
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--claim-id", action="append", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        result = queue_claims(get_database(), args.paper_id, args.claim_id, apply=args.apply)
        print(f"Requested: {result['requested']}; eligible: {len(result['eligible'])}; updated: {result['updated']}")
        print("Eligible IDs:", ", ".join(result["eligible"]))
        print("Skipped IDs:", ", ".join(result["skipped"]))
    except (PyMongoError, RuntimeError):
        raise SystemExit("Database operation failed. Check the connection configuration.")
    finally:
        close_mongo_client()


if __name__ == "__main__":
    main()
