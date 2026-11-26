"""Queue older repaired claims for re-verification without repeating repair."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymongo.errors import PyMongoError
from app.db.mongodb import close_mongo_client, get_database


_RELATIONSHIP_REASON = "Relationship is not deterministically exact and requires independent re-verification."


def recover(database, paper_id: str, apply: bool = False) -> dict:
    """Mark only legacy repairs whose latest completed check predates repair."""
    result = {"examined": 0, "eligible": 0, "updated": 0, "claim_ids": []}
    query = {"paper_id": paper_id, "verification_status": "review",
             "reverification_required": {"$ne": True},
             "duplicate_repair_history": {"$exists": True}}
    for record in database.research_claims.find(query):
        result["examined"] += 1
        history = record.get("duplicate_repair_history")
        if not isinstance(history, list) or not history:
            continue
        last = history[-1]
        if not isinstance(last, dict):
            continue
        reason = last.get("reason", "")
        requires_review = (reason == _RELATIONSHIP_REASON
                           or (reason.startswith("Possible overlap with claim ")
                               and reason.endswith("independent re-verification required.")))
        # Only the historical script omitted the explicit marker. Newer
        # repairs record the flag in history and are already queued.
        if not requires_review or "requires_reverification" in last:
            continue
        repair_at = last.get("at")
        verified_at = record.get("verified_at")
        attempted_at = record.get("verification_attempted_at")
        if not isinstance(repair_at, datetime) or not isinstance(verified_at, datetime):
            continue
        if verified_at >= repair_at or (isinstance(attempted_at, datetime) and attempted_at > repair_at):
            continue
        result["eligible"] += 1
        result["claim_ids"].append(record["claim_id"])
        if not apply:
            continue
        fields = ("verification_status", "verification_verdict", "evidence_supported", "numerical_consistency",
                  "completeness", "category_correct", "verification_explanation", "verifier_model",
                  "verified_at", "verification_attempted_at", "verification_failure_reason")
        snapshot = {field: record[field] for field in fields if field in record}
        # Guard the last repair and verification timestamps against changes
        # between previewing the document and applying the flag.
        match = {"_id": record["_id"], "verification_status": "review",
                 "reverification_required": {"$ne": True}, "duplicate_repair_history": history,
                 "verified_at": verified_at, "verification_attempted_at": attempted_at}
        write = database.research_claims.update_one(
            match,
            {"$set": {"reverification_required": True, "reverification_token": str(uuid4())},
             "$push": {"verification_history": {
                 "at": datetime.now(timezone.utc), "reason": "legacy repair queued for re-verification",
                 "previous": snapshot,
             }}},
        )
        result["updated"] += write.modified_count
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        result = recover(get_database(), args.paper_id, args.apply)
        print(f"Examined: {result['examined']}; eligible: {result['eligible']}; updated: {result['updated']}")
        for claim_id in result["claim_ids"]:
            print(claim_id)
    except (PyMongoError, RuntimeError):
        raise SystemExit("Database operation failed. Check the connection configuration.")
    finally:
        close_mongo_client()


if __name__ == "__main__":
    main()
