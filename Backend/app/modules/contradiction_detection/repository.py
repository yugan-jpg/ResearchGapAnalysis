"""MongoDB work queue and evidence snapshots for unordered cross-paper pairs."""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from uuid import uuid4

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from app.modules.claim_extraction.errors import ClaimStorageError, PaperNotFoundError
from app.modules.claim_extraction.selection import same_publication, trusted_claims
from app.schemas.comparisons import Candidate, ClaimSource, Comparison


def _digest(value) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256(data.encode("utf-8")).hexdigest()


class ComparisonRepository:
    def __init__(self, database):
        self.database = database
        self.collection = database.contradictions

    def enqueue(self, candidate: Candidate, settings) -> tuple[dict, bool]:
        if candidate.left.claim_id >= candidate.right.claim_id:
            raise ValueError("Comparison sources must be distinct and in canonical claim order.")
        data = candidate.model_dump(mode="json")
        pair_id = "pair:" + _digest([candidate.left.claim_id, candidate.right.claim_id])
        # Cache the scientific inputs and attribution, independent of ANN rank
        # or insignificant floating point variation in the cosine score.
        key = _digest([1, data["left"], data["right"], data["structural_signals"],
                       settings.llm_model_name, settings.verifier_model_name, settings.embedding_model_name])
        now = datetime.now(timezone.utc)
        pending = {
            **data, "comparison_id": pair_id, "input_key": key,
            "paper_ids": sorted([candidate.left.paper_id, candidate.right.paper_id]),
            "primary_model": settings.llm_model_name, "verifier_model": settings.verifier_model_name,
            "processing_status": "pending", "verification_status": "pending", "updated_at": now,
            "attempts": 0, "primary": None, "secondary": None, "relationship": None,
            "explanation": None, "mismatch_types": [], "failure_reason": None,
        }
        try:
            result = self.collection.update_one(
                {"_id": pair_id}, {"$setOnInsert": {**pending, "created_at": now}}, upsert=True,
            )
            inserted = result.upserted_id is not None
        except DuplicateKeyError:
            # A simultaneous discovery inserted the reverse-direction pair.
            inserted = False
        refreshed = self.collection.update_one(
            {"_id": pair_id, "$or": [{"input_key": {"$ne": key}}, {"processing_status": "stale"}]},
            {"$set": pending, "$unset": {"lease_id": "", "lease_until": ""}},
        )
        return self.collection.find_one({"_id": pair_id}), inserted or bool(refreshed.modified_count)

    def reserve(self, paper_id, other_paper_id, retry_failed, excluded, lease_seconds):
        now = datetime.now(timezone.utc)
        available = ["pending", "failed"] if retry_failed else ["pending"]
        paper_filter = {"$all": [paper_id, other_paper_id]} if other_paper_id else paper_id
        return self.collection.find_one_and_update(
            {"comparison_id": {"$exists": True}, "paper_ids": paper_filter, "_id": {"$nin": excluded},
             "$or": [{"processing_status": {"$in": available}},
                     {"processing_status": "processing", "lease_until": {"$lte": now}}]},
            {"$set": {"processing_status": "processing", "verification_status": "pending",
                      "failure_reason": None, "lease_id": uuid4().hex, "updated_at": now,
                      "lease_until": now + timedelta(seconds=lease_seconds)}, "$inc": {"attempts": 1}},
            sort=[("_id", 1)], return_document=ReturnDocument.AFTER,
        )

    def update_owned(self, record, changes, *, release=False):
        update = {"$set": {**changes, "updated_at": datetime.now(timezone.utc)}}
        if release:
            update["$unset"] = {"lease_id": "", "lease_until": ""}
        return self.collection.find_one_and_update(
            {"_id": record["_id"], "lease_id": record["lease_id"], "input_key": record["input_key"]},
            update, return_document=ReturnDocument.AFTER,
        )

    def _check_sources(self, records) -> list[bool]:
        requested = defaultdict(set)
        independent = {}
        for row in records:
            for side in ("left", "right"):
                requested[row[side]["paper_id"]].add(row[side]["claim_id"])
            papers = tuple(row["paper_ids"])
            if papers not in independent:
                try:
                    independent[papers] = not same_publication(self.database, *papers)
                except (PaperNotFoundError, ClaimStorageError):
                    # A missing paper or incomplete version chain invalidates this
                    # snapshot. Driver errors still reach the route as HTTP 503.
                    independent[papers] = False
        current = {}
        for paper_id, ids in requested.items():
            rows, _, _ = trusted_claims(self.database, paper_id=paper_id, claim_ids=list(ids), limit=len(ids))
            current.update({r["claim_id"]: ClaimSource.model_validate(r).model_dump() for r in rows})
        return [
            independent[tuple(row["paper_ids"])] and all(
                current.get(row[side]["claim_id"]) == row[side] for side in ("left", "right")
            ) for row in records
        ]

    def sources_current(self, record) -> bool:
        return self._check_sources([record])[0]

    def present(self, records) -> list[Comparison]:
        items = []
        for record, current in zip(records, self._check_sources(records), strict=True):
            public = {**record, "sources_current": current}
            if not current:
                public["verification_status"] = "stale"
            items.append(Comparison.model_validate(public))
        return items

    def list_results(self, paper_id, relationship, mismatch_type, processing_status, skip, limit):
        query = {"comparison_id": {"$exists": True}}
        for key, value in (("paper_ids", paper_id), ("relationship", relationship),
                           ("mismatch_types", mismatch_type), ("processing_status", processing_status)):
            if value is not None:
                query[key] = value
        records = self.collection.find(query).sort("_id", 1).skip(skip).limit(limit)
        return list(records), self.collection.count_documents(query)
