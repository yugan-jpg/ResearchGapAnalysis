from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import re
from typing import Any, Mapping
from uuid import uuid4

from pymongo import ASCENDING, DESCENDING
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, PyMongoError

from app.modules.claim_extraction.errors import ClaimStorageError, PaperNotFoundError
from app.schemas.claims import ClaimCategory, ClaimResponse, ExtractedClaim


@dataclass(frozen=True)
class ClaimChunk:
    chunk_id: str
    paper_id: str
    chunk_index: int
    page_number: int
    text: str


class ClaimRepository:
    """Read source chunks and persist verified claims with deterministic dedupe."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def ensure_paper(self, paper_id: str) -> None:
        try:
            if self.database.papers.find_one({"paper_id": paper_id}, {"_id": 1}) is None:
                raise PaperNotFoundError("The selected paper was not found.")
        except PaperNotFoundError:
            raise
        except PyMongoError as error:
            raise ClaimStorageError("The database could not confirm the selected paper.") from error

    def get_chunks(self, paper_id: str, model_name: str, force: bool, limit: int) -> list[ClaimChunk]:
        query: dict[str, Any] = {"paper_id": paper_id, "text": {"$type": "string", "$ne": ""}}
        if not force:
            query["$or"] = [
                {"claim_extraction_status": {"$ne": "completed"}},
                {"claim_extraction_model": {"$ne": model_name}},
            ]
        try:
            documents = self.database.paper_chunks.find(
                query,
                {"_id": 0, "chunk_id": 1, "paper_id": 1, "chunk_index": 1, "page_number": 1, "text": 1},
            ).sort([("chunk_index", ASCENDING)]).limit(limit)
            return [ClaimChunk(**document) for document in documents]
        except (PyMongoError, TypeError) as error:
            raise ClaimStorageError("The database could not read chunks for claim extraction.") from error

    def count_remaining_chunks(self, paper_id: str, model_name: str) -> int:
        try:
            return self.database.paper_chunks.count_documents(
                {
                    "paper_id": paper_id,
                    "text": {"$type": "string", "$ne": ""},
                    "$or": [
                        {"claim_extraction_status": {"$ne": "completed"}},
                        {"claim_extraction_model": {"$ne": model_name}},
                    ],
                }
            )
        except PyMongoError as error:
            raise ClaimStorageError("The database could not count remaining chunks.") from error

    def store_claims(
        self,
        chunk: ClaimChunk,
        claims: list[ExtractedClaim],
        model_name: str,
        *,
        claim_ids: list[str] | None = None,
        verification_results: Mapping[str, Any] | None = None,
        verification_cache_keys: Mapping[str, str] | None = None,
    ) -> tuple[int, int, list[ClaimResponse]]:
        """Persist source-grounded claims without replacing existing claims.

        ``claim_hash`` retains the Phase 4 per-chunk idempotency guarantee.
        ``claim_content_hash`` is intentionally non-unique: matching claims in
        different chunks keep both source references and are linked for review
        instead of silently being merged or deleted.
        """
        if claim_ids is not None and len(claim_ids) != len(claims):
            raise ClaimStorageError("Claim verification data does not match the extracted claim count.")
        inserted = 0
        duplicates = 0
        stored_claims: list[ClaimResponse] = []
        now = datetime.now(timezone.utc)
        for index, claim in enumerate(claims):
            claim_id = claim_ids[index] if claim_ids is not None else str(uuid4())
            claim_hash = self._claim_hash(chunk, claim)
            content_hash = self._claim_content_hash(chunk.paper_id, claim)
            verification = (verification_results or {}).get(claim_id)
            document = {
                "claim_id": claim_id,
                "claim_hash": claim_hash,
                "claim_content_hash": content_hash,
                "paper_id": chunk.paper_id,
                "chunk_id": chunk.chunk_id,
                "page_number": chunk.page_number,
                "claim_text": claim.claim_text,
                "category": claim.category.value,
                "supporting_evidence": claim.supporting_evidence,
                "extraction_model": model_name,
                "created_at": now,
            }
            document.update(self._verification_document(verification, now))
            cache_key = (verification_cache_keys or {}).get(claim_id)
            if cache_key:
                document["verification_cache_key"] = cache_key

            # An exact claim/evidence match in another chunk is retained as a
            # separate provenance record. It needs review rather than an
            # automatic dedupe because nearby scientific context may matter.
            previous = self.database.research_claims.find_one(
                {"paper_id": chunk.paper_id, "claim_content_hash": content_hash, "claim_hash": {"$ne": claim_hash}},
                {"_id": 0, "claim_id": 1},
                sort=[("created_at", ASCENDING)],
            )
            if previous:
                self._mark_exact_duplicate(document, previous["claim_id"])
            try:
                result = self.database.research_claims.update_one(
                    {"claim_hash": claim_hash},
                    {"$setOnInsert": document},
                    upsert=True,
                )
                if result.upserted_id is not None:
                    inserted += 1
                    stored_claims.append(ClaimResponse(**document))
                else:
                    duplicates += 1
                    existing = self.database.research_claims.find_one(
                        {"claim_hash": claim_hash},
                        self._claim_projection(),
                    )
                    if existing:
                        stored_claims.append(ClaimResponse(**existing))
            except DuplicateKeyError:
                duplicates += 1
            except PyMongoError as error:
                raise ClaimStorageError("The database could not store verified claims.") from error
        return inserted, duplicates, stored_claims

    def get_cached_verification(self, cache_key: str) -> dict[str, Any] | None:
        """Return only completed verification results safe to reuse."""
        try:
            return self.database.research_claims.find_one(
                {
                    "verification_cache_key": cache_key,
                    "verification_status": {"$in": ["accepted", "review", "rejected"]},
                },
                self._claim_projection(),
                sort=[("verified_at", DESCENDING)],
            )
        except PyMongoError as error:
            raise ClaimStorageError("The database could not read cached claim verification results.") from error

    def find_duplicate_candidates(
        self, paper_id: str, claim: ExtractedClaim, limit: int
    ) -> list[dict[str, str]]:
        """Conservatively preselect possible cross-chunk duplicates for Gemini.

        This is a lexical prefilter, not an automatic semantic deduplicator.
        Numeric mismatches are excluded before Gemini sees a candidate so that
        findings with different measurements or statistical values are kept.
        """
        if limit <= 0:
            return []
        try:
            documents = list(
                self.database.research_claims.find(
                    {"paper_id": paper_id, "category": claim.category.value},
                    {"_id": 0, "claim_id": 1, "chunk_id": 1, "claim_text": 1, "category": 1, "supporting_evidence": 1},
                )
                .sort("created_at", DESCENDING)
                .limit(250)
            )
        except PyMongoError as error:
            raise ClaimStorageError("The database could not compare existing claims for duplicates.") from error

        candidate_tokens = self._meaningful_tokens(claim.claim_text)
        candidate_numbers = self._number_tokens(claim.claim_text)
        scored: list[tuple[float, dict[str, str]]] = []
        for document in documents:
            existing_text = str(document.get("claim_text", ""))
            existing_numbers = self._number_tokens(existing_text)
            # Distinct quantities are materially meaningful in scientific
            # reporting and therefore cannot be automatic duplicate matches.
            if candidate_numbers != existing_numbers:
                continue
            existing_tokens = self._meaningful_tokens(existing_text)
            if not candidate_tokens or not existing_tokens:
                continue
            overlap = len(candidate_tokens & existing_tokens) / min(len(candidate_tokens), len(existing_tokens))
            if overlap < 0.5:
                continue
            scored.append(
                (
                    overlap,
                    {
                        "claim_id": str(document["claim_id"]),
                        "chunk_id": str(document.get("chunk_id", "")),
                        "claim_text": existing_text,
                        "category": str(document.get("category", "")),
                        "supporting_evidence": str(document.get("supporting_evidence", "")),
                    },
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in scored[:limit]]

    def list_unverified_claims(self, paper_id: str, limit: int, force: bool = False) -> list[dict[str, Any]]:
        """Read claims and original chunks for explicit, additive re-verification."""
        query: dict[str, Any] = {"paper_id": paper_id}
        if not force:
            query["$or"] = [
                {"verification_status": {"$exists": False}},
                {"verification_status": {"$in": ["pending", "unavailable"]}},
            ]
        try:
            claims = list(
                self.database.research_claims.find(query, self._claim_projection()).sort("created_at", ASCENDING).limit(limit)
            )
            if not claims:
                return []
            chunk_ids = [claim["chunk_id"] for claim in claims]
            chunks = {
                document["chunk_id"]: document
                for document in self.database.paper_chunks.find(
                    {"chunk_id": {"$in": chunk_ids}},
                    {"_id": 0, "chunk_id": 1, "text": 1},
                )
            }
            return [{**claim, "source_text": chunks.get(claim["chunk_id"], {}).get("text", "")} for claim in claims]
        except PyMongoError as error:
            raise ClaimStorageError("The database could not read claims for verification.") from error

    def update_claim_verification(
        self, claim_id: str, verification: Any, cache_key: str | None = None
    ) -> ClaimResponse | None:
        """Add verification metadata without altering scientific claim/source fields."""
        now = datetime.now(timezone.utc)
        update = self._verification_document(verification, now)
        if cache_key:
            update["verification_cache_key"] = cache_key
        try:
            self.database.research_claims.update_one({"claim_id": claim_id}, {"$set": update})
            document = self.database.research_claims.find_one({"claim_id": claim_id}, self._claim_projection())
            return ClaimResponse(**document) if document else None
        except PyMongoError as error:
            raise ClaimStorageError("The database could not update claim verification metadata.") from error

    def mark_chunk_complete(self, chunk_id: str, model_name: str, rejected_claims: int) -> None:
        try:
            self.database.paper_chunks.update_one(
                {"chunk_id": chunk_id},
                {
                    "$set": {
                        "claim_extraction_status": "completed",
                        "claim_extraction_model": model_name,
                        "claim_extracted_at": datetime.now(timezone.utc),
                        "claim_extraction_rejected_count": rejected_claims,
                    }
                },
            )
        except PyMongoError as error:
            raise ClaimStorageError("The database could not update claim extraction status.") from error

    def list_claims(self, paper_id: str, skip: int, limit: int) -> tuple[list[dict[str, Any]], int]:
        try:
            claims = list(self.database.research_claims.find({"paper_id": paper_id}, self._claim_projection()).sort("created_at", DESCENDING).skip(skip).limit(limit))
            return claims, self.database.research_claims.count_documents({"paper_id": paper_id})
        except PyMongoError as error:
            raise ClaimStorageError("The database could not retrieve claims.") from error

    @staticmethod
    def _claim_hash(chunk: ClaimChunk, claim: ExtractedClaim) -> str:
        value = "\x1f".join((chunk.paper_id, chunk.chunk_id, claim.category.value, claim.claim_text.strip(), claim.supporting_evidence.strip()))
        return sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _claim_content_hash(paper_id: str, claim: ExtractedClaim) -> str:
        value = "\x1f".join((paper_id, claim.category.value, claim.claim_text.strip(), claim.supporting_evidence.strip()))
        return sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _verification_document(verification: Any, now: datetime) -> dict[str, Any]:
        if verification is None:
            return {"verification_status": "pending"}
        document: dict[str, Any] = {
            "verification_status": verification.status,
            "verification_verdict": verification.verdict.value if verification.verdict else None,
            "evidence_supported": verification.evidence_supported,
            "numerical_consistency": verification.numerical_consistency,
            "completeness": verification.completeness,
            "category_correct": verification.category_correct,
            "verification_explanation": verification.explanation,
            "verifier_model": verification.verifier_model,
            "duplicate_of_claim_id": verification.duplicate_of_claim_id,
            "duplicate_type": verification.duplicate_type,
            "verification_attempted_at": now,
        }
        if verification.status in {"accepted", "review", "rejected"}:
            document["verified_at"] = now
        return document

    @staticmethod
    def _mark_exact_duplicate(document: dict[str, Any], previous_claim_id: str) -> None:
        document["duplicate_of_claim_id"] = previous_claim_id
        document["duplicate_type"] = "exact"
        if document.get("verification_status") != "rejected":
            document["verification_status"] = "review"
            document["verification_verdict"] = "review"
            existing_explanation = document.get("verification_explanation") or ""
            document["verification_explanation"] = (
                f"{existing_explanation} Exact claim and evidence match from another chunk retained for review."
            ).strip()

    @staticmethod
    def _claim_projection() -> dict[str, int]:
        return {
            "_id": 0,
            "claim_id": 1,
            "paper_id": 1,
            "chunk_id": 1,
            "page_number": 1,
            "claim_text": 1,
            "category": 1,
            "supporting_evidence": 1,
            "created_at": 1,
            "verification_status": 1,
            "verification_verdict": 1,
            "evidence_supported": 1,
            "numerical_consistency": 1,
            "completeness": 1,
            "category_correct": 1,
            "verification_explanation": 1,
            "verifier_model": 1,
            "verification_attempted_at": 1,
            "verified_at": 1,
            "duplicate_of_claim_id": 1,
            "duplicate_type": 1,
            "verification_cache_key": 1,
        }

    @staticmethod
    def _number_tokens(text: str) -> set[str]:
        return {token.replace(",", "") for token in re.findall(r"(?<![A-Za-z0-9])[+-]?\d[\d,]*(?:\.\d+)?%?", text)}

    @staticmethod
    def _meaningful_tokens(text: str) -> set[str]:
        stop_words = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "was", "were", "is", "are", "by", "that", "this"}
        return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2 and token not in stop_words}
