import re
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Protocol
from uuid import uuid4

from app.modules.claim_extraction.errors import ClaimStorageError
from app.modules.claim_extraction.gemini_verifier import (
    GeminiClaimVerifier,
    VerificationInput,
    VerificationOutcome,
)
from app.modules.claim_extraction.repository import ClaimChunk, ClaimRepository
from app.modules.claim_extraction.verifier_errors import GeminiVerificationError
from app.schemas.claims import ClaimResponse, ExtractedClaim, LLMClaimPayload, VerificationVerdict


class ClaimLLM(Protocol):
    settings: object

    def extract_claims(self, paper_id: str, chunk_id: str, page_number: int, chunk_text: str) -> LLMClaimPayload: ...


@dataclass(frozen=True)
class PreparedClaim:
    claim_id: str
    chunk: ClaimChunk
    claim: ExtractedClaim
    verification_input: VerificationInput
    verification_cache_key: str


@dataclass(frozen=True)
class ClaimExtractionResult:
    paper_id: str
    processed_chunks: int
    accepted_claims: int
    duplicate_claims: int
    rejected_claims: int
    remaining_chunks: int
    claims: list[ClaimResponse]


@dataclass(frozen=True)
class ClaimVerificationResult:
    paper_id: str
    processed_claims: int
    accepted_claims: int
    review_claims: int
    rejected_claims: int
    unavailable_claims: int
    claims: list[ClaimResponse]


class ClaimExtractionService:
    """Extract source-grounded claims, then optionally verify them in batches."""

    def __init__(
        self,
        repository: ClaimRepository,
        llm_client: ClaimLLM,
        model_name: str,
        verifier: GeminiClaimVerifier | None = None,
    ) -> None:
        self.repository = repository
        self.llm_client = llm_client
        self.model_name = model_name
        self.verifier = verifier

    def extract(self, paper_id: str, max_chunks: int, force: bool = False) -> ClaimExtractionResult:
        self.repository.ensure_paper(paper_id)
        chunks = self.repository.get_chunks(paper_id, self.model_name, force, max_chunks)
        rejected_claims = 0
        primary_rejections: dict[str, int] = {}
        prepared: list[PreparedClaim] = []

        # Complete primary extraction and deterministic source checks before
        # verifier calls, so Gemini receives bounded batches rather than one
        # request per candidate claim.
        for chunk in chunks:
            response = self.llm_client.extract_claims(chunk.paper_id, chunk.chunk_id, chunk.page_number, chunk.text)
            primary_rejections[chunk.chunk_id] = 0
            for claim in response.claims:
                if not self._has_exact_evidence(claim.supporting_evidence, chunk.text):
                    rejected_claims += 1
                    primary_rejections[chunk.chunk_id] += 1
                    continue
                prepared.append(self._prepare_claim(chunk, claim))

        # Include conservative candidates from the same extraction run as well
        # as previously stored claims, so two newly processed chunks can be
        # compared in the same bounded Gemini request.
        prepared = self._add_in_run_comparisons(prepared)
        outcomes = self._verify_prepared(prepared)
        prepared_by_chunk: dict[str, list[PreparedClaim]] = {chunk.chunk_id: [] for chunk in chunks}
        for item in prepared:
            prepared_by_chunk[item.chunk.chunk_id].append(item)

        accepted_claims = 0
        duplicate_claims = 0
        stored_claims: list[ClaimResponse] = []
        for chunk in chunks:
            chunk_prepared = prepared_by_chunk[chunk.chunk_id]
            inserted, duplicates, chunk_claims = self.repository.store_claims(
                chunk,
                [item.claim for item in chunk_prepared],
                self.model_name,
                claim_ids=[item.claim_id for item in chunk_prepared],
                verification_results={item.claim_id: outcomes[item.claim_id] for item in chunk_prepared},
                verification_cache_keys={item.claim_id: item.verification_cache_key for item in chunk_prepared},
            )
            # This preserves the existing Phase 4 counter meaning: a claim
            # which passed exact evidence validation was accepted by extraction.
            accepted_claims += inserted
            duplicate_claims += duplicates
            stored_claims.extend(chunk_claims)
            self.repository.mark_chunk_complete(chunk.chunk_id, self.model_name, primary_rejections[chunk.chunk_id])

        rejected_claims += sum(
            1 for item in prepared if outcomes[item.claim_id].verdict is VerificationVerdict.REJECTED
        )
        return ClaimExtractionResult(
            paper_id=paper_id,
            processed_chunks=len(chunks),
            accepted_claims=accepted_claims,
            duplicate_claims=duplicate_claims,
            rejected_claims=rejected_claims,
            remaining_chunks=self.repository.count_remaining_chunks(paper_id, self.model_name),
            claims=stored_claims,
        )

    def verify_existing(self, paper_id: str, limit: int, force: bool = False) -> ClaimVerificationResult:
        """Explicitly add metadata to historical claims without re-extracting."""
        self.repository.ensure_paper(paper_id)
        records = self.repository.list_unverified_claims(paper_id, limit, force)
        prepared: list[PreparedClaim] = []
        stored_claims: list[ClaimResponse] = []
        for record in records:
            source_text = str(record.get("source_text", ""))
            claim = ExtractedClaim(
                claim_text=record["claim_text"],
                category=record["category"],
                supporting_evidence=record["supporting_evidence"],
            )
            chunk = ClaimChunk(
                chunk_id=record["chunk_id"],
                paper_id=record["paper_id"],
                chunk_index=0,
                page_number=record["page_number"],
                text=source_text,
            )
            if not source_text or not self._has_exact_evidence(claim.supporting_evidence, source_text):
                # Never send detached historical evidence to Gemini. Only
                # additive verification metadata changes in this case.
                outcome = VerificationOutcome(
                    status="review",
                    verdict=VerificationVerdict.REVIEW,
                    evidence_supported=False,
                    explanation="Original source evidence is unavailable or no longer matches the stored claim; human review is required.",
                )
                updated = self.repository.update_claim_verification(record["claim_id"], outcome)
                if updated:
                    stored_claims.append(updated)
                continue
            prepared.append(self._prepare_claim(chunk, claim, claim_id=record["claim_id"]))

        outcomes = self._verify_prepared(prepared)
        for item in prepared:
            current = next(record for record in records if record["claim_id"] == item.claim_id)
            if (
                outcomes[item.claim_id].status == "unavailable"
                and current.get("verification_status") in {"accepted", "review", "rejected"}
            ):
                # A forced re-check cannot erase a prior completed verifier
                # assessment just because the provider is temporarily down.
                stored_claims.append(ClaimResponse(**current))
                continue
            updated = self.repository.update_claim_verification(
                item.claim_id, outcomes[item.claim_id], item.verification_cache_key
            )
            if updated:
                stored_claims.append(updated)
        return ClaimVerificationResult(
            paper_id=paper_id,
            processed_claims=len(records),
            accepted_claims=sum(item.verification_status == "accepted" for item in stored_claims),
            review_claims=sum(item.verification_status == "review" for item in stored_claims),
            rejected_claims=sum(item.verification_status == "rejected" for item in stored_claims),
            unavailable_claims=sum(item.verification_status == "unavailable" for item in stored_claims),
            claims=stored_claims,
        )

    def _prepare_claim(self, chunk: ClaimChunk, claim: ExtractedClaim, claim_id: str | None = None) -> PreparedClaim:
        try:
            candidates = self.repository.find_duplicate_candidates(
                chunk.paper_id, claim, self._verifier_comparison_limit()
            )
        except ClaimStorageError:
            # This is a non-destructive prefilter. A comparison read problem
            # must not discard otherwise source-grounded new extraction data.
            candidates = []
        verification_input = VerificationInput(
            claim_id=claim_id or str(uuid4()),
            paper_id=chunk.paper_id,
            chunk_id=chunk.chunk_id,
            page_number=chunk.page_number,
            claim_text=claim.claim_text,
            category=claim.category,
            supporting_evidence=claim.supporting_evidence,
            source_text=chunk.text,
            rule_completeness=self._is_likely_complete(claim.claim_text),
            rule_numerical_consistency=self._numbers_are_supported(
                claim.claim_text, claim.supporting_evidence
            ),
            comparison_candidates=tuple(candidates),
        )
        return PreparedClaim(
            claim_id=verification_input.claim_id,
            chunk=chunk,
            claim=claim,
            verification_input=verification_input,
            verification_cache_key=self._cache_key(verification_input),
        )

    def _verify_prepared(self, prepared: list[PreparedClaim]) -> dict[str, VerificationOutcome]:
        outcomes = {item.claim_id: VerificationOutcome.pending() for item in prepared}
        if not prepared or not self._verifier_enabled():
            return outcomes
        requests: list[PreparedClaim] = []
        for item in prepared:
            cached = self.repository.get_cached_verification(item.verification_cache_key)
            if cached:
                outcomes[item.claim_id] = self._cached_outcome(cached)
            else:
                requests.append(item)
        for offset in range(0, len(requests), self._verifier_batch_size()):
            batch = requests[offset : offset + self._verifier_batch_size()]
            try:
                batch_outcomes = self.verifier.verify_batch([item.verification_input for item in batch])  # type: ignore[union-attr]
            except GeminiVerificationError:
                batch_outcomes = {item.claim_id: VerificationOutcome.unavailable() for item in batch}
            for item in batch:
                outcomes[item.claim_id] = self._apply_rule_safeguards(
                    item.verification_input,
                    batch_outcomes.get(item.claim_id, VerificationOutcome.unavailable()),
                )
        return outcomes

    def _add_in_run_comparisons(self, prepared: list[PreparedClaim]) -> list[PreparedClaim]:
        if self._verifier_comparison_limit() <= 0:
            return prepared
        result: list[PreparedClaim] = []
        for item in prepared:
            candidates = list(item.verification_input.comparison_candidates)
            known_ids = {candidate["claim_id"] for candidate in candidates}
            for other in prepared:
                if other.claim_id == item.claim_id or other.claim_id in known_ids:
                    continue
                if not self._could_be_semantic_duplicate(item.claim, other.claim):
                    continue
                candidates.append(
                    {
                        "claim_id": other.claim_id,
                        "chunk_id": other.chunk.chunk_id,
                        "claim_text": other.claim.claim_text,
                        "category": other.claim.category.value,
                        "supporting_evidence": other.claim.supporting_evidence,
                    }
                )
                known_ids.add(other.claim_id)
                if len(candidates) >= self._verifier_comparison_limit():
                    break
            verification_input = replace(item.verification_input, comparison_candidates=tuple(candidates))
            result.append(
                replace(
                    item,
                    verification_input=verification_input,
                    verification_cache_key=self._cache_key(verification_input),
                )
            )
        return result

    @staticmethod
    def _apply_rule_safeguards(
        verification_input: VerificationInput, outcome: VerificationOutcome
    ) -> VerificationOutcome:
        """Keep deterministic checks effective even with a substituted client."""
        if outcome.verdict is VerificationVerdict.REJECTED or outcome.status in {"pending", "unavailable"}:
            return outcome
        needs_review = (
            not verification_input.rule_completeness
            or not verification_input.rule_numerical_consistency
            or outcome.evidence_supported is False
            or outcome.numerical_consistency is False
            or outcome.completeness is False
            or outcome.category_correct is False
            or bool(outcome.duplicate_of_claim_id)
        )
        if not needs_review:
            return outcome
        notes: list[str] = []
        if not verification_input.rule_completeness:
            notes.append("Rule check: the claim appears incomplete or truncated.")
        if not verification_input.rule_numerical_consistency:
            notes.append("Rule check: a number in the claim is absent from the quoted evidence.")
        if outcome.duplicate_of_claim_id:
            notes.append("Potential semantic duplicate retained for human review.")
        explanation = " ".join(part for part in [outcome.explanation or "", *notes] if part).strip()
        return replace(
            outcome,
            status="review",
            verdict=VerificationVerdict.REVIEW,
            numerical_consistency=(
                outcome.numerical_consistency and verification_input.rule_numerical_consistency
            ),
            completeness=outcome.completeness and verification_input.rule_completeness,
            explanation=explanation or "Verification requires review.",
        )

    def _verifier_enabled(self) -> bool:
        return bool(self.verifier and getattr(self.verifier.settings, "verifier_enabled", False))

    def _verifier_batch_size(self) -> int:
        return int(getattr(self.verifier.settings, "verifier_batch_size", 1)) if self.verifier else 1

    def _verifier_comparison_limit(self) -> int:
        return int(getattr(self.verifier.settings, "verifier_comparison_limit", 0)) if self.verifier else 0

    def _cache_key(self, verification_input: VerificationInput) -> str:
        model = getattr(self.verifier.settings, "gemini_model_name", "") if self.verifier else ""
        return sha256(f"{model}\x1f{verification_input.cache_key}".encode("utf-8")).hexdigest()

    @staticmethod
    def _cached_outcome(document: dict[str, object]) -> VerificationOutcome:
        raw_verdict = document.get("verification_verdict")
        return VerificationOutcome(
            status=str(document.get("verification_status", "review")),
            verdict=VerificationVerdict(raw_verdict) if raw_verdict else None,
            evidence_supported=document.get("evidence_supported") if isinstance(document.get("evidence_supported"), bool) else None,
            numerical_consistency=document.get("numerical_consistency") if isinstance(document.get("numerical_consistency"), bool) else None,
            completeness=document.get("completeness") if isinstance(document.get("completeness"), bool) else None,
            category_correct=document.get("category_correct") if isinstance(document.get("category_correct"), bool) else None,
            explanation=str(document.get("verification_explanation") or "Cached Gemini verification result."),
            verifier_model=str(document.get("verifier_model") or ""),
            duplicate_of_claim_id=str(document["duplicate_of_claim_id"]) if document.get("duplicate_of_claim_id") else None,
            duplicate_type=str(document["duplicate_type"]) if document.get("duplicate_type") else None,
        )

    @staticmethod
    def _has_exact_evidence(evidence: str, chunk_text: str) -> bool:
        return bool(evidence and evidence in chunk_text)

    @staticmethod
    def _is_likely_complete(claim_text: str) -> bool:
        text = claim_text.strip()
        if len(text) < 16 or text.endswith(("...", "…", ":", ";", ",", "-", "–", "—", "(", "[")):
            return False
        return not bool(re.search(r"\b(?:and|or|but|because|although|while)$", text.lower()))

    @staticmethod
    def _numbers_are_supported(claim_text: str, supporting_evidence: str) -> bool:
        return ClaimExtractionService._number_tokens(claim_text).issubset(
            ClaimExtractionService._number_tokens(supporting_evidence)
        )

    @staticmethod
    def _could_be_semantic_duplicate(left: ExtractedClaim, right: ExtractedClaim) -> bool:
        if left.category is not right.category:
            return False
        if ClaimExtractionService._number_tokens(left.claim_text) != ClaimExtractionService._number_tokens(right.claim_text):
            return False
        left_tokens = ClaimExtractionService._meaningful_tokens(left.claim_text)
        right_tokens = ClaimExtractionService._meaningful_tokens(right.claim_text)
        if not left_tokens or not right_tokens:
            return False
        return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens)) >= 0.5

    @staticmethod
    def _number_tokens(text: str) -> set[str]:
        return {
            token.replace(",", "")
            for token in re.findall(r"(?<![A-Za-z0-9])[+-]?\d[\d,]*(?:\.\d+)?%?", text)
        }

    @staticmethod
    def _meaningful_tokens(text: str) -> set[str]:
        stop_words = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "was", "were", "is", "are", "by", "that", "this"}
        return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2 and token not in stop_words}
