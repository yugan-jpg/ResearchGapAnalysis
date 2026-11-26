import re
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Protocol
from uuid import uuid4

from app.modules.claim_extraction.errors import ClaimStorageError
from app.modules.claim_extraction.groq_verifier import (
    GroqClaimVerifier,
    VerificationInput,
    VerificationOutcome,
)
from app.modules.claim_extraction.repository import ClaimChunk, ClaimRepository
from app.modules.claim_extraction.resolution import resolve_paper
from app.modules.claim_extraction.quality import inequality_tokens, is_factually_rejected, measurement_tokens, source_total_warning
from app.modules.claim_extraction.verifier_errors import GroqVerificationError
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
    generated_claims: int = 0
    stored_claims: int = 0
    extraction_rejections: int = 0
    exact_duplicates_skipped: int = 0
    semantic_duplicates_flagged: int = 0
    partial_overlaps_flagged: int = 0
    verification_accepted: int = 0
    verification_review: int = 0
    verification_rejected: int = 0
    verification_unavailable: int = 0


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
        verifier: GroqClaimVerifier | None = None,
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
        generated_claims = 0

        # Complete primary extraction and deterministic source checks before
        # verifier calls, so Groq receives bounded batches rather than one
        # request per candidate claim.
        for chunk in chunks:
            response = self.llm_client.extract_claims(chunk.paper_id, chunk.chunk_id, chunk.page_number, chunk.text)
            generated_claims += len(response.claims)
            primary_rejections[chunk.chunk_id] = 0
            for claim in response.claims:
                if not self._has_exact_evidence(claim.supporting_evidence, chunk.text) or not self._is_likely_complete(claim.claim_text):
                    rejected_claims += 1
                    primary_rejections[chunk.chunk_id] += 1
                    continue
                prepared.append(self._prepare_claim(chunk, claim))

        # Include conservative candidates from the same extraction run as well
        # as previously stored claims, so two newly processed chunks can be
        # compared in the same bounded Groq request.
        prepared = self._add_in_run_comparisons(prepared)
        outcomes = self._verify_prepared(prepared)
        outcomes = self._safe_duplicate_targets(prepared, outcomes)
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

        if stored_claims:
            resolve_paper(self.repository.database, paper_id, apply=True,
                          affected_ids={claim.claim_id for claim in stored_claims})
            stored_claims = [ClaimResponse(**record) for claim in stored_claims
                             if (record := self.repository.get_claim_for_duplicate(claim.claim_id))]

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
            generated_claims=generated_claims,
            stored_claims=accepted_claims,
            extraction_rejections=sum(primary_rejections.values()),
            exact_duplicates_skipped=duplicate_claims,
            semantic_duplicates_flagged=sum(c.duplicate_type == "semantic" for c in stored_claims),
            partial_overlaps_flagged=sum(c.duplicate_type in {"partial_overlap", "containment"} for c in stored_claims),
            verification_accepted=sum(c.verification_status == "accepted" for c in stored_claims),
            verification_review=sum(c.verification_status == "review" for c in stored_claims),
            verification_rejected=sum(c.verification_status == "rejected" for c in stored_claims),
            verification_unavailable=sum(c.verification_status == "unavailable" for c in stored_claims),
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
                # Never send detached historical evidence to Groq. Only
                # additive verification metadata changes in this case.
                outcome = VerificationOutcome(
                    status="review",
                    verdict=VerificationVerdict.REVIEW,
                    evidence_supported=False,
                    explanation="Original source evidence is unavailable or no longer matches the stored claim; human review is required.",
                )
                if record.get("reverification_required"):
                    # A missing original passage cannot be verified. Keep the
                    # repair request queued and retain the historical result.
                    stored_claims.append(ClaimResponse(**record))
                    continue
                updated = self.repository.update_claim_verification(record["claim_id"], outcome, previous_record=record)
                if updated:
                    stored_claims.append(updated)
                continue
            prepared.append(self._prepare_claim(chunk, claim, claim_id=record["claim_id"]))

        queued_ids = {record["claim_id"] for record in records if record.get("reverification_required")}
        outcomes = self._verify_prepared(prepared, bypass_cache=force, bypass_cache_ids=queued_ids)
        outcomes = self._safe_duplicate_targets(prepared, outcomes)
        for item in prepared:
            current = next(record for record in records if record["claim_id"] == item.claim_id)
            if (
                outcomes[item.claim_id].status in {"unavailable", "pending"}
                and (current.get("reverification_required") or current.get("verification_status") in {"accepted", "review", "rejected"})
            ):
                # A forced re-check cannot erase a prior completed verifier
                # assessment just because the provider is temporarily down.
                stored_claims.append(ClaimResponse(**current))
                continue
            updated = self.repository.update_claim_verification(
                item.claim_id, outcomes[item.claim_id], item.verification_cache_key, previous_record=current
            )
            if updated:
                stored_claims.append(updated)
        successful_ids = {item.claim_id for item in prepared
                          if outcomes[item.claim_id].status in {"accepted", "review", "rejected"}}
        if successful_ids:
            resolve_paper(self.repository.database, paper_id, apply=True, affected_ids=successful_ids)
            stored_claims = [ClaimResponse(**record) for claim in stored_claims
                             if (record := self.repository.get_claim_for_duplicate(claim.claim_id))]
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
                chunk.paper_id, claim, self._verifier_comparison_limit(), exclude_claim_id=claim_id
            )
        except ClaimStorageError:
            # This is a non-destructive prefilter. A comparison read problem
            # must not discard otherwise source-grounded new extraction data.
            candidates = []
        neighbor_context = self.repository.neighboring_context(chunk)
        numbers_supported = self._numbers_are_supported(claim.claim_text, claim.supporting_evidence)
        if not numbers_supported and re.search(r"\b(?:same period|same timeframe|same time period|same year)\b", claim.supporting_evidence, re.I):
            # The verifier still must resolve the reference and judge support;
            # this merely permits a number found in bounded same-page context.
            context = chunk.text + " " + neighbor_context
            numbers_supported = (measurement_tokens(claim.claim_text).issubset(measurement_tokens(context))
                                 and inequality_tokens(claim.claim_text).issubset(inequality_tokens(context)))
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
            rule_numerical_consistency=numbers_supported,
            comparison_candidates=tuple(candidates),
            source_warning=source_total_warning(chunk.text),
            neighboring_context=neighbor_context,
        )
        return PreparedClaim(
            claim_id=verification_input.claim_id,
            chunk=chunk,
            claim=claim,
            verification_input=verification_input,
            verification_cache_key=self._cache_key(verification_input),
        )

    def _verify_prepared(self, prepared: list[PreparedClaim], bypass_cache: bool = False,
                         bypass_cache_ids: set[str] | None = None) -> dict[str, VerificationOutcome]:
        outcomes = {item.claim_id: VerificationOutcome.pending() for item in prepared}
        if not prepared or not self._verifier_enabled():
            return outcomes
        requests: list[PreparedClaim] = []
        for item in prepared:
            cached = None if bypass_cache or item.claim_id in (bypass_cache_ids or set()) else self.repository.get_cached_verification(item.verification_cache_key)
            if cached:
                outcomes[item.claim_id] = self._cached_outcome(cached)
            else:
                requests.append(item)
        for offset in range(0, len(requests), self._verifier_batch_size()):
            batch = requests[offset : offset + self._verifier_batch_size()]
            try:
                batch_outcomes = self.verifier.verify_batch([item.verification_input for item in batch])  # type: ignore[union-attr]
            except GroqVerificationError as error:
                reason = {
                    "GroqRateLimitError": "rate_limited",
                    "GroqTimeoutError": "timeout",
                    "GroqAuthenticationError": "authentication_failed",
                    "GroqConfigurationError": "configuration_missing",
                    "GroqResponseValidationError": "malformed_response",
                }.get(type(error).__name__, "provider_unavailable")
                batch_outcomes = {item.claim_id: VerificationOutcome.unavailable(reason) for item in batch}
            for item in batch:
                outcomes[item.claim_id] = batch_outcomes.get(item.claim_id, VerificationOutcome.unavailable())
        # A candidate in the same extraction may have just been rejected.
        # Apply similarity safeguards only after every batch's verdict is known.
        for item in prepared:
            candidates = []
            for candidate in item.verification_input.comparison_candidates:
                candidate_outcome = outcomes.get(candidate["claim_id"])
                if candidate_outcome is not None and (
                    candidate_outcome.status == "rejected"
                    or candidate_outcome.factual_verdict is VerificationVerdict.REJECTED
                    or candidate_outcome.evidence_supported is False
                ):
                    continue
                if not is_factually_rejected(candidate):
                    candidates.append(candidate)
            inputs = replace(item.verification_input, comparison_candidates=tuple(candidates))
            outcomes[item.claim_id] = self._apply_rule_safeguards(inputs, outcomes[item.claim_id])
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
                if not (item.claim.supporting_evidence == other.claim.supporting_evidence or self._could_be_semantic_duplicate(item.claim, other.claim)):
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

    def _safe_duplicate_targets(self, prepared, outcomes):
        """Only link to a currently verified, same-paper, terminal target.

        Other proposed relationships remain review items, not canonical links.
        This intentionally does not infer equivalence from vector similarity.
        """
        for item in prepared:
            outcome = outcomes[item.claim_id]
            target_id = outcome.duplicate_of_claim_id or (
                outcome.duplicate_candidate_id if outcome.duplicate_assessment == "unresolved" else None
            )
            if not target_id:
                continue
            target = self.repository.get_claim_for_duplicate(target_id) if target_id != item.claim_id else None
            in_run_target = outcomes.get(target_id)
            rejected = (is_factually_rejected(target or {}) or bool(in_run_target and (
                in_run_target.status == "rejected" or in_run_target.factual_verdict is VerificationVerdict.REJECTED
                or in_run_target.evidence_supported is False)))
            if rejected and outcome.duplicate_assessment == "unresolved":
                others = []
                for candidate in item.verification_input.comparison_candidates:
                    cid = candidate["claim_id"]
                    if cid == target_id:
                        continue
                    other_outcome = outcomes.get(cid)
                    other = self.repository.get_claim_for_duplicate(cid) if other_outcome is None else None
                    if other_outcome is not None:
                        if (other_outcome.status == "rejected" or other_outcome.factual_verdict is VerificationVerdict.REJECTED
                                or other_outcome.evidence_supported is False):
                            continue
                    elif other is not None and is_factually_rejected(other):
                        continue
                    others.append(cid)
                if (not others and outcome.factual_verdict is VerificationVerdict.ACCEPTED
                        and all(value is True for value in (outcome.evidence_supported, outcome.numerical_consistency,
                                                            outcome.completeness, outcome.category_correct))
                        and not item.verification_input.source_warning):
                    outcomes[item.claim_id] = replace(
                        outcome, status="accepted", verdict=VerificationVerdict.ACCEPTED,
                        duplicate_of_claim_id=None, duplicate_type=None,
                        duplicate_assessment="rejected_candidate_excluded",
                        duplicate_explanation="The sole similarity candidate was factually rejected; no other supplied candidate remains.",
                    )
                    continue
                if others:
                    outcomes[item.claim_id] = replace(outcome, duplicate_of_claim_id=None, duplicate_candidate_id=others[0])
                    continue
            eligible = bool(target and target.get("paper_id") == item.chunk.paper_id
                            and target.get("verification_status") == "accepted"
                            and target.get("evidence_supported") is True
                            and target.get("numerical_consistency") is True
                            and target.get("completeness") is True
                            and target.get("category_correct") is True
                            and not target.get("duplicate_of_claim_id")
                            and self.repository.has_valid_source(target))
            if eligible and outcome.duplicate_type == "semantic":
                # Do not promote scientific equivalence using model or cosine
                # scores alone. Preserve differences in entities, direction,
                # negation and numeric tokens for review.
                source_tokens = self._meaningful_tokens(item.claim.claim_text)
                target_tokens = self._meaningful_tokens(target.get("claim_text", ""))
                negation = lambda text: bool(re.search(r"\b(?:not|no|without|neither|never)\b", text, re.I))
                eligible = (bool(source_tokens) and source_tokens == target_tokens
                            and measurement_tokens(item.claim.claim_text) == measurement_tokens(target.get("claim_text", ""))
                            and negation(item.claim.claim_text) == negation(target.get("claim_text", "")))
            if outcome.duplicate_assessment in {"partial_overlap", "containment", "related_distinct"}:
                outcomes[item.claim_id] = replace(outcome, duplicate_of_claim_id=None)
            elif not eligible:
                outcomes[item.claim_id] = replace(
                    outcome, duplicate_of_claim_id=None, duplicate_candidate_id=outcome.duplicate_candidate_id or target_id,
                    status="review" if outcome.status not in {"rejected", "unavailable", "pending"} else outcome.status,
                    verdict=VerificationVerdict.REVIEW if outcome.status not in {"rejected", "unavailable", "pending"} else outcome.verdict,
                    duplicate_explanation=(outcome.duplicate_explanation or "Potential related claim requires independent review.")
                    + " No eligible terminal canonical target was established.",
                )
        return outcomes

    @staticmethod
    def _apply_rule_safeguards(
        verification_input: VerificationInput, outcome: VerificationOutcome
    ) -> VerificationOutcome:
        """Keep deterministic checks effective even with a substituted client."""
        if outcome.verdict is VerificationVerdict.REJECTED or outcome.status in {"pending", "unavailable"}:
            return outcome
        # Shared evidence alone is not sufficient for a duplicate. A high
        # content-word overlap additionally makes active/passive restatements
        # candidates for review, without removing either atomic claim.
        if not outcome.duplicate_of_claim_id and not outcome.duplicate_assessment:
            left = ClaimExtractionService._meaningful_tokens(verification_input.claim_text)
            for candidate in verification_input.comparison_candidates:
                if is_factually_rejected(candidate):
                    continue
                right = ClaimExtractionService._meaningful_tokens(candidate.get("claim_text", ""))
                same_evidence = candidate.get("supporting_evidence", "").strip() == verification_input.supporting_evidence.strip()
                overlap = len(left & right) / max(len(left), len(right)) if left and right else 0
                if same_evidence and overlap >= 0.65:
                    outcome = replace(
                        outcome,
                        duplicate_of_claim_id=candidate["claim_id"],
                        duplicate_type="partial_overlap", duplicate_candidate_id=candidate["claim_id"],
                        duplicate_assessment="unresolved",
                        duplicate_similarity=candidate.get("similarity"),
                        duplicate_explanation="Same source passage and substantial wording overlap suggest a possible paraphrase; compare the methods, populations and findings before treating them as equivalent.",
                    )
                    break
        needs_review = (
            not verification_input.rule_completeness
            or not verification_input.rule_numerical_consistency
            or outcome.evidence_supported is False
            or outcome.numerical_consistency is False
            or outcome.completeness is False
            or outcome.category_correct is False
            or outcome.duplicate_assessment in {"unresolved", "confirmed_equivalent"}
            or bool(verification_input.source_warning)
        )
        factual_verdict = outcome.factual_verdict or outcome.verdict
        factual_problem = (not verification_input.rule_completeness or not verification_input.rule_numerical_consistency
                           or outcome.evidence_supported is False or outcome.numerical_consistency is False
                           or outcome.completeness is False or outcome.category_correct is False
                           or bool(verification_input.source_warning))
        if factual_problem and factual_verdict is not VerificationVerdict.REJECTED:
            factual_verdict = VerificationVerdict.REVIEW
        if not needs_review:
            return replace(outcome, factual_verdict=factual_verdict)
        notes: list[str] = []
        if not verification_input.rule_completeness:
            notes.append("Rule check: the claim appears incomplete or truncated.")
        if not verification_input.rule_numerical_consistency:
            notes.append("Rule check: a number in the claim is absent from the quoted evidence.")
        if outcome.duplicate_of_claim_id:
            notes.append("Potential semantic duplicate retained for human review.")
        if verification_input.source_warning:
            notes.append(verification_input.source_warning)
        explanation = outcome.explanation or ""
        for note in notes:
            if note not in explanation:
                explanation = f"{explanation} {note}".strip()
        return replace(
            outcome,
            status="review",
            verdict=VerificationVerdict.REVIEW,
            factual_verdict=factual_verdict,
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
        model = getattr(self.verifier.settings, "verifier_model_name", "") if self.verifier else ""
        return sha256(f"{model}\x1f{verification_input.cache_key}".encode("utf-8")).hexdigest()

    @staticmethod
    def _cached_outcome(document: dict[str, object]) -> VerificationOutcome:
        raw_verdict = document.get("verification_verdict")
        raw_factual = document.get("factual_verdict")
        return VerificationOutcome(
            status=str(document.get("verification_status", "review")),
            verdict=VerificationVerdict(raw_verdict) if raw_verdict else None,
            factual_verdict=VerificationVerdict(raw_factual) if raw_factual else None,
            evidence_supported=document.get("evidence_supported") if isinstance(document.get("evidence_supported"), bool) else None,
            numerical_consistency=document.get("numerical_consistency") if isinstance(document.get("numerical_consistency"), bool) else None,
            completeness=document.get("completeness") if isinstance(document.get("completeness"), bool) else None,
            category_correct=document.get("category_correct") if isinstance(document.get("category_correct"), bool) else None,
            explanation=str(document.get("verification_explanation") or "Cached Groq verification result."),
            verifier_model=str(document.get("verifier_model") or ""),
            duplicate_of_claim_id=str(document["duplicate_of_claim_id"]) if document.get("duplicate_of_claim_id") else None,
            duplicate_type=str(document["duplicate_type"]) if document.get("duplicate_type") else None,
            duplicate_similarity=document.get("duplicate_similarity"),
            duplicate_explanation=document.get("duplicate_explanation"),
            duplicate_candidate_id=document.get("duplicate_candidate_id"),
            duplicate_assessment=document.get("duplicate_assessment"),
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
        return (measurement_tokens(claim_text).issubset(measurement_tokens(supporting_evidence))
                and inequality_tokens(claim_text).issubset(inequality_tokens(supporting_evidence)))

    @staticmethod
    def _could_be_semantic_duplicate(left: ExtractedClaim, right: ExtractedClaim) -> bool:
        if left.category is not right.category:
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
