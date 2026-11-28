from app.modules.claim_extraction.errors import (
    ClaimExtractionError, ClaimResponseValidationError, LLMAuthenticationError, LLMConfigurationError,
    LLMProviderError, LLMRateLimitError, LLMTimeoutError,
)
from app.modules.claim_extraction.repository import ClaimRepository
from app.modules.claim_extraction.selection import same_publication
from app.schemas.comparisons import Candidate, DetectionResponse, Judgment, Mismatch, Relationship


# Only these codes and HTTP statuses leave the provider boundary.
FAILURES = (
    (LLMRateLimitError, "rate_limited", 429),
    (LLMTimeoutError, "provider_timeout", 504),
    (LLMAuthenticationError, "provider_authentication", 503),
    (LLMConfigurationError, "missing_configuration", 503),
    (ClaimResponseValidationError, "malformed_response", 502),
    (LLMProviderError, "provider_unavailable", 502),
)


def reconcile(primary: Judgment, secondary: Judgment) -> dict:
    same_relationship = primary.relationship == secondary.relationship
    relationship = primary.relationship if same_relationship else Relationship.INSUFFICIENT
    aligned = (same_relationship and primary.comparable_scope == secondary.comparable_scope
               and set(primary.mismatch_types) == set(secondary.mismatch_types))
    signals = set(primary.mismatch_types) | set(secondary.mismatch_types)
    findings = {Relationship.AGREEMENT, Relationship.DIRECT, Relationship.PARTIAL}
    # Preserve both raw judgments when their own evidence caveats undermine
    # an apparent conclusion. A shared, explicitly bounded subset can still
    # support a partial contradiction across otherwise different populations.
    unsupported = (relationship in findings and bool(signals & {Mismatch.CLAIM_SCOPE, Mismatch.EVIDENCE_METHOD})
                   or relationship == Relationship.DIRECT and Mismatch.POPULATION_CONTEXT in signals)
    if unsupported:
        relationship = Relationship.INSUFFICIENT
        explanation = "A reported difference in scope, method or population limits the finding. Review both passages."
    elif aligned:
        explanation = secondary.explanation
    else:
        explanation = "The independent assessments differ. Review both explanations and their cited passages."
    return {
        "primary": primary.model_dump(mode="json"), "secondary": secondary.model_dump(mode="json"),
        "relationship": relationship.value,
        "verification_status": "verified" if aligned and relationship != Relationship.INSUFFICIENT else "review",
        "mismatch_types": sorted(x.value for x in set(primary.mismatch_types) & set(secondary.mismatch_types)),
        "explanation": explanation, "processing_status": "completed",
    }


class ComparisonService:
    def __init__(self, repository, evaluator, settings):
        self.repository = repository
        self.evaluator = evaluator
        self.settings = settings

    def run(self, request, limit: int) -> tuple[DetectionResponse, int]:
        if request.other_paper_id:
            if same_publication(self.repository.database, request.paper_id, request.other_paper_id):
                raise ValueError("Choose two independent publications.")
        else:
            ClaimRepository(self.repository.database).ensure_paper(request.paper_id)
        records = []
        attempted_ids = []
        http_status = 200
        # Cover configured transport attempts, quota waits and malformed-output
        # retries. An abandoned reservation becomes eligible again after expiry.
        s = self.settings
        longest_call = max(s.llm_timeout_seconds, s.verifier_timeout_seconds)
        attempts = max(s.llm_max_retries, s.verifier_max_retries) + 1
        lease_seconds = (2 + s.verifier_malformed_retries) * attempts * (
            longest_call + 2 * s.groq_max_throttle_wait_seconds + 40
        ) + 60
        for _ in range(limit):
            record = self.repository.reserve(request.paper_id, request.other_paper_id,
                                             request.retry_failed, attempted_ids, lease_seconds)
            if record is None:
                break
            attempted_ids.append(record["_id"])
            try:
                record = self._evaluate(record)
            except ClaimExtractionError as error:
                failure = next(((code, status) for cls, code, status in FAILURES if isinstance(error, cls)), None)
                if failure is None:
                    raise
                reason, http_status = failure
                record = self.repository.update_owned(record, {
                    "processing_status": "failed", "verification_status": "unavailable", "failure_reason": reason,
                }, release=True)
            if record:
                records.append(record)
            if http_status != 200:
                break  # Do not spend quota on the rest of a batch after a provider failure.
        return DetectionResponse(
            attempted=len(attempted_ids), completed=sum(r["processing_status"] == "completed" for r in records),
            failed=sum(r["processing_status"] == "failed" for r in records),
            stale=sum(r["processing_status"] == "stale" for r in records), comparisons=self.repository.present(records),
        ), http_status

    def _evaluate(self, record):
        if (not self.repository.sources_current(record)
                or record["primary_model"] != self.settings.llm_model_name
                or record["verifier_model"] != self.settings.verifier_model_name):
            return self._stale(record)
        if not self.settings.verifier_enabled or not self.settings.verifier_model_name:
            raise LLMConfigurationError("Independent comparison verification requires a configured verifier.")
        if self.settings.llm_model_name == self.settings.verifier_model_name:
            raise LLMConfigurationError("Independent comparison verification requires distinct configured models.")
        candidate = Candidate.model_validate(record)
        if record.get("primary"):
            primary = Judgment.model_validate(record["primary"])
        else:
            primary = self.evaluator.classify(candidate)
            if self.repository.update_owned(record, {"primary": primary.model_dump(mode="json")}) is None:
                return None  # Another owner or a newer evidence revision superseded this attempt.
        secondary = self.evaluator.verify(candidate)
        if not self.repository.sources_current(record):
            return self._stale(record)
        return self.repository.update_owned(record, reconcile(primary, secondary), release=True)

    def _stale(self, record):
        return self.repository.update_owned(record, {
            "processing_status": "stale", "verification_status": "stale", "failure_reason": "inputs_changed",
            "explanation": "A source or model configuration changed. Discover candidates again.",
        }, release=True)
