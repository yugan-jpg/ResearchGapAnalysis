import json

import mongomock
import pytest

from app.core.config import Settings
from app.modules.claim_extraction.gemini_verifier import (
    GeminiClaimVerifier,
    VerificationInput,
    VerificationOutcome,
)
from app.modules.claim_extraction.repository import ClaimRepository
from app.modules.claim_extraction.service import ClaimExtractionService
from app.modules.claim_extraction.verifier_errors import (
    GeminiConfigurationError,
    GeminiProviderError,
    GeminiRateLimitError,
    GeminiResponseValidationError,
    GeminiTimeoutError,
)
from app.schemas.claims import ClaimCategory, ExtractedClaim, LLMClaimPayload, VerificationVerdict


class FakeGeminiResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeGeminiModels:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return FakeGeminiResponse(response)


class FakeGeminiClient:
    def __init__(self, responses) -> None:
        self.models = FakeGeminiModels(responses)


class FakeClaimLLM:
    def __init__(self, claims) -> None:
        self.claims = claims

    def extract_claims(self, *_args):
        return LLMClaimPayload(claims=self.claims)


class StubVerifier:
    def __init__(self, settings: Settings, outcome_factory) -> None:
        self.settings = settings
        self.outcome_factory = outcome_factory
        self.calls = []

    def verify_batch(self, inputs):
        self.calls.append(inputs)
        return {item.claim_id: self.outcome_factory(item) for item in inputs}


class RateLimitError(Exception):
    status_code = 429


def _settings(**overrides) -> Settings:
    values = {
        "gemini_api_key": "test-gemini-key",
        "gemini_model_name": "gemini-2.5-flash",
        "verifier_enabled": True,
        "gemini_max_retries": 0,
        "verifier_batch_size": 3,
        "verifier_comparison_limit": 5,
    }
    values.update(overrides)
    return Settings(**values)


def _input(claim_id="claim-1", comparisons=()):
    return VerificationInput(
        claim_id=claim_id,
        paper_id="paper-1",
        chunk_id="chunk-1",
        page_number=3,
        claim_text="The intervention improved validation accuracy by 12%.",
        category=ClaimCategory.FINDING,
        supporting_evidence="The intervention improved validation accuracy by 12%.",
        source_text="The intervention improved validation accuracy by 12%.",
        rule_completeness=True,
        rule_numerical_consistency=True,
        comparison_candidates=comparisons,
    )


def _verification_json(claim_id="claim-1", **overrides) -> str:
    item = {
        "claim_id": claim_id,
        "verdict": "accepted",
        "evidence_supported": True,
        "numerical_consistency": True,
        "completeness": True,
        "category_correct": True,
        "explanation": "The result and percentage are directly stated in the source passage.",
        "duplicate_of_claim_id": None,
    }
    item.update(overrides)
    return json.dumps({"verifications": [item]})


def _database_with_source_chunks():
    database = mongomock.MongoClient().claim_verifier
    database.papers.insert_one({"paper_id": "paper-1"})
    database.paper_chunks.insert_many(
        [
            {
                "chunk_id": "chunk-1",
                "paper_id": "paper-1",
                "chunk_index": 0,
                "page_number": 3,
                "text": "The intervention improved validation accuracy by 12%.",
            },
            {
                "chunk_id": "chunk-2",
                "paper_id": "paper-1",
                "chunk_index": 1,
                "page_number": 4,
                "text": "The intervention improved validation accuracy by 12% after training.",
            },
        ]
    )
    return database


def test_gemini_verifier_accepts_valid_structured_batch() -> None:
    client = FakeGeminiClient([_verification_json()])
    outcomes = GeminiClaimVerifier(_settings(), client=client, sleep=lambda _: None).verify_batch([_input()])

    assert outcomes["claim-1"].status == "accepted"
    assert outcomes["claim-1"].verdict is VerificationVerdict.ACCEPTED
    assert client.models.calls[0]["model"] == "gemini-2.5-flash"
    assert len(client.models.calls) == 1


def test_gemini_verifier_rejects_malformed_or_unknown_claim_json() -> None:
    malformed = GeminiClaimVerifier(_settings(), client=FakeGeminiClient(["not-json"]), sleep=lambda _: None)
    with pytest.raises(GeminiResponseValidationError):
        malformed.verify_batch([_input()])

    unknown = GeminiClaimVerifier(
        _settings(), client=FakeGeminiClient([_verification_json(claim_id="other-claim")]), sleep=lambda _: None
    )
    with pytest.raises(GeminiResponseValidationError):
        unknown.verify_batch([_input()])


def test_gemini_verifier_handles_rate_limits_timeouts_and_missing_config() -> None:
    with pytest.raises(GeminiRateLimitError):
        GeminiClaimVerifier(
            _settings(), client=FakeGeminiClient([RateLimitError("rate limited")]), sleep=lambda _: None
        ).verify_batch([_input()])
    with pytest.raises(GeminiTimeoutError):
        GeminiClaimVerifier(
            _settings(), client=FakeGeminiClient([TimeoutError("timed out")]), sleep=lambda _: None
        ).verify_batch([_input()])
    with pytest.raises(GeminiConfigurationError):
        GeminiClaimVerifier(_settings(gemini_api_key=""), client=FakeGeminiClient([])).verify_batch([_input()])


def test_service_marks_rejected_and_incomplete_claims_without_deleting_them() -> None:
    database = _database_with_source_chunks()
    grounded = ExtractedClaim(
        claim_text="The intervention improved validation accuracy by 12%,",
        category=ClaimCategory.FINDING,
        supporting_evidence="The intervention improved validation accuracy by 12%.",
    )
    verifier = StubVerifier(
        _settings(),
        lambda _item: VerificationOutcome(
            status="accepted",
            verdict=VerificationVerdict.ACCEPTED,
            evidence_supported=True,
            numerical_consistency=True,
            completeness=True,
            category_correct=True,
            explanation="Supported.",
            verifier_model="gemini-2.5-flash",
        ),
    )
    result = ClaimExtractionService(
        ClaimRepository(database), FakeClaimLLM([grounded]), "groq-gpt-oss-120b", verifier
    ).extract("paper-1", max_chunks=1)

    assert result.accepted_claims == 1
    assert result.claims[0].verification_status == "review"
    assert result.claims[0].completeness is False
    assert database.research_claims.count_documents({}) == 1

    rejected_verifier = StubVerifier(
        _settings(gemini_model_name="gemini-test-rejected"),
        lambda _item: VerificationOutcome(
            status="rejected",
            verdict=VerificationVerdict.REJECTED,
            evidence_supported=False,
            numerical_consistency=True,
            completeness=True,
            category_correct=True,
            explanation="The claim extends beyond the supplied evidence.",
            verifier_model="gemini-2.5-flash",
        ),
    )
    result = ClaimExtractionService(
        ClaimRepository(database), FakeClaimLLM([grounded]), "other-extraction-model", rejected_verifier
    ).extract("paper-1", max_chunks=1, force=True)
    assert result.rejected_claims == 1
    assert database.research_claims.count_documents({}) == 1


def test_cross_chunk_duplicate_is_retained_for_review_and_cache_skips_extra_call() -> None:
    database = _database_with_source_chunks()
    existing = ExtractedClaim(
        claim_text="The intervention improved validation accuracy by 12%.",
        category=ClaimCategory.FINDING,
        supporting_evidence="The intervention improved validation accuracy by 12%.",
    )
    repository = ClaimRepository(database)
    repository.store_claims(
        repository.get_chunks("paper-1", "unused", True, 1)[0], [existing], "old-model"
    )
    new_claim = ExtractedClaim(
        claim_text="Validation accuracy improved by 12% after the intervention.",
        category=ClaimCategory.FINDING,
        supporting_evidence="The intervention improved validation accuracy by 12% after training.",
    )

    def duplicate_outcome(item):
        duplicate_id = item.comparison_candidates[0]["claim_id"]
        return VerificationOutcome(
            status="review",
            verdict=VerificationVerdict.REVIEW,
            evidence_supported=True,
            numerical_consistency=True,
            completeness=True,
            category_correct=True,
            explanation="Potential duplicate retained for review.",
            verifier_model="gemini-2.5-flash",
            duplicate_of_claim_id=duplicate_id,
            duplicate_type="semantic",
        )

    verifier = StubVerifier(_settings(), duplicate_outcome)
    service = ClaimExtractionService(repository, FakeClaimLLM([new_claim]), "groq-gpt-oss-120b", verifier)
    first = service.extract("paper-1", max_chunks=2)
    second = service.extract("paper-1", max_chunks=2, force=True)

    semantic = next(item for item in first.claims if item.chunk_id == "chunk-2")
    assert semantic.verification_status == "review"
    assert semantic.duplicate_type == "semantic"
    assert database.research_claims.count_documents({}) == 2
    # The second pass reuses the verifier metadata keyed by unchanged claim and
    # source evidence rather than making a second Gemini request.
    assert len(verifier.calls) == 1
    assert second.duplicate_claims == 1


def test_exact_cross_chunk_match_is_linked_not_deleted() -> None:
    database = _database_with_source_chunks()
    repository = ClaimRepository(database)
    chunks = repository.get_chunks("paper-1", "unused", True, 2)
    exact = ExtractedClaim(
        claim_text="The intervention improved validation accuracy by 12%.",
        category=ClaimCategory.FINDING,
        supporting_evidence="The intervention improved validation accuracy by 12%",
    )

    repository.store_claims(chunks[0], [exact], "old-model")
    repository.store_claims(chunks[1], [exact], "old-model")
    second = database.research_claims.find_one({"chunk_id": "chunk-2"})

    assert database.research_claims.count_documents({}) == 2
    assert second["duplicate_type"] == "exact"
    assert second["verification_status"] == "review"
    assert second["duplicate_of_claim_id"]


def test_verifier_failure_preserves_new_claim_as_unavailable() -> None:
    database = _database_with_source_chunks()
    grounded = ExtractedClaim(
        claim_text="The intervention improved validation accuracy by 12%.",
        category=ClaimCategory.FINDING,
        supporting_evidence="The intervention improved validation accuracy by 12%.",
    )

    class FailingVerifier:
        settings = _settings()

        def verify_batch(self, _inputs):
            raise GeminiProviderError("provider unavailable")

    result = ClaimExtractionService(
        ClaimRepository(database), FakeClaimLLM([grounded]), "groq-gpt-oss-120b", FailingVerifier()
    ).extract("paper-1", max_chunks=1)

    assert result.accepted_claims == 1
    assert result.claims[0].verification_status == "unavailable"
    assert result.claims[0].verification_verdict is None
    assert database.research_claims.count_documents({}) == 1

