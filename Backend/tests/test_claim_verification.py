import json

import mongomock
import pytest
import httpx

from app.core.config import Settings
from app.modules.claim_extraction.groq_verifier import (
    GroqClaimVerifier,
    VerificationInput,
    VerificationOutcome,
)
from app.modules.claim_extraction.repository import ClaimRepository
from app.modules.claim_extraction.service import ClaimExtractionService
from app.modules.claim_extraction.verifier_errors import (
    GroqConfigurationError,
    GroqProviderError,
    GroqRateLimitError,
    GroqResponseValidationError,
    GroqTimeoutError,
)
from app.schemas.claims import ClaimCategory, ExtractedClaim, LLMClaimPayload, VerificationVerdict


class FakeGroqResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code
        self.headers = {}

    def json(self):
        return {"choices": [{"message": {"content": self.text}}]}


class FakeGroqClient:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, int):
            return FakeGroqResponse("", response)
        return FakeGroqResponse(response)


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
        "llm_api_key": "test-key",
        "llm_api_url": "https://api.groq.test/openai/v1/chat/completions",
        "verifier_model_name": "openai/gpt-oss-120b",
        "verifier_enabled": True,
        "verifier_max_retries": 0,
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


def test_groq_verifier_accepts_valid_structured_batch() -> None:
    client = FakeGroqClient([_verification_json()])
    outcomes = GroqClaimVerifier(_settings(), client=client, sleep=lambda _: None).verify_batch([_input()])

    assert outcomes["claim-1"].status == "accepted"
    assert outcomes["claim-1"].verdict is VerificationVerdict.ACCEPTED
    assert client.calls[0]["json"]["model"] == "openai/gpt-oss-120b"
    assert client.calls[0]["url"] == "https://api.groq.test/openai/v1/chat/completions"
    assert client.calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert client.calls[0]["timeout"] == 60
    assert len(client.calls) == 1


def test_duplicate_candidates_include_same_evidence_exclude_current_claim_and_flag_paraphrases() -> None:
    db = _database_with_source_chunks()
    evidence = "Samples were genotyped using SpotMalaria and assessed for resistance."
    db.research_claims.insert_many([
        {"paper_id": "paper-1", "claim_id": "old", "chunk_id": "chunk-1",
         "claim_text": "Samples were genotyped using SpotMalaria.", "category": "methodology",
         "supporting_evidence": evidence},
        {"paper_id": "paper-1", "claim_id": "self", "chunk_id": "chunk-1",
         "claim_text": "SpotMalaria genotyped samples.", "category": "methodology",
         "supporting_evidence": evidence},
    ])
    claim = ExtractedClaim(claim_text="SpotMalaria genotyped samples.", category="methodology", supporting_evidence=evidence)
    candidates = ClaimRepository(db).find_duplicate_candidates("paper-1", claim, 5, exclude_claim_id="self")
    assert [candidate["claim_id"] for candidate in candidates] == ["old"]
    item = _input()
    from dataclasses import replace
    item = replace(item, claim_text=claim.claim_text, supporting_evidence=evidence, comparison_candidates=tuple(candidates))
    accepted = VerificationOutcome(status="accepted", verdict=VerificationVerdict.ACCEPTED, evidence_supported=True,
                                   numerical_consistency=True, completeness=True, category_correct=True)
    review = ClaimExtractionService._apply_rule_safeguards(item, accepted)
    assert review.status == "review"
    assert review.duplicate_of_claim_id == "old"
    assert "possible paraphrase" in review.duplicate_explanation


def test_shared_evidence_alone_does_not_mark_distinct_claims_duplicate() -> None:
    from dataclasses import replace
    item = replace(_input(), claim_text="Samples were collected in Cambodia.", supporting_evidence="Same passage.",
                   comparison_candidates=({"claim_id": "x", "claim_text": "Sequencing used SpotMalaria.",
                                           "supporting_evidence": "Same passage."},))
    accepted = VerificationOutcome(status="accepted", verdict=VerificationVerdict.ACCEPTED, evidence_supported=True,
                                   numerical_consistency=True, completeness=True, category_correct=True)
    assert ClaimExtractionService._apply_rule_safeguards(item, accepted).status == "accepted"


def test_groq_verifier_rejects_malformed_or_unknown_claim_json() -> None:
    malformed = GroqClaimVerifier(_settings(), client=FakeGroqClient(["not-json"]), sleep=lambda _: None)
    with pytest.raises(GroqResponseValidationError):
        malformed.verify_batch([_input()])

    unknown = GroqClaimVerifier(
        _settings(), client=FakeGroqClient([_verification_json(claim_id="other-claim")]), sleep=lambda _: None
    )
    with pytest.raises(GroqResponseValidationError):
        unknown.verify_batch([_input()])


def test_groq_verifier_handles_rate_limits_timeouts_and_missing_config() -> None:
    with pytest.raises(GroqRateLimitError):
        GroqClaimVerifier(
            _settings(), client=FakeGroqClient([429]), sleep=lambda _: None
        ).verify_batch([_input()])
    with pytest.raises(GroqTimeoutError):
        GroqClaimVerifier(
            _settings(), client=FakeGroqClient([httpx.ReadTimeout("timed out")]), sleep=lambda _: None
        ).verify_batch([_input()])
    with pytest.raises(GroqConfigurationError):
        GroqClaimVerifier(_settings(llm_api_key=""), client=FakeGroqClient([])).verify_batch([_input()])


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
            verifier_model="openai/gpt-oss-120b",
        ),
    )
    result = ClaimExtractionService(
        ClaimRepository(database), FakeClaimLLM([grounded]), "groq-gpt-oss-120b", verifier
    ).extract("paper-1", max_chunks=1)

    assert result.accepted_claims == 0
    assert result.extraction_rejections == 1
    assert database.research_claims.count_documents({}) == 0

    grounded = grounded.model_copy(update={"claim_text": "The intervention improved validation accuracy by 12%."})

    rejected_verifier = StubVerifier(
        _settings(verifier_model_name="other-verifier-model"),
        lambda _item: VerificationOutcome(
            status="rejected",
            verdict=VerificationVerdict.REJECTED,
            evidence_supported=False,
            numerical_consistency=True,
            completeness=True,
            category_correct=True,
            explanation="The claim extends beyond the supplied evidence.",
            verifier_model="openai/gpt-oss-120b",
        ),
    )
    result = ClaimExtractionService(
        ClaimRepository(database), FakeClaimLLM([grounded]), "other-extraction-model", rejected_verifier
    ).extract("paper-1", max_chunks=1, force=True)
    assert result.rejected_claims == 1
    assert database.research_claims.count_documents({}) == 1


def test_cross_chunk_duplicate_is_retained_and_new_comparison_invalidates_cache() -> None:
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
            verifier_model="openai/gpt-oss-120b",
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
    # The stored claim itself is now among the candidates for this newly
    # extracted claim, so the comparison set changes and the cache is stale.
    assert len(verifier.calls) == 2
    assert second.duplicate_claims == 1


def test_exact_cross_chunk_match_preserves_source_reference() -> None:
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
    stored = database.research_claims.find_one({"chunk_id": "chunk-1"})

    assert database.research_claims.count_documents({}) == 1
    assert {ref["chunk_id"] for ref in stored["source_references"]} == {"chunk-1", "chunk-2"}


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
            raise GroqProviderError("provider unavailable")

    result = ClaimExtractionService(
        ClaimRepository(database), FakeClaimLLM([grounded]), "groq-gpt-oss-120b", FailingVerifier()
    ).extract("paper-1", max_chunks=1)

    assert result.accepted_claims == 1
    assert result.claims[0].verification_status == "unavailable"
    assert result.claims[0].verification_verdict is None
    assert database.research_claims.count_documents({}) == 1

