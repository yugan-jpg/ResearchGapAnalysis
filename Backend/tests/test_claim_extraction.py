import pytest
import mongomock
from pydantic import ValidationError

from app.core.config import Settings
from app.modules.claim_extraction.errors import (
    ClaimResponseValidationError,
    LLMRateLimitError,
    PaperNotFoundError,
)
from app.modules.claim_extraction.llm_client import QwenClaimClient
from app.modules.claim_extraction.repository import ClaimRepository
from app.modules.claim_extraction.service import ClaimExtractionService
from app.schemas.claims import (
    ClaimCategory,
    ClaimExtractionRequest,
    ClaimVerificationRequest,
    ExtractedClaim,
    LLMClaimPayload,
)


class FakeHttpResponse:
    def __init__(self, status_code: int, body, headers=None) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def json(self):
        return self._body


class FakeHttpClient:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


class FakeClaimLLM:
    def __init__(self, claims: list[ExtractedClaim]) -> None:
        self.claims = claims
        self.calls = []

    def extract_claims(self, paper_id, chunk_id, page_number, chunk_text):
        self.calls.append((paper_id, chunk_id, page_number, chunk_text))
        return LLMClaimPayload(claims=self.claims)


def _settings(**overrides) -> Settings:
    values = {
        "llm_api_url": "https://provider.example.test/chat/completions",
        "llm_api_key": "test-key",
        "llm_model_name": "qwen3-8b",
        "llm_max_retries": 0,
    }
    values.update(overrides)
    return Settings(**values)


def _database_with_source_chunk():
    database = mongomock.MongoClient().phase4_test
    database.papers.insert_one({"paper_id": "paper-1"})
    database.paper_chunks.insert_one(
        {
            "chunk_id": "chunk-1",
            "paper_id": "paper-1",
            "chunk_index": 0,
            "page_number": 4,
            "text": "The intervention improved accuracy by 12% in the validation cohort. External validation was not performed.",
        }
    )
    return database


def test_qwen_client_validates_openai_compatible_json_response() -> None:
    response = FakeHttpResponse(
        200,
        {
            "choices": [
                {
                    "message": {
                        "content": '{"claims":[{"claim_text":"Accuracy improved by 12%.","category":"finding","supporting_evidence":"improved accuracy by 12%"}]}'
                    }
                }
            ]
        },
    )
    http_client = FakeHttpClient([response])
    claims = QwenClaimClient(_settings(), http_client=http_client, sleep=lambda _: None).extract_claims("paper-1", "chunk-1", 4, "improved accuracy by 12%")

    assert claims.claims[0].category == ClaimCategory.FINDING
    assert http_client.calls[0][0] == "https://provider.example.test/chat/completions"
    assert http_client.calls[0][1]["json"]["model"] == "qwen3-8b"


def test_qwen_client_rejects_invalid_claim_json() -> None:
    http_client = FakeHttpClient([FakeHttpResponse(200, {"choices": [{"message": {"content": "not-json"}}]})])

    with pytest.raises(ClaimResponseValidationError):
        QwenClaimClient(_settings(), http_client=http_client, sleep=lambda _: None).extract_claims("paper-1", "chunk-1", 4, "source text")


def test_qwen_client_reports_rate_limit_without_real_request() -> None:
    http_client = FakeHttpClient([FakeHttpResponse(429, {}, {"retry-after": "1"})])

    with pytest.raises(LLMRateLimitError):
        QwenClaimClient(_settings(), http_client=http_client, sleep=lambda _: None).extract_claims("paper-1", "chunk-1", 4, "source text")


def test_claim_extraction_stores_only_exact_source_evidence_and_prevents_duplicates() -> None:
    database = _database_with_source_chunk()
    grounded_claim = ExtractedClaim(
        claim_text="The intervention improved validation accuracy by 12%.",
        category=ClaimCategory.FINDING,
        supporting_evidence="The intervention improved accuracy by 12% in the validation cohort.",
    )
    service = ClaimExtractionService(ClaimRepository(database), FakeClaimLLM([grounded_claim]), "qwen3-8b")

    first = service.extract("paper-1", max_chunks=10)
    repeated = service.extract("paper-1", max_chunks=10, force=True)

    stored = database.research_claims.find_one({"paper_id": "paper-1"})
    assert first.accepted_claims == 1
    assert repeated.duplicate_claims == 1
    assert database.research_claims.count_documents({}) == 1
    assert stored["chunk_id"] == "chunk-1"
    assert stored["page_number"] == 4
    assert stored["supporting_evidence"] in database.paper_chunks.find_one({"chunk_id": "chunk-1"})["text"]


def test_claim_extraction_rejects_fabricated_evidence() -> None:
    database = _database_with_source_chunk()
    fabricated_claim = ExtractedClaim(
        claim_text="The intervention was externally validated.",
        category=ClaimCategory.FINDING,
        supporting_evidence="The intervention was externally validated across hospitals.",
    )
    result = ClaimExtractionService(ClaimRepository(database), FakeClaimLLM([fabricated_claim]), "qwen3-8b").extract("paper-1", max_chunks=10)

    assert result.accepted_claims == 0
    assert result.rejected_claims == 1
    assert database.research_claims.count_documents({}) == 0


def test_missing_paper_and_request_validation() -> None:
    repository = ClaimRepository(mongomock.MongoClient().phase4_test)
    with pytest.raises(PaperNotFoundError):
        repository.ensure_paper("missing-paper")
    assert ClaimExtractionRequest(max_chunks=10).max_chunks == 10
    with pytest.raises(ValidationError):
        ClaimExtractionRequest(max_chunks=101)
    with pytest.raises(ValidationError):
        ClaimVerificationRequest(limit=101)
