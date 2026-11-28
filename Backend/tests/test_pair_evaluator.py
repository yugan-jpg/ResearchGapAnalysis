import json

import httpx
import pytest

from app.core.config import Settings
from app.modules.claim_extraction.errors import (
    ClaimResponseValidationError, LLMAuthenticationError, LLMConfigurationError,
    LLMProviderError, LLMRateLimitError, LLMTimeoutError,
)
from app.modules.claim_extraction.usage import UsageMonitor
from app.modules.contradiction_detection.evaluator import PairEvaluator
from app.schemas.comparisons import Candidate, ClaimSource


LEFT = "Under 5 MPa at 20 °C, the coating increased film thickness relative to an uncoated control."
RIGHT = "Under 5 MPa at 20 °C, the coating decreased film thickness relative to an uncoated control."


def pair(left=LEFT, right=RIGHT):
    def source(side, text):
        return ClaimSource(claim_id=side, paper_id=f"paper-{side}", chunk_id=f"chunk-{side}",
                           page_number=2, claim_text=text, category="finding", supporting_evidence=text)

    return Candidate(left=source("L", left), right=source("R", right),
                     similarity_score=0.91, structural_signals={})


def judgment(**changes):
    values = dict(relationship="direct_contradiction", comparable_scope=True, mismatch_types=[],
                  explanation="The same treatment and control produced opposing thickness changes at 5 MPa and 20 °C.",
                  left_quote=LEFT, right_quote=RIGHT)
    return values | changes


class FakeHTTP:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def post(self, url, **kwargs):
        self.requests.append(kwargs["json"])
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, int):
            return httpx.Response(result, json={"error": "provider body, intentionally private"})
        content = result if isinstance(result, str) else json.dumps(result)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def evaluator(responses, **overrides):
    config = dict(llm_api_key="test", llm_model_name="pair-test-primary", verifier_model_name="pair-test-secondary",
                  llm_api_url="https://example.invalid", verifier_enabled=True, llm_max_retries=0,
                  verifier_max_retries=0, verifier_malformed_retries=0)
    fake = FakeHTTP(responses)
    return PairEvaluator(Settings(_env_file=None, **(config | overrides)), fake, sleep=lambda _: None), fake


@pytest.fixture(autouse=True)
def isolated_quota(monkeypatch):
    monkeypatch.setattr("app.modules.claim_extraction.llm_client.usage_monitor", UsageMonitor())


def test_models_independently_compare_evidence_and_use_shared_transport():
    result = judgment()
    client, fake = evaluator([result, result])
    candidate = pair()
    assert client.classify(candidate).relationship == "direct_contradiction"
    assert client.verify(candidate).relationship == "direct_contradiction"
    primary, secondary = fake.requests
    assert primary["model"] == "pair-test-primary"
    assert secondary["model"] == "pair-test-secondary"
    assert primary["response_format"]["type"] == "json_object"
    assert secondary["response_format"]["json_schema"]["strict"] is True
    assert primary["messages"] == secondary["messages"]
    rules, data = secondary["messages"]
    assert rules["role"] == "system" and "untrusted research data" in rules["content"]
    assert data["role"] == "user"
    proof = json.loads(data["content"])
    assert proof["left"]["supporting_evidence"] == LEFT and proof["right"]["supporting_evidence"] == RIGHT
    assert result["explanation"] not in data["content"]


@pytest.mark.parametrize("left,right,relationship,signals,explanation,comparable", [
    ("The estimate improved in adults.", "The estimate worsened in children.", "contextual_differences",
     ["population_context"], "The studies report estimates in different age groups.", False),
    ("We observed faster recovery in volunteers.", "Random assignment was needed to establish causal benefit.",
     "complementary_findings", ["claim_scope", "evidence_method"], "The sampled volunteers and observational design constrain causal scope.", False),
    ("In operations research the queue discipline prioritized throughput.", "In clinical logistics the same queue discipline prioritized acuity.",
     "contextual_differences", ["cross_field"], "The two disciplines report different priorities for the same approach.", False),
    ("Accuracy improved on the private test set.", "Only mean latency was collected.",
     "insufficient_evidence", [], "Latency does not establish an accuracy effect.", False),
])
def test_scientific_relationships_and_potential_mismatch_signals(left, right, relationship, signals, explanation, comparable):
    client, _ = evaluator([judgment(relationship=relationship, mismatch_types=signals, explanation=explanation,
                                    comparable_scope=comparable, left_quote=left, right_quote=right)])
    output = client.classify(pair(left, right))
    assert output.relationship == relationship
    assert output.mismatch_types == signals


@pytest.mark.parametrize("malformed", [
    "incomplete JSON", judgment(left_quote=RIGHT), judgment(right_quote="A fabricated passage."),
    judgment(comparable_scope=False), judgment(relationship="agreement", comparable_scope=False),
    judgment(relationship="confirmed_research_gap"),
    judgment(mismatch_types=["population_context", "population_context"]),
])
def test_invalid_judgments_fail_closed(malformed):
    client, _ = evaluator([malformed])
    with pytest.raises(ClaimResponseValidationError):
        client.classify(pair())


def test_secondary_retries_only_within_configured_budget():
    client, fake = evaluator(["{", judgment()], verifier_malformed_retries=1)
    assert client.verify(pair()).relationship == "direct_contradiction"
    assert len(fake.requests) == 2


@pytest.mark.parametrize("error,kind", [(429, LLMRateLimitError), (503, LLMProviderError),
                                        (403, LLMAuthenticationError), (httpx.ReadTimeout("read"), LLMTimeoutError)])
def test_provider_failures_remain_distinct_and_sanitized(error, kind):
    client, fake = evaluator([error])
    with pytest.raises(kind) as caught:
        client.verify(pair())
    assert "intentionally private" not in str(caught.value)
    assert len(fake.requests) == 1


@pytest.mark.parametrize("overrides", [{"verifier_enabled": False}, {"verifier_model_name": ""},
                                       {"verifier_model_name": "pair-test-primary"}])
def test_independent_verification_requires_distinct_configured_model(overrides):
    client, fake = evaluator([], **overrides)
    with pytest.raises(LLMConfigurationError):
        client.verify(pair())
    assert not fake.requests
