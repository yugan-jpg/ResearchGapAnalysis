from dataclasses import replace
import json
import logging
import mongomock
import pytest

from app.core.config import Settings
from app.modules.claim_extraction.quality import inequality_tokens, measurement_tokens
from app.modules.claim_extraction.repository import ClaimRepository, ClaimChunk
from app.modules.claim_extraction.service import ClaimExtractionService
from app.schemas.claims import ExtractedClaim
from app.modules.claim_extraction.repair import plan_repairs, repair_paper
from app.modules.claim_extraction.usage import UsageMonitor, parse_duration
from app.modules.claim_extraction.groq_verifier import GroqClaimVerifier, VerificationInput
from app.modules.claim_extraction.verifier_errors import GroqResponseValidationError
from app.schemas.claims import ClaimCategory


def claim(cid, text, status="accepted", target=None, paper="p"):
    return {"claim_id": cid, "paper_id": paper, "claim_text": text, "chunk_id": "c",
            "page_number": 1, "supporting_evidence": text, "verification_status": status,
            "evidence_supported": True, "completeness": True, "numerical_consistency": True,
            "category_correct": True, "duplicate_of_claim_id": target, "duplicate_type": "semantic"}


def test_cycles_exact_duplicates_choose_eligible_canonical_and_preserve_history():
    db = mongomock.MongoClient().test
    db.research_claims.insert_many([claim("a", "Drug reduced fever.", target="b"),
                                    claim("b", "Drug reduced fever.", target="a")])
    db.paper_chunks.insert_one({"paper_id": "p", "chunk_id": "c", "page_number": 1, "text": "Drug reduced fever."})
    result = repair_paper(db, "p", apply=False)
    assert result["planned"] == 2 and result["applied"] == 0
    assert db.research_claims.find_one({"claim_id": "a"})["duplicate_of_claim_id"] == "b"
    assert repair_paper(db, "p", apply=True)["applied"] == 2
    assert db.research_claims.find_one({"claim_id": "a"})["duplicate_of_claim_id"] is None
    assert db.research_claims.find_one({"claim_id": "b"})["duplicate_of_claim_id"] == "a"
    assert db.research_claims.find_one({"claim_id": "b"})["verification_status"] == "review"
    assert len(db.research_claims.find_one({"claim_id": "a"})["duplicate_repair_history"]) == 1
    assert repair_paper(db, "p", apply=True)["applied"] == 0


def test_rejected_exact_claim_does_not_suppress_new_chunk():
    from app.modules.claim_extraction.groq_verifier import VerificationOutcome
    from app.schemas.claims import VerificationVerdict
    db = mongomock.MongoClient().test
    repository = ClaimRepository(db)
    extracted = ExtractedClaim(claim_text="The study examined 20 patients.", category="methodology",
                               supporting_evidence="The study examined 20 patients.")
    old = ClaimChunk("old", "p", 0, 1, extracted.supporting_evidence)
    new = ClaimChunk("new", "p", 1, 2, extracted.supporting_evidence)
    rejected = VerificationOutcome(status="rejected", verdict=VerificationVerdict.REJECTED)
    accepted = VerificationOutcome(status="accepted", verdict=VerificationVerdict.ACCEPTED,
                                   evidence_supported=True, completeness=True,
                                   numerical_consistency=True, category_correct=True)
    repository.store_claims(old, [extracted], "old", claim_ids=["old-id"], verification_results={"old-id": rejected})
    inserted, _, records = repository.store_claims(new, [extracted], "new", claim_ids=["new-id"],
                                                   verification_results={"new-id": accepted})
    assert inserted == 1 and records[0].claim_id == "new-id"
    assert db.research_claims.find_one({"claim_id": "old-id"})["verification_status"] == "rejected"


@pytest.mark.parametrize("first,second", [
    ("Malaria declined in Attapeu.", "Malaria declined in Sainyabuli."),
    ("Malaria prevalence decreased.", "PPQ-R prevalence decreased."),
    ("The model improved accuracy.", "The model increased latency."),
    ("The treatment reduced fever.", "The treatment did not reduce fever."),
    ("The survey recorded income in rural districts.", "The survey recorded income in urban districts."),
    ("Temperature increased in reaction A.", "Temperature decreased in reaction A."),
    ("Particle velocity increased in vacuum.", "Particle velocity decreased in vacuum."),
])
def test_distinct_links_and_rejected_targets_are_unlinked_for_review(first, second):
    records = [claim("a", first, target="b"), claim("b", second, status="rejected")]
    plan = plan_repairs(records)
    assert len(plan) == 1 and plan[0]["requires_reverification"]
    assert plan[0]["new"]["duplicate_of_claim_id"] is None
    assert plan[0]["new"]["verification_status"] == "review"


def test_chain_and_partial_overlap_are_not_automatically_declared_equivalent():
    records = [claim("a", "Drug reduced fever.", target="b"),
               claim("b", "Drug reduced fever and nausea.", target="c"),
               claim("c", "Drug reduced nausea.")]
    plan = plan_repairs(records)
    assert {item["claim_id"] for item in plan} == {"a", "b"}
    assert all(item["requires_reverification"] for item in plan)


def test_unlinked_shared_evidence_paraphrases_are_reviewed_not_merged():
    first = claim("a", "The spread of KEL1 strains was tracked by populations.")
    second = claim("b", "The study tracked the spread of KEL1 strains by populations.")
    first["supporting_evidence"] = second["supporting_evidence"] = "We tracked the spread of KEL1 strains by populations."
    plan = plan_repairs([first, second])
    assert {item["claim_id"] for item in plan} == {"a", "b"}
    assert all(item["requires_reverification"] and item["new"]["verification_status"] == "review" for item in plan)
    assert all("duplicate_of_claim_id" not in item["new"] for item in plan)


@pytest.mark.parametrize("a,b", [
    ("β = −0.5, p < 1e-3, R² = 0.6", "β = -0.5, p < 0.001, R² = 0.6"),
    ("A chemistry measurement was 12mg", "A chemistry measurement was 12 mg"),
    ("Between 2017¹⁶ and 2019²⁰ there were 321 samples", "Between 2017 and 2019 there were 321 samples"),
])
def test_scientific_number_normalization(a, b):
    assert measurement_tokens(a) == measurement_tokens(b)


def test_scientific_number_signs_and_identifiers():
    assert measurement_tokens("β = −0.5") != measurement_tokens("β = 0.5")
    assert measurement_tokens("KEL1/PLA1, pfk13, H97Y, plasmepsin2/3") == set()
    assert measurement_tokens("1e-3") == measurement_tokens("10⁻³")
    assert inequality_tokens("p ≤ 0.05") == inequality_tokens("p <= .05")
    assert inequality_tokens("p > 0.05") != inequality_tokens("p < 0.05")


def test_same_period_uses_bounded_same_page_context():
    db = mongomock.MongoClient().test
    db.paper_chunks.insert_many([
        {"paper_id": "p", "chunk_id": "prev", "chunk_index": 0, "page_number": 1,
         "text": "The study ran from 2019 to 2020."},
        {"paper_id": "p", "chunk_id": "c", "chunk_index": 1, "page_number": 1,
         "text": "Over the same period, 40 patients were assessed."},
    ])
    chunk = ClaimChunk("c", "p", 1, 1, "Over the same period, 40 patients were assessed.")
    claim = ExtractedClaim(claim_text="During 2019 to 2020, 40 patients were assessed.",
                           category="methodology", supporting_evidence=chunk.text)
    service = ClaimExtractionService(ClaimRepository(db), object(), "extractor")
    prepared = service._prepare_claim(chunk, claim)
    assert prepared.verification_input.rule_numerical_consistency
    assert "2019 to 2020" in prepared.verification_input.neighboring_context


@pytest.mark.parametrize("left,right", [
    ("Malaria prevalence decreased in Attapeu", "Malaria prevalence decreased in Sainyabuli"),
    ("Malaria prevalence decreased in 2020", "PPQ-R prevalence decreased in 2020"),
    ("Drug reduced fever in adults", "Drug reduced fever in children"),
    ("The algorithm increased model accuracy", "The algorithm decreased model accuracy"),
])
def test_distinct_context_is_not_canonicalized(left, right):
    from app.modules.claim_extraction.groq_verifier import VerificationOutcome
    from app.schemas.claims import VerificationVerdict
    db = mongomock.MongoClient().test
    db.research_claims.insert_one(claim("target", right))
    service = ClaimExtractionService(ClaimRepository(db), object(), "extractor")
    chunk = ClaimChunk("c", "p", 0, 1, "Common source text.")
    prepared = service._prepare_claim(chunk, ExtractedClaim(claim_text=left, category="finding",
                                                               supporting_evidence=chunk.text), "new")
    outcome = VerificationOutcome(status="review", verdict=VerificationVerdict.REVIEW,
                                  duplicate_of_claim_id="target", duplicate_type="semantic")
    result = service._safe_duplicate_targets([prepared], {"new": outcome})["new"]
    assert result.duplicate_of_claim_id is None
    assert result.status == "review"


def test_usage_monitor_tracks_models_and_throttles_using_headers(caplog):
    current = [0.0]
    monitor = UsageMonitor(clock=lambda: current[0], sleep=lambda n: current.__setitem__(0, current[0] + n))
    settings = Settings(_env_file=None, llm_model_name="q", verifier_model_name="v",
                        llm_tokens_per_minute=100, verifier_tokens_per_minute=100,
                        groq_rate_safety_fraction=0.9)
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    with monitor.slot("v", 50, settings):
        monitor.observe("v", {"x-ratelimit-limit-requests": "100",
                              "x-ratelimit-remaining-requests": "9",
                              "x-ratelimit-limit-tokens": "100",
                              "x-ratelimit-remaining-tokens": "10",
                              "x-ratelimit-reset-tokens": "2s",
                              "x-ratelimit-reset-requests": "1m"}, {"total_tokens": 50}, 60)
    with monitor.slot("q", 50, settings):
        monitor.observe("q", {}, None, 50)
    assert current[0] == 0
    with monitor.slot("v", 50, settings):
        pass
    assert current[0] == 60
    assert "90% threshold" in caplog.text
    assert "provider_reported_tokens=50" in caplog.text
    assert "locally_estimated_tokens=50" in caplog.text
    assert parse_duration("2m3.5s") == 123.5


def test_verifier_retries_only_malformed_response_then_succeeds():
    response = {"verifications": [{"claim_id": "id", "verdict": "accepted", "evidence_supported": True,
                                   "numerical_consistency": True, "completeness": True, "category_correct": True,
                                   "explanation": "Directly stated.", "duplicate_of_claim_id": None}]}
    class Response:
        status_code = 200
        headers = {}
        def __init__(self, value): self.value = value
        def json(self): return {"choices": [{"message": {"content": self.value}}]}
    class Client:
        def __init__(self): self.responses = ["bad json", json.dumps(response)]; self.calls = 0
        def post(self, *args, **kwargs):
            self.calls += 1
            return Response(self.responses.pop(0))
    client = Client()
    settings = Settings(_env_file=None, llm_api_key="test", llm_api_url="https://example.test",
                        verifier_model_name="verifier", verifier_malformed_retries=1)
    verifier = GroqClaimVerifier(settings, client=client, sleep=lambda _: None)
    item = VerificationInput("id", "p", "c", 1, "The sample contained 20 items.",
                             ClaimCategory.METHODOLOGY, "20 items", "The sample contained 20 items.", True, True)
    assert verifier.verify_batch([item])["id"].status == "accepted"
    assert client.calls == 2


def test_http_retry_after_and_usage_never_logs_key(monkeypatch, caplog):
    from app.modules.claim_extraction.llm_client import QwenClaimClient
    class Response:
        def __init__(self, code):
            self.status_code = code
            self.headers = {"retry-after": "2"} if code == 429 else {}
        def json(self):
            return {"usage": {"total_tokens": 25}}
    class Client:
        def __init__(self): self.responses = [Response(429), Response(200)]
        def post(self, *args, **kwargs): return self.responses.pop(0)
    sleeps = []
    current = [0.0]
    def sleep(seconds):
        sleeps.append(seconds)
        current[0] += seconds
    monkeypatch.setattr("app.modules.claim_extraction.llm_client.usage_monitor", UsageMonitor(clock=lambda: current[0], sleep=sleep))
    monkeypatch.setattr("app.modules.claim_extraction.llm_client.random.uniform", lambda *_: 0)
    settings = Settings(_env_file=None, llm_api_key="sensitive-test-key",
                        llm_model_name="retry-model", llm_api_url="https://example.test", llm_max_retries=1)
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    result = QwenClaimClient(settings, http_client=Client(), sleep=sleep)._request(
        {"model": "retry-model", "messages": []}
    )
    assert result["usage"]["total_tokens"] == 25
    assert sleeps == [2]
    assert "rate limited" in caplog.text
    assert "sensitive-test-key" not in caplog.text
