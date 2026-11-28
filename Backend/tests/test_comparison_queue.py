from datetime import datetime, timedelta, timezone

import mongomock
import pytest

from app.core.config import Settings
from app.db.indexes import initialize_database
from app.modules.claim_extraction.errors import LLMRateLimitError
from app.modules.contradiction_detection.repository import ComparisonRepository
from app.modules.contradiction_detection.service import ComparisonService, reconcile
from app.schemas.comparisons import Candidate, ClaimSource, DetectionRequest, Judgment


def source(side, change):
    text = f"The coating {change} mean friction on steel at 20 °C."
    return ClaimSource(claim_id=side, chunk_id="chunk-" + side, paper_id="paper-" + side,
                       page_number=3, claim_text=text, supporting_evidence=text, category="finding")


@pytest.fixture
def queue():
    db = mongomock.MongoClient().comparisons
    initialize_database(db)
    settings = Settings(_env_file=None, llm_model_name="primary", verifier_model_name="secondary", verifier_enabled=True)
    pair = Candidate(left=source("a", "increased"), right=source("b", "decreased"),
                     similarity_score=0.82, structural_signals={"variables_outcomes": ["friction"]})
    for side in (pair.left, pair.right):
        db.papers.insert_one({"paper_id": side.paper_id})
        db.research_claims.insert_one({
            **side.model_dump(), "claim_hash": side.claim_id,
            "verification_status": "accepted", "factual_verdict": "accepted", "verification_verdict": "accepted",
            "source_verified": True, "verifier_model": "secondary", "verified_at": datetime.now(timezone.utc),
            "evidence_supported": True, "completeness": True, "numerical_consistency": True, "category_correct": True,
        })
        db.paper_chunks.insert_one({"chunk_id": side.chunk_id, "paper_id": side.paper_id,
                                    "page_number": side.page_number, "text": side.supporting_evidence})
    repo = ComparisonRepository(db)
    return repo, settings, pair


class Evaluator:
    def __init__(self):
        self.calls = []
        self.interrupt = False

    def classify(self, pair):
        self.calls.append("primary")
        return decision(pair)

    def verify(self, pair):
        self.calls.append("secondary")
        if self.interrupt:
            raise LLMRateLimitError("simulated private provider message")
        return decision(pair)


def decision(pair, relationship="direct_contradiction", mismatch_types=()):
    return Judgment(relationship=relationship, explanation="Both passages name steel, friction and temperature.",
                    left_quote=pair.left.supporting_evidence, right_quote=pair.right.supporting_evidence,
                    comparable_scope=True, mismatch_types=list(mismatch_types))


def run(repo, config, evaluator, retry_failed=False):
    request = DetectionRequest(paper_id="paper-a", retry_failed=retry_failed)
    return ComparisonService(repo, evaluator, config).run(request, limit=2)


def test_secondary_failure_preserves_primary_for_explicit_retry(queue):
    repo, settings, candidate = queue
    record, created = repo.enqueue(candidate, settings)
    evaluator = Evaluator()
    evaluator.interrupt = True
    failure, status = run(repo, settings, evaluator)
    assert created and status == 429 and failure.failed == 1
    row = failure.comparisons[0]
    assert row.primary and row.secondary is None and row.relationship is None
    assert row.failure_reason == "rate_limited" and row.verification_status == "unavailable"
    assert "private" not in failure.model_dump_json()
    assert run(repo, settings, evaluator)[0].attempted == 0
    evaluator.interrupt = False
    report, status = run(repo, settings, evaluator, retry_failed=True)
    assert status == 200 and report.completed == 1 and report.attempted == 1
    row = report.comparisons[0]
    assert row.verification_status == "verified" and row.relationship == "direct_contradiction"
    assert row.sources_current and row.attempts == 2 and row.comparison_id == record["_id"]
    assert not repo.enqueue(candidate, settings)[1]
    assert run(repo, settings, evaluator, retry_failed=True)[0].attempted == 0
    assert evaluator.calls == ["primary", "secondary", "secondary"]


def test_lease_expiry_and_changed_inputs_fence_old_workers(queue):
    repo, settings, pair = queue
    record, _ = repo.enqueue(pair, settings)
    args = ("paper-b", "paper-a", False, [], 100)
    first = repo.reserve(*args)
    assert first and repo.reserve(*args) is None
    repo.collection.update_one({"_id": record["_id"]},
                               {"$set": {"lease_until": datetime.now(timezone.utc) - timedelta(seconds=1)}})
    second = repo.reserve(*args)
    assert second and second["lease_id"] != first["lease_id"]
    assert repo.update_owned(first, {"explanation": "old result"}, release=True) is None
    revised = pair.model_copy(update={"left": pair.left.model_copy(update={"page_number": 4})})
    updated, changed = repo.enqueue(revised, settings)
    assert changed and updated["comparison_id"] == record["comparison_id"]
    assert updated["primary"] is None and updated["processing_status"] == "pending"
    assert repo.update_owned(second, {"processing_status": "completed"}, release=True) is None
    assert repo.collection.count_documents({}) == 1


def test_source_revocation_before_evaluation_and_after_completion(queue):
    repo, settings, pair = queue
    repo.enqueue(pair, settings)
    evaluator = Evaluator()
    repo.database.research_claims.update_one({"claim_id": "b"}, {"$set": {"reverification_required": True}})
    report, _ = run(repo, settings, evaluator)
    assert report.stale == 1 and not evaluator.calls
    assert report.comparisons[0].verification_status == "stale"
    repo.database.research_claims.update_one({"claim_id": "b"}, {"$unset": {"reverification_required": ""}})
    assert repo.enqueue(pair, settings)[1]
    assert run(repo, settings, evaluator)[0].completed == 1
    repo.database.paper_chunks.update_one({"chunk_id": "chunk-a"}, {"$set": {"text": "Updated passage."}})
    rows, _ = repo.list_results(None, None, None, "completed", 0, 10)
    stored = repo.present(rows)[0]
    assert not stored.sources_current and stored.verification_status == "stale"
    assert stored.primary and stored.primary.left_quote == pair.left.supporting_evidence


@pytest.mark.parametrize("primary,secondary,expected", [
    ("agreement", "direct_contradiction", "insufficient_evidence"),
    ("insufficient_evidence", "insufficient_evidence", "insufficient_evidence"),
    ("partial_contradiction", "contextual_differences", "insufficient_evidence"),
])
def test_mixed_or_insufficient_evidence_stays_under_review(queue, primary, secondary, expected):
    _, _, pair = queue
    result = reconcile(decision(pair, primary), decision(pair, secondary))
    assert result["relationship"] == expected and result["verification_status"] == "review"


def test_only_independently_identified_mismatch_signals_are_reported(queue):
    _, _, pair = queue
    first = decision(pair, "contextual_differences", ["evidence_method", "population_context"])
    second = decision(pair, "contextual_differences", ["population_context"])
    result = reconcile(first, second)
    assert result["mismatch_types"] == ["population_context"] and result["verification_status"] == "review"


@pytest.mark.parametrize("signal", ["evidence_method", "claim_scope", "population_context"])
def test_doubt_about_evidence_or_confounding_blocks_verified_direct_contradiction(queue, signal):
    _, _, pair = queue
    assessment = decision(pair, mismatch_types=[signal])
    result = reconcile(assessment, assessment)
    assert result["relationship"] == "insufficient_evidence" and result["verification_status"] == "review"
    assert result["primary"]["relationship"] == "direct_contradiction" and result["mismatch_types"] == [signal]


def test_verifier_disabled_does_not_consume_primary_quota(queue):
    repo, settings, pair = queue
    settings.verifier_enabled = False
    repo.enqueue(pair, settings)
    evaluator = Evaluator()
    report, status = run(repo, settings, evaluator)
    assert status == 503 and report.comparisons[0].failure_reason == "missing_configuration"
    assert not evaluator.calls
