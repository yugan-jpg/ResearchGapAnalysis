from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import mongomock

from app.modules.claim_extraction.groq_verifier import VerificationOutcome
from app.modules.claim_extraction.repair import repair_paper
from app.modules.claim_extraction.repository import ClaimRepository
from app.modules.claim_extraction.selection import eligible_claims
from app.modules.claim_extraction.service import ClaimExtractionService
from app.modules.claim_extraction.verifier_errors import GroqTimeoutError
from app.schemas.claims import VerificationVerdict
from scripts.recover_reverification_queue import recover


OLD_REASON = "Relationship is not deterministically exact and requires independent re-verification."


def fixture_database():
    db = mongomock.MongoClient().phase4
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    db.papers.insert_one({"paper_id": "paper"})
    db.paper_chunks.insert_one({"paper_id": "paper", "chunk_id": "chunk", "chunk_index": 0, "page_number": 1,
                               "text": "The trial enrolled 20 patients."})
    base = {"paper_id": "paper", "chunk_id": "chunk", "page_number": 1, "category": "methodology",
            "claim_text": "The trial enrolled 20 patients.", "supporting_evidence": "The trial enrolled 20 patients.",
            "created_at": now, "verification_status": "review", "verification_verdict": "review",
            "verified_at": now, "verification_attempted_at": now, "verification_explanation": "Old assessment.",
            "evidence_supported": True, "numerical_consistency": True, "completeness": True, "category_correct": True}
    old = {"at": now + timedelta(hours=1), "reason": OLD_REASON, "previous": {"duplicate_of_claim_id": "old-target"}}
    db.research_claims.insert_many([
        {**base, "claim_id": "repaired", "duplicate_repair_history": [old]},
        {**base, "claim_id": "unrelated-review"},
        {**base, "claim_id": "already-checked", "duplicate_repair_history": [old],
         "verified_at": now + timedelta(hours=2), "verification_attempted_at": now + timedelta(hours=2)},
        {**base, "claim_id": "unrelated-accepted", "verification_status": "accepted"},
        {**base, "claim_id": "unrelated-rejected", "verification_status": "rejected", "duplicate_repair_history": [old]},
    ])
    return db


class FakeVerifier:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []
        self.settings = SimpleNamespace(verifier_enabled=True, verifier_batch_size=3,
                                        verifier_comparison_limit=0, verifier_model_name="test-verifier")

    def verify_batch(self, items):
        self.calls.append([item.claim_id for item in items])
        if self.fail:
            raise GroqTimeoutError("test timeout")
        return {item.claim_id: VerificationOutcome(
            status="accepted", verdict=VerificationVerdict.ACCEPTED, evidence_supported=True,
            numerical_consistency=True, completeness=True, category_correct=True,
            explanation="New assessment.", verifier_model="test-verifier") for item in items}


def service_for(db, verifier):
    return ClaimExtractionService(ClaimRepository(db), object(), "extractor", verifier)


def test_legacy_recovery_queues_only_stale_repairs_and_success_is_not_repeated():
    db = fixture_database()
    preview = recover(db, "paper")
    assert preview["claim_ids"] == ["repaired"] and preview["updated"] == 0
    assert recover(db, "paper", apply=True)["updated"] == 1
    assert recover(db, "paper", apply=True)["updated"] == 0
    original = db.research_claims.find_one({"claim_id": "repaired"})
    assert original["verification_status"] == "review"
    assert original["verification_history"][0]["previous"]["verification_explanation"] == "Old assessment."
    assert all(record["claim_id"] != "repaired" for record in eligible_claims(db, ["paper"]))

    verifier = FakeVerifier()
    service = service_for(db, verifier)
    # A queued repair must not be satisfied by an old cached assessment.
    service.repository.get_cached_verification = lambda _: (_ for _ in ()).throw(AssertionError("stale cache used"))
    result = service.verify_existing("paper", limit=5, force=False)
    assert result.processed_claims == 1 and verifier.calls == [["repaired"]]
    updated = db.research_claims.find_one({"claim_id": "repaired"})
    assert updated["verification_status"] == "accepted"
    assert updated["reverification_required"] is False
    assert "reverification_token" not in updated
    assert updated["verification_history"][-1]["previous"]["verification_explanation"] == "Old assessment."
    assert service.verify_existing("paper", limit=5, force=False).processed_claims == 0
    assert db.research_claims.find_one({"claim_id": "unrelated-review"})["verification_status"] == "review"


def test_failed_or_disabled_verifier_keeps_old_result_and_queue():
    db = fixture_database()
    recover(db, "paper", apply=True)
    verifier = FakeVerifier(fail=True)
    result = service_for(db, verifier).verify_existing("paper", 5, False)
    stored = db.research_claims.find_one({"claim_id": "repaired"})
    assert result.processed_claims == 1
    assert stored["verification_status"] == "review" and stored["reverification_required"] is True
    assert stored["verification_explanation"] == "Old assessment."
    assert service_for(db, FakeVerifier()).verify_existing("paper", 5, False).processed_claims == 1


def test_disabled_verifier_preserves_the_queued_claim():
    db = fixture_database()
    recover(db, "paper", apply=True)
    verifier = FakeVerifier()
    verifier.settings.verifier_enabled = False
    assert service_for(db, verifier).verify_existing("paper", 5, False).processed_claims == 1
    stored = db.research_claims.find_one({"claim_id": "repaired"})
    assert stored["verification_status"] == "review" and stored["reverification_required"] is True
    assert verifier.calls == []


def test_successful_review_is_not_retried_but_still_excluded():
    db = fixture_database()
    recover(db, "paper", apply=True)
    verifier = FakeVerifier()
    def review(items):
        return {item.claim_id: VerificationOutcome(
            status="review", verdict=VerificationVerdict.REVIEW, evidence_supported=True,
            numerical_consistency=True, completeness=True, category_correct=True,
            explanation="Human review remains necessary.", verifier_model="test-verifier") for item in items}
    verifier.verify_batch = review
    service = service_for(db, verifier)
    assert service.verify_existing("paper", 5, False).processed_claims == 1
    record = db.research_claims.find_one({"claim_id": "repaired"})
    assert record["verification_status"] == "review" and record["reverification_required"] is False
    assert service.verify_existing("paper", 5, False).processed_claims == 0
    assert "repaired" not in {c["claim_id"] for c in eligible_claims(db, ["paper"])}


def test_accepted_claim_with_new_repair_request_is_ineligible():
    db = fixture_database()
    db.research_claims.update_one({"claim_id": "unrelated-accepted"}, {"$set": {"reverification_required": True}})
    assert "unrelated-accepted" not in {c["claim_id"] for c in eligible_claims(db, ["paper"])}


def test_new_repair_request_during_verification_is_not_cleared():
    db = fixture_database()
    recover(db, "paper", apply=True)
    verifier = FakeVerifier()
    original = verifier.verify_batch
    def concurrent(items):
        db.research_claims.update_one({"claim_id": "repaired"}, {"$set": {"reverification_token": "new-request"}})
        return original(items)
    verifier.verify_batch = concurrent
    service_for(db, verifier).verify_existing("paper", 5, False)
    stored = db.research_claims.find_one({"claim_id": "repaired"})
    assert stored["reverification_required"] is True and stored["reverification_token"] == "new-request"
    assert stored["verification_explanation"] == "Old assessment."


def test_future_repairs_mark_only_cases_requiring_reverification():
    db = fixture_database()
    db.research_claims.insert_one({
        "paper_id": "paper", "claim_id": "new-repair", "chunk_id": "chunk", "page_number": 1,
        "claim_text": "The trial enrolled twenty patients in total.", "supporting_evidence": "The trial enrolled 20 patients.",
        "verification_status": "review", "verification_verdict": "review", "duplicate_of_claim_id": "other",
        "evidence_supported": True, "completeness": True, "numerical_consistency": True, "category_correct": True,
    })
    result = repair_paper(db, "paper", apply=True)
    assert result["applied"] >= 1
    assert db.research_claims.find_one({"claim_id": "new-repair"})["reverification_required"] is True


def test_verify_endpoint_processes_only_the_queued_claim(monkeypatch):
    from fastapi.testclient import TestClient
    import app.api.routes.claims as claims_routes
    from app.main import app
    db = fixture_database()
    recover(db, "paper", apply=True)
    verifier = FakeVerifier()
    monkeypatch.setattr(claims_routes, "get_database", lambda: db)
    monkeypatch.setattr(claims_routes, "get_settings", lambda: SimpleNamespace(llm_model_name="extractor", verifier_enabled=True))
    monkeypatch.setattr(claims_routes, "get_embedding_service", lambda: None)
    monkeypatch.setattr(claims_routes, "QwenClaimClient", lambda _: object())
    monkeypatch.setattr(claims_routes, "GroqClaimVerifier", lambda _: verifier)
    client = TestClient(app)
    response = client.post("/api/claims/verify/paper", json={"limit": 5, "force": False})
    assert response.status_code == 200
    assert response.json()["processed_claims"] == 1
    assert verifier.calls == [["repaired"]]
    assert client.post("/api/claims/verify/paper", json={"limit": 5, "force": False}).json()["processed_claims"] == 0
