from datetime import datetime, timezone

import mongomock

from app.modules.claim_extraction.groq_verifier import GroqClaimVerifier, VerificationInput, VerificationOutcome
from app.modules.claim_extraction.resolution import plan_resolutions, resolve_paper
from app.modules.claim_extraction.selection import eligible_claims
from app.modules.claim_extraction.repository import ClaimRepository
from app.modules.claim_extraction.service import ClaimExtractionService
from app.schemas.claims import ClaimCategory, VerificationVerdict
from app.schemas.verification import GroqVerificationPayload
from scripts.queue_claim_reverification import queue_claims


def _record(claim_id, text, paper="paper", status="review", **extra):
    return {
        "claim_id": claim_id, "paper_id": paper, "chunk_id": f"chunk-{claim_id}", "page_number": 1,
        "claim_text": text, "category": "finding", "supporting_evidence": f"Source passage for {claim_id}.",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "verification_status": status, "verification_verdict": status, "factual_verdict": "accepted",
        "verified_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "verifier_model": "independent-verifier",
        "evidence_supported": True, "numerical_consistency": True, "completeness": True,
        "category_correct": True, **extra,
    }


def _database(*records):
    db = mongomock.MongoClient().claims
    db.research_claims.insert_many([dict(record) for record in records])
    for paper_id in {record["paper_id"] for record in records}:
        db.papers.insert_one({"paper_id": paper_id})
    for record in records:
        db.paper_chunks.insert_one({"paper_id": record["paper_id"], "chunk_id": record["chunk_id"],
                                    "page_number": 1, "text": record["supporting_evidence"]})
    return db


def test_review_only_equivalent_group_gets_one_canonical_and_keeps_audit():
    a = _record("a", "The intervention reduced blood pressure in adults.",
                duplicate_assessment="confirmed_equivalent", duplicate_candidate_id="b")
    b = _record("b", "Blood pressure was reduced by the intervention in adults.",
                duplicate_assessment="confirmed_equivalent", duplicate_candidate_id="a")
    db = _database(a, b)
    preview = resolve_paper(db, "paper")
    assert preview["planned"] == 2 and preview["applied"] == 0
    assert resolve_paper(db, "paper", apply=True)["applied"] == 2
    records = list(db.research_claims.find({}))
    canonical = next(r for r in records if r["duplicate_assessment"] == "canonical")
    duplicate = next(r for r in records if r["duplicate_assessment"] == "confirmed_duplicate")
    assert canonical["claim_id"] == "b"
    assert duplicate["duplicate_of_claim_id"] == "b" and canonical["duplicate_of_claim_id"] is None
    assert len(eligible_claims(db, ["paper"])) == 1
    assert all(r["verification_status"] == "accepted" and r["factual_verdict"] == "accepted" for r in records)
    assert all(len(r["canonical_resolution_history"]) == 1 for r in records)
    assert resolve_paper(db, "paper", apply=True)["applied"] == 0


def test_explicit_partial_overlap_preserves_supported_unique_claim():
    a = _record("a", "The intervention reduced blood pressure in adults.",
                duplicate_assessment="partial_overlap", duplicate_type="partial_overlap", duplicate_candidate_id="b")
    b = _record("b", "The intervention reduced blood pressure and heart rate.", status="accepted")
    db = _database(a, b)
    resolve_paper(db, "paper", apply=True)
    assert {r["claim_id"] for r in eligible_claims(db, ["paper"])} == {"a", "b"}


def test_uncertain_and_unsupported_claims_are_never_promoted():
    unresolved = _record("a", "The intervention reduced blood pressure.",
                         duplicate_assessment="unresolved", duplicate_candidate_id="b")
    other = _record("b", "The intervention reduced heart rate.", status="accepted")
    unsupported = _record("c", "The intervention reduced blood pressure.",
                          factual_verdict="rejected", evidence_supported=False)
    db = _database(unresolved, other, unsupported)
    assert resolve_paper(db, "paper", apply=True)["applied"] == 0
    assert {r["claim_id"] for r in eligible_claims(db, ["paper"])} == {"b"}


def test_distinct_papers_are_not_merged_and_legacy_reviews_stay_unresolved():
    a = _record("a", "The trial reduced fever.", paper="one", status="accepted")
    b = _record("b", "The trial reduced fever.", paper="two", status="accepted")
    c = _record("c", "The trial reduced fever.", paper="one", factual_verdict=None)
    db = _database(a, b, c)
    assert resolve_paper(db, "one", apply=True)["applied"] == 1
    legacy = db.research_claims.find_one({"claim_id": "c"})
    assert legacy["verification_status"] == "review" and legacy["duplicate_of_claim_id"] == "a"
    assert {r["claim_id"] for r in eligible_claims(db, ["one", "two"])} == {"a", "b"}


def test_false_equivalence_different_population_or_negation_is_not_resolved():
    a = _record("a", "The trial reduced fever in adults.",
                duplicate_assessment="confirmed_equivalent", duplicate_candidate_id="b")
    b = _record("b", "The trial did not reduce fever in children.")
    db = _database(a, b)
    assert resolve_paper(db, "paper", apply=True)["applied"] == 0


def test_different_first_word_geography_and_acronyms_are_not_merged():
    a = _record("a", "Attapeu showed lower prevalence after treatment.",
                duplicate_assessment="confirmed_equivalent", duplicate_candidate_id="b")
    b = _record("b", "Sainyabuli showed lower prevalence after treatment.")
    c = _record("c", "The PPQ-R prevalence declined in adults.",
                duplicate_assessment="confirmed_equivalent", duplicate_candidate_id="d")
    d = _record("d", "The ART-R prevalence declined in adults.")
    db = _database(a, b, c, d)
    assert resolve_paper(db, "paper", apply=True)["applied"] == 0


def test_equivalence_chain_without_pairwise_confirmation_stays_unresolved():
    a = _record("a", "The intervention reduced blood pressure in adults.",
                duplicate_assessment="confirmed_equivalent", duplicate_candidate_id="b")
    b = _record("b", "Blood pressure decreased in adults following the intervention.",
                duplicate_assessment="confirmed_equivalent", duplicate_candidate_id="c")
    c = _record("c", "Blood pressure declined in adults following the intervention.")
    db = _database(a, b, c)
    assert resolve_paper(db, "paper", apply=True)["applied"] == 0


def test_rejected_and_unavailable_claims_never_become_canonical():
    for status in ("rejected", "unavailable"):
        a = _record("a", "The intervention reduced blood pressure in adults.", status=status,
                    duplicate_assessment="confirmed_equivalent", duplicate_candidate_id="b")
        b = _record("b", "Blood pressure was reduced by the intervention in adults.",
                    duplicate_assessment="confirmed_equivalent", duplicate_candidate_id="a")
        db = _database(a, b)
        assert resolve_paper(db, "paper", apply=True)["applied"] == 0


def test_more_complete_supported_claim_survives_incomplete_paraphrase():
    complete = _record("a", "The intervention reduced blood pressure in adults after treatment.",
                       duplicate_assessment="confirmed_equivalent", duplicate_candidate_id="b")
    incomplete = _record("b", "The intervention reduced blood pressure in adults.",
                         factual_verdict="review", completeness=False)
    db = _database(complete, incomplete)
    assert resolve_paper(db, "paper", apply=True)["applied"] == 1
    assert {r["claim_id"] for r in eligible_claims(db, ["paper"])} == {"a"}
    assert db.research_claims.find_one({"claim_id": "b"})["verification_status"] == "review"


def test_verifier_keeps_factual_acceptance_separate_from_partial_overlap():
    from app.core.config import Settings
    verifier = GroqClaimVerifier(Settings(_env_file=None))
    item = VerificationInput("a", "p", "c", 1, "The trial reduced fever.",
                             ClaimCategory.FINDING, "trial reduced fever", "trial reduced fever",
                             True, True, comparison_candidates=({"claim_id": "b"},))
    payload = GroqVerificationPayload.model_validate({"verifications": [{
        "claim_id": "a", "verdict": "accepted", "evidence_supported": True,
        "numerical_consistency": True, "completeness": True, "category_correct": True,
        "explanation": "Supported.", "duplicate_of_claim_id": "b", "duplicate_type": "partial_overlap",
        "duplicate_explanation": "One outcome overlaps."
    }]})
    outcome = verifier._to_outcomes([item], payload)["a"]
    assert outcome.factual_verdict == VerificationVerdict.ACCEPTED
    assert outcome.status == "accepted" and outcome.duplicate_assessment == "partial_overlap"


def test_automatic_resolution_after_reverification_without_prior_accepted_target():
    a = _record("a", "The intervention reduced blood pressure in adults.",
                reverification_required=True, reverification_token="a")
    b = _record("b", "Blood pressure was reduced by the intervention in adults.",
                reverification_required=True, reverification_token="b")
    db = _database(a, b)
    from types import SimpleNamespace
    class Verifier:
        settings = SimpleNamespace(verifier_enabled=True, verifier_batch_size=3,
                                   verifier_comparison_limit=5, verifier_model_name="test")
        def verify_batch(self, inputs):
            return {item.claim_id: VerificationOutcome(
                status="review", verdict=VerificationVerdict.REVIEW, factual_verdict=VerificationVerdict.ACCEPTED,
                evidence_supported=True, numerical_consistency=True, completeness=True, category_correct=True,
                explanation="Supported.", verifier_model="test", duplicate_assessment="confirmed_equivalent", duplicate_type="semantic",
                duplicate_of_claim_id="b" if item.claim_id == "a" else "a",
                duplicate_candidate_id="b" if item.claim_id == "a" else "a") for item in inputs}
    service = ClaimExtractionService(ClaimRepository(db), object(), "extractor", Verifier())
    result = service.verify_existing("paper", 5)
    assert result.processed_claims == 2
    assert len(eligible_claims(db, ["paper"])) == 1
    assert service.verify_existing("paper", 5).processed_claims == 0


def test_targeted_legacy_queue_is_opt_in_and_idempotent():
    a = _record("a", "The intervention reduced blood pressure in adults.", factual_verdict=None)
    b = _record("b", "Blood pressure was reduced by the intervention in adults.", status="accepted")
    db = _database(a, b)
    assert queue_claims(db, "paper", ["a", "b"])["eligible"] == ["a"]
    assert "reverification_required" not in db.research_claims.find_one({"claim_id": "a"})
    assert queue_claims(db, "paper", ["a", "b"], apply=True)["updated"] == 1
    assert queue_claims(db, "paper", ["a", "b"], apply=True)["updated"] == 0
    stored = db.research_claims.find_one({"claim_id": "a"})
    assert stored["reverification_required"] is True
    assert stored["verification_history"][0]["previous"]["verification_status"] == "review"
