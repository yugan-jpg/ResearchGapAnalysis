from datetime import datetime, timezone
from types import SimpleNamespace

import mongomock
import pytest

from app.modules.claim_extraction.groq_verifier import VerificationInput, VerificationOutcome
from app.modules.claim_extraction.repository import ClaimChunk, ClaimRepository
from app.modules.claim_extraction.resolution import resolve_paper
from app.modules.claim_extraction.selection import eligible_claims
from app.modules.claim_extraction.service import ClaimExtractionService, PreparedClaim
from app.schemas.claims import ClaimCategory, ExtractedClaim, VerificationVerdict


def record(claim_id, text, *, factual="accepted", status="review", complete=True, **kwargs):
    return {
        "paper_id": "p", "claim_id": claim_id, "chunk_id": "chunk-" + claim_id,
        "page_number": 1, "category": "finding", "claim_text": text,
        "supporting_evidence": text, "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "verification_status": status, "verification_verdict": status,
        "factual_verdict": factual, "evidence_supported": True, "numerical_consistency": True,
        "completeness": complete, "category_correct": True, "verifier_model": "verifier",
        "verified_at": datetime(2025, 1, 1, tzinfo=timezone.utc), **kwargs,
    }


def database(*claims):
    db = mongomock.MongoClient().claims
    db.papers.insert_one({"paper_id": "p"})
    for claim in claims:
        db.research_claims.insert_one(dict(claim))
        db.paper_chunks.insert_one({"paper_id": "p", "chunk_id": claim["chunk_id"], "page_number": 1,
                                    "text": claim["supporting_evidence"], "chunk_index": 0})
    return db


def complete_and_legacy(incomplete_text):
    full = record(
        "full", "At North Clinic, coffee improved memory in adults aged 20 to 25 after 8 hours of sleep.",
        duplicate_assessment="confirmed_equivalent", duplicate_candidate_id="short",
    )
    short = record("short", incomplete_text, factual=None, complete=False)
    return full, short


def test_missing_entity_and_sleep_condition_release_complete_claim_only():
    full, short = complete_and_legacy("Coffee improved memory in adults aged 20 to 25.")
    db = database(full, short)
    assert resolve_paper(db, "p", apply=False)["planned"] == 1
    assert resolve_paper(db, "p", apply=True)["applied"] == 1
    assert {item["claim_id"] for item in eligible_claims(db, ["p"])} == {"full"}
    remaining = db.research_claims.find_one({"claim_id": "short"})
    assert remaining["verification_status"] == "review"
    assert remaining["factual_verdict"] is None
    assert not remaining.get("duplicate_of_claim_id")
    assert resolve_paper(db, "p", apply=True)["applied"] == 0


@pytest.mark.parametrize("conflict", [
    "At South Clinic, coffee improved memory in adults aged 20 to 25.",
    "Coffee improved memory in children aged 20 to 25.",
    "Coffee improved attention in adults aged 20 to 25.",
    "Coffee worsened memory in adults aged 20 to 25.",
    "Coffee did not improve memory in adults aged 20 to 25.",
    "Coffee improved memory in adults aged 20 to 25 after 6 hours of sleep.",
    "Coffee improved memory in adults aged 30 to 35.",
    "Memory improved coffee in adults aged 20 to 25.",
])
def test_conflicting_or_reordered_context_is_not_treated_as_incomplete_match(conflict):
    full, short = complete_and_legacy(conflict)
    db = database(full, short)
    assert resolve_paper(db, "p", apply=True)["applied"] == 0


def test_incomplete_legacy_claim_requires_verified_metadata_and_exact_source():
    full, short = complete_and_legacy("Coffee improved memory in adults aged 20 to 25.")
    short.pop("verified_at")
    db = database(full, short)
    assert resolve_paper(db, "p", apply=True)["applied"] == 0
    db.research_claims.update_one({"claim_id": "short"}, {"$set": {"verified_at": datetime.now(timezone.utc)}})
    db.paper_chunks.update_one({"chunk_id": "chunk-short"}, {"$set": {"text": "No matching evidence."}})
    assert resolve_paper(db, "p", apply=True)["applied"] == 0


def test_legacy_complete_review_is_not_promoted_as_an_incomplete_claim():
    full, short = complete_and_legacy("Coffee improved memory in adults aged 20 to 25.")
    short["completeness"] = True
    db = database(full, short)
    assert resolve_paper(db, "p", apply=True)["applied"] == 0


def test_rejected_similarity_candidate_no_longer_blocks_supported_claim():
    valid = record("valid", "The trial improved memory in adults.", duplicate_assessment="unresolved",
                   duplicate_type="partial_overlap", duplicate_candidate_id="bad")
    bad = record("bad", "The trial improved memory in children.", status="rejected",
                 factual="rejected", evidence_supported=False, duplicate_repair_history=[{"reason": "retain"}])
    db = database(valid, bad)
    assert resolve_paper(db, "p", apply=True)["applied"] == 1
    assert {item["claim_id"] for item in eligible_claims(db, ["p"])} == {"valid"}
    rejected = db.research_claims.find_one({"claim_id": "bad"})
    assert rejected["verification_status"] == "rejected"
    assert rejected["duplicate_repair_history"] == [{"reason": "retain"}]
    assert resolve_paper(db, "p", apply=True)["applied"] == 0


@pytest.mark.parametrize("other_status,other_factual", [("accepted", "accepted"), ("review", "review"), ("unavailable", None)])
def test_other_viable_candidates_still_block_automatic_release(other_status, other_factual):
    valid = record("valid", "The trial improved memory in adults.", duplicate_assessment="unresolved",
                   duplicate_type="partial_overlap", duplicate_candidate_id="bad")
    bad = record("bad", "The trial improved memory in children.", status="rejected",
                 factual="rejected", evidence_supported=False)
    other = record("other", "The trial improved memory in older adults.", status=other_status,
                   factual=other_factual)
    db = database(valid, bad, other)
    assert resolve_paper(db, "p", apply=True)["applied"] == 0
    assert db.research_claims.find_one({"claim_id": "valid"})["verification_status"] == "review"


def test_genuine_factual_review_is_not_released():
    valid = record("valid", "The trial improved memory in adults.", factual="review",
                   duplicate_assessment="unresolved", duplicate_candidate_id="bad")
    bad = record("bad", "The trial improved memory in children.", status="rejected", factual="rejected",
                 evidence_supported=False)
    db = database(valid, bad)
    assert resolve_paper(db, "p", apply=True)["applied"] == 0


def test_rejected_candidates_do_not_consume_limit_before_eligible_candidates():
    db = mongomock.MongoClient().claims
    evidence = "The analysis used a scientific method."
    db.research_claims.insert_many([
        {"paper_id": "p", "claim_id": f"bad-{index}", "category": "methodology",
         "claim_text": "A scientific method was used.", "supporting_evidence": evidence,
         "verification_status": "rejected", "evidence_supported": False,
         "created_at": datetime(2025, 1, 2, tzinfo=timezone.utc)}
        for index in range(251)
    ])
    db.research_claims.insert_one({"paper_id": "p", "claim_id": "viable", "category": "methodology",
                                   "claim_text": "A scientific method was used.", "supporting_evidence": evidence,
                                   "verification_status": "accepted", "evidence_supported": True,
                                   "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc)})
    extracted = ExtractedClaim(claim_text="The scientific method was used.", category="methodology",
                               supporting_evidence=evidence)
    candidates = ClaimRepository(db).find_duplicate_candidates("p", extracted, 2)
    assert [candidate["claim_id"] for candidate in candidates] == ["viable"]


def test_same_batch_rejected_candidate_is_filtered_after_all_verdicts():
    db = mongomock.MongoClient().claims
    repo = ClaimRepository(db)
    service = ClaimExtractionService(repo, object(), "extractor")
    verifier = SimpleNamespace(settings=SimpleNamespace(verifier_enabled=True, verifier_batch_size=1))
    def verify_batch(items):
        return {item.claim_id: VerificationOutcome(
            status="rejected" if item.claim_id == "bad" else "accepted",
            verdict=VerificationVerdict.REJECTED if item.claim_id == "bad" else VerificationVerdict.ACCEPTED,
            factual_verdict=VerificationVerdict.REJECTED if item.claim_id == "bad" else VerificationVerdict.ACCEPTED,
            evidence_supported=item.claim_id != "bad", numerical_consistency=True, completeness=True, category_correct=True,
        ) for item in items}
    verifier.verify_batch = verify_batch
    service.verifier = verifier
    good = ExtractedClaim(claim_text="The trial improved memory in adults.", category=ClaimCategory.FINDING,
                          supporting_evidence="The trial improved memory in adults.")
    bad = ExtractedClaim(claim_text="The trial improved memory in children.", category=ClaimCategory.FINDING,
                         supporting_evidence=good.supporting_evidence)
    chunk = ClaimChunk("chunk", "p", 0, 1, good.supporting_evidence)
    def prepared(cid, claim, candidates):
        inputs = VerificationInput(cid, "p", "chunk", 1, claim.claim_text, claim.category, claim.supporting_evidence,
                                   chunk.text, True, True, comparison_candidates=tuple(candidates))
        return PreparedClaim(cid, chunk, claim, inputs, "no-cache-" + cid)
    good_item = prepared("good", good, [{"claim_id": "bad", "claim_text": bad.claim_text,
                                       "supporting_evidence": bad.supporting_evidence}])
    bad_item = prepared("bad", bad, [])
    outcomes = service._verify_prepared([good_item, bad_item])
    assert outcomes["good"].status == "accepted"
    assert outcomes["bad"].status == "rejected"


@pytest.mark.parametrize("other_candidate,factual,expected", [
    (False, VerificationVerdict.ACCEPTED, "accepted"),
    (True, VerificationVerdict.ACCEPTED, "review"),
    (False, VerificationVerdict.REVIEW, "review"),
])
def test_cached_similarity_to_rejected_target_checks_other_candidates(other_candidate, factual, expected):
    db = mongomock.MongoClient().claims
    service = ClaimExtractionService(ClaimRepository(db), object(), "extractor")
    claim = ExtractedClaim(claim_text="The treatment improved memory in adults.", category="finding",
                           supporting_evidence="The treatment improved memory in adults.")
    chunk = ClaimChunk("chunk", "p", 0, 1, claim.supporting_evidence)
    candidates = [{"claim_id": "bad", "claim_text": "The treatment improved memory in children.",
                   "supporting_evidence": claim.supporting_evidence}]
    if other_candidate:
        candidates.append({"claim_id": "other", "claim_text": "Memory improved among adults.",
                           "supporting_evidence": claim.supporting_evidence})
    inputs = VerificationInput("valid", "p", "chunk", 1, claim.claim_text, claim.category,
                               claim.supporting_evidence, chunk.text, True, True, comparison_candidates=tuple(candidates))
    item = PreparedClaim("valid", chunk, claim, inputs, "key")
    outcomes = {
        "valid": VerificationOutcome(status="review", verdict=VerificationVerdict.REVIEW, factual_verdict=factual,
                                     evidence_supported=True, numerical_consistency=True, completeness=True,
                                     category_correct=True, duplicate_assessment="unresolved",
                                     duplicate_candidate_id="bad", duplicate_type="partial_overlap"),
        "bad": VerificationOutcome(status="rejected", verdict=VerificationVerdict.REJECTED, evidence_supported=False),
    }
    if other_candidate:
        outcomes["other"] = VerificationOutcome(status="accepted", verdict=VerificationVerdict.ACCEPTED,
                                                 evidence_supported=True)
    result = service._safe_duplicate_targets([item], outcomes)["valid"]
    assert result.status == expected
    if other_candidate:
        assert result.duplicate_candidate_id == "other"
