from datetime import datetime, timezone

import mongomock
import pytest

from app.modules.claim_extraction.errors import ClaimStorageError, PaperNotFoundError
from app.modules.claim_extraction.selection import same_publication, trusted_claims


def source_claim(db, claim_id, *, paper_id="paper", chunk_text="Original passage.", **fields):
    record = {
        "claim_id": claim_id, "paper_id": paper_id, "chunk_id": f"{paper_id}-{claim_id}",
        "page_number": 2, "claim_text": f"Claim {claim_id}.",
        "supporting_evidence": "Original passage.", "category": "finding",
        "verification_status": "accepted", "evidence_supported": True,
        "verification_verdict": "accepted", "verifier_model": "verifier", "verified_at": datetime.now(timezone.utc),
        "numerical_consistency": True, "completeness": True, "category_correct": True,
        **fields,
    }
    db.research_claims.insert_one(record)
    db.paper_chunks.insert_one({
        "chunk_id": record["chunk_id"], "paper_id": paper_id,
        "page_number": 2, "text": chunk_text,
    })


def test_cursor_advances_even_when_an_entire_page_fails_source_checks():
    db = mongomock.MongoClient().sources
    source_claim(db, "c", factual_verdict="accepted")
    source_claim(db, "b", page_number=1)
    source_claim(db, "a", chunk_text="The quotation is absent.")
    items, cursor, scanned = trusted_claims(db, paper_id="paper", limit=2)
    assert (items, cursor, scanned) == ([], "b", 2)
    items, cursor, scanned = trusted_claims(db, paper_id="paper", after_claim_id=cursor, limit=2)
    assert ([c["claim_id"] for c in items], cursor, scanned) == (["c"], None, 1)


def test_claim_and_chunk_filters_cannot_return_sources_from_another_paper():
    db = mongomock.MongoClient().sources
    source_claim(db, "a")
    source_claim(db, "b")
    source_claim(db, "c", paper_id="unrelated")
    items, cursor, scanned = trusted_claims(
        db, paper_id="paper", chunk_ids=["paper-b", "unrelated-c"], claim_ids=["b", "c"],
    )
    assert ([c["claim_id"] for c in items], cursor, scanned) == (["b"], None, 1)
    assert trusted_claims(db, paper_id="paper", chunk_ids=[]) == ([], None, 0)
    assert trusted_claims(db, paper_id="paper", claim_ids=[]) == ([], None, 0)


def test_stale_accepted_status_cannot_override_contradictory_verification_or_source_metadata():
    db = mongomock.MongoClient().sources
    for i, fields in enumerate([
        {"verification_status": "review"}, {"factual_verdict": "rejected"},
        {"factual_verdict": "review"}, {"verification_verdict": "rejected"},
        {"source_verified": False}, {"duplicate_assessment": "unresolved"},
        {"reverification_required": True}, {"completeness": False}, {"numerical_consistency": False},
        {"verified_at": None}, {"verifier_model": ""}, {"verification_verdict": None},
    ]):
        source_claim(db, str(i), **fields)
    source_claim(db, "eligible")
    records, _, _ = trusted_claims(db, paper_id="paper")
    assert [r["claim_id"] for r in records] == ["eligible"]


def test_publication_identity_follows_ancestor_doi_and_source_file_hash():
    db = mongomock.MongoClient().sources
    db.papers.insert_many([
        {"paper_id": "original", "content_sha256": "abc", "metadata": {"doi": "10.1000/Trial"}},
        {"paper_id": "v2", "version_of_paper_id": "original"},
        {"paper_id": "v3", "version_of_paper_id": "v2"},
        {"paper_id": "alternate", "metadata": {"doi": "https://doi.org/10.1000/TRIAL"}},
        {"paper_id": "hash-only", "source_content_sha256": "ABC"},
        {"paper_id": "independent", "original_filename": "file.pdf"},
    ])
    for same in ("original", "v2", "v3", "alternate", "hash-only"):
        assert same_publication(db, "v3", same)
    assert not same_publication(db, "v3", "independent")
    with pytest.raises(PaperNotFoundError):
        same_publication(db, "independent", "missing")
    with pytest.raises(PaperNotFoundError):
        same_publication(db, "missing", "missing")


@pytest.mark.parametrize("second_parent", ["first", "missing", 9])
def test_unverifiable_lineage_cannot_count_as_independent_evidence(second_parent):
    db = mongomock.MongoClient().sources
    db.papers.insert_many([
        {"paper_id": "first", "version_of_paper_id": "second"},
        {"paper_id": "second", "version_of_paper_id": second_parent},
        {"paper_id": "independent"},
    ])
    with pytest.raises(ClaimStorageError):
        same_publication(db, "independent", "first")
