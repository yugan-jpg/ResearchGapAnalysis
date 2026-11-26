import mongomock

from app.modules.claim_extraction.quality import measurement_tokens, normalize_claim, source_total_warning
from app.modules.claim_extraction.repository import ClaimRepository
from app.modules.claim_extraction.selection import eligible_claims


def test_normalization_keeps_scientific_differences():
    assert normalize_claim("  Accuracy  improved by 12%. ") == normalize_claim("accuracy improved by 12%")
    assert normalize_claim("The treatment did not reduce mortality.") != normalize_claim("The treatment reduced mortality.")
    assert normalize_claim("A dose of 5 mg.") != normalize_claim("A dose of 5 g.")


def test_ambiguous_total_gets_review_warning_not_correction():
    warning = source_total_warning("A total of 1146 samples were collected: 321, 825, and 155.")
    assert "1301" in warning
    assert "overlapping" in warning
    assert source_total_warning("A total of 1146 samples were collected: 321 and 825.") is None


def test_gene_identifiers_and_superscript_citations_are_not_measurements():
    claim = "In 2017, plasmepsin2/3 and KEL1/PLA1 and pfk13 H97Y were tested in 321 samples."
    evidence = "In 2017¹⁶–¹⁹, plasmepsin2/3 and KEL1/PLA1 and pfk13 H97Y were tested in 321 samples [12]."
    assert measurement_tokens(claim) == {"2017", "321"}
    assert measurement_tokens(claim) <= measurement_tokens(evidence)
    assert measurement_tokens("12mg and 5%") == {"12", "5%"}
    assert measurement_tokens("825 samples") != measurement_tokens("321 samples")


def test_model_change_does_not_reprocess_completed_chunk():
    db = mongomock.MongoClient().claims
    db.paper_chunks.insert_one({"paper_id": "p", "chunk_id": "c", "chunk_index": 0,
                                "page_number": 1, "text": "Source text.",
                                "claim_extraction_status": "completed", "claim_extraction_model": "old"})
    repo = ClaimRepository(db)
    assert repo.get_chunks("p", "new", False, 10) == []
    assert len(repo.get_chunks("p", "new", True, 10)) == 1


def test_selection_preserves_cross_paper_corroboration_and_excludes_reviews():
    db = mongomock.MongoClient().claims
    common = {"verification_status": "accepted", "evidence_supported": True,
              "numerical_consistency": True, "completeness": True, "category_correct": True,
              "chunk_id": "c", "page_number": 1, "supporting_evidence": "Evidence."}
    db.research_claims.insert_many([
        {**common, "claim_id": "a", "paper_id": "p1"},
        {**common, "claim_id": "b", "paper_id": "p2"},
        {**common, "claim_id": "c", "paper_id": "p2", "verification_status": "review"},
        {**common, "claim_id": "d", "paper_id": "p2", "duplicate_of_claim_id": "b", "duplicate_type": "partial_overlap"},
    ])
    db.paper_chunks.insert_many([
        {"paper_id": "p1", "chunk_id": "c", "page_number": 1, "text": "Evidence."},
        {"paper_id": "p2", "chunk_id": "c", "page_number": 1, "text": "Evidence."},
    ])
    assert {c["claim_id"] for c in eligible_claims(db, ["p1", "p2"])} == {"a", "b"}
