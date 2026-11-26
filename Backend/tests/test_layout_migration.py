from hashlib import sha256
from pathlib import Path

import mongomock
import pytest

from scripts.backfill_claim_sources import backfill
from scripts.reingest_layout_version import reingest


def test_reingest_is_dry_run_then_additive_and_repeatable(tmp_path: Path, pdf_factory):
    pdf = tmp_path / "original.pdf"
    content = pdf_factory(["First paragraph with scientific findings."])
    pdf.write_bytes(content)
    db = mongomock.MongoClient().test
    original = {"paper_id": "original", "content_sha256": sha256(content).hexdigest(),
                "original_filename": "original.pdf"}
    db.papers.insert_one(original.copy())
    db.paper_chunks.insert_one({"paper_id": "original", "chunk_id": "old", "embedding": [1, 2]})
    db.research_claims.insert_one({"paper_id": "original", "claim_id": "old-claim"})
    preview = reingest(db, "original", pdf)
    assert not preview["applied"] and db.papers.count_documents({}) == 1
    result = reingest(db, "original", pdf, apply=True)
    assert result["version_paper_id"] == preview["version_paper_id"]
    assert reingest(db, "original", pdf, apply=True) == result
    assert db.papers.count_documents({}) == 2
    assert db.paper_chunks.find_one({"chunk_id": "old"})["embedding"] == [1, 2]
    assert db.research_claims.count_documents({"paper_id": "original"}) == 1
    assert db.paper_chunks.find_one({"paper_id": result["version_paper_id"]})["embedding_status"] == "pending"
    other = tmp_path / "other.pdf"
    other.write_bytes(pdf_factory(["Different source document."]))
    with pytest.raises(ValueError):
        reingest(db, "original", other, apply=True)


def test_backfill_only_adds_existing_verifiable_source():
    db = mongomock.MongoClient().test
    db.paper_chunks.insert_one({"paper_id": "p", "chunk_id": "c", "page_number": 2, "text": "Original evidence."})
    db.research_claims.insert_many([
        {"paper_id": "p", "claim_id": "good", "chunk_id": "c", "page_number": 2, "supporting_evidence": "Original evidence."},
        {"paper_id": "p", "claim_id": "bad", "chunk_id": "c", "page_number": 2, "supporting_evidence": "Invented evidence."},
    ])
    assert backfill(db, "p") == {"examined": 2, "eligible": 1, "updated": 0, "skipped": 1}
    assert backfill(db, "p", True)["updated"] == 1
    assert db.research_claims.find_one({"claim_id": "good"})["source_references"][0]["chunk_id"] == "c"
    assert "source_references" not in db.research_claims.find_one({"claim_id": "bad"})
    assert backfill(db, "p", True)["updated"] == 0
