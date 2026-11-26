"""Select traceable, independently verified claims for future consumers."""

from typing import Any
from app.modules.claim_extraction.quality import normalize_claim


def eligible_claims(database: Any, paper_ids: list[str], limit: int = 200) -> list[dict[str, Any]]:
    """Keep each paper's independent evidence, including unique partial overlaps."""
    if not paper_ids or limit < 1:
        return []
    candidates = database.research_claims.find(
        {
            "paper_id": {"$in": paper_ids},
            "verification_status": "accepted",
            "reverification_required": {"$ne": True},
            "evidence_supported": True,
            "numerical_consistency": True,
            "completeness": True,
            "category_correct": True,
            "duplicate_type": {"$nin": ["exact", "semantic"]},
            "duplicate_assessment": {"$nin": ["confirmed_duplicate", "confirmed_equivalent", "unresolved"]},
            "$or": [
                {"duplicate_of_claim_id": None},
                {"duplicate_type": "related_distinct"},
            ],
        },
        {"_id": 0},
    ).limit(limit)
    selected = []
    seen = set()
    papers = {p["paper_id"]: p for p in database.papers.find({"paper_id": {"$in": paper_ids}}, {"paper_id": 1, "version_of_paper_id": 1})}
    for record in candidates:
        if not all(record.get(field) for field in ("claim_id", "paper_id", "chunk_id", "page_number", "supporting_evidence")):
            continue
        chunk = database.paper_chunks.find_one(
            {"paper_id": record["paper_id"], "chunk_id": record["chunk_id"], "page_number": record["page_number"]},
            {"text": 1},
        )
        if chunk and record["supporting_evidence"] in chunk.get("text", ""):
            publication = papers.get(record["paper_id"], {}).get("version_of_paper_id") or record["paper_id"]
            identity = (publication, normalize_claim(record.get("claim_text", "")))
            if identity not in seen:
                seen.add(identity)
                selected.append(record)
    return selected
