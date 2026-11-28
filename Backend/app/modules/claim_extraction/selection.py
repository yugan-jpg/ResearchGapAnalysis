"""Select traceable, independently verified claims for downstream comparison."""

from typing import Any

from app.modules.claim_extraction.errors import ClaimStorageError, PaperNotFoundError
from app.modules.claim_extraction.quality import normalize_claim


def _eligibility_filter() -> dict[str, Any]:
    return {
        "verification_status": "accepted",
        "reverification_required": {"$ne": True},
        "evidence_supported": True,
        "numerical_consistency": True,
        "completeness": True,
        "category_correct": True,
        "duplicate_type": {"$nin": ["exact", "semantic"]},
        "duplicate_assessment": {"$nin": ["confirmed_duplicate", "confirmed_equivalent", "unresolved"]},
        "$or": [{"duplicate_of_claim_id": None}, {"duplicate_type": "related_distinct"}],
    }


def _source_checked(database: Any, records: list[dict]) -> list[dict]:
    """Check exact passages against their original paper, chunk and page in one read."""
    if not records:
        return []
    records = [
        r for r in records if isinstance(r.get("chunk_id"), str)
        and type(r.get("page_number")) is int and r["page_number"] > 0
    ]
    chunks = database.paper_chunks.find(
        {"chunk_id": {"$in": list({r["chunk_id"] for r in records})}},
        {"_id": 0, "paper_id": 1, "chunk_id": 1, "page_number": 1, "text": 1},
    )
    fields = ("paper_id", "chunk_id", "page_number")
    texts = {tuple(c.get(k) for k in fields): c.get("text") for c in chunks}
    checked = []
    for record in records:
        source = tuple(record.get(k) for k in fields)
        quote = record.get("supporting_evidence")
        text = texts.get(source)
        if (record.get("claim_id") and all(source)
                and isinstance(quote, str) and quote.strip()
                and isinstance(text, str) and quote in text):
            checked.append(record)
    return checked


def eligible_claims(database: Any, paper_ids: list[str], limit: int = 200) -> list[dict[str, Any]]:
    """Keep each paper's independent evidence, including unique partial overlaps."""
    if not paper_ids or limit < 1:
        return []
    query = {**_eligibility_filter(), "paper_id": {"$in": paper_ids}}
    candidates = list(database.research_claims.find(query, {"_id": 0}).limit(limit))
    papers = {p["paper_id"]: p for p in database.papers.find(
        {"paper_id": {"$in": paper_ids}}, {"paper_id": 1, "version_of_paper_id": 1},
    )}
    selected = []
    seen = set()
    for record in _source_checked(database, candidates):
        publication = papers.get(record["paper_id"], {}).get("version_of_paper_id") or record["paper_id"]
        identity = (publication, normalize_claim(record.get("claim_text", "")))
        if identity not in seen:
            seen.add(identity)
            selected.append(record)
    return selected


def trusted_claims(
    database: Any, *, paper_id: str, chunk_ids: list[str] | None = None,
    claim_ids: list[str] | None = None, after_claim_id: str | None = None, limit: int = 200,
) -> tuple[list[dict], str | None, int]:
    """Read a bounded page. The cursor advances even if all passages on a page fail."""
    if limit < 1:
        raise ValueError("The claim page size must be positive.")
    ids: dict[str, Any] = {"$type": "string", "$gt": after_claim_id or ""}
    if claim_ids is not None:
        ids["$in"] = claim_ids
    query = {
        **_eligibility_filter(), "paper_id": paper_id, "claim_id": ids,
        # Legacy accepted records may have no separate factual verdict. An
        # explicit failed check or review cannot be reconciled by selection.
        "factual_verdict": {"$in": [None, "accepted"]},
        "verification_verdict": {"$in": [None, "accepted"]},
        "$and": [{"$or": [{"factual_verdict": "accepted"}, {"verification_verdict": "accepted"}]}],
        "verifier_model": {"$type": "string", "$regex": r"\S"},
        "verified_at": {"$type": "date"},
        "source_verified": {"$ne": False},
    }
    if chunk_ids is not None:
        query["chunk_id"] = {"$in": chunk_ids}
    # Do not transfer stored audit histories or vectors during candidate discovery.
    projection = {"_id": 0, "claim_id": 1, "paper_id": 1, "chunk_id": 1, "page_number": 1,
                  "claim_text": 1, "category": 1, "supporting_evidence": 1}
    records = list(database.research_claims.find(query, projection)
                   .sort("claim_id", 1).limit(limit + 1))
    page = records[:limit]
    next_cursor = page[-1]["claim_id"] if len(records) > limit else None
    checked = [
        r for r in _source_checked(database, page)
        if isinstance(r.get("claim_text"), str) and r["claim_text"].strip()
        and isinstance(r.get("category"), str) and r["category"].strip()
    ]
    return checked, next_cursor, len(page)


def _publication_identifiers(database: Any, paper_id: str) -> set[tuple[str, str]]:
    """Collect explicit identifiers across the stored version chain."""
    seen = set()
    identifiers = set()
    if not paper_id:
        raise PaperNotFoundError("A selected paper could not be found.")
    current = paper_id
    while current:
        if current in seen or len(seen) >= 32:
            raise ClaimStorageError("Publication version history could not be verified.")
        paper = database.papers.find_one(
            {"paper_id": current},
            {"paper_id": 1, "version_of_paper_id": 1, "metadata.doi": 1,
             "source_content_sha256": 1, "content_sha256": 1},
        )
        if paper is None:
            if not seen:
                raise PaperNotFoundError("A selected paper could not be found.")
            raise ClaimStorageError("An entry in the publication version history is missing.")
        identifiers.add(("id", current))
        seen.add(current)
        for name in ("source_content_sha256", "content_sha256"):
            digest = paper.get(name)
            if isinstance(digest, str) and digest.strip():
                identifiers.add(("hash", digest.strip().lower()))
        metadata = paper.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ClaimStorageError("Publication metadata could not be verified.")
        doi = (metadata or {}).get("doi")
        if isinstance(doi, str) and doi.strip():
            normalized = doi.strip().casefold()
            for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:"):
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):].strip()
                    break
            if normalized:
                identifiers.add(("doi", normalized))
        current = paper.get("version_of_paper_id")
        if current is not None and (not isinstance(current, str) or not current.strip()):
            raise ClaimStorageError("Publication version history contains an invalid paper identifier.")
    return identifiers


def same_publication(database: Any, left_paper_id: str, right_paper_id: str) -> bool:
    """Compare version lineage, DOI and file identity before treating sources as independent."""
    left = _publication_identifiers(database, left_paper_id)
    if left_paper_id == right_paper_id:
        return True
    right = _publication_identifiers(database, right_paper_id)
    return bool(left & right)
