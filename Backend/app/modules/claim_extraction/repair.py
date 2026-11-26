"""Conservative duplicate-graph inspection and explicit, audited repair."""
from collections import defaultdict
from datetime import datetime, timezone
import re
from uuid import uuid4

from app.modules.claim_extraction.quality import normalize_claim


def eligible_canonical(record):
    return (record.get("verification_status") == "accepted"
            and record.get("source_verified", True) is True
            and all(record.get(field) is True for field in
                    ("evidence_supported", "completeness", "numerical_consistency", "category_correct"))
            and bool(record.get("supporting_evidence") and record.get("chunk_id") and record.get("page_number")))


def plan_repairs(records):
    """Resolve only identical normalized text. Everything else requires review."""
    by_id = {record["claim_id"]: record for record in records}
    groups = defaultdict(list)
    for record in records:
        groups[(record.get("paper_id"), normalize_claim(record.get("claim_text", "")))].append(record)
    canonical = {}
    for key, group in groups.items():
        candidates = [r for r in group if eligible_canonical(r)]
        if len(group) > 1 and key[1] and candidates:
            # Exact-text groups: retain the most informative supporting quote;
            # stable claim ID breaks ties. Never use rejected/unavailable claims.
            candidates.sort(key=lambda r: (-len(r.get("supporting_evidence", "")), r["claim_id"]))
            canonical[key] = candidates[0]["claim_id"]
    plans = []
    for record in records:
        target = record.get("duplicate_of_claim_id")
        key = (record.get("paper_id"), normalize_claim(record.get("claim_text", "")))
        representative = canonical.get(key)
        if not target and (not representative or record["claim_id"] == representative
                           or record.get("verification_status") != "accepted"):
            continue
        if representative:
            new_target = None if record["claim_id"] == representative else representative
            reason = "Identical normalized claim text; canonical selected from verified supported claims."
            kind = "exact" if new_target else None
        else:
            # Semantic links, cycles, cross-paper links and missing or rejected
            # targets cannot establish equivalence. Preserve them as audit data.
            new_target = None
            kind = record.get("duplicate_type")
            reason = "Relationship is not deterministically exact and requires independent re-verification."
        update = {"duplicate_of_claim_id": new_target, "duplicate_type": kind, "duplicate_explanation": reason}
        if not representative and record.get("verification_status") == "accepted":
            update.update(verification_status="review", verification_verdict="review")
        elif representative and new_target and record.get("verification_status") == "accepted":
            update.update(verification_status="review", verification_verdict="review")
        if all(record.get(k) == v for k, v in update.items()):
            continue
        plans.append({"claim_id": record["claim_id"], "previous_target": target, "new": update,
                      "requires_reverification": not bool(representative),
                      "previous": {k: record.get(k) for k in update}})
    # Previously accepted paraphrases may have no relationship at all. Flag
    # candidates for re-verification only; shared evidence and word overlap do
    # not prove equivalence and never cause automatic merging.
    stop = {"the", "and", "for", "with", "was", "were", "are", "that", "this", "from", "using"}
    def tokens(text):
        return {x for x in re.findall(r"[a-z0-9]+", text.lower()) if len(x) > 2 and x not in stop}
    planned_ids = {p["claim_id"] for p in plans}
    for index, left in enumerate(records):
        if not eligible_canonical(left) or left.get("duplicate_of_claim_id") or not left.get("supporting_evidence"):
            continue
        for right in records[index + 1:]:
            if (not eligible_canonical(right) or right.get("duplicate_of_claim_id")
                    or left.get("paper_id") != right.get("paper_id")
                    or left.get("supporting_evidence") != right.get("supporting_evidence")
                    or normalize_claim(left.get("claim_text", "")) == normalize_claim(right.get("claim_text", ""))):
                continue
            left_tokens, right_tokens = tokens(left.get("claim_text", "")), tokens(right.get("claim_text", ""))
            overlap = len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens)) if left_tokens and right_tokens else 0
            if overlap < 0.65:
                continue
            for record, other in ((left, right), (right, left)):
                if record["claim_id"] in planned_ids:
                    continue
                update = {"verification_status": "review", "verification_verdict": "review",
                          "duplicate_explanation": f"Possible overlap with claim {other['claim_id']}; independent re-verification required."}
                plans.append({"claim_id": record["claim_id"], "previous_target": None, "new": update,
                              "requires_reverification": True, "previous": {k: record.get(k) for k in update}})
                planned_ids.add(record["claim_id"])
    return plans


def repair_paper(database, paper_id, apply=False):
    records = list(database.research_claims.find({"paper_id": paper_id}))
    chunk_ids = {record.get("chunk_id") for record in records if record.get("chunk_id")}
    chunks = {chunk["chunk_id"]: chunk for chunk in database.paper_chunks.find(
        {"paper_id": paper_id, "chunk_id": {"$in": list(chunk_ids)}},
        {"chunk_id": 1, "page_number": 1, "text": 1},
    )}
    for record in records:
        chunk = chunks.get(record.get("chunk_id"), {})
        evidence = record.get("supporting_evidence")
        record["source_verified"] = bool(
            isinstance(evidence, str) and evidence and chunk.get("page_number") == record.get("page_number")
            and evidence in chunk.get("text", "")
        )
    plans = plan_repairs(records)
    applied = 0
    if apply:
        records_by_id = {record["claim_id"]: record for record in records}
        for plan in plans:
            original = records_by_id[plan["claim_id"]]
            update = dict(plan["new"])
            if plan["requires_reverification"]:
                update.update(reverification_required=True, reverification_token=str(uuid4()))
            snapshot_fields = ("verification_status", "verification_verdict", "factual_verdict", "evidence_supported",
                               "numerical_consistency", "completeness", "category_correct",
                               "verification_explanation", "verifier_model", "verified_at",
                               "verification_attempted_at", "verification_failure_reason")
            result = database.research_claims.update_one(
                {"paper_id": paper_id, "claim_id": plan["claim_id"],
                 "duplicate_of_claim_id": plan["previous_target"]},
                {"$set": update, "$push": {"duplicate_repair_history": {
                    "at": datetime.now(timezone.utc), "previous": plan["previous"],
                    "reason": plan["new"]["duplicate_explanation"],
                    "requires_reverification": plan["requires_reverification"],
                    "verification_snapshot": {key: original[key] for key in snapshot_fields if key in original}}}},
            )
            applied += result.modified_count
    return {"examined": len(records), "planned": len(plans), "applied": applied, "changes": plans}
