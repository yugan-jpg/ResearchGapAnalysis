"""Resolve explicitly supported equivalence without treating similarity as proof."""
from collections import defaultdict
from datetime import datetime, timezone
import re
import unicodedata

from app.modules.claim_extraction.quality import inequality_tokens, is_factually_rejected, measurement_tokens, normalize_claim


def _factual_acceptance(record):
    # Legacy accepted verdicts are usable. A legacy review with four true
    # flags is not enough: its original factual verdict was not retained.
    verdict = record.get("factual_verdict")
    if verdict is None:
        verdict = record.get("verification_verdict") if record.get("verification_status") == "accepted" else None
    return (verdict == "accepted" and not record.get("reverification_required")
            and record.get("verification_status") not in {"rejected", "unavailable", "pending"}
            and record.get("source_verified") is True
            and bool(record.get("verifier_model") and record.get("verified_at"))
            and all(record.get(field) is True for field in
                    ("evidence_supported", "numerical_consistency", "completeness", "category_correct")))


def _context_compatible(left, right):
    a, b = left.get("claim_text", ""), right.get("claim_text", "")
    if left.get("category") != right.get("category"):
        return False
    if measurement_tokens(a) != measurement_tokens(b) or inequality_tokens(a) != inequality_tokens(b):
        return False
    negation = r"\b(?:not|no|without|neither|never|non-significant|insignificant)\b"
    if bool(re.search(negation, a, re.I)) != bool(re.search(negation, b, re.I)):
        return False
    # Keep scientifically meaningful letter-digit identifiers and proper
    # names distinct. The verifier supplies the equivalence judgment; these
    # are only conservative vetoes, never evidence of equivalence.
    def entities(text, other):
        words = re.findall(r"\b[A-Za-z][A-Za-z0-9/-]*\b", text)
        other_words = {word.casefold() for word in re.findall(r"\b[A-Za-z][A-Za-z0-9/-]*\b", other)}
        openers = {"the", "a", "an", "in", "between", "during", "this", "these", "our", "we"}
        return {word.casefold() for i, word in enumerate(words)
                if (any(char.isdigit() for char in word)
                    or (word.isupper() and len(word) > 1)
                    or (word[0].isupper() and
                        (i > 0 or (word.casefold() not in openers and word.casefold() not in other_words))))}
    return entities(a, b) == entities(b, a)


def _incomplete_context_compatible(complete, incomplete):
    """Allow omitted context, but not conflicting or reordered details.

    This is intentionally stricter than semantic similarity. Every
    substantive token in the incomplete claim must occur in order in the
    complete claim; numbers, inequalities, and negation have extra checks.
    The incomplete claim is not merged or promoted by this comparison.
    """
    if complete.get("category") != incomplete.get("category"):
        return False
    full, partial = complete.get("claim_text", ""), incomplete.get("claim_text", "")
    if (not measurement_tokens(partial).issubset(measurement_tokens(full))
            or not inequality_tokens(partial).issubset(inequality_tokens(full))):
        return False
    negation = r"\b(?:not|no|without|neither|never|non-significant|insignificant)\b"
    if bool(re.search(negation, full, re.I)) != bool(re.search(negation, partial, re.I)):
        return False
    # Omit only grammatical glue and a neutral event verb; retain scientific
    # entities, conditions, qualifiers, outcomes, units, and measurements.
    glue = {"a", "an", "the", "of", "to", "and", "in", "by", "for", "with",
            "was", "were", "is", "are", "be", "been", "being", "that", "this",
            "as", "it", "occur", "occurred", "occurs"}
    def tokens(text):
        return [word for word in re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())
                if word not in glue]
    complete_tokens, partial_tokens = tokens(full), tokens(partial)
    if not partial_tokens:
        return False
    iterator = iter(complete_tokens)
    return all(any(word == part for word in iterator) for part in partial_tokens)


def _verified_incomplete(record):
    """Accept explicit or legacy review metadata only for incomplete claims."""
    verdict = record.get("factual_verdict")
    legacy_review = verdict is None and record.get("verification_verdict") == "review"
    return (
        record.get("verification_status") == "review"
        and (verdict == "review" or legacy_review)
        and record.get("completeness") is False
        and record.get("source_verified") is True
        and not record.get("reverification_required")
        and bool(record.get("verifier_model") and record.get("verified_at"))
        and all(record.get(field) is True for field in
                ("evidence_supported", "numerical_consistency", "category_correct"))
    )


def _other_viable_candidate(record, rejected_id, records):
    """Fail closed when another stored same-paper candidate may overlap."""
    stop = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "was",
            "were", "is", "are", "by", "that", "this"}
    def tokens(text):
        return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2 and word not in stop}
    left = tokens(record.get("claim_text", ""))
    for other in records:
        if (other["claim_id"] in {record["claim_id"], rejected_id}
                or other.get("paper_id") != record.get("paper_id")
                or other.get("category") != record.get("category")
                or is_factually_rejected(other)):
            continue
        if other.get("duplicate_candidate_id") == record["claim_id"] or other.get("duplicate_of_claim_id") == record["claim_id"]:
            return True
        right = tokens(other.get("claim_text", ""))
        overlap = len(left & right) / min(len(left), len(right)) if left and right else 0
        if overlap >= 0.5 or (
            record.get("supporting_evidence") and record.get("supporting_evidence") == other.get("supporting_evidence")
        ):
            return True
    return False


def plan_resolutions(records, affected_ids=None):
    """Plan complete, same-paper groups with independently verified facts."""
    eligible = {r["claim_id"]: r for r in records if _factual_acceptance(r)}
    neighbors = defaultdict(set)
    ids = sorted(eligible)
    for index, left_id in enumerate(ids):
        left = eligible[left_id]
        for right_id in ids[index + 1:]:
            right = eligible[right_id]
            if left.get("paper_id") != right.get("paper_id") or not _context_compatible(left, right):
                continue
            exact = normalize_claim(left["claim_text"]) == normalize_claim(right["claim_text"])
            explicit = ((left.get("duplicate_assessment") == "confirmed_equivalent" and
                         left.get("duplicate_candidate_id") == right_id) or
                        (right.get("duplicate_assessment") == "confirmed_equivalent" and
                         right.get("duplicate_candidate_id") == left_id))
            if exact or explicit:
                neighbors[left_id].add(right_id)
                neighbors[right_id].add(left_id)

    plans = []
    visited = set()
    conflicts = {
        claim_id for claim_id, record in eligible.items()
        if record.get("duplicate_assessment") == "confirmed_equivalent"
        and record.get("duplicate_candidate_id") in eligible
        and record.get("duplicate_candidate_id") not in neighbors[claim_id]
    }
    for root in ids:
        if root in visited or root not in neighbors:
            continue
        component, stack = set(), [root]
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(neighbors[current] - component)
        visited |= component
        if affected_ids is not None and not (component & affected_ids):
            continue
        # A chain A~B~C does not establish A~C. Leave non-cliques for review.
        if component & conflicts or any(not (component - {cid}).issubset(neighbors[cid]) for cid in component):
            continue
        members = [eligible[cid] for cid in component]
        # More complete supported wording first; retain an existing accepted
        # claim only as a tie-breaker, followed by a stable claim ID.
        members.sort(key=lambda r: (-len(re.findall(r"\w+", r["claim_text"])),
                                    -len(r.get("supporting_evidence", "")),
                                    0 if r.get("verification_status") == "accepted" else 1,
                                    r["claim_id"]))
        canonical = members[0]["claim_id"]
        for record in members:
            target = None if record["claim_id"] == canonical else canonical
            update = {
                "verification_status": "accepted", "verification_verdict": "accepted",
                "factual_verdict": "accepted",
                "duplicate_of_claim_id": target,
                "duplicate_type": "semantic" if target else None,
                "duplicate_assessment": "confirmed_duplicate" if target else "canonical",
                "duplicate_explanation": (
                    f"Confirmed equivalent; canonical claim {canonical} was chosen by completeness and stable tie-breaks."
                    if target else "Canonical representative selected from independently verified equivalent claims."
                ),
            }
            if any(record.get(k) != v for k, v in update.items()):
                plans.append({"claim_id": record["claim_id"], "canonical_id": canonical, "new": update,
                              "previous": {k: record.get(k) for k in update}})

    # A legacy review did not retain its independent factual verdict. Never
    # promote it, but an exact-text match can be linked to an already accepted
    # source-checked canonical while the legacy claim remains under review.
    planned_ids = {plan["claim_id"] for plan in plans}
    for record in records:
        if (record["claim_id"] in planned_ids or record.get("verification_status") != "review"
                or record.get("factual_verdict") is not None or record.get("source_verified") is not True
                or record.get("reverification_required")
                or not all(record.get(field) is True for field in
                           ("evidence_supported", "numerical_consistency", "completeness", "category_correct"))):
            continue
        candidates = [candidate for candidate in eligible.values()
                      if candidate["claim_id"] != record["claim_id"]
                      and candidate.get("paper_id") == record.get("paper_id")
                      and candidate.get("verification_status") == "accepted"
                      and not candidate.get("duplicate_of_claim_id")
                      and normalize_claim(candidate["claim_text"]) == normalize_claim(record["claim_text"])
                      and _context_compatible(candidate, record)]
        if not candidates:
            continue
        candidates.sort(key=lambda r: (-len(r.get("supporting_evidence", "")), r["claim_id"]))
        canonical_id = candidates[0]["claim_id"]
        if affected_ids is not None and not ({record["claim_id"], canonical_id} & affected_ids):
            continue
        update = {"duplicate_of_claim_id": canonical_id, "duplicate_type": "exact",
                  "duplicate_assessment": "confirmed_duplicate",
                  "duplicate_explanation": "Exact normalized claim text matches an independently verified canonical; legacy factual review is preserved.",
                  "verification_status": "review"}
        if any(record.get(k) != v for k, v in update.items()):
            plans.append({"claim_id": record["claim_id"], "canonical_id": canonical_id, "new": update,
                          "previous": {k: record.get(k) for k in update}})
            planned_ids.add(record["claim_id"])

    # A complete verified claim can stand on its own when its only confirmed
    # equivalent is an incomplete, source-supported paraphrase. The latter
    # remains under review, preserving its distinct ID and evidence.
    planned_ids = {plan["claim_id"] for plan in plans}
    for complete in eligible.values():
        if complete["claim_id"] in planned_ids:
            continue
        linked = [
            other for other in records if other["claim_id"] != complete["claim_id"]
            and other.get("paper_id") == complete.get("paper_id")
            and ((complete.get("duplicate_assessment") == "confirmed_equivalent"
                  and complete.get("duplicate_candidate_id") == other["claim_id"])
                 or (other.get("duplicate_assessment") == "confirmed_equivalent"
                     and other.get("duplicate_candidate_id") == complete["claim_id"]))
        ]
        if not linked or (affected_ids is not None and not
                          ({complete["claim_id"]} | {item["claim_id"] for item in linked}) & affected_ids):
            continue
        if any(not _verified_incomplete(other) or not _incomplete_context_compatible(complete, other)
               for other in linked):
            continue
        update = {"verification_status": "accepted", "verification_verdict": "accepted",
                  "duplicate_of_claim_id": None, "duplicate_type": None,
                  "duplicate_assessment": "incomplete_counterpart",
                  "duplicate_explanation": "Complete verified claim retained; the incomplete related claim remains under review without being merged."}
        if any(complete.get(k) != v for k, v in update.items()):
            plans.append({"claim_id": complete["claim_id"], "canonical_id": complete["claim_id"], "new": update,
                          "previous": {k: complete.get(k) for k in update}})
            planned_ids.add(complete["claim_id"])

    # A known incorrect claim cannot remain the sole similarity blocker of an
    # independently verified claim. Do not release it if any other plausible
    # same-paper candidate remains or if the factual checks are unresolved.
    by_id = {record["claim_id"]: record for record in records}
    for record in eligible.values():
        if affected_ids is not None and record["claim_id"] not in affected_ids:
            continue
        if any(plan["claim_id"] == record["claim_id"] for plan in plans):
            continue
        target_id = record.get("duplicate_candidate_id")
        target = by_id.get(target_id)
        if (record.get("duplicate_assessment") != "unresolved" or not target
                or target.get("paper_id") != record.get("paper_id")
                or not is_factually_rejected(target)
                or _other_viable_candidate(record, target_id, records)):
            continue
        update = {
            "verification_status": "accepted", "verification_verdict": "accepted",
            "duplicate_of_claim_id": None, "duplicate_type": None,
            "duplicate_assessment": "rejected_candidate_excluded",
            "duplicate_explanation": "The sole plausible stored similarity candidate was factually rejected; no other plausible stored candidate was found.",
        }
        if any(record.get(k) != v for k, v in update.items()):
            plans.append({"claim_id": record["claim_id"], "canonical_id": None, "new": update,
                          "previous": {k: record.get(k) for k in update}})

    # Explicitly classified partial overlaps retain their supported content.
    for record in eligible.values():
        if affected_ids is not None and record["claim_id"] not in affected_ids:
            continue
        if record.get("duplicate_assessment") not in {"partial_overlap", "containment", "related_distinct"}:
            continue
        if any(plan["claim_id"] == record["claim_id"] for plan in plans):
            continue
        update = {"verification_status": "accepted", "verification_verdict": "accepted", "duplicate_of_claim_id": None}
        if any(record.get(k) != v for k, v in update.items()):
            plans.append({"claim_id": record["claim_id"], "canonical_id": None, "new": update,
                          "previous": {k: record.get(k) for k in update}})
    return plans


def resolve_paper(database, paper_id, *, apply=False, affected_ids=None):
    records = list(database.research_claims.find({"paper_id": paper_id}))
    chunks = {chunk["chunk_id"]: chunk for chunk in database.paper_chunks.find(
        {"paper_id": paper_id}, {"chunk_id": 1, "page_number": 1, "text": 1})}
    for record in records:
        chunk = chunks.get(record.get("chunk_id"), {})
        evidence = record.get("supporting_evidence")
        record["source_verified"] = bool(isinstance(evidence, str) and evidence
                                         and chunk.get("page_number") == record.get("page_number")
                                         and evidence in chunk.get("text", ""))
    plans = plan_resolutions(records, affected_ids)
    applied = 0
    if apply:
        # Clear/establish canonical representatives first so that no new link
        # points into a cycle. Concurrent updates cause the group to be skipped.
        succeeded = set()
        by_id = {record["claim_id"]: record for record in records}
        for plan in sorted(plans, key=lambda p: p["new"].get("duplicate_of_claim_id") is not None):
            canonical_id = plan["canonical_id"]
            if canonical_id and plan["claim_id"] != canonical_id and canonical_id not in succeeded:
                # A canonical with no change must still be an eligible terminal.
                canonical = by_id.get(canonical_id, {})
                if (canonical_id in {p["claim_id"] for p in plans}
                        or canonical.get("duplicate_of_claim_id")):
                    continue
            old = by_id[plan["claim_id"]]
            query = {"_id": old["_id"], "verification_status": old.get("verification_status"),
                     "duplicate_of_claim_id": old.get("duplicate_of_claim_id"),
                     "reverification_required": {"$ne": True},
                     "verified_at": old.get("verified_at"),
                     "factual_verdict": old.get("factual_verdict"),
                     "duplicate_assessment": old.get("duplicate_assessment"),
                     "duplicate_candidate_id": old.get("duplicate_candidate_id"),
                     "evidence_supported": old.get("evidence_supported"),
                     "numerical_consistency": old.get("numerical_consistency"),
                     "completeness": old.get("completeness"),
                     "category_correct": old.get("category_correct")}
            result = database.research_claims.update_one(
                query, {"$set": plan["new"], "$push": {"canonical_resolution_history": {
                    "at": datetime.now(timezone.utc), "previous": plan["previous"],
                    "reason": plan["new"].get("duplicate_explanation", "Supported partial overlap retained."),
                }}})
            if result.modified_count:
                applied += 1
                if plan["claim_id"] == canonical_id:
                    succeeded.add(canonical_id)
    return {"examined": len(records), "planned": len(plans), "applied": applied, "changes": plans}
