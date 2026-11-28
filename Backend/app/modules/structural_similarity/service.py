"""Bounded chunk retrieval followed by claim-level cosine and structural screening."""

import math
import re

from app.core.config import Settings
from app.modules.claim_extraction.selection import same_publication, trusted_claims
from app.modules.embeddings.errors import EmbeddingModelError, VectorSearchError
from app.modules.embeddings.repository import EmbeddingRepository
from app.modules.embeddings.service import LocalEmbeddingService
from app.schemas.comparisons import Candidate, ClaimSource


# These are grammatical cues, not a dictionary of scientific entities. Keep the
# claim and quotation untouched; the extracted terms only guide candidate retrieval.
_CUES = {
    "research_objectives": r"aim(?:ed|s)? to|objective (?:was|is) to|to (?:assess|evaluate|determine|investigate|test)",
    "methodology": r"using|assessed (?:by|via)|measured (?:by|via)|method of|estimated (?:by|via)",
    "population_sample": r"among|cohort of|sample of|enroll(?:ed|ing)|recruit(?:ed|ing)|participants? (?:with|aged)",
    "variables_outcomes": r"measured|outcomes? (?:was|were|included)|levels? of|(?:increases?|decreases?|reductions?) in",
    "experimental_conditions": r"under|after|before|during|in the presence of|exposed to|at a temperature of",
    "scientific_findings": r"associated with|led to|resulted in|increased|decreased|reduced|improved|predicted",
    "research_context": r"in the context of|in the setting of|in the field of|within|at sites? in",
}
_PATTERNS = {
    field: re.compile(r"\b(?:" + cues + r")\s+([^.;!?\n]{1,100})", re.I)
    for field, cues in _CUES.items()
}
_WORD = re.compile(r"[^\W_]+(?:[-/][^\W_]+)*", re.UNICODE)
_GENERIC = frozenset("""
    a an and are as at be because been before between both by can could did does during each
    for from had has have how in into is it its may more most no not of on one or other our
    per than that the their them these they this those through to toward two under was we were
    which while with within without would year years after among compared versus found observed
    showed reported suggested study studies research result results finding findings outcome outcomes
    method methods methodology analysis analyses sample samples size sizes data group groups compared
    treatment treatments effect effects level levels primary secondary using used measure measures
    measuring assessed investigate tested test evaluate based cohort population participants
    significantly significant statistically higher lower improved improvement better worse different
    differences difference associated association increased increases increasing increase reduced
    reduction decreased decreases decreasing decrease indicated measured value values rates total
    average mean respectively experimental experiment experiments evidence objective objectives
    approach approaches model models systems system paper papers further authors compared baseline
    additional across all also carried changes change considered demonstrated identified included
    intervention interventions investigated large limited our performed showed shows standard subjects
    suggest tested together trial trials use validation work percent percentage units
    mg kg ml mm cm nm hz khz mhz ghz ms ns seconds minutes hours weeks months
""".split())


def _terms(text: str) -> set[str]:
    words = (word.casefold() for word in _WORD.findall(text))
    return {word for word in words if len(word) > 1 and any(c.isalpha() for c in word)
            and word not in _GENERIC and not word[0].isdecimal()}


def _attributes(claim: ClaimSource) -> dict[str, set[str]]:
    text = f"{claim.claim_text}\n{claim.supporting_evidence}"
    fields = {}
    for field, pattern in _PATTERNS.items():
        values = set()
        for match in pattern.finditer(text):
            # Stop at the next clause. Ambiguous or missing attributes stay unknown.
            phrase = re.split(r",|\b(?:and then|whereas|because|although|using|among|under|after)\b",
                              match[1], maxsplit=1, flags=re.I)[0]
            values.update(_terms(" ".join(phrase.split()[:6])))
        fields[field] = values
    fields["shared_claim_terms"] = _terms(claim.claim_text)
    return fields


def _shared_attributes(left: dict[str, set[str]], right: dict[str, set[str]]) -> dict[str, list[str]]:
    common = {field: terms & right[field] for field, terms in left.items()}
    claim_terms = common["shared_claim_terms"]
    # A shared explicitly described outcome, method or objective can retrieve
    # terse claims. Otherwise require at least two content terms in both claims.
    anchors = common["variables_outcomes"] | common["methodology"] | common["research_objectives"]
    if len(claim_terms) < 2 and not claim_terms.intersection(anchors):
        return {}
    return {field: sorted(terms)[:10] for field, terms in common.items() if terms}


def _cosine(left: list[float], right: list[float]) -> float:
    denominator = math.hypot(*left) * math.hypot(*right)
    if not math.isfinite(denominator) or denominator == 0:
        raise EmbeddingModelError("The embedding model returned an invalid claim vector.")
    value = sum(a * b for a, b in zip(left, right, strict=True)) / denominator
    return max(-1.0, min(1.0, value))


def _source(record: dict) -> ClaimSource:
    return ClaimSource(**{field: record[field] for field in ClaimSource.model_fields})


class StructuralSimilarityService:
    def __init__(self, database, embedder: LocalEmbeddingService, settings: Settings):
        self.database = database
        self.embedder = embedder
        self.settings = settings
        self.chunks = EmbeddingRepository(database)

    def discover(
        self, source_paper_id: str, target_paper_id: str, after_claim_id: str | None,
        limit: int, min_similarity: float,
    ) -> tuple[list[Candidate], str | None, int]:
        if same_publication(self.database, source_paper_id, target_paper_id):
            raise ValueError("Choose two independent publications for comparison.")
        records, cursor, scanned = trusted_claims(
            self.database, paper_id=source_paper_id, after_claim_id=after_claim_id, limit=limit,
        )
        sources = [_source(record) for record in records]
        if not sources:
            return [], cursor, scanned
        # Read bounded pages to check the exact evidence of each target claim.
        # Fail explicitly if the shortlist would exceed the filter budget.
        targets_by_chunk = {}
        target_cursor = None
        scanned_targets = 0
        while True:
            page, target_cursor, target_scanned = trusted_claims(
                self.database, paper_id=target_paper_id, after_claim_id=target_cursor, limit=200,
            )
            scanned_targets += target_scanned
            for record in page:
                targets_by_chunk.setdefault(record["chunk_id"], []).append(record)
            if len(targets_by_chunk) > 10000 or scanned_targets > 100000:
                raise VectorSearchError("The target paper exceeds the comparison shortlist limit.")
            if target_cursor is None:
                break
        if not targets_by_chunk:
            return [], cursor, scanned
        neighbors = self.settings.comparison_neighbors
        direct_chunks = sorted(targets_by_chunk) if len(targets_by_chunk) <= neighbors else None
        embedding_filter = {"embedding_status": "ready", "embedding_model": self.settings.embedding_model_name,
                            "embedding_dimensions": self.settings.embedding_dimensions}
        if direct_chunks is None:
            ready = self.database.paper_chunks.find_one(
                {**embedding_filter, "paper_id": target_paper_id, "chunk_id": {"$in": list(targets_by_chunk)},
                 "embedding": {"$type": "array"}}, {"_id": 1},
            )
            if not ready:
                raise VectorSearchError("The target paper has no ready embeddings for trusted claim chunks. "
                                        "Run /api/embeddings/backfill for that paper.")
        vectors = self.embedder.embed_texts([claim.claim_text for claim in sources])
        candidates = []
        # Request-local reuse when several anchors retrieve the same target chunks.
        target_profiles = {}
        target_vectors = {}
        for source, vector in zip(sources, vectors, strict=True):
            if direct_chunks is not None:
                chunk_ids = direct_chunks
            else:
                hits = self.chunks.vector_search(
                    vector, self.settings.vector_search_index_name, top_k=neighbors,
                    num_candidates=min(10_000, neighbors * self.settings.vector_search_candidate_multiplier),
                    paper_id=target_paper_id, embedding_filter=embedding_filter,
                    chunk_ids=list(targets_by_chunk),
                )
                chunk_ids = list(dict.fromkeys(
                    hit["chunk_id"] for hit in hits
                    if hit["paper_id"] == target_paper_id and hit["chunk_id"] in targets_by_chunk
                ))
                if not chunk_ids:
                    raise VectorSearchError("Trusted target embeddings exist, but Atlas returned no indexed chunks. "
                                            "Check the configured vector index and its filter fields, then retry.")
            targets = [record for chunk_id in chunk_ids for record in targets_by_chunk[chunk_id]]
            source_profile = _attributes(source)
            qualified = []
            for record in targets:
                target = _source(record)
                if target.claim_id not in target_profiles:
                    target_profiles[target.claim_id] = _attributes(target)
                signals = _shared_attributes(source_profile, target_profiles[target.claim_id])
                if signals:
                    qualified.append((target, signals))
            # Only embed claims that passed the inexpensive structural screen.
            missing = [target for target, _ in qualified if target.claim_id not in target_vectors]
            if missing:
                encoded = self.embedder.embed_texts([target.claim_text for target in missing])
                target_vectors.update((target.claim_id, embedding) for target, embedding in zip(missing, encoded, strict=True))
            ranked = []
            for target, signals in qualified:
                cosine = _cosine(vector, target_vectors[target.claim_id])
                if cosine >= min_similarity:
                    left, right = sorted((source, target), key=lambda item: item.claim_id)
                    ranked.append(Candidate(left=left, right=right, similarity_score=cosine, structural_signals=signals))
            ranked.sort(key=lambda pair: (-pair.similarity_score, pair.left.claim_id, pair.right.claim_id))
            candidates.extend(ranked[:neighbors])
        return candidates, cursor, scanned
