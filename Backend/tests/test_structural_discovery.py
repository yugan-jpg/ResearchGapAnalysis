from datetime import datetime, timezone
from types import SimpleNamespace

import mongomock
import pytest

from app.modules.embeddings.repository import EmbeddingRepository
from app.modules.embeddings.errors import VectorSearchError
from app.modules.structural_similarity.service import StructuralSimilarityService


class Encoder:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(texts)
        return [self.vectors[text] for text in texts]


def database():
    db = mongomock.MongoClient().comparison
    db.papers.insert_many([{"paper_id": "paper-a"}, {"paper_id": "paper-b"}])
    return db


def claim(db, claim_id, paper_id, chunk_id, text, *, status="accepted", evidence=None):
    quote = evidence or text
    db.research_claims.insert_one({
        "claim_id": claim_id, "paper_id": paper_id, "chunk_id": chunk_id,
        "page_number": 3, "category": "finding", "claim_text": text, "supporting_evidence": quote,
        "verification_status": status, "factual_verdict": status, "verification_verdict": status,
        "verifier_model": "verifier", "verified_at": datetime.now(timezone.utc),
        "source_verified": True, "evidence_supported": True, "numerical_consistency": True,
        "completeness": True, "category_correct": True,
    })
    existing = db.paper_chunks.find_one({"chunk_id": chunk_id}) or {}
    db.paper_chunks.update_one({"chunk_id": chunk_id}, {"$set": {
        "paper_id": paper_id, "text": existing.get("text", "") + "\n" + quote, "page_number": 3,
        "embedding_status": "ready", "embedding_model": "test-model", "embedding_dimensions": 2, "embedding": [1, 0],
    }}, upsert=True)


def discovery(db, vectors, monkeypatch, hits, neighbors=3):
    service = StructuralSimilarityService(db, Encoder(vectors), SimpleNamespace(
        comparison_neighbors=neighbors, vector_search_index_name="configured-index",
        vector_search_candidate_multiplier=10, embedding_model_name="test-model", embedding_dimensions=2,
    ))
    calls = []

    def vector_search(vector, index, *, top_k, num_candidates, paper_id, embedding_filter, chunk_ids):
        assert embedding_filter == {"embedding_status": "ready", "embedding_model": "test-model", "embedding_dimensions": 2}
        calls.append((vector, index, top_k, num_candidates, paper_id, chunk_ids))
        return [{"paper_id": paper_id, "chunk_id": chunk, "score": 0.998} for chunk in hits]

    monkeypatch.setattr(service.chunks, "vector_search", vector_search)
    return service, calls


def test_chunk_ann_is_prefiltered_by_target_paper():
    captured = []

    class Chunks:
        def aggregate(self, pipeline):
            captured.extend(pipeline)
            return []

    repository = EmbeddingRepository(SimpleNamespace(paper_chunks=Chunks()))
    repository.vector_search([0, 1], "custom", top_k=5, num_candidates=50, paper_id="target",
                             embedding_filter={"embedding_model": "test-model"})
    assert captured[0]["$vectorSearch"]["filter"] == {"$and": [
        {"embedding_model": {"$eq": "test-model"}}, {"paper_id": {"$eq": "target"}},
    ]}
    assert captured[0]["$vectorSearch"]["limit"] == 5


def test_reranks_multiple_claims_with_actual_cosine_and_preserves_both_sources(monkeypatch):
    db = database()
    left = "Surface roughness increased under cyclic loading."
    right = "Cyclic loading decreased surface roughness."
    related = "Surface roughness stayed stable under continuous loading."
    claim(db, "z", "paper-a", "source", left)
    claim(db, "a", "paper-b", "hit", right)
    claim(db, "b", "paper-b", "hit", related)
    claim(db, "rejected", "paper-b", "hit", left, status="rejected")
    service, calls = discovery(db, {left: [2, 0], right: [4, 3], related: [1, 1]}, monkeypatch, ["hit"])

    pairs, cursor, scanned = service.discover("paper-a", "paper-b", None, 5, 0.7)
    assert cursor is None and scanned == 1
    assert [pair.left.claim_id for pair in pairs] == ["a", "b"]
    assert [pair.similarity_score for pair in pairs] == pytest.approx([0.8, 2 ** -0.5])
    assert pairs[0].right.claim_id == "z" and pairs[0].left.supporting_evidence == right
    assert pairs[0].left.page_number == pairs[0].right.page_number == 3
    assert "roughness" in pairs[0].structural_signals["shared_claim_terms"]
    assert calls == []
    assert len(service.discover("paper-a", "paper-b", None, 5, 0.75)[0]) == 1


def test_shared_generic_language_and_numbers_are_not_structural_evidence(monkeypatch):
    db = database()
    left = "The study reported a statistically significant difference of 15 percent."
    right = "The trial reported a statistically significant difference of 15 percent."
    claim(db, "A", "paper-a", "source", left)
    claim(db, "B", "paper-b", "hit", right)
    service, _ = discovery(db, {left: [1, 0], right: [1, 0]}, monkeypatch, ["hit"])
    pairs, _, _ = service.discover("paper-a", "paper-b", None, 1, 0.0)
    assert pairs == []
    assert service.embedder.calls == [[left]]


def test_method_objective_population_outcome_conditions_findings_and_context(monkeypatch):
    db = database()
    evidence = (
        "We aimed to investigate vocabulary retention. Using recall tasks, we measured vocabulary retention "
        "among bilingual students. Under noise, testing resulted in lower vocabulary retention. "
        "In the setting of bilingual classrooms, scores declined."
    )
    a, b = "Noise reduced vocabulary retention.", "Noise improved vocabulary retention."
    claim(db, "a", "paper-a", "source", a, evidence=evidence + " " + a)
    claim(db, "b", "paper-b", "hit", b, evidence=evidence + " " + b)
    service, _ = discovery(db, {a: [1, 0], b: [1, 0]}, monkeypatch, ["hit"])
    pair = service.discover("paper-a", "paper-b", None, 5, 0.7)[0][0]
    expected = {
        "research_objectives": "retention", "methodology": "recall", "variables_outcomes": "vocabulary",
        "population_sample": "bilingual", "experimental_conditions": "noise",
        "scientific_findings": "retention", "research_context": "classrooms",
    }
    for field, term in expected.items():
        assert term in pair.structural_signals[field]


def test_cursor_and_request_local_target_encoding_cache(monkeypatch):
    db = database()
    a = "Cache compression reduced median latency."
    b = "Request batching reduced median latency."
    target = "Cache compression and request batching increased median latency."
    claim(db, "a", "paper-a", "src-a", a)
    claim(db, "b", "paper-a", "src-b", b)
    claim(db, "c", "paper-b", "hit", target)
    service, _ = discovery(db, {s: [1, 0] for s in (a, b, target)}, monkeypatch, ["hit"])
    first, cursor, scanned = service.discover("paper-a", "paper-b", None, 1, 0.6)
    assert scanned == 1 and cursor == "a" and first[0].left.claim_id == "a"
    second, cursor, scanned = service.discover("paper-a", "paper-b", cursor, 1, 0.6)
    assert scanned == 1 and cursor is None and second[0].left.claim_id == "b"
    service.embedder.calls.clear()
    assert len(service.discover("paper-a", "paper-b", None, 2, 0.6)[0]) == 2
    assert service.embedder.calls == [[a, b], [target]]


def test_versions_are_excluded_before_embedding_or_atlas_search(monkeypatch):
    db = database()
    db.papers.update_one({"paper_id": "paper-b"}, {"$set": {"version_of_paper_id": "paper-a"}})
    service, calls = discovery(db, {}, monkeypatch, [])
    with pytest.raises(ValueError, match="independent publications"):
        service.discover("paper-a", "paper-b", None, 1, 0.7)
    assert not calls and not service.embedder.calls


def test_missing_matching_embeddings_and_index_lag_have_actionable_errors(monkeypatch):
    db = database()
    text = "Surface roughness increased under cyclic loading."
    claim(db, "a", "paper-a", "source", text)
    claim(db, "b", "paper-b", "target", text)
    service, _ = discovery(db, {text: [1, 0]}, monkeypatch, [], neighbors=0)
    with pytest.raises(VectorSearchError, match="indexed chunks"):
        service.discover("paper-a", "paper-b", None, 1, 0.6)
    db.paper_chunks.update_one({"chunk_id": "target"}, {"$set": {"embedding_model": "old-model"}})
    with pytest.raises(VectorSearchError, match="backfill"):
        service.discover("paper-a", "paper-b", None, 1, 0.6)


def test_sparse_claim_bearing_chunks_are_all_considered_without_atlas(monkeypatch):
    db = database()
    source = "Carbon coating reduced surface roughness."
    target = "Carbon coating increased surface roughness."
    claim(db, "a", "paper-a", "source", source)
    claim(db, "b", "paper-b", "claim-chunk", target)
    for i in range(30):
        db.paper_chunks.insert_one({"paper_id": "paper-b", "chunk_id": f"empty-{i}", "text": "unrelated"})
    service, calls = discovery(db, {source: [1, 0], target: [1, 0]}, monkeypatch, [])
    pairs, _, _ = service.discover("paper-a", "paper-b", None, 1, 0.6)
    assert len(pairs) == 1 and pairs[0].right.claim_id == "b"
    assert calls == []


def test_large_target_filters_atlas_by_trusted_chunk_ids(monkeypatch):
    db = database()
    source = "Carbon coating reduced surface roughness."
    claim(db, "a", "paper-a", "source", source)
    vectors = {source: [1, 0]}
    for i in range(4):
        text = f"Carbon coating increased surface roughness in sample {i}."
        vectors[text] = [1, 0]
        claim(db, f"b{i}", "paper-b", f"chunk-{i}", text)
    claim(db, "rejected", "paper-b", "rejected-chunk", source, status="rejected")
    service, calls = discovery(db, vectors, monkeypatch, ["chunk-2", "rejected-chunk"], neighbors=3)
    pairs, _, _ = service.discover("paper-a", "paper-b", None, 1, 0.6)
    assert [p.right.claim_id for p in pairs] == ["b2"]
    assert set(calls[0][-1]) == {"chunk-0", "chunk-1", "chunk-2", "chunk-3"}
    assert calls[0][2:4] == (3, 30)


def test_target_without_trusted_claims_returns_empty_without_model_or_atlas(monkeypatch):
    db = database()
    text = "Carbon coating reduced surface roughness."
    claim(db, "a", "paper-a", "source", text)
    claim(db, "b", "paper-b", "target", text, status="rejected")
    service, calls = discovery(db, {}, monkeypatch, [])
    assert service.discover("paper-a", "paper-b", None, 1, 0.6) == ([], None, 1)
    assert calls == [] and service.embedder.calls == []


def test_atlas_chunk_filter_uses_in_operator():
    captured = []
    class Chunks:
        def aggregate(self, pipeline):
            captured.extend(pipeline)
            return []
    repository = EmbeddingRepository(SimpleNamespace(paper_chunks=Chunks()))
    repository.vector_search([1, 0], "idx", 3, 30, paper_id="p", chunk_ids=["a", "b"])
    assert {"chunk_id": {"$in": ["a", "b"]}} in captured[0]["$vectorSearch"]["filter"]["$and"]
