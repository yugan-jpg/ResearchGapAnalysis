import mongomock
import pytest
from pydantic import ValidationError

from app.modules.embeddings.backfill import EmbeddingBackfillService
from app.modules.embeddings.errors import EmbeddingDimensionError, EmbeddingModelError
from app.modules.embeddings.repository import EmbeddingRepository
from app.modules.embeddings.service import LocalEmbeddingService
from app.schemas.embeddings import BackfillRequest, SemanticSearchRequest


class FakeVectors(list):
    def tolist(self):
        return list(self)


class FakeModel:
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    def encode(self, sentences, **_kwargs):
        self.calls.append(list(sentences))
        return FakeVectors([[float(index) for index in range(self.dimensions)] for _ in sentences])


def _embedder(model: FakeModel | None = None) -> LocalEmbeddingService:
    return LocalEmbeddingService(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        batch_size=16,
        model=model or FakeModel(),
    )


def _database_with_chunks():
    database = mongomock.MongoClient().phase3_test
    database.paper_chunks.insert_many([
        {"chunk_id": "chunk-1", "paper_id": "paper-1", "chunk_index": 0, "page_number": 1, "text": "Neural methods improve forecasting.", "embedding_status": "pending"},
        {"chunk_id": "chunk-2", "paper_id": "paper-1", "chunk_index": 1, "page_number": 2, "text": "External validation remains limited.", "embedding_status": "pending"},
    ])
    return database


def test_generates_384_dimension_embeddings() -> None:
    model = FakeModel()
    vectors = _embedder(model).embed_texts(["A scientific sentence."])

    assert len(vectors) == 1
    assert len(vectors[0]) == 384
    assert model.calls == [["A scientific sentence."]]


def test_rejects_unexpected_embedding_dimensions() -> None:
    with pytest.raises(EmbeddingDimensionError):
        _embedder(FakeModel(dimensions=8)).embed_texts(["A scientific sentence."])


def test_reports_encoder_failures_without_exposing_driver_details() -> None:
    class BrokenModel:
        def encode(self, *_args, **_kwargs):
            raise RuntimeError("underlying model failure")

    with pytest.raises(EmbeddingModelError, match="could not encode"):
        _embedder(BrokenModel()).embed_texts(["A scientific sentence."])


def test_backfill_stores_embeddings_and_is_idempotent() -> None:
    database = _database_with_chunks()
    repository = EmbeddingRepository(database)
    service = EmbeddingBackfillService(repository, _embedder())

    first = service.backfill(paper_id="paper-1", limit=50)
    second = service.backfill(paper_id="paper-1", limit=50)

    stored = list(database.paper_chunks.find({}, {"_id": 0, "embedding": 1, "embedding_status": 1, "embedding_model": 1}))
    assert first.candidate_chunks == 2
    assert first.embedded_chunks == 2
    assert first.remaining_chunks == 0
    assert second.candidate_chunks == 0
    assert all(item["embedding_status"] == "ready" for item in stored)
    assert all(len(item["embedding"]) == 384 for item in stored)
    assert all(item["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2" for item in stored)


def test_backfill_respects_paper_filter() -> None:
    database = _database_with_chunks()
    database.paper_chunks.insert_one({"chunk_id": "chunk-3", "paper_id": "paper-2", "chunk_index": 0, "page_number": 1, "text": "Other paper text.", "embedding_status": "pending"})

    result = EmbeddingBackfillService(EmbeddingRepository(database), _embedder()).backfill(paper_id="paper-1", limit=50)

    assert result.embedded_chunks == 2
    assert database.paper_chunks.find_one({"chunk_id": "chunk-3"}).get("embedding") is None


def test_vector_search_builds_atlas_pipeline_and_returns_scores() -> None:
    class CapturingChunks:
        def __init__(self):
            self.pipeline = None

        def aggregate(self, pipeline):
            self.pipeline = pipeline
            return [{"chunk_id": "chunk-1", "paper_id": "paper-1", "chunk_index": 0, "page_number": 1, "text": "Relevant text", "score": 0.91}]

    class CapturingDatabase:
        paper_chunks = CapturingChunks()

    database = CapturingDatabase()
    results = EmbeddingRepository(database).vector_search([0.1] * 384, "paper_chunks_vector_index", top_k=3, num_candidates=30)

    vector_stage = database.paper_chunks.pipeline[0]["$vectorSearch"]
    assert vector_stage["index"] == "paper_chunks_vector_index"
    assert vector_stage["path"] == "embedding"
    assert vector_stage["limit"] == 3
    assert vector_stage["numCandidates"] == 30
    assert results[0]["score"] == 0.91


def test_api_request_validation() -> None:
    assert BackfillRequest(limit=10).limit == 10
    assert SemanticSearchRequest(query="semantic retrieval", top_k=3).top_k == 3
    with pytest.raises(ValidationError):
        SemanticSearchRequest(query="no", top_k=3)
    with pytest.raises(ValidationError):
        SemanticSearchRequest(query="valid search", top_k=51)
