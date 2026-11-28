from datetime import datetime, timezone
import json

import httpx
import mongomock
import pytest
from fastapi.testclient import TestClient
from pymongo.errors import OperationFailure

from app.api.routes import comparisons
from app.core.config import Settings
from app.main import app
from app.modules.claim_extraction.errors import ClaimStorageError
from app.modules.contradiction_detection.evaluator import PairEvaluator
from app.modules.embeddings.repository import EmbeddingRepository


@pytest.fixture
def comparison_api(monkeypatch):
    db = mongomock.MongoClient().comparison_api
    settings = Settings(_env_file=None, llm_api_url="https://example.invalid", llm_api_key="test",
                        llm_model_name="mock-primary", verifier_model_name="mock-verifier", verifier_enabled=True,
                        llm_max_retries=0, verifier_max_retries=0, verifier_malformed_retries=0)
    for paper, claim_id, text in (
        ("experiment-a", "anchor-1", "During cyclic loading, carbon coatings increased surface roughness."),
        ("experiment-a", "anchor-2", "Carbon coatings increased surface roughness during cyclic loading."),
        ("experiment-b", "other", "During cyclic loading, carbon coatings decreased surface roughness."),
    ):
        db.papers.update_one({"paper_id": paper}, {"$setOnInsert": {"paper_id": paper}}, upsert=True)
        db.paper_chunks.insert_one({"paper_id": paper, "chunk_id": claim_id, "page_number": 2, "text": text,
                                    "embedding": [1.0] * 384, "embedding_model": settings.embedding_model_name,
                                    "embedding_dimensions": 384, "embedding_status": "ready"})
        db.research_claims.insert_one({
            "paper_id": paper, "claim_id": claim_id, "chunk_id": claim_id, "page_number": 2,
            "category": "finding", "claim_text": text, "supporting_evidence": text,
            "verification_status": "accepted", "factual_verdict": "accepted", "evidence_supported": True,
            "numerical_consistency": True, "completeness": True, "category_correct": True,
            "verified_at": datetime.now(timezone.utc), "verifier_model": "mock-verifier",
        })

    class Encoder:
        def embed_texts(self, texts):
            return [[1.0] * 384 for _ in texts]

    class HTTP:
        def __init__(self):
            self.models = []
            self.status = 200

        def post(self, url, **kwargs):
            payload = kwargs["json"]
            self.models.append(payload["model"])
            if self.status != 200:
                return httpx.Response(self.status, json={"error": "sensitive-provider-error"})
            evidence = json.loads(payload["messages"][1]["content"])
            result = {"relationship": "partial_contradiction", "mismatch_types": [], "comparable_scope": True,
                      "left_quote": evidence["left"]["supporting_evidence"],
                      "right_quote": evidence["right"]["supporting_evidence"],
                      "explanation": "The coating findings have opposite directions under cyclic loading."}
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)}}]})

    def search(repo, vector, index, top_k, num_candidates, *, paper_id, embedding_filter, chunk_ids):
        return list(db.paper_chunks.find({"paper_id": paper_id, **embedding_filter}).limit(top_k))

    http = HTTP()
    monkeypatch.setattr(comparisons, "get_settings", lambda: settings)
    monkeypatch.setattr(comparisons, "get_database", lambda: db)
    monkeypatch.setattr(comparisons, "get_embedding_service", Encoder)
    monkeypatch.setattr(comparisons, "PairEvaluator", lambda s: PairEvaluator(s, http))
    monkeypatch.setattr(EmbeddingRepository, "vector_search", search)
    # Client sends ASGI requests in-process. Application lifespan is omitted to
    # isolate this test from any local AUTO_INITIALIZE_DATABASE configuration.
    return TestClient(app), db, http


DISCOVER = {"source_paper_id": "experiment-a", "target_paper_id": "experiment-b", "limit": 1}
DETECT = {"paper_id": "experiment-a", "other_paper_id": "experiment-b", "limit": 1}


def test_discovery_cursor_detection_cache_and_result_filters(comparison_api):
    client, db, http = comparison_api
    base = "/api/comparisons"
    first = client.post(base + "/similar", json=DISCOVER)
    assert first.status_code == 200, first.text
    page = first.json()
    assert page["queued"] == page["scanned"] == 1 and page["next_cursor"] == "anchor-1"
    second = client.post(base + "/similar", json={**DISCOVER, "after_claim_id": page["next_cursor"]}).json()
    assert second["queued"] == 1 and second["next_cursor"] is None
    reverse = client.post(base + "/similar", json={**DISCOVER, "source_paper_id": "experiment-b",
                                                   "target_paper_id": "experiment-a"}).json()
    assert len(reverse["comparisons"]) == 2 and reverse["queued"] == 0 and not http.models
    detected = client.post(base + "/detect", json={**DETECT, "limit": 2})
    assert detected.status_code == 200, detected.text
    assert detected.json()["completed"] == 2 and len(http.models) == 4
    assert client.post(base + "/detect", json=DETECT).json()["attempted"] == 0
    filters = {"paper_id": "experiment-b", "relationship": "partial_contradiction", "limit": 1}
    results = client.get(base, params=filters).json()
    row = results["comparisons"][0]
    assert results["total"] == 2 and row["verification_status"] == "verified"
    assert row["sources_current"] and row["left"]["page_number"] == row["right"]["page_number"] == 2
    next_row = client.get(base, params={**filters, "skip": 1}).json()["comparisons"][0]
    assert row["comparison_id"] != next_row["comparison_id"]
    assert client.get(base, params={**filters, "mismatch_type": "cross_field"}).json()["total"] == 0
    db.research_claims.update_one({"claim_id": "other"}, {"$set": {"verification_status": "rejected"}})
    stale = client.get(base, params=filters).json()["comparisons"][0]
    assert stale["verification_status"] == "stale" and not stale["sources_current"]
    assert len(http.models) == 4


def test_http_errors_and_non_destructive_retry(comparison_api):
    client, _, http = comparison_api
    base = "/api/comparisons"
    assert client.post(base + "/similar", json={**DISCOVER, "target_paper_id": "missing"}).status_code == 404
    assert client.post(base + "/similar", json={**DISCOVER, "min_similarity": 2}).status_code == 422
    assert client.post(base + "/similar", json={**DISCOVER, "target_paper_id": "experiment-a"}).status_code == 422
    assert client.post(base + "/detect", json={**DETECT, "limit": 3}).status_code == 422
    assert client.get(base, params={"mismatch_type": "research_gap"}).status_code == 422
    assert client.post(base + "/similar", json=DISCOVER).status_code == 200
    http.status = 429
    error = client.post(base + "/detect", json=DETECT)
    assert error.status_code == 429 and "sensitive-provider-error" not in error.text
    row = error.json()["comparisons"][0]
    assert row["failure_reason"] == "rate_limited" and row["verification_status"] == "unavailable"
    http.status = 200
    assert client.post(base + "/detect", json=DETECT).json()["attempted"] == 0
    retry = client.post(base + "/detect", json={**DETECT, "retry_failed": True}).json()
    assert retry["completed"] == 1 and retry["comparisons"][0]["comparison_id"] == row["comparison_id"]


@pytest.mark.parametrize("failure", [ClaimStorageError, OperationFailure, RuntimeError])
def test_storage_503_logs_redacted_chained_traceback(monkeypatch, caplog, failure):
    settings = Settings(_env_file=None, llm_api_key="configured-secret-123",
                        mongodb_uri="mongodb+srv://test-user:db%40secret@atlas.invalid/")
    monkeypatch.setattr(comparisons, "get_settings", lambda: settings)
    header_secret = "another-header-secret"
    driver_message = (f"failed at source read; {settings.mongodb_uri} db@secret "
                      f"{settings.llm_api_key.get_secret_value()} Authorization: Bearer {header_secret}")

    def read_source():
        raise OperationFailure(driver_message, code=13)

    def failing_database():
        try:
            read_source()
        except OperationFailure as cause:
            raise failure("Could not read comparison source.") from cause

    monkeypatch.setattr(comparisons, "get_database", failing_database)
    response = TestClient(app).post("/api/comparisons/similar", json=DISCOVER)
    assert response.status_code == 503
    assert response.json() == {"detail": "Comparison storage or source provenance is unavailable."}
    logs = [r for r in caplog.records if r.name == "uvicorn.error"]
    assert len(logs) == 1 and logs[0].exc_info is None
    trace = logs[0].getMessage()
    assert trace.count("Traceback (most recent call last)") == 2
    assert "in find_similar" in trace and "in read_source" in trace
    assert "OperationFailure: failed at source read" in trace
    assert f"{failure.__name__}: Could not read comparison source." in trace
    for value in ("test-user", "db%40secret", "db@secret", "atlas.invalid", "configured-secret-123", header_secret):
        assert value not in trace and value not in response.text


def test_configuration_failure_preserves_original_503_without_logging_unknown_values(monkeypatch, caplog):
    def broken_settings():
        raise RuntimeError("sensitive configuration failure")

    monkeypatch.setattr(comparisons, "get_settings", broken_settings)
    response = TestClient(app).post("/api/comparisons/similar", json=DISCOVER)
    assert response.status_code == 503
    assert "in broken_settings" in caplog.text and "RuntimeError" in caplog.text
    assert "sensitive configuration failure" not in caplog.text
