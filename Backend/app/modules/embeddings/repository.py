from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING
from pymongo.database import Database
from pymongo.errors import OperationFailure, PyMongoError

from app.modules.embeddings.errors import EmbeddingStorageError, VectorSearchError


@dataclass(frozen=True)
class EmbeddingCandidate:
    chunk_id: str
    paper_id: str
    text: str


class EmbeddingRepository:
    """MongoDB operations for embedding backfills and Atlas Vector Search."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def count_candidates(self, model_name: str, dimensions: int, paper_id: str | None, force: bool) -> int:
        return self.database.paper_chunks.count_documents(self._candidate_filter(model_name, dimensions, paper_id, force))

    def get_candidates(
        self,
        model_name: str,
        dimensions: int,
        paper_id: str | None,
        force: bool,
        limit: int,
    ) -> list[EmbeddingCandidate]:
        try:
            documents = self.database.paper_chunks.find(
                self._candidate_filter(model_name, dimensions, paper_id, force),
                {"_id": 0, "chunk_id": 1, "paper_id": 1, "text": 1},
            ).sort([("paper_id", ASCENDING), ("chunk_index", ASCENDING)]).limit(limit)
            return [EmbeddingCandidate(**document) for document in documents]
        except (PyMongoError, TypeError) as error:
            raise EmbeddingStorageError("The database could not read chunks for embedding.") from error

    def store_embeddings(
        self,
        candidates: list[EmbeddingCandidate],
        vectors: list[list[float]],
        model_name: str,
        dimensions: int,
    ) -> int:
        if len(candidates) != len(vectors):
            raise EmbeddingStorageError("The number of chunks and embeddings does not match.")
        now = datetime.now(timezone.utc)
        if not candidates:
            return 0
        try:
            modified_count = 0
            for candidate, vector in zip(candidates, vectors, strict=True):
                result = self.database.paper_chunks.update_one(
                    {"chunk_id": candidate.chunk_id},
                    {
                        "$set": {
                            "embedding": vector,
                            "embedding_model": model_name,
                            "embedding_dimensions": dimensions,
                            "embedding_status": "ready",
                            "embedded_at": now,
                        }
                    },
                )
                modified_count += result.modified_count
            return modified_count
        except PyMongoError as error:
            raise EmbeddingStorageError("The database could not store generated embeddings.") from error

    def vector_search(
        self,
        query_vector: list[float],
        index_name: str,
        top_k: int,
        num_candidates: int,
    ) -> list[dict[str, Any]]:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": num_candidates,
                    "limit": top_k,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "chunk_id": 1,
                    "paper_id": 1,
                    "chunk_index": 1,
                    "page_number": 1,
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        try:
            return list(self.database.paper_chunks.aggregate(pipeline))
        except (OperationFailure, PyMongoError, TypeError) as error:
            raise VectorSearchError(
                "Vector search is unavailable. Confirm that the Atlas Vector Search index is created, active, and named correctly."
            ) from error

    @staticmethod
    def _candidate_filter(model_name: str, dimensions: int, paper_id: str | None, force: bool) -> dict[str, Any]:
        query: dict[str, Any] = {"text": {"$type": "string", "$ne": ""}}
        if paper_id:
            query["paper_id"] = paper_id
        if not force:
            query["$or"] = [
                {"embedding": {"$exists": False}},
                {"embedding_model": {"$ne": model_name}},
                {"embedding_dimensions": {"$ne": dimensions}},
                {"embedding_status": {"$ne": "ready"}},
            ]
        return query
