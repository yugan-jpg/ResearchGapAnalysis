from dataclasses import dataclass

from app.modules.embeddings.repository import EmbeddingRepository
from app.modules.embeddings.service import LocalEmbeddingService


@dataclass(frozen=True)
class BackfillResult:
    candidate_chunks: int
    embedded_chunks: int
    remaining_chunks: int
    model_name: str
    dimensions: int


class EmbeddingBackfillService:
    """Idempotently fill missing or stale chunk embeddings in bounded batches."""

    def __init__(self, repository: EmbeddingRepository, embedder: LocalEmbeddingService) -> None:
        self.repository = repository
        self.embedder = embedder

    def backfill(self, paper_id: str | None, limit: int, force: bool = False) -> BackfillResult:
        candidates = self.repository.get_candidates(
            model_name=self.embedder.model_name,
            dimensions=self.embedder.dimensions,
            paper_id=paper_id,
            force=force,
            limit=limit,
        )
        if not candidates:
            return BackfillResult(0, 0, 0, self.embedder.model_name, self.embedder.dimensions)

        vectors = self.embedder.embed_texts([candidate.text for candidate in candidates])
        embedded_chunks = self.repository.store_embeddings(
            candidates=candidates,
            vectors=vectors,
            model_name=self.embedder.model_name,
            dimensions=self.embedder.dimensions,
        )
        remaining_chunks = self.repository.count_candidates(
            model_name=self.embedder.model_name,
            dimensions=self.embedder.dimensions,
            paper_id=paper_id,
            # Remaining work means missing/stale embeddings, even after a
            # caller deliberately forced this particular batch to re-embed.
            force=False,
        )
        return BackfillResult(
            candidate_chunks=len(candidates),
            embedded_chunks=embedded_chunks,
            remaining_chunks=remaining_chunks,
            model_name=self.embedder.model_name,
            dimensions=self.embedder.dimensions,
        )
