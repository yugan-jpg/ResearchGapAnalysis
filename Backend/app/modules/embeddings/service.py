from functools import lru_cache
from typing import Protocol, Sequence

from app.core.config import get_settings
from app.modules.embeddings.errors import EmbeddingDimensionError, EmbeddingModelError


class SentenceEncoder(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs: object) -> object: ...


class LocalEmbeddingService:
    """CPU-friendly wrapper around a lazily loaded sentence-transformers model."""

    def __init__(
        self,
        model_name: str,
        dimensions: int,
        batch_size: int,
        device: str = "cpu",
        model: SentenceEncoder | None = None,
    ) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.device = device
        self._model = model

    @property
    def model(self) -> SentenceEncoder:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as error:
                raise EmbeddingModelError(
                    "The local embedding model could not be loaded. Check the model name, internet access for the first download, and available disk space."
                ) from error
        return self._model

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            values = vectors.tolist() if hasattr(vectors, "tolist") else vectors
        except EmbeddingModelError:
            raise
        except Exception as error:
            raise EmbeddingModelError("The local embedding model could not encode the supplied text.") from error

        if len(values) != len(texts):
            raise EmbeddingModelError("The embedding model returned an unexpected number of vectors.")
        normalized_vectors: list[list[float]] = []
        for vector in values:
            if len(vector) != self.dimensions:
                raise EmbeddingDimensionError(
                    f"The embedding model returned {len(vector)} dimensions; expected {self.dimensions}."
                )
            normalized_vectors.append([float(value) for value in vector])
        return normalized_vectors


@lru_cache
def get_embedding_service() -> LocalEmbeddingService:
    settings = get_settings()
    return LocalEmbeddingService(
        model_name=settings.embedding_model_name,
        dimensions=settings.embedding_dimensions,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
    )
