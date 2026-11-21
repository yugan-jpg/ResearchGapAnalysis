class EmbeddingError(Exception):
    """Base error for local embedding and vector-retrieval operations."""


class EmbeddingModelError(EmbeddingError):
    """The local sentence-transformers model could not load or encode text."""


class EmbeddingDimensionError(EmbeddingError):
    """A model returned a vector with an unexpected dimensionality."""


class EmbeddingStorageError(EmbeddingError):
    """MongoDB could not persist generated embeddings safely."""


class VectorSearchError(EmbeddingError):
    """Atlas Vector Search could not execute the configured query."""
