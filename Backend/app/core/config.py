from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIRECTORY = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        # Resolve independently of the command's working directory.
        env_file=BACKEND_DIRECTORY / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    mongodb_uri: str = Field(default="", description="MongoDB Atlas connection string")
    mongodb_database: str = Field(default="scientific_literature_gap_discovery")
    llm_api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    llm_model_name: str = Field(default="")
    llm_api_url: str = Field(default="")
    llm_api_auth_header: str = Field(default="Authorization")
    llm_api_auth_prefix: str = Field(default="Bearer")
    llm_timeout_seconds: int = Field(default=60, ge=1, le=300)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    claim_extraction_max_claims: int = Field(default=8, ge=1, le=20)
    verifier_model_name: str = Field(default="")
    verifier_enabled: bool = Field(default=False)
    verifier_timeout_seconds: int = Field(default=60, ge=1, le=300)
    verifier_max_retries: int = Field(default=2, ge=0, le=5)
    verifier_batch_size: int = Field(default=3, ge=1, le=20)
    verifier_comparison_limit: int = Field(default=8, ge=0, le=20)
    llm_tokens_per_minute: int = Field(default=0, ge=0)
    verifier_tokens_per_minute: int = Field(default=0, ge=0)
    llm_requests_per_minute: int = Field(default=0, ge=0)
    verifier_requests_per_minute: int = Field(default=0, ge=0)
    groq_rate_safety_fraction: float = Field(default=0.9, gt=0, le=1)
    groq_estimated_output_tokens: int = Field(default=1024, ge=0, le=16000)
    groq_max_throttle_wait_seconds: int = Field(default=120, ge=1, le=3600)
    verifier_malformed_retries: int = Field(default=1, ge=0, le=3)
    frontend_url: str = Field(default="http://localhost:5173")
    auto_initialize_database: bool = Field(default=False)
    app_name: str = Field(default="Scientific Literature Gap Discovery API")
    api_prefix: str = Field(default="/api")
    mongodb_timeout_ms: int = Field(default=5_000, ge=1_000, le=30_000)
    pdf_max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1_024, le=100 * 1024 * 1024)
    pdf_chunk_size: int = Field(default=1_200, ge=200, le=5_000)
    pdf_chunk_overlap: int = Field(default=200, ge=0, le=2_500)
    embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dimensions: int = Field(default=384, ge=1, le=4_096)
    embedding_batch_size: int = Field(default=16, ge=1, le=128)
    embedding_device: str = Field(default="cpu")
    vector_search_index_name: str = Field(default="paper_chunks_vector_index")
    vector_search_candidate_multiplier: int = Field(default=10, ge=1, le=100)
    comparison_min_similarity: float = Field(default=0.60, ge=0, le=1)
    comparison_neighbors: int = Field(default=5, ge=1, le=20)
    comparison_discovery_limit: int = Field(default=5, ge=1, le=25)
    comparison_detection_limit: int = Field(default=2, ge=1, le=25)


@lru_cache
def get_settings() -> Settings:
    return Settings()
