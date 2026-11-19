from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    mongodb_uri: str = Field(default="", description="MongoDB Atlas connection string")
    mongodb_database: str = Field(default="scientific_literature_gap_discovery")
    llm_api_key: str = Field(default="", repr=False)
    llm_model_name: str = Field(default="")
    frontend_url: str = Field(default="http://localhost:5173")
    auto_initialize_database: bool = Field(default=False)
    app_name: str = Field(default="Scientific Literature Gap Discovery API")
    api_prefix: str = Field(default="/api")
    mongodb_timeout_ms: int = Field(default=5_000, ge=1_000, le=30_000)


@lru_cache
def get_settings() -> Settings:
    return Settings()
