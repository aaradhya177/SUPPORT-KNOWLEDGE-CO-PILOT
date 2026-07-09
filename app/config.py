"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the Support Knowledge Copilot service."""

    embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        alias="EMBEDDING_MODEL_NAME",
    )
    llm_provider: str = Field(default="gemini", alias="LLM_PROVIDER")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model_name: str = Field(default="gemini-flash-lite-latest", alias="LLM_MODEL_NAME")
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    dense_index_path: str = Field(default="indexes/dense", alias="DENSE_INDEX_PATH")
    bm25_index_path: str = Field(default="indexes/sparse", alias="BM25_INDEX_PATH")
    confidence_threshold: float = Field(default=0.55, alias="CONFIDENCE_THRESHOLD")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
