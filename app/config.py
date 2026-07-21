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
    support_copilot_api_key: str = Field(default="", alias="SUPPORT_COPILOT_API_KEY")
    support_copilot_admin_api_key: str = Field(
        default="",
        alias="SUPPORT_COPILOT_ADMIN_API_KEY",
    )
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    dense_index_path: str = Field(default="indexes/dense", alias="DENSE_INDEX_PATH")
    bm25_index_path: str = Field(default="indexes/sparse", alias="BM25_INDEX_PATH")
    enable_reranker: bool = Field(default=False, alias="ENABLE_RERANKER")
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="RERANKER_MODEL_NAME",
    )
    ingestion_jobs_db_path: str = Field(
        default="data/processed/ingestion_jobs.sqlite3",
        alias="INGESTION_JOBS_DB_PATH",
    )
    feedback_db_path: str = Field(
        default="data/processed/feedback.sqlite3",
        alias="FEEDBACK_DB_PATH",
    )
    enable_judge_cache: bool = Field(default=True, alias="ENABLE_JUDGE_CACHE")
    judge_cache_db_path: str = Field(
        default="data/processed/judge_cache.sqlite3",
        alias="JUDGE_CACHE_DB_PATH",
    )
    confidence_threshold: float = Field(default=0.55, alias="CONFIDENCE_THRESHOLD")
    query_rate_limit_requests: int = Field(default=60, alias="QUERY_RATE_LIMIT_REQUESTS")
    query_rate_limit_window_seconds: int = Field(
        default=60,
        alias="QUERY_RATE_LIMIT_WINDOW_SECONDS",
    )
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
