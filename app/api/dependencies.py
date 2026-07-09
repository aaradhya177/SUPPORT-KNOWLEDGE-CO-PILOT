"""FastAPI dependency providers for expensive application services."""

from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.generation.generator import AnswerGenerator
from app.llm.client import LLMClient, create_llm_client
from app.pipeline import RAGPipeline
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse import BM25Retriever
from app.scoring.confidence import ConfidenceScorer
from app.verification.judge import CitationVerifier


@lru_cache
def get_llm_client() -> LLMClient:
    """Return a cached LLM client instance."""
    return create_llm_client()


@lru_cache
def get_pipeline() -> RAGPipeline:
    """Return a cached RAG pipeline with indexes loaded once."""
    settings = get_settings()

    dense = DenseRetriever()
    dense.load_index(Path(settings.dense_index_path))

    sparse = BM25Retriever()
    sparse.load_index(Path(settings.bm25_index_path))

    llm_client = get_llm_client()
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
    return RAGPipeline(
        retriever=hybrid,
        generator=AnswerGenerator(llm_client=llm_client),
        verifier=CitationVerifier(llm_client=llm_client),
        scorer=ConfidenceScorer(),
    )


def clear_dependency_caches() -> None:
    """Clear cached services, useful after rebuilding indexes."""
    get_pipeline.cache_clear()
    get_llm_client.cache_clear()
