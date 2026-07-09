"""Run a retrieval-augmented support query from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.generation.generator import AnswerGenerator
from app.llm.client import create_llm_client
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse import BM25Retriever
from app.verification.judge import CitationVerifier


def _build_parser() -> argparse.ArgumentParser:
    """Build the query command-line parser.

    Returns:
        Configured argument parser.
    """
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ask a grounded question over support docs.")
    parser.add_argument("--query", required=True, help="Support question to answer.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of hybrid chunks to use.")
    parser.add_argument("--dense-dir", type=Path, default=Path(settings.dense_index_path))
    parser.add_argument("--sparse-dir", type=Path, default=Path(settings.bm25_index_path))
    return parser


def main() -> None:
    """Run hybrid retrieval and grounded answer generation."""
    parser = _build_parser()
    args = parser.parse_args()

    dense = DenseRetriever()
    dense.load_index(args.dense_dir)

    sparse = BM25Retriever()
    sparse.load_index(args.sparse_dir)

    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
    retrieved_chunks = hybrid.retrieve(args.query, top_k=args.top_k)

    llm_client = create_llm_client()
    generator = AnswerGenerator(llm_client=llm_client)
    generated = generator.generate(query=args.query, retrieved_chunks=retrieved_chunks)
    verifier = CitationVerifier(llm_client=llm_client)
    verified = verifier.verify(generated=generated, retrieved_chunks=retrieved_chunks)

    print("\nAnswer\n" + "=" * 80)
    print(verified.answer_text)

    print("\nVerified Citations\n" + "=" * 80)
    if not verified.verified_citations:
        print("No verified citations returned.")
    for citation in verified.verified_citations:
        print(f"- [{citation.chunk_id}] doc_id={citation.doc_id} source={citation.source_path}")
        if citation.quoted_text:
            print(f"  snippet: {citation.quoted_text}")

    print("\nFlagged Citations\n" + "=" * 80)
    if not verified.flagged_citations:
        print("No flagged citations.")
    for citation in verified.flagged_citations:
        print(f"- [{citation.chunk_id}] doc_id={citation.doc_id} source={citation.source_path}")
        if citation.quoted_text:
            print(f"  snippet: {citation.quoted_text}")

    print("\nJudge Verdicts\n" + "=" * 80)
    for verdict in verified.verdicts:
        print(
            f"- [{verdict.chunk_id}] {verdict.verdict.value}: "
            f"{verdict.claim_excerpt} | {verdict.judge_reasoning}"
        )


if __name__ == "__main__":
    main()
