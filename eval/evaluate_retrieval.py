"""Evaluate dense, sparse, and hybrid retrieval quality on a JSONL eval set."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.retrieval.base import BaseRetriever, RetrievedChunk
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.sparse import BM25Retriever


@dataclass(frozen=True)
class RetrievalExample:
    """Single retrieval evaluation example."""

    query: str
    expected_doc_id: str | None
    expected_chunk_ids: frozenset[str]
    expected_keywords: tuple[str, ...]


def _build_parser() -> argparse.ArgumentParser:
    """Build the retrieval evaluation command-line parser.

    Returns:
        Configured argument parser.
    """
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate dense, sparse, and hybrid retrieval.")
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=Path("eval/retrieval_eval_set.jsonl"),
        help="JSONL file containing retrieval evaluation examples.",
    )
    parser.add_argument(
        "--dense-dir",
        type=Path,
        default=Path(settings.dense_index_path),
        help="Directory containing dense FAISS artifacts.",
    )
    parser.add_argument(
        "--sparse-dir",
        type=Path,
        default=Path(settings.bm25_index_path),
        help="Directory containing sparse BM25 artifacts.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/retrieval_comparison.md"),
        help="Markdown report output path.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Hit-rate cutoff.")
    return parser


def _load_eval_set(path: Path) -> list[RetrievalExample]:
    """Load retrieval examples from JSONL.

    Args:
        path: Evaluation JSONL path.

    Returns:
        Parsed retrieval examples.
    """
    examples: list[RetrievalExample] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            payload = json.loads(line)
            examples.append(
                RetrievalExample(
                    query=str(payload["query"]),
                    expected_doc_id=payload.get("expected_doc_id"),
                    expected_chunk_ids=frozenset(payload.get("expected_chunk_ids", [])),
                    expected_keywords=tuple(payload.get("expected_keywords", [])),
                )
            )
    return examples


def _is_hit(results: Iterable[RetrievedChunk], example: RetrievalExample) -> bool:
    """Return whether any retrieved result satisfies an eval example.

    A result is considered a hit if it matches an expected chunk ID, matches the
    expected document ID, or contains at least one expected keyword.

    Args:
        results: Retrieved chunks.
        example: Evaluation example.

    Returns:
        True when the expected evidence appears in the results.
    """
    expected_keywords = [keyword.lower() for keyword in example.expected_keywords]

    for result in results:
        if result.chunk_id in example.expected_chunk_ids:
            return True
        if example.expected_doc_id and result.doc_id == example.expected_doc_id:
            return True
        text = result.text.lower()
        if any(keyword in text for keyword in expected_keywords):
            return True

    return False


def _hit_rate_at_k(
    retriever: BaseRetriever,
    examples: list[RetrievalExample],
    top_k: int,
) -> tuple[float, list[bool]]:
    """Compute Hit Rate@K for a retriever.

    Args:
        retriever: Retriever to evaluate.
        examples: Evaluation examples.
        top_k: Retrieval cutoff.

    Returns:
        Hit rate and per-example hit flags.
    """
    hits: list[bool] = []
    for example in examples:
        results = retriever.retrieve(example.query, top_k=top_k)
        hits.append(_is_hit(results, example))

    if not hits:
        return 0.0, []
    return sum(hits) / len(hits), hits


def _write_report(report_path: Path, table: pd.DataFrame, top_k: int, example_count: int) -> None:
    """Write the retrieval comparison markdown report.

    Args:
        report_path: Report output path.
        table: Metrics table.
        top_k: Retrieval cutoff.
        example_count: Number of evaluation examples.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metric_column = f"Hit Rate@{top_k}"
    table_markdown = _dataframe_to_markdown(table)
    best_row = table.sort_values(metric_column, ascending=False).iloc[0]

    report = (
        "# Retrieval Accuracy Comparison\n\n"
        f"Evaluation examples: {example_count}\n\n"
        f"Metric: Hit Rate@{top_k}, counted as a hit when the expected chunk, "
        "expected document, or expected support evidence keyword appears in the top results.\n\n"
        f"{table_markdown}\n\n"
        "## Interpretation\n\n"
        f"The strongest retriever in this run is `{best_row['Retriever']}` with "
        f"{best_row[metric_column]:.2%} Hit Rate@{top_k}. Hybrid retrieval is expected "
        "to outperform either individual retriever when the eval set mixes exact lexical "
        "needs and paraphrased intent: BM25 is strong for precise terms such as HTTP 429, "
        "invoice, and tax exemption, while dense retrieval can recover semantically similar "
        "support questions that do not reuse the document wording. The reported numbers are "
        "computed from this eval set and the currently built indexes; they should be rerun "
        "whenever the corpus, chunking strategy, embedding model, or fusion settings change.\n\n"
        "## Resume Claim Guidance\n\n"
        "Do not claim a retrieval lift such as `72% to 88%` unless this report actually "
        "shows those measured values on a sufficiently large and non-trivial eval set. "
        "When the corpus is tiny or the number of indexed chunks is less than or close to "
        "the Hit Rate cutoff, Hit Rate@K can saturate and stop distinguishing retrievers.\n"
    )
    report_path.write_text(report, encoding="utf-8")


def _dataframe_to_markdown(table: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-style markdown table.

    Args:
        table: DataFrame to render.

    Returns:
        Markdown table string.
    """
    headers = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for _, row in table.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def main() -> None:
    """Run the retrieval evaluation and write a comparison report."""
    parser = _build_parser()
    args = parser.parse_args()

    examples = _load_eval_set(args.eval_set)

    dense = DenseRetriever()
    dense.load_index(args.dense_dir)

    sparse = BM25Retriever()
    sparse.load_index(args.sparse_dir)

    settings = get_settings()
    reranker = (
        CrossEncoderReranker(model_name=settings.reranker_model_name)
        if settings.enable_reranker
        else None
    )
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, reranker=reranker)
    hybrid_name = "Hybrid RRF + Reranker" if settings.enable_reranker else "Hybrid RRF"

    rows: list[dict[str, object]] = []
    for name, retriever in [
        ("Dense", dense),
        ("BM25", sparse),
        (hybrid_name, hybrid),
    ]:
        hit_rate, hits = _hit_rate_at_k(retriever, examples, top_k=args.top_k)
        rows.append(
            {
                "Retriever": name,
                f"Hit Rate@{args.top_k}": hit_rate,
                "Hits": sum(hits),
                "Total": len(hits),
            }
        )

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    _write_report(args.report, table, top_k=args.top_k, example_count=len(examples))
    print(f"\nWrote report to {args.report}")


if __name__ == "__main__":
    main()
