"""Run the full golden-set evaluation for the RAG pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.generation.generator import AnswerGenerator
from app.llm.client import LLMClient, create_llm_client
from app.pipeline import RAGPipeline
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse import BM25Retriever
from app.scoring.confidence import ConfidenceScorer
from app.verification.judge import CitationVerifier
from app.verification.schemas import CitationVerdict, VerifiedAnswer
from eval.golden_set_schema import GoldenQuestion
from eval.metrics import (
    answer_correctness_llm_graded,
    citation_faithfulness_rate,
    no_answer_precision_recall,
    retrieval_hit_rate,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the evaluation command-line parser."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run golden-set RAG evaluation.")
    parser.add_argument("--golden-set", type=Path, default=Path("eval/golden_set.jsonl"))
    parser.add_argument("--results", type=Path, default=Path("reports/eval_run_results.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("reports/eval_summary.md"))
    parser.add_argument("--dense-dir", type=Path, default=Path(settings.dense_index_path))
    parser.add_argument("--sparse-dir", type=Path, default=Path(settings.bm25_index_path))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional pause between questions to stay within hosted LLM rate limits.",
    )
    return parser


def _load_golden_set(path: Path, sample_size: int | None = None) -> list[GoldenQuestion]:
    """Load and optionally truncate the golden set."""
    questions: list[GoldenQuestion] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            questions.append(GoldenQuestion.model_validate_json(line))
    if sample_size is not None:
        return questions[:sample_size]
    return questions


def _build_pipeline(llm_client: LLMClient, dense_dir: Path, sparse_dir: Path) -> RAGPipeline:
    """Construct a production pipeline for evaluation."""
    dense = DenseRetriever()
    dense.load_index(dense_dir)

    sparse = BM25Retriever()
    sparse.load_index(sparse_dir)

    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
    return RAGPipeline(
        retriever=hybrid,
        generator=AnswerGenerator(llm_client=llm_client),
        verifier=CitationVerifier(llm_client=llm_client),
        scorer=ConfidenceScorer(),
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_summary(path: Path, metrics: dict[str, float], total: int) -> None:
    """Write a markdown evaluation summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        [
            {"Metric": "Retrieval Hit Rate", "Value": metrics["retrieval_hit_rate"]},
            {"Metric": "Avg Answer Correctness", "Value": metrics["avg_answer_correctness"]},
            {"Metric": "Avg Citation Faithfulness", "Value": metrics["avg_citation_faithfulness"]},
            {"Metric": "No-Answer Precision", "Value": metrics["no_answer_precision"]},
            {"Metric": "No-Answer Recall", "Value": metrics["no_answer_recall"]},
        ]
    )
    lines = [
        "# Golden Set Evaluation Summary",
        "",
        f"Questions evaluated: {total}",
        "",
        _dataframe_to_markdown(table),
        "",
        "## Interpretation",
        "",
        "Retrieval hit rate measures whether at least one expected source document appeared in the retrieved context. "
        "Answer correctness is LLM-graded against the human-authored expected summary. Citation faithfulness measures "
        "how many judge verdicts were fully supported. No-answer precision and recall specifically evaluate whether "
        "the system refuses plausible but unsupported questions without over-refusing answerable ones.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dataframe_to_markdown(table: pd.DataFrame) -> str:
    """Render a small DataFrame as markdown without optional tabulate dependency."""
    lines = ["| Metric | Value |", "| --- | --- |"]
    for _, row in table.iterrows():
        lines.append(f"| {row['Metric']} | {float(row['Value']):.4f} |")
    return "\n".join(lines)


def run_golden_eval(
    golden_set_path: Path = Path("eval/golden_set.jsonl"),
    results_path: Path = Path("reports/eval_run_results.jsonl"),
    summary_path: Path = Path("reports/eval_summary.md"),
    dense_dir: Path | None = None,
    sparse_dir: Path | None = None,
    top_k: int = 5,
    sample_size: int | None = None,
    sleep_seconds: float = 0.0,
    llm_client: LLMClient | None = None,
    pipeline: RAGPipeline | None = None,
) -> dict[str, float]:
    """Run the golden-set evaluation as an importable function.

    Args:
        golden_set_path: Golden-set JSONL path.
        results_path: Per-question JSONL output path.
        summary_path: Markdown summary output path.
        dense_dir: Dense index directory. Uses config when omitted.
        sparse_dir: Sparse index directory. Uses config when omitted.
        top_k: Retrieval cutoff.
        sample_size: Optional number of examples to run.
        sleep_seconds: Optional pause between questions for rate-limited LLM APIs.
        llm_client: Optional LLM client dependency.
        pipeline: Optional prebuilt pipeline dependency.

    Returns:
        Aggregate metrics.
    """
    settings = get_settings()
    resolved_dense_dir = dense_dir or Path(settings.dense_index_path)
    resolved_sparse_dir = sparse_dir or Path(settings.bm25_index_path)
    resolved_llm_client = llm_client or create_llm_client()
    resolved_pipeline = pipeline or _build_pipeline(
        resolved_llm_client,
        dense_dir=resolved_dense_dir,
        sparse_dir=resolved_sparse_dir,
    )
    golden = _load_golden_set(golden_set_path, sample_size=sample_size)

    rows: list[dict[str, Any]] = []
    correctness_scores: list[float] = []
    faithfulness_scores: list[float] = []
    retrieval_hits: list[bool] = []

    for question in golden:
        retrieved = resolved_pipeline.retriever.retrieve(question.question, top_k=top_k)
        retrieved_doc_ids = [chunk.doc_id for chunk in retrieved]
        retrieval_hit = retrieval_hit_rate(retrieved_doc_ids, question.expected_doc_ids)
        error: str | None = None
        try:
            response = resolved_pipeline.answer_query(question.question, top_k=top_k)
        except Exception as exc:
            error = str(exc)
            response = {
                "answer": "",
                "status": "error",
                "confidence": 0.0,
                "verdicts": [],
            }

        if question.expected_answerable and response["status"] == "answered":
            try:
                correctness = answer_correctness_llm_graded(
                    generated_answer=str(response["answer"]),
                    expected_summary=question.expected_answer_summary,
                    llm_client=resolved_llm_client,
                )
            except Exception as exc:
                error = f"{error}; {exc}" if error else str(exc)
                correctness = 0.0
        elif question.expected_answerable:
            correctness = 0.0
        else:
            correctness = 1.0 if response["status"] == "no_answer" else 0.0

        verdicts = [
            CitationVerdict.model_validate(verdict) for verdict in response.get("verdicts", [])
        ]
        verified = VerifiedAnswer(
            answer_text=str(response.get("answer", "")),
            verified_citations=[],
            flagged_citations=[],
            verdicts=verdicts,
            all_supported=all(verdict.verdict.value != "UNSUPPORTED" for verdict in verdicts),
        )
        faithfulness = citation_faithfulness_rate(verified)

        retrieval_hits.append(retrieval_hit)
        correctness_scores.append(correctness)
        faithfulness_scores.append(faithfulness)
        rows.append(
            {
                "id": question.id,
                "question": question.question,
                "category": question.category,
                "expected_doc_ids": question.expected_doc_ids,
                "expected_answer_summary": question.expected_answer_summary,
                "expected_answerable": question.expected_answerable,
                "retrieved_doc_ids": retrieved_doc_ids,
                "retrieval_hit": retrieval_hit,
                "answer": response.get("answer"),
                "status": response.get("status"),
                "confidence": response.get("confidence"),
                "answer_correctness": correctness,
                "citation_faithfulness": faithfulness,
                "error": error,
            }
        )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    no_answer = no_answer_precision_recall(rows, golden)
    aggregate_metrics = {
        "retrieval_hit_rate": sum(retrieval_hits) / len(retrieval_hits) if retrieval_hits else 0.0,
        "avg_answer_correctness": mean(correctness_scores) if correctness_scores else 0.0,
        "avg_citation_faithfulness": mean(faithfulness_scores) if faithfulness_scores else 0.0,
        "no_answer_precision": no_answer["precision"],
        "no_answer_recall": no_answer["recall"],
    }

    _write_jsonl(results_path, rows)
    _write_summary(summary_path, aggregate_metrics, total=len(golden))
    return aggregate_metrics


def main() -> None:
    """Run the golden-set evaluation."""
    args = _build_parser().parse_args()
    aggregate_metrics = run_golden_eval(
        golden_set_path=args.golden_set,
        results_path=args.results,
        summary_path=args.summary,
        dense_dir=args.dense_dir,
        sparse_dir=args.sparse_dir,
        top_k=args.top_k,
        sample_size=args.sample_size,
        sleep_seconds=args.sleep_seconds,
    )
    print(pd.DataFrame([aggregate_metrics]).to_string(index=False))
    print(f"Wrote per-question results to {args.results}")
    print(f"Wrote summary to {args.summary}")


if __name__ == "__main__":
    main()
