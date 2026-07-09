"""Tests for evaluation metrics."""

from typing import Any

from app.generation.generator import Citation
from app.llm.client import LLMClient
from app.verification.schemas import CitationVerdict, Verdict, VerifiedAnswer
from eval.golden_set_schema import GoldenQuestion
from eval.metrics import (
    answer_correctness_llm_graded,
    citation_faithfulness_rate,
    no_answer_precision_recall,
    retrieval_hit_rate,
)


class FakeGraderLLM(LLMClient):
    """Fake grader client returning fixed JSON."""

    def __init__(self, response: str) -> None:
        """Initialize with a response string."""
        self.response = response

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        """Return the fixed response."""
        return self.response


def test_retrieval_hit_rate_detects_overlap() -> None:
    """Assert retrieval hit rate is true when any expected doc is retrieved."""
    assert retrieval_hit_rate(["a", "b"], ["c", "b"]) is True
    assert retrieval_hit_rate(["a"], ["b"]) is False


def test_answer_correctness_llm_graded_parses_score() -> None:
    """Assert LLM-graded correctness parses fenced JSON scores."""
    llm = FakeGraderLLM('```json\n{"score": 0.75, "reasoning": "mostly correct"}\n```')

    score = answer_correctness_llm_graded(
        generated_answer="answer",
        expected_summary="expected",
        llm_client=llm,
    )

    assert score == 0.75


def test_answer_correctness_llm_graded_returns_zero_on_bad_output() -> None:
    """Assert malformed grader output returns zero."""
    llm = FakeGraderLLM("not json")

    assert answer_correctness_llm_graded("answer", "expected", llm) == 0.0


def test_citation_faithfulness_rate_counts_supported_only() -> None:
    """Assert faithfulness is supported verdicts divided by all verdicts."""
    verified = VerifiedAnswer(
        answer_text="answer",
        verified_citations=[
            Citation(chunk_id="a", doc_id="doc", source_path="doc.md", quoted_text="quote")
        ],
        flagged_citations=[],
        verdicts=[
            CitationVerdict(
                chunk_id="a",
                claim_excerpt="claim",
                verdict=Verdict.SUPPORTED,
                judge_reasoning="ok",
            ),
            CitationVerdict(
                chunk_id="b",
                claim_excerpt="claim",
                verdict=Verdict.PARTIALLY_SUPPORTED,
                judge_reasoning="partial",
            ),
            CitationVerdict(
                chunk_id="c",
                claim_excerpt="claim",
                verdict=Verdict.UNSUPPORTED,
                judge_reasoning="bad",
            ),
        ],
        all_supported=False,
    )

    assert citation_faithfulness_rate(verified) == 1 / 3


def test_no_answer_precision_recall() -> None:
    """Assert no-answer precision and recall are computed from golden labels."""
    golden = [
        GoldenQuestion(
            id="1",
            question="q1",
            expected_doc_ids=[],
            expected_answer_summary="none",
            expected_answerable=False,
            category="x",
        ),
        GoldenQuestion(
            id="2",
            question="q2",
            expected_doc_ids=["a"],
            expected_answer_summary="answer",
            expected_answerable=True,
            category="x",
        ),
        GoldenQuestion(
            id="3",
            question="q3",
            expected_doc_ids=[],
            expected_answer_summary="none",
            expected_answerable=False,
            category="x",
        ),
    ]
    results: list[dict[str, Any]] = [
        {"id": "1", "status": "no_answer"},
        {"id": "2", "status": "no_answer"},
        {"id": "3", "status": "answered"},
    ]

    metrics = no_answer_precision_recall(results, golden)

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
