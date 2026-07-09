"""Evaluation metrics for the RAG pipeline."""

from __future__ import annotations

import json
import re
from typing import Any

from app.llm.client import LLMClient
from app.verification.schemas import Verdict, VerifiedAnswer
from eval.golden_set_schema import GoldenQuestion

GRADING_SYSTEM_PROMPT = """You are an exacting support-answer evaluator.

Compare the generated answer with the expected answer summary. Grade only factual correctness and completeness relative to the expected summary.
Return strict JSON only: {"score": 0.0-1.0, "reasoning": "..."}.

Scoring guide:
- 1.0: Fully correct and complete.
- 0.7: Mostly correct, minor omission.
- 0.4: Partially correct, major omission or ambiguity.
- 0.0: Incorrect, unsupported, or refuses despite the expected summary being answerable.
"""


def retrieval_hit_rate(retrieved_doc_ids: list[str], expected_doc_ids: list[str]) -> bool:
    """Return whether any expected document appears in retrieved documents."""
    return bool(set(retrieved_doc_ids).intersection(expected_doc_ids))


def answer_correctness_llm_graded(
    generated_answer: str,
    expected_summary: str,
    llm_client: LLMClient,
) -> float:
    """Grade answer correctness against an expected summary with an LLM judge.

    Args:
        generated_answer: Pipeline answer text.
        expected_summary: Human-authored expected answer summary.
        llm_client: LLM client used for grading.

    Returns:
        Score from 0.0 to 1.0. Malformed grader output returns 0.0.
    """
    user_prompt = (
        "Expected answer summary:\n"
        f"{expected_summary}\n\n"
        "Generated answer:\n"
        f"{generated_answer}\n\n"
        "Return strict JSON only."
    )
    raw_output = llm_client.complete(
        system_prompt=GRADING_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=256,
    )
    try:
        payload = _extract_json_object(raw_output)
        return _clamp01(float(payload["score"]))
    except Exception:
        return 0.0


def citation_faithfulness_rate(verified: VerifiedAnswer) -> float:
    """Return the proportion of judge verdicts that are fully supported."""
    if not verified.verdicts:
        return 0.0
    supported = sum(1 for verdict in verified.verdicts if verdict.verdict == Verdict.SUPPORTED)
    return supported / len(verified.verdicts)


def no_answer_precision_recall(
    results: list[dict[str, Any]],
    golden: list[GoldenQuestion],
) -> dict[str, float]:
    """Compute precision and recall for no-answer detection.

    Precision answers: when the system says no-answer, how often was the question
    truly unanswerable? Recall answers: of all truly unanswerable questions, how
    many did the system correctly refuse?
    """
    golden_by_id = {question.id: question for question in golden}
    predicted_no_answer = [result for result in results if result.get("status") == "no_answer"]
    expected_unanswerable = [question for question in golden if not question.expected_answerable]
    true_positive = 0
    for result in predicted_no_answer:
        question = golden_by_id.get(str(result.get("id")))
        if question is not None and not question.expected_answerable:
            true_positive += 1

    precision = true_positive / len(predicted_no_answer) if predicted_no_answer else 0.0
    recall = true_positive / len(expected_unanswerable) if expected_unanswerable else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "true_positive": float(true_positive),
        "predicted_no_answer": float(len(predicted_no_answer)),
        "expected_unanswerable": float(len(expected_unanswerable)),
    }


def _extract_json_object(raw_output: str) -> dict[str, Any]:
    """Extract a JSON object from grader output."""
    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*?\}", cleaned, flags=re.DOTALL)
    if match is None:
        raise ValueError("No JSON object found.")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("JSON payload is not an object.")
    return parsed


def _clamp01(value: float) -> float:
    """Clamp a numeric value to 0..1."""
    return max(0.0, min(1.0, value))
