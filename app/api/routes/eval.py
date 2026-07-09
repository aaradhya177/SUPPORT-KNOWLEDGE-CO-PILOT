"""Evaluation API routes."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app.api.schemas import EvalRunRequest, EvalRunResponse
from app.utils.logger import get_logger
from eval.run_eval import run_golden_eval

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/eval/run",
    response_model=EvalRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Run golden-set evaluation",
    description=(
        "Runs the golden-set evaluation pipeline and writes per-question and "
        "summary reports under reports/."
    ),
)
def run_eval_endpoint(request: EvalRunRequest) -> EvalRunResponse:
    """Run the golden-set evaluation and return aggregate metrics."""
    report_path = Path("reports/eval_summary.md")
    try:
        summary = run_golden_eval(sample_size=request.sample_size, summary_path=report_path)
    except Exception as exc:
        logger.exception("Evaluation run failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evaluation run failed.",
        ) from exc

    return EvalRunResponse(summary=summary, report_path=str(report_path))
