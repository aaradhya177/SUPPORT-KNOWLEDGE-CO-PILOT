"""Streamlit frontend for Support Knowledge Copilot."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.api_client import (
    BackendAPIError,
    get_backend_health,
    get_ingest_status,
    query_backend,
    trigger_eval,
    trigger_ingest,
)

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
CONFIDENCE_THRESHOLD = 0.55
DEMO_QUESTIONS = [
    "What should I do if my password reset email never arrives?",
    "How do invoice corrections work after a payment has already been made?",
    "What are the API rate limits and what should clients do after a 429?",
    "What is the maximum webhook payload size for Northstar Desk?",
]


def _initialize_state() -> None:
    """Initialize Streamlit session state defaults."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_eval_summary", _load_last_eval_summary())


def _load_last_eval_summary() -> dict[str, Any] | None:
    """Load a lightweight summary from the generated eval report if present.

    Returns:
        Parsed metric labels and values when the report exists.
    """
    report_path = Path("reports/eval_summary.md")
    if not report_path.exists():
        return None

    summary: dict[str, Any] = {"report_path": str(report_path), "metrics": {}}
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| ---") or "Metric" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 2:
            summary["metrics"][cells[0]] = cells[1]
    return summary


def _render_sidebar() -> tuple[str, int]:
    """Render sidebar settings and admin controls.

    Returns:
        Backend URL and top-k setting.
    """
    with st.sidebar:
        st.header("Settings")
        backend_url = st.text_input(
            "Backend URL",
            value=DEFAULT_BACKEND_URL,
            help="FastAPI backend base URL.",
        )
        _render_backend_status(backend_url)
        top_k = st.slider(
            "Retrieved chunks",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of hybrid retrieval chunks sent to generation.",
        )

        st.divider()
        st.subheader("Knowledge Base")
        if st.button("Re-ingest Documents", use_container_width=True):
            _run_ingest_flow(backend_url)

        st.divider()
        with st.expander("Admin", expanded=False):
            _render_admin_panel(backend_url)

    return backend_url, top_k


def _render_backend_status(backend_url: str) -> None:
    """Render a compact backend connectivity indicator."""
    try:
        payload = get_backend_health(backend_url)
        if payload.get("status") == "ok":
            st.success("Backend connected")
        else:
            st.warning("Backend responded, but health status was unexpected.")
    except BackendAPIError as exc:
        st.error(f"Backend unavailable: {exc}")


def _run_ingest_flow(backend_url: str) -> None:
    """Trigger ingestion and poll backend status."""
    try:
        with st.spinner("Starting ingestion and index rebuild..."):
            trigger_ingest(backend_url)

        status_placeholder = st.empty()
        for _ in range(90):
            status = get_ingest_status(backend_url)
            status_text = str(status.get("status", "unknown"))
            status_placeholder.info(
                f"Status: {status_text} | files={status.get('files_processed', 0)} | "
                f"chunks={status.get('chunks_created', 0)}"
            )
            if status_text in {"completed", "failed", "idle"}:
                break
            time.sleep(2)

        final_status = get_ingest_status(backend_url)
        if final_status.get("status") == "completed":
            st.success(
                "Ingestion complete: "
                f"{final_status.get('files_processed', 0)} files, "
                f"{final_status.get('chunks_created', 0)} chunks."
            )
        elif final_status.get("status") == "failed":
            st.error("Ingestion failed. Check backend logs for details.")
    except BackendAPIError as exc:
        st.error(str(exc))


def _render_admin_panel(backend_url: str) -> None:
    """Render evaluation controls and last summary."""
    st.caption("Evaluation")
    sample_size = st.number_input(
        "Sample size",
        min_value=1,
        max_value=75,
        value=10,
        help="Run a smaller eval for quick checks.",
    )
    if st.button("Run Eval", use_container_width=True):
        try:
            with st.spinner("Running evaluation..."):
                response = trigger_eval(backend_url, sample_size=int(sample_size))
            st.session_state.last_eval_summary = {
                "report_path": response.get("report_path"),
                "metrics": response.get("summary", {}),
            }
            st.success("Evaluation complete.")
        except BackendAPIError as exc:
            st.error(str(exc))

    summary = st.session_state.get("last_eval_summary")
    if not summary:
        st.info("No eval summary found yet.")
        return

    st.caption(f"Last eval report: `{summary.get('report_path', 'unknown')}`")
    metrics = summary.get("metrics", {})
    if not metrics:
        st.info("No metrics available.")
        return

    for label, value in metrics.items():
        st.metric(label.replace("_", " ").title(), _format_metric_value(value))


def _render_chat_history() -> None:
    """Render prior chat messages."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_assistant_response(message["content"])
            else:
                st.markdown(message["content"])


def _render_assistant_response(response: dict[str, Any]) -> None:
    """Render an assistant response payload."""
    status = response.get("status")
    if status == "no_answer":
        st.warning(response.get("answer", "I don't have enough verified information."))
        reason = response.get("reason")
        if reason:
            st.caption(f"Reason: {reason}")
        confidence = float(response.get("confidence", 0.0))
        st.caption(f"Confidence: {confidence:.2f}")
        return

    st.markdown(str(response.get("answer", "")))
    _render_confidence_badge(float(response.get("confidence", 0.0)))
    _render_citations(response)


def _render_confidence_badge(confidence: float) -> None:
    """Render a color-coded confidence badge."""
    if confidence > 0.8:
        color = "#16794c"
        label = "High confidence"
    elif confidence >= CONFIDENCE_THRESHOLD:
        color = "#9a6700"
        label = "Meets threshold"
    else:
        color = "#b42318"
        label = "Below threshold"

    st.markdown(
        f"""
        <div style="display:inline-block;padding:0.35rem 0.65rem;border-radius:999px;
                    background:{color};color:white;font-weight:600;margin:0.35rem 0;">
            {label}: {confidence:.2f}
        </div>
        <div style="font-size:0.85rem;color:#666;margin-bottom:0.5rem;">
            Confidence threshold: {CONFIDENCE_THRESHOLD:.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_citations(response: dict[str, Any]) -> None:
    """Render verified and flagged citations."""
    citations = list(response.get("citations", []))
    flagged = list(response.get("flagged_citations", []))
    verdicts = {
        verdict.get("chunk_id"): verdict
        for verdict in response.get("verdicts", [])
        if isinstance(verdict, dict)
    }

    with st.expander("Citation verification", expanded=True):
        if not citations and not flagged:
            st.info("No citations returned.")
            return

        for citation in citations:
            verdict = verdicts.get(citation.get("chunk_id"), {})
            st.markdown(
                f"✅ **Verified** `[{citation.get('chunk_id')}]` "
                f"from `{citation.get('source_path', 'unknown')}`"
            )
            if verdict.get("judge_reasoning"):
                st.caption(str(verdict["judge_reasoning"]))
            if citation.get("quoted_text"):
                st.code(str(citation["quoted_text"]), language="text")

        for citation in flagged:
            verdict = verdicts.get(citation.get("chunk_id"), {})
            st.markdown(
                f"""
                <div style="border-left:4px solid #b42318;padding:0.5rem 0.75rem;
                            background:#fff1f0;margin:0.5rem 0;">
                    ⚠️ <strong>Flagged / unverified</strong>
                    <code>[{citation.get('chunk_id')}]</code>
                    from <code>{citation.get('source_path', 'unknown')}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if verdict.get("judge_reasoning"):
                st.caption(str(verdict["judge_reasoning"]))
            if citation.get("quoted_text"):
                st.code(str(citation["quoted_text"]), language="text")


def _format_metric_value(value: Any) -> str:
    """Format metric values for display."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if 0.0 <= numeric <= 1.0:
        return f"{numeric:.0%}"
    return f"{numeric:.3f}"


def _submit_query(backend_url: str, top_k: int, query: str) -> None:
    """Submit a query and append user/assistant messages to history."""
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching, generating, and verifying citations..."):
                response = query_backend(backend_url=backend_url, query=query, top_k=top_k)
            _render_assistant_response(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except BackendAPIError as exc:
            st.error(str(exc))
            error_response = {
                "answer": "The backend request failed.",
                "confidence": 0.0,
                "status": "no_answer",
                "reason": str(exc),
                "citations": [],
            }
            st.session_state.messages.append({"role": "assistant", "content": error_response})


def main() -> None:
    """Render the Streamlit frontend."""
    st.set_page_config(
        page_title="Support Knowledge Copilot",
        layout="wide",
    )
    _initialize_state()

    backend_url, top_k = _render_sidebar()

    st.title("Support Knowledge Copilot")
    st.caption("Hybrid RAG with LLM-verified citations and confidence-based no-answer detection.")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Retrieval", "Dense + BM25")
    col_b.metric("Citation Check", "LLM Judge")
    col_c.metric("No-Answer Gate", f"{CONFIDENCE_THRESHOLD:.2f}")

    _render_chat_history()

    if not st.session_state.messages:
        st.subheader("Try a demo question")
        cols = st.columns(2)
        for index, demo_question in enumerate(DEMO_QUESTIONS):
            if cols[index % 2].button(demo_question, use_container_width=True):
                _submit_query(backend_url=backend_url, top_k=top_k, query=demo_question)

    query = st.chat_input("Ask a support question...")
    if query:
        _submit_query(backend_url=backend_url, top_k=top_k, query=query)


if __name__ == "__main__":
    main()
