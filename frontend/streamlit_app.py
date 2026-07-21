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
    stream_query_backend,
    submit_feedback,
    trigger_eval,
    trigger_ingest,
)

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DEFAULT_API_KEY = os.getenv("SUPPORT_COPILOT_API_KEY", "")
DEFAULT_ADMIN_API_KEY = os.getenv("SUPPORT_COPILOT_ADMIN_API_KEY", "")
CONFIDENCE_THRESHOLD = 0.55
FILTER_CATEGORIES = [
    "",
    "api-rate-limits",
    "billing-faq",
    "data-export-retention",
    "notification-settings",
    "password-reset",
    "sso-login",
    "webhook-troubleshooting",
]
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


def _render_sidebar() -> tuple[str, str, str, int, dict[str, Any] | None, bool]:
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
        api_key = st.text_input(
            "API key",
            value=DEFAULT_API_KEY,
            type="password",
            help="Required for support question requests.",
        )
        admin_api_key = st.text_input(
            "Admin API key",
            value=DEFAULT_ADMIN_API_KEY,
            type="password",
            help="Required for ingestion and evaluation.",
        )
        _render_backend_status(backend_url)
        top_k = st.slider(
            "Retrieved chunks",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of hybrid retrieval chunks sent to generation.",
        )
        use_streaming = st.toggle(
            "Stream progress",
            value=False,
            help="Use the streaming query endpoint and show backend progress events.",
        )
        st.divider()
        st.subheader("Filters")
        category = st.selectbox(
            "Category",
            options=FILTER_CATEGORIES,
            format_func=lambda value: "All" if not value else value,
            help="Restrict retrieval to a source category.",
        )
        source_path = st.text_input(
            "Source path",
            value="",
            help="Restrict retrieval to a matching source path or filename.",
        )
        section = st.text_input(
            "Section",
            value="",
            help="Restrict retrieval to a matching document section.",
        )
        filters = _build_filters(category=category, source_path=source_path, section=section)

        st.divider()
        st.subheader("Knowledge Base")
        if st.button("Re-ingest Documents", use_container_width=True):
            _run_ingest_flow(backend_url, admin_api_key)

        st.divider()
        with st.expander("Admin", expanded=False):
            _render_admin_panel(backend_url, admin_api_key)

    return backend_url, api_key, admin_api_key, top_k, filters, use_streaming


def _build_filters(
    category: str,
    source_path: str,
    section: str,
) -> dict[str, Any] | None:
    """Build API filters from sidebar controls."""
    filters: dict[str, Any] = {}
    if category:
        filters["category"] = [category]
    if source_path.strip():
        filters["source_path"] = [source_path.strip()]
    if section.strip():
        filters["section"] = [section.strip()]
    return filters or None


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


def _run_ingest_flow(backend_url: str, admin_api_key: str) -> None:
    """Trigger ingestion and poll backend status."""
    try:
        with st.spinner("Starting ingestion and index rebuild..."):
            trigger_ingest(backend_url, api_key=admin_api_key)

        status_placeholder = st.empty()
        for _ in range(90):
            status = get_ingest_status(backend_url, api_key=admin_api_key)
            status_text = str(status.get("status", "unknown"))
            status_placeholder.info(
                f"Status: {status_text} | files={status.get('files_processed', 0)} | "
                f"chunks={status.get('chunks_created', 0)}"
            )
            if status_text in {"completed", "failed", "idle"}:
                break
            time.sleep(2)

        final_status = get_ingest_status(backend_url, api_key=admin_api_key)
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


def _render_admin_panel(backend_url: str, admin_api_key: str) -> None:
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
                response = trigger_eval(
                    backend_url,
                    api_key=admin_api_key,
                    sample_size=int(sample_size),
                )
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
    backend_url = st.session_state.get("backend_url", DEFAULT_BACKEND_URL)
    api_key = st.session_state.get("api_key", DEFAULT_API_KEY)
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_assistant_response(
                    message["content"],
                    backend_url=backend_url,
                    api_key=api_key,
                    feedback_key=f"history-{index}",
                )
            else:
                st.markdown(message["content"])


def _render_assistant_response(
    response: dict[str, Any],
    backend_url: str | None = None,
    api_key: str | None = None,
    feedback_key: str | None = None,
) -> None:
    """Render an assistant response payload."""
    status = response.get("status")
    if status == "no_answer":
        st.warning(response.get("answer", "I don't have enough verified information."))
        reason = response.get("reason")
        if reason:
            st.caption(f"Reason: {reason}")
        confidence = float(response.get("confidence", 0.0))
        st.caption(f"Confidence: {confidence:.2f}")
        _render_confidence_breakdown(response)
        _render_feedback_controls(response, backend_url, api_key, feedback_key)
        return

    st.markdown(str(response.get("answer", "")))
    _render_confidence_badge(float(response.get("confidence", 0.0)))
    _render_confidence_breakdown(response)
    _render_citation_verification(response)
    _render_feedback_controls(response, backend_url, api_key, feedback_key)


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


def _render_confidence_breakdown(response: dict[str, Any]) -> None:
    """Render confidence score components when the backend returns them."""
    breakdown = response.get("confidence_breakdown")
    if not isinstance(breakdown, dict):
        return

    with st.expander("Confidence breakdown", expanded=False):
        metric_specs = [
            ("Retrieval", breakdown.get("retrieval_score_component")),
            ("Retriever agreement", breakdown.get("rrf_agreement_component")),
            ("Verification", breakdown.get("verification_score_component")),
            ("Final", breakdown.get("final_confidence")),
        ]
        columns = st.columns(4)
        for column, (label, value) in zip(columns, metric_specs):
            column.metric(label, _format_metric_value(value))

        confidence_state = "Confident" if breakdown.get("is_confident") else "Below threshold"
        st.caption(f"Decision: {confidence_state}")
        reason = breakdown.get("reason")
        if reason:
            st.info(str(reason))


def _render_citation_verification(response: dict[str, Any]) -> None:
    """Render verified and flagged citations as expandable evidence cards."""
    citations = list(response.get("citations", []))
    flagged = list(response.get("flagged_citations", []))
    verdicts = {
        verdict.get("chunk_id"): verdict
        for verdict in response.get("verdicts", [])
        if isinstance(verdict, dict)
    }

    st.markdown("#### Citation Verification")
    if not citations and not flagged:
        st.info("No citations returned.")
        return

    _render_citation_group(
        title="Verified Citations",
        citations=citations,
        verdicts=verdicts,
        default_verdict="SUPPORTED",
        empty_message="No verified citations.",
    )
    _render_citation_group(
        title="Flagged Citations",
        citations=flagged,
        verdicts=verdicts,
        default_verdict="UNSUPPORTED",
        empty_message="No flagged citations.",
    )


def _render_citation_group(
    title: str,
    citations: list[Any],
    verdicts: dict[Any, dict[str, Any]],
    default_verdict: str,
    empty_message: str,
) -> None:
    """Render one citation group with a clear count and expandable rows."""
    st.markdown(f"**{title} ({len(citations)})**")
    if not citations:
        st.caption(empty_message)
        return

    for index, citation in enumerate(citations, start=1):
        if not isinstance(citation, dict):
            continue
        verdict = verdicts.get(citation.get("chunk_id"), {})
        _render_citation_card(
            citation=citation,
            verdict=verdict,
            default_verdict=default_verdict,
            index=index,
        )


def _render_citation_card(
    citation: dict[str, Any],
    verdict: dict[str, Any],
    default_verdict: str,
    index: int,
) -> None:
    """Render a single citation as an expandable card."""
    chunk_id = str(citation.get("chunk_id") or "unknown")
    source_path = str(citation.get("source_path") or "unknown")
    section = str(citation.get("section") or "Unknown section")
    verdict_label = str(verdict.get("verdict") or default_verdict)
    label = f"{index}. {chunk_id} | {verdict_label} | {source_path}"

    with st.expander(label, expanded=False):
        col_a, col_b = st.columns(2)
        col_a.caption("Chunk ID")
        col_a.code(chunk_id, language="text")
        col_b.caption("Verdict")
        col_b.markdown(f"**{verdict_label.replace('_', ' ').title()}**")

        col_c, col_d = st.columns(2)
        col_c.caption("Source path")
        col_c.code(source_path, language="text")
        col_d.caption("Section")
        col_d.write(section)

        claim_excerpt = verdict.get("claim_excerpt")
        if claim_excerpt:
            st.caption("Claim checked")
            st.write(str(claim_excerpt))

        st.caption("Judge reasoning")
        st.write(str(verdict.get("judge_reasoning") or "No judge reasoning returned."))

        quoted_text = citation.get("quoted_text")
        st.caption("Quoted text")
        if quoted_text:
            st.code(str(quoted_text), language="text")
        else:
            st.write("No quoted text returned.")


def _render_feedback_controls(
    response: dict[str, Any],
    backend_url: str | None,
    api_key: str | None,
    feedback_key: str | None,
) -> None:
    """Render thumbs up/down feedback controls for an assistant response."""
    query = str(response.get("query", "")).strip()
    answer = str(response.get("answer", "")).strip()
    if not backend_url or not api_key or not query or not answer:
        return

    key_base = feedback_key or str(response.get("request_id") or id(response))
    existing_feedback = st.session_state.get(f"feedback_submitted_{key_base}")
    if existing_feedback:
        st.caption(f"Feedback recorded: {existing_feedback}")
        return

    comment = st.text_input(
        "Feedback note",
        key=f"feedback_comment_{key_base}",
        placeholder="Optional note",
        label_visibility="collapsed",
    )
    col_up, col_down = st.columns(2)
    with col_up:
        if st.button("Thumbs up", key=f"feedback_up_{key_base}", use_container_width=True):
            _submit_feedback(
                response=response,
                backend_url=backend_url,
                api_key=api_key,
                rating="up",
                comment=comment,
                feedback_key=key_base,
            )
    with col_down:
        if st.button("Thumbs down", key=f"feedback_down_{key_base}", use_container_width=True):
            _submit_feedback(
                response=response,
                backend_url=backend_url,
                api_key=api_key,
                rating="down",
                comment=comment,
                feedback_key=key_base,
            )


def _submit_feedback(
    response: dict[str, Any],
    backend_url: str,
    api_key: str,
    rating: str,
    comment: str | None,
    feedback_key: str,
) -> None:
    """Submit feedback and update UI state."""
    citation_chunk_ids = [
        str(citation.get("chunk_id"))
        for citation in response.get("citations", [])
        if isinstance(citation, dict) and citation.get("chunk_id")
    ]
    try:
        submit_feedback(
            backend_url=backend_url,
            api_key=api_key,
            query=str(response.get("query", "")),
            answer=str(response.get("answer", "")),
            status=str(response.get("status", "")),
            confidence=float(response.get("confidence", 0.0)),
            rating=rating,
            comment=comment or None,
            citation_chunk_ids=citation_chunk_ids,
        )
        st.session_state[f"feedback_submitted_{feedback_key}"] = rating
        st.success("Feedback recorded.")
    except BackendAPIError as exc:
        st.error(str(exc))


def _format_metric_value(value: Any) -> str:
    """Format metric values for display."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if 0.0 <= numeric <= 1.0:
        return f"{numeric:.0%}"
    return f"{numeric:.3f}"


def _submit_query(
    backend_url: str,
    api_key: str,
    top_k: int,
    query: str,
    filters: dict[str, Any] | None,
    use_streaming: bool,
) -> None:
    """Submit a query and append user/assistant messages to history."""
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            if use_streaming:
                response = _run_streaming_query(
                    backend_url=backend_url,
                    api_key=api_key,
                    query=query,
                    top_k=top_k,
                    filters=filters,
                )
            else:
                with st.spinner("Searching, generating, and verifying citations..."):
                    response = query_backend(
                        backend_url=backend_url,
                        query=query,
                        top_k=top_k,
                        api_key=api_key,
                        filters=filters,
                    )
            response["query"] = query
            _render_assistant_response(
                response,
                backend_url=backend_url,
                api_key=api_key,
                feedback_key=f"live-{len(st.session_state.messages)}",
            )
            st.session_state.messages.append({"role": "assistant", "content": response})
        except BackendAPIError as exc:
            st.error(str(exc))
            error_response = {
                "answer": "The backend request failed.",
                "confidence": 0.0,
                "status": "no_answer",
                "reason": str(exc),
                "citations": [],
                "query": query,
            }
            st.session_state.messages.append({"role": "assistant", "content": error_response})


def _run_streaming_query(
    backend_url: str,
    api_key: str,
    query: str,
    top_k: int,
    filters: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run a streaming query and return the completed response payload."""
    status_placeholder = st.empty()
    final_response: dict[str, Any] | None = None

    for event in stream_query_backend(
        backend_url=backend_url,
        query=query,
        top_k=top_k,
        api_key=api_key,
        filters=filters,
    ):
        event_name = str(event.get("event", "message"))
        if event_name == "retrieval_started":
            status_placeholder.info("Retrieving evidence...")
        elif event_name == "retrieval_completed":
            status_placeholder.info(
                f"Retrieved {event.get('retrieved_chunks_count', 0)} candidate chunks."
            )
        elif event_name == "generation_started":
            status_placeholder.info("Generating grounded answer...")
        elif event_name == "verification_started":
            status_placeholder.info("Verifying citations...")
        elif event_name == "completed":
            status_placeholder.empty()
            final_response = dict(event.get("result", {}))
            final_response["request_id"] = event.get("request_id", "")
        elif event_name == "error":
            status_placeholder.empty()
            raise BackendAPIError(str(event.get("message", "Streaming query failed.")))

    if final_response is None:
        raise BackendAPIError("Streaming query ended before a completed event was received.")

    return final_response


def main() -> None:
    """Render the Streamlit frontend."""
    st.set_page_config(
        page_title="Support Knowledge Copilot",
        layout="wide",
    )
    _initialize_state()

    backend_url, api_key, _admin_api_key, top_k, filters, use_streaming = _render_sidebar()
    st.session_state.backend_url = backend_url
    st.session_state.api_key = api_key

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
                _submit_query(
                    backend_url=backend_url,
                    api_key=api_key,
                    top_k=top_k,
                    query=demo_question,
                    filters=filters,
                    use_streaming=use_streaming,
                )

    query = st.chat_input("Ask a support question...")
    if query:
        _submit_query(
            backend_url=backend_url,
            api_key=api_key,
            top_k=top_k,
            query=query,
            filters=filters,
            use_streaming=use_streaming,
        )


if __name__ == "__main__":
    main()
