"""Reusable Streamlit render functions for an analysis and its reasoning trace.

Kept separate from the page wiring so the History page (a later ticket) can reuse
`render_analysis_card` verbatim. These functions take ORM rows + read the JSONL
trace; they never query the DB for analyses themselves.
"""

import json

import streamlit as st

from dashboard.data import read_reasoning_trace
from storage.models import Analysis

# Recommendation → coloured badge. Unknown/None falls back to a neutral marker.
_REC_BADGES = {"buy": "🟢", "sell": "🔴", "hold": "🟡"}

# Strong action calls auto-expand so nothing important is buried on page load.
_AUTO_EXPAND_CONFIDENCE = 0.6


def _as_float(value: object) -> float:
    """Coerce a JSON-decoded numeric value to float, defaulting to 0.0."""
    return float(value) if isinstance(value, (int, float)) else 0.0


def render_analysis_card(
    analysis: Analysis,
    *,
    prompt_version: str | None = None,
    prior_recommendation: str | None = None,
) -> None:
    """Render one analysis as a colour-coded, expandable card.

    When `prompt_version` is given (the History page passes it), the card's date and the
    prompt version tag are appended to the label so rows from different runs are
    distinguishable. The Today page omits it, leaving its single-run labels unchanged.

    When `prior_recommendation` is given and differs from the current call (the Today
    page passes it), a "was <PRIOR>" suffix flags the change so a recurring reviewer
    spots it without opening History.
    """
    badge = _REC_BADGES.get(analysis.recommendation or "", "⚪")
    recommendation = (analysis.recommendation or "n/a").upper()
    confidence = analysis.confidence or 0.0
    dq_notes = analysis.data_quality_notes or []
    dq_suffix = " ⚠️" if dq_notes else ""
    changed_suffix = (
        f" · was {prior_recommendation.upper()}"
        if prior_recommendation is not None and prior_recommendation != analysis.recommendation
        else ""
    )

    auto_expand = analysis.recommendation != "hold" and confidence > _AUTO_EXPAND_CONFIDENCE

    label = (
        f"{badge} **{analysis.ticker}** — {recommendation} "
        f"(confidence: {confidence:.0%}){dq_suffix}{changed_suffix}"
    )
    if prompt_version is not None:
        created = analysis.created_at.strftime("%Y-%m-%d") if analysis.created_at else "—"
        label += f" · {created} · [{prompt_version}]"
    with st.expander(label, expanded=auto_expand):
        if analysis.thesis:
            st.markdown(analysis.thesis)

        if dq_notes:
            st.warning("⚠️ Data quality notes:\n" + "\n".join(f"- {n}" for n in dq_notes))

        col_lynch, col_buffett = st.columns(2)
        with col_lynch:
            st.markdown("**Lynch signals**")
            for signal in analysis.lynch_signals or []:
                st.markdown(f"- {signal}")
        with col_buffett:
            st.markdown("**Buffett signals**")
            for signal in analysis.buffett_signals or []:
                st.markdown(f"- {signal}")

        st.markdown("**Key risks**")
        for risk in analysis.key_risks or []:
            st.markdown(f"- {risk}")

        with st.expander("🔍 Reasoning trace"):
            render_reasoning_trace(analysis.run_id, analysis.ticker)


def render_reasoning_trace(run_id: str, ticker: str) -> None:
    """Render the full reasoning trace for a ticker, step by step in log order.

    Every tool_call and llm_call event is shown sequentially: tool calls include
    their args and output payload (large outputs are sidecar-truncated by the logger,
    so we link to the sidecar instead of dumping it); LLM turns show model + token
    cost. This mirrors the actual sequence the agent followed.
    """
    events = read_reasoning_trace(run_id, ticker)
    if not events:
        st.caption("No reasoning trace found for this ticker.")
        return
    for step, event in enumerate(events, start=1):
        if event.get("event") == "tool_call":
            _render_tool_step(step, event)
        elif event.get("event") == "llm_call":
            _render_llm_step(step, event)
        st.divider()


def _render_tool_step(step: int, event: dict[str, object]) -> None:
    cached = " · cached" if event.get("cached") else ""
    status = event.get("status")
    status_suffix = "" if status in (None, "ok") else f" · {status}"
    st.markdown(
        f"**{step}. 🔧 `{event.get('tool')}`** — {event.get('latency_ms')}ms{cached}{status_suffix}"
    )
    tool_input = event.get("input")
    if tool_input:
        st.caption("args")
        st.json(tool_input, expanded=False)
    st.caption("output")
    _render_tool_output(event.get("output"))
    error_msg = event.get("error_msg")
    if error_msg:
        st.error(str(error_msg))


def _render_llm_step(step: int, event: dict[str, object]) -> None:
    cost = _as_float(event.get("cost_usd"))
    st.markdown(
        f"**{step}. 💬 LLM turn** (`{event.get('model')}`) — "
        f"{event.get('input_tokens')} in / {event.get('output_tokens')} out — ${cost:.4f}"
    )


def _render_tool_output(output: object) -> None:
    """Render a tool's output payload, gracefully handling sidecar-truncated outputs."""
    if output is None:
        st.caption("—")
        return
    text = output if isinstance(output, str) else json.dumps(output)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        st.code(text)
        return
    if isinstance(parsed, dict) and parsed.get("truncated"):
        path = parsed.get("path")
        sha = str(parsed.get("sha256") or "")
        st.caption(f"output truncated — full payload at `{path}` (sha256 {sha[:12]}…)")
        return
    st.json(parsed, expanded=False)
