"""Reusable Streamlit render functions for an analysis and its reasoning trace.

Kept separate from the page wiring so the History page (a later ticket) can reuse
`render_analysis_card` verbatim. These functions take ORM rows + read the JSONL
trace; they never query the DB for analyses themselves.
"""

import json

import streamlit as st
from pydantic import ValidationError

from agent.models import DirtDecisionContract
from dashboard.data import read_reasoning_trace
from storage.models import Analysis

# Recommendation → coloured badge. Unknown/None falls back to a neutral marker.
_REC_BADGES = {"buy": "🟢", "sell": "🔴", "hold": "🟡"}
_DECISION_BADGES = {"buy": "🟢 BUY", "watchlist": "🟡 WATCHLIST", "pass": "⚪ PASS"}

# Strong action calls auto-expand so nothing important is buried on page load.
_AUTO_EXPAND_CONFIDENCE = 0.6


def _as_float(value: object) -> float:
    """Coerce a JSON-decoded numeric value to float, defaulting to 0.0."""
    return float(value) if isinstance(value, (int, float)) else 0.0


def _parse_dirt_decision(value: object) -> DirtDecisionContract | None:
    if not isinstance(value, dict):
        return None
    try:
        return DirtDecisionContract.model_validate(value)
    except ValidationError:
        return None


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
    decision = _parse_dirt_decision(analysis.dirt_decision)
    decision_outcome = None if decision is None else decision.outcome
    integrity_warning: str | None = None
    if analysis.dirt_decision is not None and decision is None:
        integrity_warning = "Stored DIRT decision is invalid and was not rendered."
    elif decision is not None and analysis.decision_outcome != decision.outcome:
        integrity_warning = (
            "Stored decision projection disagrees with the contract; the contract is authoritative."
        )
    if decision_outcome in _DECISION_BADGES:
        badge_and_call = _DECISION_BADGES[decision_outcome]
    else:
        badge = _REC_BADGES.get(analysis.recommendation or "", "⚪")
        badge_and_call = f"{badge} {(analysis.recommendation or 'n/a').upper()}"
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
        f"{badge_and_call} · **{analysis.ticker}** "
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
        if integrity_warning is not None:
            st.warning(integrity_warning)

        if decision is not None:
            render_dirt_decision(decision)

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


def render_dirt_decision(decision: DirtDecisionContract) -> None:
    """Render a computed DIRT decision contract without reinterpreting its math."""
    st.markdown("**DIRT decision contract**")
    weighted = decision.probability_weighted_irr
    hurdle = decision.hurdle_irr
    required_price = decision.required_entry_price
    currency = decision.currency
    metric_irr, metric_hurdle, metric_entry = st.columns(3)
    metric_irr.metric("Weighted IRR", f"{weighted:.1%}")
    metric_hurdle.metric("Hurdle", f"{hurdle:.1%}")
    metric_entry.metric(
        "Required entry",
        f"{required_price:,.2f} {currency}",
    )

    payload = decision.model_dump(mode="json")
    scenarios = payload.get("scenarios")
    if isinstance(scenarios, list) and scenarios:
        rows = []
        for raw in scenarios:
            if not isinstance(raw, dict):
                continue
            rows.append(
                {
                    "Case": str(raw.get("case", "—")).upper(),
                    "Probability": raw.get("probability"),
                    "Terminal price": raw.get("terminal_price"),
                    "Dividends": raw.get("total_dividends"),
                    "Total value": raw.get("total_value"),
                    "Total return": raw.get("total_return"),
                    "IRR": raw.get("irr"),
                    "Terminal date": raw.get("terminal_date"),
                    "Assumption": raw.get("assumption"),
                    "Rationale": raw.get("rationale"),
                }
            )
        if rows:
            st.markdown("**Scenarios**")
            st.table(rows)

    floor = payload.get("downside_floor")
    st.markdown("**Downside floor**")
    if isinstance(floor, dict):
        st.markdown(
            f"{str(floor.get('basis', 'none')).replace('_', ' ').title()}: "
            f"gross `{floor.get('gross', '—')} {currency}`, haircut "
            f"`{floor.get('haircut', '—')}`, adjusted `{floor.get('adjusted', '—')} {currency}`, "
            f"coverage `{floor.get('coverage', '—')}` · confidence "
            f"`{floor.get('confidence', 'unavailable')}`"
        )
        st.caption(f"As of {floor.get('as_of', '—')} · source {floor.get('source_ref', '—')}")
        adjustments = floor.get("adjustments")
        if isinstance(adjustments, list) and adjustments:
            st.caption("Adjustments: " + "; ".join(str(item) for item in adjustments))
    else:
        st.caption("Unavailable")

    _render_decision_list("Catalysts", payload.get("catalysts"))
    st.markdown("**Failure thesis**")
    st.markdown(decision.failure_thesis)
    _render_decision_list("Entry conditions", payload.get("entry_conditions"))
    _render_decision_list("Blocking unknowns", payload.get("blocking_unknowns"))
    _render_decision_list("Monitoring", payload.get("monitoring_metrics"))


def _render_decision_list(label: str, value: object) -> None:
    st.markdown(f"**{label}**")
    if not isinstance(value, list) or not value:
        st.caption("None")
        return
    for item in value:
        if isinstance(item, dict):
            primary = item.get("description") or item.get("metric") or item.get("category")
            details = ", ".join(
                f"{key.replace('_', ' ')}: {field_value}"
                for key, field_value in item.items()
                if key not in {"description", "metric"} and field_value is not None
            )
            st.markdown(f"- {primary or 'Item'}" + (f" — {details}" if details else ""))
        else:
            st.markdown(f"- {item}")


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
