"""Today page — the most recent run's analysis as cards with reasoning traces.

Queries the latest run from `warren.db` and the JSONL logs, renders a run-metadata
header, then holdings and discovery cards in Tech Spec §9.Q3 order. `set_page_config`
lives in `app.py`, not here.

This page is read-only with one deliberate exception: the sidebar "Run now" button,
a human-clicked dev convenience that shells out to `python -m agent.run` for testing
prompt changes without waiting for the 2 AM schedule. No other code path here writes
to the database or triggers an analysis.
"""

import subprocess
import sys
from pathlib import Path

# Make `dashboard.*` / `storage.*` importable when Streamlit runs this file directly
# (the bootstrap is a no-op once the repo root is already on sys.path, e.g. in tests).
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

from dashboard.components.analysis_card import render_analysis_card  # noqa: E402
from dashboard.data import (  # noqa: E402
    MONTHLY_WARNING_THRESHOLD_USD,
    cooldown_suppressed_count,
    get_analyses_for_run,
    get_latest_run,
    monthly_cost,
    previous_recommendation,
    run_duration_seconds,
)
from storage.engine import get_session  # noqa: E402
from storage.models import Analysis  # noqa: E402

_RUN_NOW_TIMEOUT_SECONDS = 1800  # 30 min max

st.title("Warren · Today's Analysis")

with st.sidebar:
    st.divider()
    if st.button("▶ Run now", type="primary", help="Trigger a full analysis run immediately"):
        with st.spinner("Running analysis..."):
            result = subprocess.run(
                ["python", "-m", "agent.run"],
                capture_output=True,
                text=True,
                timeout=_RUN_NOW_TIMEOUT_SECONDS,
            )
        if result.returncode == 0:
            st.success("Run completed! Refresh the page to see results.")
        else:
            st.error(f"Run failed:\n```\n{result.stderr[-2000:]}\n```")

with get_session() as session:
    run = get_latest_run(session)
    if run is None:
        st.info("No runs yet. Run `python -m agent.run` to generate your first analysis.")
        st.stop()

    if run.status not in (None, "success"):
        st.error(f"⚠️ Last run finished with status `{run.status}`: {run.error_msg or 'no details'}")

    monthly = monthly_cost(session, months=1)
    if monthly and monthly[0].total_cost_usd > MONTHLY_WARNING_THRESHOLD_USD:
        st.warning(
            f"⚠️ This month's cost is ${monthly[0].total_cost_usd:.2f} "
            f"— approaching the $20 ceiling. See the Metrics page for the trend."
        )

    analyses = get_analyses_for_run(session, run.id)

    duration = run_duration_seconds(run)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Run date", run.started_at.strftime("%Y-%m-%d") if run.started_at else "—")
    col2.metric(
        "Total cost", f"${run.total_cost_usd:.3f}" if run.total_cost_usd is not None else "—"
    )
    col3.metric("Tickers analysed", str(len(analyses)))
    col4.metric("Duration", f"{int(duration // 60)}m {int(duration % 60)}s" if duration else "—")
    col5.metric("Suppressed (cooldown)", cooldown_suppressed_count(run.id))

    holdings = [a for a in analyses if a.analysis_type == "holding"]
    discoveries = [a for a in analyses if a.analysis_type == "discovery"]

    def _render(analysis: Analysis) -> None:
        prior = (
            previous_recommendation(session, analysis.ticker, analysis.created_at)
            if analysis.created_at is not None
            else None
        )
        render_analysis_card(analysis, prior_recommendation=prior)

    st.header(f"Portfolio Holdings ({len(holdings)})")
    for analysis in holdings:
        _render(analysis)

    st.header(f"Discovery Candidates ({len(discoveries)})")
    for analysis in discoveries:
        _render(analysis)
