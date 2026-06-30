"""Today page — the most recent run's analysis as cards with reasoning traces.

Read-only: queries the latest run from `warren.db` and the JSONL logs, renders a
run-metadata header, then holdings and discovery cards in Tech Spec §9.Q3 order.
Never triggers an analysis. `set_page_config` lives in `app.py`, not here.
"""

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
    get_analyses_for_run,
    get_latest_run,
    run_duration_seconds,
)
from storage.engine import get_session  # noqa: E402

st.title("Warren · Today's Analysis")

with get_session() as session:
    run = get_latest_run(session)
    if run is None:
        st.info("No runs yet. Run `python -m agent.run` to generate your first analysis.")
        st.stop()

    analyses = get_analyses_for_run(session, run.id)

    duration = run_duration_seconds(run)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Run date", run.started_at.strftime("%Y-%m-%d") if run.started_at else "—")
    col2.metric(
        "Total cost", f"${run.total_cost_usd:.3f}" if run.total_cost_usd is not None else "—"
    )
    col3.metric("Tickers analysed", str(len(analyses)))
    col4.metric("Duration", f"{int(duration // 60)}m {int(duration % 60)}s" if duration else "—")

    holdings = [a for a in analyses if a.analysis_type == "holding"]
    discoveries = [a for a in analyses if a.analysis_type == "discovery"]

    st.header(f"Portfolio Holdings ({len(holdings)})")
    for analysis in holdings:
        render_analysis_card(analysis)

    st.header(f"Discovery Candidates ({len(discoveries)})")
    for analysis in discoveries:
        render_analysis_card(analysis)
