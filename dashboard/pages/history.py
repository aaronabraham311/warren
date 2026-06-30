"""History page — a searchable, filterable archive of every past recommendation.

Read-only: sidebar filters (ticker text search, recommendation multi-select, date
range, confidence range) drive a single `search_analyses` query, and each result is an
expandable card reusing the Today page's `render_analysis_card`. The query is backed by
`idx_analyses_ticker_created` and capped at 500 rows, so it stays fast as the DB grows.
`set_page_config` lives in `app.py`, not here.
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
from dashboard.data import search_analyses  # noqa: E402
from storage.engine import get_session  # noqa: E402

st.title("Warren · History")

with st.sidebar:
    st.header("Filters")
    ticker_search = st.text_input("Ticker", placeholder="AAPL")
    rec_filter = st.multiselect(
        "Recommendation", ["buy", "sell", "hold"], default=["buy", "sell", "hold"]
    )
    date_from = st.date_input("From", value=None)
    date_to = st.date_input("To", value=None)
    conf_min, conf_max = st.slider("Confidence range", 0.0, 1.0, (0.0, 1.0), step=0.05)

with get_session() as session:
    results = search_analyses(
        session,
        ticker=ticker_search,
        recommendations=rec_filter,
        date_from=date_from,
        date_to=date_to,
        conf_min=conf_min,
        conf_max=conf_max,
    )

    if not results:
        st.info("No results match the current filters.")
        st.stop()

    st.caption(f"{len(results)} results")
    for result in results:
        render_analysis_card(
            result.analysis, prompt_version=result.prompt_version or "unknown version"
        )
