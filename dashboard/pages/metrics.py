"""Metrics page — cost, token, cache, and recommendation-distribution charts.

Read-only: renders cost/token/cache-hit-rate/recommendation charts from `runs`,
`tool_calls`, and `analyses` so cost trends and Week 3 caching ROI are visible at a
glance. This is the proof-of-concept for the PRD's $20/month budget goal. Never
triggers an analysis. `set_page_config` lives in `app.py`, not here.
"""

import sys
from pathlib import Path

# Make `dashboard.*` / `storage.*` importable when Streamlit runs this file directly
# (the bootstrap is a no-op once the repo root is already on sys.path, e.g. in tests).
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.data import (  # noqa: E402
    MONTHLY_WARNING_THRESHOLD_USD,
    cache_hit_rate,
    get_recent_runs_with_tokens,
    monthly_cost,
    recommendation_distribution,
)
from storage.engine import get_session  # noqa: E402

# Tech Spec success criterion: stay under $20/mo. Bars/banner flag runs and months
# approaching that ceiling.
_COST_ALERT_THRESHOLD_USD = 1.25

st.title("Warren · Run Metrics")

with get_session() as session:
    runs = get_recent_runs_with_tokens(session, limit=30)
    if not runs:
        st.info("No runs yet. Run `python -m agent.run` to generate your first analysis.")
        st.stop()

    runs_df = pd.DataFrame(
        {
            "date": [run.started_at for run in runs],
            "cost_usd": [run.total_cost_usd or 0.0 for run in runs],
            "input_tokens": [run.total_input_tokens or 0 for run in runs],
            "output_tokens": [run.total_output_tokens or 0 for run in runs],
            "cache_read_tokens": [run.cache_read_tokens for run in runs],
            "status": [run.status or "unknown" for run in runs],
        }
    )

    st.subheader("Cost per run (USD)")
    cost_chart = (
        alt.Chart(runs_df)
        .mark_bar()
        .encode(
            x=alt.X("date:T", title="Run date"),
            y=alt.Y("cost_usd:Q", title="Cost (USD)"),
            color=alt.condition(
                alt.datum.cost_usd > _COST_ALERT_THRESHOLD_USD,
                alt.value("red"),
                alt.value("steelblue"),
            ),
            tooltip=["date:T", "cost_usd:Q", "status:N"],
        )
    )
    st.altair_chart(cost_chart, use_container_width=True)

    st.subheader("Token consumption per run")
    hit_rate = cache_hit_rate(session)
    st.metric(
        "Cache hit rate",
        f"{hit_rate:.0%}" if hit_rate is not None else "—",
        help="Fraction of tool calls served from cache. Target: >60% after Week 3.",
    )
    token_df = runs_df.melt(
        id_vars="date",
        value_vars=["input_tokens", "output_tokens", "cache_read_tokens"],
        var_name="token_type",
        value_name="tokens",
    )
    token_chart = (
        alt.Chart(token_df)
        .mark_bar()
        .encode(
            x=alt.X("date:T", title="Run date"),
            y=alt.Y("tokens:Q", title="Tokens", stack="zero"),
            color=alt.Color("token_type:N", title="Token type"),
            tooltip=["date:T", "token_type:N", "tokens:Q"],
        )
    )
    st.altair_chart(token_chart, use_container_width=True)

    st.subheader("Recommendation distribution (all time)")
    rec_counts = recommendation_distribution(session)
    rec_df = pd.DataFrame(
        {
            "recommendation": [rec.recommendation for rec in rec_counts],
            "count": [rec.count for rec in rec_counts],
        }
    ).set_index("recommendation")
    st.bar_chart(rec_df)

    st.subheader("Monthly cost (USD)")
    monthly = monthly_cost(session)
    monthly_df = pd.DataFrame(
        {
            "month": [row.month for row in monthly],
            "monthly_cost": [row.total_cost_usd for row in monthly],
        }
    )
    st.dataframe(monthly_df, use_container_width=True)
    if not monthly_df.empty and monthly_df.iloc[0]["monthly_cost"] > MONTHLY_WARNING_THRESHOLD_USD:
        st.warning(
            f"⚠️ This month's cost is ${monthly_df.iloc[0]['monthly_cost']:.2f} "
            f"— approaching the $20 ceiling."
        )
