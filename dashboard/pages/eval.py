"""Eval page — pass-rate trend + side-by-side diff of any two eval runs.

Read-only prompt-iteration tool over `eval_runs`. Two sections separated by a divider:

1. **Pass rate over time** — one point per eval run, labelled by prompt `version_tag`,
   so a regression in a new persona prompt is visible at a glance. A dataframe below
   carries the same numbers for inspection.
2. **Diff two runs** — pick a baseline and a current run; a net-change banner summarises
   fixes vs regressions, then per-ticker expanders show only the checks that flipped
   (green = fixed, red = regression), with the expected/actual envelope for each.

All SQL + the diff computation live in `dashboard.data` (pure, ORM-backed); this page is
display-only. `set_page_config` lives in `app.py`, not here.
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
    EvalRunSummary,
    diff_eval_runs,
    eval_run_summaries,
    load_eval_grades,
)
from storage.engine import get_session  # noqa: E402

st.title("Warren · Eval")

with get_session() as session:
    summaries = eval_run_summaries(session)
    if not summaries:
        st.info("No eval runs yet. Run `python -m agent.eval --golden-set` to generate results.")
        st.stop()

    # ------------------------------------------------------------------ #
    # Pass rate by prompt version
    # ------------------------------------------------------------------ #
    st.header("Pass rate by prompt version")

    summary_df = pd.DataFrame(
        {
            "date": [s.started_at for s in summaries],
            "version": [s.version_tag or "unknown" for s in summaries],
            "passed": [s.passed for s in summaries],
            "total": [s.total for s in summaries],
            "pass_rate": [s.pass_rate for s in summaries],
        }
    )

    pass_rate_chart = (
        alt.Chart(summary_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Run date"),
            y=alt.Y("pass_rate:Q", title="Pass rate", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("date:T", title="Run date"),
                alt.Tooltip("version:N", title="Prompt version"),
                alt.Tooltip("passed:Q", title="Passed"),
                alt.Tooltip("total:Q", title="Total"),
                alt.Tooltip("pass_rate:Q", title="Pass rate", format=".0%"),
            ],
        )
    )
    st.altair_chart(pass_rate_chart, use_container_width=True)
    st.dataframe(
        summary_df[["date", "version", "passed", "total", "pass_rate"]],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ------------------------------------------------------------------ #
    # Diff two eval runs
    # ------------------------------------------------------------------ #
    st.header("Diff two eval runs")

    if len(summaries) < 2:
        st.info("Need at least 2 eval runs to diff. Run evals again after changing the prompt.")
        st.stop()

    # Key options by run_id (unique) so same-day / same-version runs never collide.
    def _label(s: EvalRunSummary) -> str:
        day = s.started_at.date().isoformat() if s.started_at else "unknown date"
        return f"{day} · [{s.version_tag or 'unknown'}] · {s.run_id}"

    options = {_label(s): s.run_id for s in summaries}
    labels = list(options)

    col_a, col_b = st.columns(2)
    with col_a:
        baseline_label = st.selectbox("Baseline run", labels, index=1)
    with col_b:
        current_label = st.selectbox("Current run", labels, index=0)

    baseline_id = options[baseline_label]
    current_id = options[current_label]

    diff = diff_eval_runs(
        baseline_id,
        current_id,
        load_eval_grades(session, baseline_id),
        load_eval_grades(session, current_id),
    )

    # Net-change banner.
    if diff.fixes == 0 and diff.regressions == 0:
        st.info("No differences between these two runs.")
    elif diff.regressions == 0:
        st.success(f"🎉 Net improvement: +{diff.fixes} checks fixed, 0 regressions")
    else:
        st.error(f"⚠️ Net change: +{diff.fixes} fixed, -{diff.regressions} regressions")

    st.subheader("Check-level diff")
    if not diff.ticker_diffs:
        st.caption("Every ticker's checks are identical across the two runs.")
    for ticker_diff in diff.ticker_diffs:
        n = len(ticker_diff.changes)
        with st.expander(f"**{ticker_diff.ticker}** — {n} check(s) changed"):
            for change in ticker_diff.changes:
                detail = f"expected {change.expected!r}, actual {change.actual!r}"
                if change.kind == "fix":
                    st.success(
                        f"✅ `{change.check_name}` — FIXED (was failing, now passing) · {detail}"
                    )
                elif change.kind == "regression":
                    st.error(
                        f"❌ `{change.check_name}` — REGRESSION "
                        f"(was passing, now failing) · {detail}"
                    )
                else:
                    st.warning(f"⚠️ `{change.check_name}` — {change.old} → {change.new} · {detail}")
