"""Read-only data access for the Streamlit dashboard.

Pure functions over the existing ORM (`storage.models`) and the JSONL run logs.
No Streamlit imports here so this layer stays trivially unit-testable and reusable
across pages (Today, and the History/Eval pages that land later). The dashboard is
strictly read-only — it never writes to `warren.db` or triggers an analysis.
"""

import json
import os
from pathlib import Path

from sqlalchemy import case, select
from sqlalchemy.orm import Session

from storage.models import Analysis, Run

# Event kinds from RunLogger that make up a ticker's reasoning trace.
_TRACE_EVENTS = {"tool_call", "llm_call"}


def get_latest_run(session: Session) -> Run | None:
    """The most recent run by start time, or None when no runs exist yet."""
    return session.scalars(select(Run).order_by(Run.started_at.desc()).limit(1)).first()


def get_analyses_for_run(session: Session, run_id: str) -> list[Analysis]:
    """Analyses for a run in Tech Spec §9.Q3 order.

    Non-hold recommendations (buy/sell) sort before holds; within each group,
    higher confidence appears first — so high-confidence action calls surface to
    the top of the page.
    """
    hold_last = case((Analysis.recommendation == "hold", 1), else_=0)
    return list(
        session.scalars(
            select(Analysis)
            .where(Analysis.run_id == run_id)
            .order_by(hold_last.asc(), Analysis.confidence.desc())
        )
    )


def run_duration_seconds(run: Run) -> float | None:
    """Wall-clock duration of a run, or None if it never completed."""
    if run.started_at is None or run.completed_at is None:
        return None
    return (run.completed_at - run.started_at).total_seconds()


def logs_dir() -> Path:
    """Base directory for JSONL run logs (overridable via WARREN_LOGS_DIR for tests)."""
    return Path(os.environ.get("WARREN_LOGS_DIR", "logs/runs"))


def read_reasoning_trace(
    run_id: str, ticker: str, base_dir: Path | None = None
) -> list[dict[str, object]]:
    """Tool-call and LLM-call events for one ticker, in log order.

    Reads `{base_dir}/{run_id}.jsonl` and keeps the `tool_call` / `llm_call` events
    tagged with this ticker. Returns an empty list when the log file is absent.
    """
    base = base_dir if base_dir is not None else logs_dir()
    log_path = base / f"{run_id}.jsonl"
    if not log_path.exists():
        return []
    events: list[dict[str, object]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event: dict[str, object] = json.loads(line)
        if event.get("ticker") == ticker and event.get("event") in _TRACE_EVENTS:
            events.append(event)
    return events
