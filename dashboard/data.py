"""Read-only data access for the Streamlit dashboard.

Pure functions over the existing ORM (`storage.models`) and the JSONL run logs.
No Streamlit imports here so this layer stays trivially unit-testable and reusable
across pages (Today, and the History/Eval pages that land later). The dashboard is
read-only — it never writes to `warren.db` — with one deliberate exception: the
Today page's "Run now" button, a human-clicked dev convenience that shells out to
`python -m agent.run` (see `dashboard/pages/today.py`). No other code path here
triggers an analysis or writes to the database.
"""

import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from storage.models import Analysis, PromptVersion, Run, ToolCall

# Event kinds from RunLogger that make up a ticker's reasoning trace.
_TRACE_EVENTS = {"tool_call", "llm_call"}

# Tech Spec success criterion: stay under $20/mo. Shared by the Metrics page's
# monthly-cost banner and the Today page's budget guardrail banner.
MONTHLY_WARNING_THRESHOLD_USD = 18.0


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


def previous_recommendation(session: Session, ticker: str, before: datetime) -> str | None:
    """The most recent recommendation for `ticker` from any analysis strictly before `before`.

    Used by the Today page to show a "prior call → today's call" delta so a recurring
    reviewer can spot changes at a glance without opening History. Returns None when the
    ticker has no earlier analysis (e.g. a brand-new discovery candidate).
    """
    stmt = (
        select(Analysis.recommendation)
        .where(Analysis.ticker == ticker, Analysis.created_at < before)
        .order_by(Analysis.created_at.desc())
        .limit(1)
    )
    return session.scalar(stmt)


@dataclass
class AnalysisSearchResult:
    """One History-page row: an analysis plus the prompt version that produced it.

    `prompt_version` is the `version_tag` joined through `runs → prompt_versions`, or
    `None` when the run has no linked prompt version (older runs, or seeds without one).
    """

    analysis: Analysis
    prompt_version: str | None


def search_analyses(
    session: Session,
    *,
    ticker: str | None = None,
    recommendations: list[str] | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    conf_min: float = 0.0,
    conf_max: float = 1.0,
    limit: int = 500,
) -> list[AnalysisSearchResult]:
    """Filtered, newest-first search across every analysis for the History page.

    Each active filter narrows the query: `ticker` is a case-insensitive `LIKE` substring
    match, `recommendations` an `IN` set, `date_from`/`date_to` bound the calendar day of
    `created_at`, and `conf_min`/`conf_max` bound confidence. Results are ordered newest
    first and capped at `limit` (the `idx_analyses_ticker_created` index keeps this fast).
    The prompt version tag is joined in via `runs → prompt_versions`.
    """
    stmt = (
        select(Analysis, PromptVersion.version_tag)
        .outerjoin(Run, Analysis.run_id == Run.id)
        .outerjoin(PromptVersion, Run.prompt_version_id == PromptVersion.id)
    )
    if ticker:
        stmt = stmt.where(Analysis.ticker.like(f"%{ticker.upper()}%"))
    if recommendations:
        stmt = stmt.where(Analysis.recommendation.in_(recommendations))
    if date_from is not None:
        stmt = stmt.where(func.date(Analysis.created_at) >= str(date_from))
    if date_to is not None:
        stmt = stmt.where(func.date(Analysis.created_at) <= str(date_to))
    stmt = (
        stmt.where(Analysis.confidence.between(conf_min, conf_max))
        .order_by(Analysis.created_at.desc())
        .limit(limit)
    )
    return [
        AnalysisSearchResult(analysis=analysis, prompt_version=version_tag)
        for analysis, version_tag in session.execute(stmt)
    ]


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


def cooldown_suppressed_count(run_id: str, base_dir: Path | None = None) -> int:
    """`suppressed_count` from this run's `discovery_cooldown_applied` event, or 0.

    Single-ticker runs and runs recorded before this event existed have no such event,
    in which case 0 (nothing suppressed) is the correct display value, not an error.
    """
    base = base_dir if base_dir is not None else logs_dir()
    log_path = base / f"{run_id}.jsonl"
    if not log_path.exists():
        return 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event: dict[str, object] = json.loads(line)
        if event.get("event") == "discovery_cooldown_applied":
            count = event.get("suppressed_count")
            return count if isinstance(count, int) else 0
    return 0


def cache_read_tokens_for_run(run_id: str, base_dir: Path | None = None) -> int:
    """Sum of `cache_read_tokens` across a run's `llm_call` events.

    The `runs` table has no cache-read column — that figure only exists per `llm_call`
    event in the JSONL trace — so the Metrics page's token chart reads it from here.
    Returns 0 when the log file is absent.
    """
    base = base_dir if base_dir is not None else logs_dir()
    log_path = base / f"{run_id}.jsonl"
    if not log_path.exists():
        return 0
    total = 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event: dict[str, object] = json.loads(line)
        if event.get("event") == "llm_call":
            tokens = event.get("cache_read_tokens")
            if isinstance(tokens, int):
                total += tokens
    return total


@dataclass
class RunCostRow:
    """One row of the Metrics page's cost/token charts."""

    run_id: str
    started_at: datetime | None
    total_cost_usd: float | None
    total_input_tokens: int | None
    total_output_tokens: int | None
    cache_read_tokens: int
    status: str | None


def get_recent_runs_with_tokens(
    session: Session, *, limit: int = 30, base_dir: Path | None = None
) -> list[RunCostRow]:
    """The most recent `limit` runs, newest first, with cache_read_tokens joined from the trace."""
    runs = session.scalars(select(Run).order_by(Run.started_at.desc()).limit(limit)).all()
    return [
        RunCostRow(
            run_id=run.id,
            started_at=run.started_at,
            total_cost_usd=run.total_cost_usd,
            total_input_tokens=run.total_input_tokens,
            total_output_tokens=run.total_output_tokens,
            cache_read_tokens=cache_read_tokens_for_run(run.id, base_dir),
            status=run.status,
        )
        for run in runs
    ]


def cache_hit_rate(session: Session) -> float | None:
    """Fraction of all `tool_calls` rows served from cache, or None when there are none yet."""
    total = session.scalar(select(func.count()).select_from(ToolCall))
    if not total:
        return None
    cached = session.scalar(
        select(func.count()).select_from(ToolCall).where(ToolCall.cached.is_(True))
    )
    return (cached or 0) / total


@dataclass
class RecommendationCount:
    """All-time count of one recommendation bucket, for the bias-check chart."""

    recommendation: str
    count: int


def recommendation_distribution(session: Session) -> list[RecommendationCount]:
    """All-time buy/sell/hold counts across every analysis."""
    stmt = select(Analysis.recommendation, func.count()).group_by(Analysis.recommendation)
    return [
        RecommendationCount(recommendation=recommendation or "unknown", count=count)
        for recommendation, count in session.execute(stmt)
    ]


@dataclass
class MonthlyCost:
    """One calendar month's total run cost, for the budget-ceiling table."""

    month: str
    total_cost_usd: float


def monthly_cost(session: Session, *, months: int = 6) -> list[MonthlyCost]:
    """Total run cost per calendar month, newest first, capped at `months`."""
    month_expr = func.strftime("%Y-%m", Run.started_at)
    stmt = (
        select(month_expr, func.sum(Run.total_cost_usd))
        .group_by(month_expr)
        .order_by(month_expr.desc())
        .limit(months)
    )
    return [
        MonthlyCost(month=month, total_cost_usd=total_cost or 0.0)
        for month, total_cost in session.execute(stmt)
        if month is not None
    ]
