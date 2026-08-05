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
from typing import Literal

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from storage.models import Analysis, EvalRun, PromptVersion, Run, ToolCall

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
    decision_order = case(
        (Analysis.decision_outcome == "buy", 0),
        (Analysis.decision_outcome == "watchlist", 1),
        (Analysis.decision_outcome == "pass", 2),
        else_=3,
    )
    return list(
        session.scalars(
            select(Analysis)
            .where(Analysis.run_id == run_id)
            .order_by(
                decision_order.asc(),
                Analysis.probability_weighted_irr.desc().nullslast(),
                hold_last.asc(),
                Analysis.confidence.desc(),
            )
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
    decision_outcomes: list[str] | None = None,
    weighted_irr_min: float | None = None,
    weighted_irr_max: float | None = None,
    limit: int = 500,
) -> list[AnalysisSearchResult]:
    """Filter analyses and rank decisions buy/watchlist/pass, then weighted IRR.

    Each active filter narrows the query: `ticker` is a case-insensitive `LIKE` substring
    match, `recommendations` an `IN` set, `date_from`/`date_to` bound the calendar day of
    `created_at`, and confidence/outcome/IRR bounds narrow results. Decision outcomes sort
    buy, watchlist, pass, then legacy rows; weighted IRR breaks ties before recency. Results
    are capped at `limit` (the ticker/date index keeps broad history queries bounded).
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
    if decision_outcomes:
        stmt = stmt.where(Analysis.decision_outcome.in_(decision_outcomes))
    if weighted_irr_min is not None:
        stmt = stmt.where(Analysis.probability_weighted_irr >= weighted_irr_min)
    if weighted_irr_max is not None:
        stmt = stmt.where(Analysis.probability_weighted_irr <= weighted_irr_max)
    outcome_order = case(
        (Analysis.decision_outcome == "buy", 0),
        (Analysis.decision_outcome == "watchlist", 1),
        (Analysis.decision_outcome == "pass", 2),
        else_=3,
    )
    stmt = (
        stmt.where(Analysis.confidence.between(conf_min, conf_max))
        .order_by(
            outcome_order.asc(),
            Analysis.probability_weighted_irr.desc().nullslast(),
            Analysis.created_at.desc(),
        )
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


# --------------------------------------------------------------------------- #
# Eval page — pass-rate summaries + run-to-run diff
# --------------------------------------------------------------------------- #


@dataclass
class EvalRunSummary:
    """One eval run's aggregate pass rate — a point on the trend chart / a table row.

    `passed`/`total` count `eval_runs` rows for the run (one row per golden-set example);
    `passed` is the number whose `must` checks all held (`EvalRun.passed`). `version_tag`
    is joined through `runs → prompt_versions`, or None when the run has no linked version.
    """

    run_id: str
    started_at: datetime | None
    version_tag: str | None
    total: int
    passed: int

    @property
    def pass_rate(self) -> float:
        """Fraction of examples that passed, or 0.0 for an empty run (no division by zero)."""
        return self.passed / self.total if self.total else 0.0


def eval_run_summaries(session: Session, *, limit: int = 20) -> list[EvalRunSummary]:
    """The most recent eval runs, newest first, with per-run pass counts and version tag.

    Groups `eval_runs` by `run_id` (each row is one graded example), joins `runs` for the
    start time and `prompt_versions` for the tag. `passed` sums the boolean `passed` column
    via a CASE so a NULL grade counts as not-passed rather than poisoning the sum.
    """
    passed_sum = func.sum(case((EvalRun.passed.is_(True), 1), else_=0))
    stmt = (
        select(
            EvalRun.run_id,
            Run.started_at,
            PromptVersion.version_tag,
            func.count().label("total"),
            passed_sum.label("passed"),
        )
        .join(Run, EvalRun.run_id == Run.id)
        .outerjoin(PromptVersion, Run.prompt_version_id == PromptVersion.id)
        .group_by(EvalRun.run_id)
        .order_by(Run.started_at.desc())
        .limit(limit)
    )
    return [
        EvalRunSummary(
            run_id=run_id,
            started_at=started_at,
            version_tag=version_tag,
            total=total or 0,
            passed=passed or 0,
        )
        for run_id, started_at, version_tag, total, passed in session.execute(stmt)
    ]


@dataclass
class EvalCheckResult:
    """One parsed `check_results` entry — mirrors `eval.grader.CheckResult` on disk.

    Stored as JSON in `EvalRun.check_results`; carries the expected/actual envelope so the
    diff view can show *why* a check failed, not just that it flipped.
    """

    check_name: str
    passed: bool
    expected: str
    actual: str
    severity: str


def load_eval_grades(session: Session, run_id: str) -> dict[str, dict[str, EvalCheckResult]]:
    """`{ticker: {check_name: EvalCheckResult}}` for one eval run.

    `EvalRun.check_results` is a JSON *string* (Text column), so it is parsed here. A row with
    a missing/empty payload contributes an empty check map for its ticker rather than raising.
    """
    rows = session.execute(
        select(EvalRun.example_ticker, EvalRun.check_results).where(EvalRun.run_id == run_id)
    ).all()
    grades: dict[str, dict[str, EvalCheckResult]] = {}
    for ticker, check_results in rows:
        if ticker is None:
            continue
        checks = json.loads(check_results) if check_results else []
        grades[ticker] = {
            c["check_name"]: EvalCheckResult(
                check_name=c["check_name"],
                passed=c["passed"],
                expected=c.get("expected", ""),
                actual=c.get("actual", ""),
                severity=c.get("severity", ""),
            )
            for c in checks
        }
    return grades


ChangeKind = Literal["fix", "regression", "other"]


@dataclass
class CheckChange:
    """A single check whose pass/fail state differs between baseline and current runs.

    `old`/`new` are the check's `passed` value in each run, or None when the check is absent
    from that run (a check added/removed between runs, or a ticker present in only one).
    """

    check_name: str
    old: bool | None
    new: bool | None
    expected: str
    actual: str

    @property
    def kind(self) -> ChangeKind:
        """`fix` for a False→True flip, `regression` for True→False, else `other`.

        None-involving transitions (a check that appeared or vanished) are `other`: they are
        shown for inspection but not counted as fixes/regressions, keeping the net summary honest.
        """
        if self.old is False and self.new is True:
            return "fix"
        if self.old is True and self.new is False:
            return "regression"
        return "other"


@dataclass
class TickerDiff:
    """The changed checks for one ticker (unchanged checks are dropped)."""

    ticker: str
    changes: list[CheckChange]


@dataclass
class EvalRunDiff:
    """A baseline→current diff: per-ticker changes plus fix/regression totals.

    `fixes`/`regressions` count `CheckChange`s of the matching `kind` across every ticker, so
    the net-change banner and the per-ticker detail are always derived from the same set.
    """

    baseline_run_id: str
    current_run_id: str
    ticker_diffs: list[TickerDiff]
    fixes: int
    regressions: int


def diff_eval_runs(
    baseline_run_id: str,
    current_run_id: str,
    grades_a: dict[str, dict[str, EvalCheckResult]],
    grades_b: dict[str, dict[str, EvalCheckResult]],
) -> EvalRunDiff:
    """Diff two runs' grade maps (from `load_eval_grades`) into per-ticker changed checks.

    For each ticker across both runs, a check is a change when its `passed` value differs
    (a check missing from one side compares as None). Tickers with no changed checks are
    omitted entirely, so the diff view shows only what moved.
    """
    ticker_diffs: list[TickerDiff] = []
    fixes = regressions = 0
    for ticker in sorted(set(grades_a) | set(grades_b)):
        a = grades_a.get(ticker, {})
        b = grades_b.get(ticker, {})
        changes: list[CheckChange] = []
        for check_name in sorted(set(a) | set(b)):
            old = a[check_name].passed if check_name in a else None
            new = b[check_name].passed if check_name in b else None
            if old == new:
                continue
            # Prefer the current run's envelope for detail; fall back to the baseline's.
            detail = b.get(check_name) or a.get(check_name)
            change = CheckChange(
                check_name=check_name,
                old=old,
                new=new,
                expected=detail.expected if detail else "",
                actual=detail.actual if detail else "",
            )
            changes.append(change)
            if change.kind == "fix":
                fixes += 1
            elif change.kind == "regression":
                regressions += 1
        if changes:
            ticker_diffs.append(TickerDiff(ticker=ticker, changes=changes))
    return EvalRunDiff(
        baseline_run_id=baseline_run_id,
        current_run_id=current_run_id,
        ticker_diffs=ticker_diffs,
        fixes=fixes,
        regressions=regressions,
    )
