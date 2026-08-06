"""Typed, read-only access to Warren's persisted terminal state.

History and run details come from the SQLite projection.  Trace events deliberately
come from the per-run JSONL write-ahead log, which remains the authoritative record.
None of these queries needs an Anthropic API key or invokes an agent tool.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import func, select

from storage.engine import get_session
from storage.models import Analysis, Holding, Run, Watchlist

DEFAULT_HISTORY_LIMIT = 20
MAX_HISTORY_LIMIT = 100
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    run_id: str
    started_at: datetime | None
    completed_at: datetime | None
    status: str | None
    tickers: tuple[str, ...]
    total_cost_usd: float | None
    num_tool_calls: int | None


@dataclass(frozen=True, slots=True)
class AnalysisDetail:
    ticker: str
    analysis_type: str | None
    recommendation: str | None
    confidence: float | None
    thesis: str | None
    lynch_signals: dict[str, list[str]]
    buffett_signals: dict[str, list[str]]
    key_risks: tuple[str, ...]
    data_quality_notes: tuple[str, ...]
    tool_calls_made: int | None
    tokens_used: int | None
    termination_reason: str | None
    dirt_signals: dict[str, object] | None
    dirt_decision: dict[str, object] | None
    decision_outcome: str | None
    probability_weighted_irr: float | None
    created_at: datetime | None


@dataclass(frozen=True, slots=True)
class StoredTickerResult:
    ticker: str
    analysis: AnalysisDetail | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RunDetail:
    run_id: str
    started_at: datetime | None
    completed_at: datetime | None
    status: str | None
    total_input_tokens: int | None
    total_output_tokens: int | None
    total_cost_usd: float | None
    num_tool_calls: int | None
    error_msg: str | None
    analyses: tuple[AnalysisDetail, ...]
    ticker_results: tuple[StoredTickerResult, ...]


@dataclass(frozen=True, slots=True)
class TraceEvent:
    sequence: int
    timestamp: str | None
    event: str | None
    ticker: str | None
    summary: str


@dataclass(frozen=True, slots=True)
class QueryWarning:
    line_number: int | None
    message: str


@dataclass(frozen=True, slots=True)
class TraceResult:
    run_id: str
    events: tuple[TraceEvent, ...]
    warnings: tuple[QueryWarning, ...]


@dataclass(frozen=True, slots=True)
class PortfolioEntry:
    ticker: str
    shares: float | None
    cost_basis: float | None
    current_price: float | None
    purchase_date: date | None
    updated_at: datetime | None


@dataclass(frozen=True, slots=True)
class WatchlistEntry:
    ticker: str
    notes: str | None
    added_at: datetime | None


def _history_limit(limit: int) -> int:
    if isinstance(limit, bool) or limit <= 0:
        return 0
    return min(limit, MAX_HISTORY_LIMIT)


def _analysis_detail(analysis: Analysis) -> AnalysisDetail:
    return AnalysisDetail(
        ticker=analysis.ticker,
        analysis_type=analysis.analysis_type,
        recommendation=analysis.recommendation,
        confidence=analysis.confidence,
        thesis=analysis.thesis,
        lynch_signals=dict(analysis.lynch_signals or {}),
        buffett_signals=dict(analysis.buffett_signals or {}),
        key_risks=tuple(analysis.key_risks or ()),
        data_quality_notes=tuple(analysis.data_quality_notes or ()),
        tool_calls_made=analysis.tool_calls_made,
        tokens_used=analysis.tokens_used,
        termination_reason=analysis.termination_reason,
        dirt_signals=dict(analysis.dirt_signals) if analysis.dirt_signals is not None else None,
        dirt_decision=(
            dict(analysis.dirt_decision) if analysis.dirt_decision is not None else None
        ),
        decision_outcome=analysis.decision_outcome,
        probability_weighted_irr=analysis.probability_weighted_irr,
        created_at=analysis.created_at,
    )


def list_history(
    ticker: str | None = None,
    *,
    limit: int = DEFAULT_HISTORY_LIMIT,
) -> tuple[HistoryEntry, ...]:
    """Return recent projected runs, newest first, optionally containing ``ticker``."""
    bounded_limit = _history_limit(limit)
    if bounded_limit == 0:
        return ()

    with get_session() as session:
        statement = select(Run)
        if ticker is not None:
            statement = statement.where(
                select(Analysis.id)
                .where(
                    Analysis.run_id == Run.id,
                    func.upper(Analysis.ticker) == ticker.upper(),
                )
                .exists()
            )
        runs = tuple(
            session.scalars(
                statement.order_by(
                    Run.started_at.is_(None),
                    Run.started_at.desc(),
                    Run.id.desc(),
                ).limit(bounded_limit)
            )
        )
        if not runs:
            return ()

        run_ids = tuple(run.id for run in runs)
        ticker_rows = session.execute(
            select(Analysis.run_id, Analysis.ticker)
            .where(Analysis.run_id.in_(run_ids))
            .order_by(Analysis.run_id, Analysis.id, Analysis.ticker)
        )
        tickers_by_run: dict[str, list[str]] = {run_id: [] for run_id in run_ids}
        for run_id, row_ticker in ticker_rows:
            tickers_by_run[run_id].append(row_ticker)

        return tuple(
            HistoryEntry(
                run_id=run.id,
                started_at=run.started_at,
                completed_at=run.completed_at,
                status=run.status,
                tickers=tuple(tickers_by_run[run.id]),
                total_cost_usd=run.total_cost_usd,
                num_tool_calls=run.num_tool_calls,
            )
            for run in runs
        )


def get_run(run_id: str) -> RunDetail | None:
    """Return one projected run and its analyses in persistence/input order."""
    with get_session() as session:
        run = session.get(Run, run_id)
        if run is None:
            return None
        analyses = tuple(
            _analysis_detail(analysis)
            for analysis in session.scalars(
                select(Analysis)
                .where(Analysis.run_id == run_id)
                .order_by(Analysis.id, Analysis.ticker)
            )
        )
        detail = RunDetail(
            run_id=run.id,
            started_at=run.started_at,
            completed_at=run.completed_at,
            status=run.status,
            total_input_tokens=run.total_input_tokens,
            total_output_tokens=run.total_output_tokens,
            total_cost_usd=run.total_cost_usd,
            num_tool_calls=run.num_tool_calls,
            error_msg=run.error_msg,
            analyses=analyses,
            ticker_results=(),
        )
    return RunDetail(
        run_id=detail.run_id,
        started_at=detail.started_at,
        completed_at=detail.completed_at,
        status=detail.status,
        total_input_tokens=detail.total_input_tokens,
        total_output_tokens=detail.total_output_tokens,
        total_cost_usd=detail.total_cost_usd,
        num_tool_calls=detail.num_tool_calls,
        error_msg=detail.error_msg,
        analyses=detail.analyses,
        ticker_results=_stored_ticker_results(detail),
    )


def _stored_ticker_results(detail: RunDetail) -> tuple[StoredTickerResult, ...]:
    """Rebuild requested ticker order and failures from the authoritative WAL."""

    analysis_by_ticker = {analysis.ticker: analysis for analysis in detail.analyses}
    trace_path = Path(os.environ.get("WARREN_LOGS_DIR", "logs/runs")) / f"{detail.run_id}.jsonl"
    records: list[dict[str, object]] = []
    if trace_path.is_file():
        for line in trace_path.read_bytes().splitlines():
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                continue
            if isinstance(value, dict):
                records.append(value)

    ordered: list[str] = []
    started = next((record for record in records if record.get("event") == "run_started"), None)
    if started is not None:
        tickers = started.get("tickers")
        if isinstance(tickers, list):
            ordered.extend(ticker for ticker in tickers if isinstance(ticker, str))
    if not ordered:
        for record in records:
            ticker = record.get("ticker")
            if record.get("event") == "ticker_started" and isinstance(ticker, str):
                if ticker not in ordered:
                    ordered.append(ticker)
    for analysis in detail.analyses:
        if analysis.ticker not in ordered:
            ordered.append(analysis.ticker)

    failures = {
        ticker: error
        for record in records
        if record.get("event") == "ticker_failed"
        for ticker, error in [(record.get("ticker"), record.get("error"))]
        if isinstance(ticker, str) and isinstance(error, str)
    }
    return tuple(
        StoredTickerResult(
            ticker=ticker,
            analysis=analysis_by_ticker.get(ticker),
            error=(
                None
                if ticker in analysis_by_ticker
                else failures.get(ticker, detail.error_msg or "analysis unavailable")
            ),
        )
        for ticker in ordered
    )


def _latest_run_id() -> str | None:
    with get_session() as session:
        return session.scalar(
            select(Run.id)
            .order_by(Run.started_at.is_(None), Run.started_at.desc(), Run.id.desc())
            .limit(1)
        )


def _trace_event(sequence: int, record: dict[str, object]) -> TraceEvent:
    timestamp = record.get("ts")
    event = record.get("event")
    ticker = record.get("ticker")
    summary_parts: list[str] = []
    for key in (
        "phase",
        "tool",
        "model",
        "status",
        "cached",
        "retry_count",
        "latency_ms",
        "input_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
        "output_tokens",
        "cost_usd",
        "duration_seconds",
    ):
        value = record.get(key)
        if isinstance(value, (str, int, float, bool)):
            summary_parts.append(f"{key}={value}")
    error_msg = record.get("error_msg")
    if isinstance(error_msg, str) and error_msg:
        summary_parts.append(f"error={error_msg[:160]}")
    output = record.get("output")
    if isinstance(output, str):
        try:
            marker = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            marker = None
        if isinstance(marker, dict) and marker.get("truncated") is True:
            sidecar = marker.get("path")
            if isinstance(sidecar, str):
                summary_parts.append(f"output_file={sidecar}")
    return TraceEvent(
        sequence=sequence,
        timestamp=timestamp if isinstance(timestamp, str) else None,
        event=event if isinstance(event, str) else None,
        ticker=ticker if isinstance(ticker, str) else None,
        summary=", ".join(summary_parts),
    )


def get_trace(
    run_id: str | None = None,
    *,
    log_dir: Path | None = None,
) -> TraceResult | None:
    """Read an authoritative JSONL trace, preserving file order.

    A malformed final record can be left by a hard crash.  It is ignored with an
    explicit warning while every prior durable record remains available.  Other
    malformed records are also skipped and surfaced as corruption warnings rather
    than making the interactive shell fail.
    """
    selected_run_id = run_id if run_id is not None else _latest_run_id()
    if selected_run_id is None or _RUN_ID_RE.fullmatch(selected_run_id) is None:
        return None

    base_dir = (
        log_dir if log_dir is not None else Path(os.environ.get("WARREN_LOGS_DIR", "logs/runs"))
    )
    trace_path = base_dir / f"{selected_run_id}.jsonl"
    if not trace_path.is_file():
        return TraceResult(
            run_id=selected_run_id,
            events=(),
            warnings=(QueryWarning(None, "The authoritative trace file is unavailable."),),
        )

    # Parse bytes one line at a time so even a torn multi-byte UTF-8 character in the
    # final record cannot hide the preceding durable events.
    lines = trace_path.read_bytes().splitlines()
    last_content_line = max(
        (index for index, line in enumerate(lines, 1) if line.strip()),
        default=0,
    )
    events: list[TraceEvent] = []
    warnings: list[QueryWarning] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            message = (
                "Ignored a torn final JSONL record."
                if line_number == last_content_line
                else "Ignored a malformed JSONL record."
            )
            warnings.append(QueryWarning(line_number, message))
            continue
        if not isinstance(record, dict):
            warnings.append(QueryWarning(line_number, "Ignored a non-object JSONL record."))
            continue
        events.append(_trace_event(len(events) + 1, record))

    return TraceResult(
        run_id=selected_run_id,
        events=tuple(events),
        warnings=tuple(warnings),
    )


def list_portfolio() -> tuple[PortfolioEntry, ...]:
    """Return the stored portfolio without refreshing prices or calling agent tools."""
    with get_session() as session:
        holdings = session.scalars(select(Holding).order_by(Holding.ticker)).all()
        return tuple(
            PortfolioEntry(
                ticker=holding.ticker,
                shares=holding.shares,
                cost_basis=holding.cost_basis,
                current_price=holding.current_price,
                purchase_date=holding.purchase_date,
                updated_at=holding.updated_at,
            )
            for holding in holdings
        )


def list_watchlist() -> tuple[WatchlistEntry, ...]:
    """Return the stored watchlist without consulting CSV or external sources."""

    with get_session() as session:
        entries = session.scalars(select(Watchlist).order_by(Watchlist.ticker)).all()
        return tuple(
            WatchlistEntry(ticker=entry.ticker, notes=entry.notes, added_at=entry.added_at)
            for entry in entries
        )
