"""Typed, display-safe events emitted by the shared run service.

These events intentionally contain only progress and aggregate metadata.  Prompts,
model text, tool inputs/outputs, and hidden reasoning remain solely in their existing
storage boundaries and are never copied into the terminal event stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Protocol, TypeAlias


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class RunStarted:
    run_id: str
    mode: str | None = None
    tickers: tuple[str, ...] = ()
    timestamp: str = field(default_factory=_utc_now_iso)
    kind: Literal["run_started"] = "run_started"


@dataclass(frozen=True, slots=True)
class ScreeningStarted:
    run_id: str
    total: int | None = None
    timestamp: str = field(default_factory=_utc_now_iso)
    kind: Literal["screening_started"] = "screening_started"


@dataclass(frozen=True, slots=True)
class ScreeningProgress:
    run_id: str
    completed: int
    total: int
    ticker: str | None = None
    timestamp: str = field(default_factory=_utc_now_iso)
    kind: Literal["screening_progress"] = "screening_progress"


@dataclass(frozen=True, slots=True)
class CandidateSelected:
    run_id: str
    ticker: str
    rank: int | None = None
    timestamp: str = field(default_factory=_utc_now_iso)
    kind: Literal["candidate_selected"] = "candidate_selected"


@dataclass(frozen=True, slots=True)
class TickerStarted:
    run_id: str
    ticker: str
    index: int | None = None
    total: int | None = None
    timestamp: str = field(default_factory=_utc_now_iso)
    kind: Literal["ticker_started"] = "ticker_started"


@dataclass(frozen=True, slots=True)
class LlmCallCompleted:
    run_id: str
    ticker: str | None
    model: str | None
    latency_ms: int | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    timestamp: str = field(default_factory=_utc_now_iso)
    kind: Literal["llm_call_completed"] = "llm_call_completed"


@dataclass(frozen=True, slots=True)
class ToolCallStarted:
    run_id: str
    ticker: str | None
    tool_name: str
    timestamp: str = field(default_factory=_utc_now_iso)
    kind: Literal["tool_call_started"] = "tool_call_started"


@dataclass(frozen=True, slots=True)
class ToolCallCompleted:
    run_id: str
    ticker: str | None
    tool_name: str
    status: str
    cached: bool
    latency_ms: int | None
    retry_count: int
    error_summary: str | None = None
    timestamp: str = field(default_factory=_utc_now_iso)
    kind: Literal["tool_call_completed"] = "tool_call_completed"


@dataclass(frozen=True, slots=True)
class AnalysisCompleted:
    run_id: str
    ticker: str
    recommendation: str | None
    confidence: float | None
    timestamp: str = field(default_factory=_utc_now_iso)
    kind: Literal["analysis_completed"] = "analysis_completed"


@dataclass(frozen=True, slots=True)
class RunCompleted:
    run_id: str
    status: str
    total_cost_usd: float
    duration_seconds: float | None
    timestamp: str = field(default_factory=_utc_now_iso)
    kind: Literal["run_completed"] = "run_completed"


@dataclass(frozen=True, slots=True)
class RunFailed:
    run_id: str
    error_msg: str | None = None
    timestamp: str = field(default_factory=_utc_now_iso)
    kind: Literal["run_failed"] = "run_failed"


@dataclass(frozen=True, slots=True)
class RunCancelled:
    run_id: str
    timestamp: str = field(default_factory=_utc_now_iso)
    kind: Literal["run_cancelled"] = "run_cancelled"


RunEvent: TypeAlias = (
    RunStarted
    | ScreeningStarted
    | ScreeningProgress
    | CandidateSelected
    | TickerStarted
    | LlmCallCompleted
    | ToolCallStarted
    | ToolCallCompleted
    | AnalysisCompleted
    | RunCompleted
    | RunFailed
    | RunCancelled
)


class EventSink(Protocol):
    """Consumer of the safe, typed display event stream."""

    def emit(self, event: RunEvent) -> None: ...


class NullEventSink:
    """No-op event sink used by existing non-interactive callers."""

    def emit(self, event: RunEvent) -> None:
        del event


def emit_safely(sink: EventSink, event: RunEvent) -> None:
    """Emit progress without allowing a presentation failure to abort a run."""
    try:
        sink.emit(event)
    except Exception:
        # The durable logger reports its own diagnostic; direct transient progress is
        # best-effort and intentionally cannot affect run control flow.
        return


def _str(record: dict[str, object], key: str) -> str | None:
    value = record.get(key)
    return value if isinstance(value, str) else None


def _int(record: dict[str, object], key: str, default: int = 0) -> int:
    value = record.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _optional_int(record: dict[str, object], key: str) -> int | None:
    value = record.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _float(record: dict[str, object], key: str, default: float = 0.0) -> float:
    value = record.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _timestamp(record: dict[str, object]) -> str:
    return _str(record, "ts") or _utc_now_iso()


def event_from_wal_record(record: dict[str, object]) -> RunEvent | None:
    """Project a durable WAL record into its redacted display event, if applicable."""
    run_id = _str(record, "run_id")
    event = _str(record, "event")
    if run_id is None or event is None:
        return None
    if event == "run_started":
        raw_tickers = record.get("tickers")
        tickers = (
            tuple(item for item in raw_tickers if isinstance(item, str))
            if isinstance(raw_tickers, list)
            else ()
        )
        return RunStarted(
            run_id,
            mode=_str(record, "mode"),
            tickers=tickers,
            timestamp=_timestamp(record),
        )
    if event == "phase_started" and record.get("phase") == "screening":
        total = _int(record, "universe_size", -1)
        return ScreeningStarted(
            run_id,
            total=None if total < 0 else total,
            timestamp=_timestamp(record),
        )
    if event == "screening_progress":
        return ScreeningProgress(
            run_id=run_id,
            completed=_int(record, "completed"),
            total=_int(record, "total"),
            ticker=_str(record, "ticker"),
            timestamp=_timestamp(record),
        )
    if event == "candidate_selected":
        ticker = _str(record, "ticker")
        if ticker is not None:
            return CandidateSelected(
                run_id=run_id,
                ticker=ticker,
                rank=_optional_int(record, "rank"),
                timestamp=_timestamp(record),
            )
    if event == "ticker_started":
        ticker = _str(record, "ticker")
        if ticker is not None:
            return TickerStarted(run_id, ticker, timestamp=_timestamp(record))
    if event == "llm_call":
        return LlmCallCompleted(
            run_id=run_id,
            ticker=_str(record, "ticker"),
            model=_str(record, "model"),
            latency_ms=_optional_int(record, "latency_ms"),
            input_tokens=_int(record, "input_tokens"),
            output_tokens=_int(record, "output_tokens"),
            cost_usd=_float(record, "cost_usd"),
            cache_read_tokens=_int(record, "cache_read_tokens"),
            cache_creation_tokens=_int(record, "cache_creation_tokens"),
            timestamp=_timestamp(record),
        )
    if event == "tool_call":
        tool_name = _str(record, "tool")
        if tool_name is not None:
            return ToolCallCompleted(
                run_id=run_id,
                ticker=_str(record, "ticker"),
                tool_name=tool_name,
                status=_str(record, "status") or "unknown",
                cached=record.get("cached") is True,
                latency_ms=_optional_int(record, "latency_ms"),
                retry_count=_int(record, "retry_count"),
                error_summary=_str(record, "error_msg"),
                timestamp=_timestamp(record),
            )
    if event == "ticker_completed":
        ticker = _str(record, "ticker")
        if ticker is not None:
            confidence = record.get("confidence")
            return AnalysisCompleted(
                run_id=run_id,
                ticker=ticker,
                recommendation=_str(record, "recommendation"),
                confidence=(
                    float(confidence)
                    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
                    else None
                ),
                timestamp=_timestamp(record),
            )
    if event == "run_completed":
        status = _str(record, "status") or "failed"
        if status == "cancelled":
            return RunCancelled(run_id, timestamp=_timestamp(record))
        if status == "failed":
            return RunFailed(
                run_id,
                error_msg=_str(record, "error_msg"),
                timestamp=_timestamp(record),
            )
        duration = record.get("duration_seconds")
        return RunCompleted(
            run_id=run_id,
            status=status,
            total_cost_usd=_float(record, "total_cost_usd"),
            duration_seconds=(
                float(duration)
                if isinstance(duration, (int, float)) and not isinstance(duration, bool)
                else None
            ),
            timestamp=_timestamp(record),
        )
    return None
