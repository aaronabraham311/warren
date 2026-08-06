"""Pure, display-safe activity reduction shared by diagnostics and renderers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

from agent.events import (
    AnalysisCompleted,
    CandidateSelected,
    LlmCallCompleted,
    LlmCallFailed,
    LlmCallStarted,
    RunCancelled,
    RunCompleted,
    RunEvent,
    RunFailed,
    RunStarted,
    ScreeningProgress,
    ScreeningStarted,
    TickerStarted,
    ToolCallCompleted,
    ToolCallStarted,
)

TerminalOutcome: TypeAlias = Literal["completed", "failed", "cancelled"]
OperationKind: TypeAlias = Literal["setup", "screening", "ticker", "model", "tool"]


@dataclass(frozen=True, slots=True)
class ActivityState:
    """Display-safe projection of the latest accepted lifecycle event."""

    run_id: str | None = None
    ticker: str | None = None
    operation: OperationKind | None = None
    operation_name: str | None = None
    operation_started_at: float | None = None
    last_event_at: float | None = None
    external_wait: bool = False
    completed_tools: int = 0
    completed_models: int = 0
    cancellation_requested: bool = False
    outcome: TerminalOutcome | None = None

    def operation_age(self, now: float) -> float | None:
        if self.operation_started_at is None:
            return None
        return max(0.0, now - self.operation_started_at)

    def semantic_event_age(self, now: float) -> float | None:
        if self.last_event_at is None:
            return None
        return max(0.0, now - self.last_event_at)


def reduce_activity(state: ActivityState, event: RunEvent, *, now: float) -> ActivityState:
    """Reduce one typed event without allowing terminal states to regress."""

    if isinstance(event, RunStarted):
        return ActivityState(
            run_id=event.run_id,
            operation="setup",
            operation_name="preparing",
            operation_started_at=now,
            last_event_at=now,
        )
    if state.outcome is not None or (state.run_id is not None and event.run_id != state.run_id):
        return state
    current = replace(state, run_id=event.run_id, last_event_at=now)
    if isinstance(event, ScreeningStarted):
        return replace(
            current,
            operation="screening",
            operation_name="screening",
            operation_started_at=now,
            external_wait=False,
        )
    if isinstance(event, ScreeningProgress):
        return replace(current, ticker=event.ticker or state.ticker)
    if isinstance(event, CandidateSelected):
        return replace(current, ticker=event.ticker)
    if isinstance(event, TickerStarted):
        return replace(
            current,
            ticker=event.ticker,
            operation="ticker",
            operation_name=event.ticker,
            operation_started_at=now,
            external_wait=False,
        )
    if isinstance(event, LlmCallStarted):
        return replace(
            current,
            ticker=event.ticker or state.ticker,
            operation="model",
            operation_name=event.purpose,
            operation_started_at=now,
            external_wait=True,
        )
    if isinstance(event, LlmCallCompleted):
        return replace(
            current,
            ticker=event.ticker or state.ticker,
            operation="ticker",
            operation_name=event.ticker or state.ticker,
            operation_started_at=now,
            external_wait=False,
            completed_models=state.completed_models + 1,
        )
    if isinstance(event, LlmCallFailed):
        return replace(
            current,
            ticker=event.ticker or state.ticker,
            operation="ticker",
            operation_name=event.ticker or state.ticker,
            operation_started_at=now,
            external_wait=False,
        )
    if isinstance(event, ToolCallStarted):
        return replace(
            current,
            ticker=event.ticker or state.ticker,
            operation="tool",
            operation_name=event.tool_name,
            operation_started_at=now,
            external_wait=True,
        )
    if isinstance(event, ToolCallCompleted):
        return replace(
            current,
            ticker=event.ticker or state.ticker,
            operation="ticker",
            operation_name=event.ticker or state.ticker,
            operation_started_at=now,
            external_wait=False,
            completed_tools=state.completed_tools + 1,
        )
    if isinstance(event, AnalysisCompleted):
        return replace(
            current,
            ticker=event.ticker,
            operation="ticker",
            operation_name=event.ticker,
            external_wait=False,
        )
    if isinstance(event, RunCancelled):
        return replace(
            current,
            operation=None,
            operation_name=None,
            external_wait=False,
            cancellation_requested=True,
            outcome="cancelled",
        )
    if isinstance(event, RunFailed):
        return replace(
            current,
            operation=None,
            operation_name=None,
            external_wait=False,
            outcome="failed",
        )
    if isinstance(event, RunCompleted):
        return replace(
            current,
            operation=None,
            operation_name=None,
            external_wait=False,
            outcome="completed",
        )
