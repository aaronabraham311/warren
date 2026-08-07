"""Deterministic terminal-state reduction and semantic screen capture.

The lab deliberately delegates VT parsing to :mod:`pyte`.  Warren owns only the
domain-specific lifecycle reducer and the small scenario vocabulary that connects
typed events to the real renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from io import StringIO
from typing import Literal, TypeAlias

import pyte

from agent.events import (
    AnalysisCompleted,
    CandidateSelected,
    LlmCallCompleted,
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
from agent.terminal.renderer import ColorMode, TerminalRenderer

TerminalOutcome: TypeAlias = Literal["completed", "failed", "cancelled"]
OperationKind: TypeAlias = Literal["setup", "screening", "ticker", "model", "tool"]


@dataclass(slots=True)
class FakeClock:
    """A monotonic clock advanced explicitly by terminal scenarios."""

    current: float = 1.0

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("monotonic time cannot move backwards")
        self.current += seconds
        return self.current


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


class _CaptureStream(StringIO):
    def __init__(self, *, tty: bool, writes: list[str]) -> None:
        super().__init__()
        self._tty = tty
        self._writes = writes

    def isatty(self) -> bool:
        return self._tty

    def write(self, value: str) -> int:
        self._writes.append(value)
        return super().write(value)


@dataclass(frozen=True, slots=True)
class ScreenSnapshot:
    """Normalized terminal state suitable for semantic snapshot assertions."""

    name: str
    cells: tuple[str, ...]
    cursor: tuple[int, int]
    cursor_visible: bool
    scrollback: tuple[str, ...]
    stdout: str
    stderr: str
    activity: ActivityState


def _history_line(line: dict[int, pyte.screens.Char]) -> str:
    if not line:
        return ""
    return "".join(line[column].data for column in range(max(line) + 1)).rstrip()


class TerminalScenario:
    """Drive the real renderer with typed events and capture named VT checkpoints."""

    def __init__(
        self,
        renderer_factory: type[TerminalRenderer] = TerminalRenderer,
        *,
        width: int = 80,
        height: int = 24,
        tty: bool = True,
        color: ColorMode = "always",
        clock: FakeClock | None = None,
    ) -> None:
        self.clock = clock or FakeClock()
        self.width = width
        self.height = height
        self.activity = ActivityState()
        self._writes: list[str] = []
        self._fed_writes = 0
        self._screen = pyte.HistoryScreen(width, height, history=1_000)
        self._stream = pyte.Stream(self._screen)
        self.stdout = _CaptureStream(tty=tty, writes=self._writes)
        self.stderr = _CaptureStream(tty=tty, writes=self._writes)
        self.renderer = renderer_factory(
            stdout=self.stdout,
            stderr=self.stderr,
            color=color,
            animation=tty,
            width=width,
            clock=self.clock,
            terminal_type="xterm-256color" if tty else "dumb",
            no_color=color == "never",
        )

    def start(self, message: str = "Preparing analysis…") -> TerminalScenario:
        self.renderer.start_activity(message)
        return self

    def emit(self, event: RunEvent) -> TerminalScenario:
        self.activity = reduce_activity(self.activity, event, now=self.clock())
        self.renderer.emit(event)
        return self

    def advance(self, seconds: float) -> TerminalScenario:
        self.clock.advance(seconds)
        self.renderer.refresh_activity()
        return self

    def resize(self, width: int, height: int | None = None) -> TerminalScenario:
        self._feed_pending()
        self.width = width
        if height is not None:
            self.height = height
        self._screen.resize(lines=self.height, columns=self.width)
        self.renderer.resize(self.width)
        return self

    def checkpoint(self, name: str) -> ScreenSnapshot:
        self._feed_pending()
        return ScreenSnapshot(
            name=name,
            cells=tuple(line.rstrip() for line in self._screen.display),
            cursor=(self._screen.cursor.x, self._screen.cursor.y),
            cursor_visible=not self._screen.cursor.hidden,
            scrollback=tuple(_history_line(line) for line in self._screen.history.top),
            stdout=self.stdout.getvalue(),
            stderr=self.stderr.getvalue(),
            activity=self.activity,
        )

    def _feed_pending(self) -> None:
        pending = self._writes[self._fed_writes :]
        if pending:
            self._stream.feed("".join(pending))
            self._fed_writes = len(self._writes)

    def close(self) -> TerminalScenario:
        self.renderer.stop_live()
        return self
