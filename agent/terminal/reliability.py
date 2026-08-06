"""Deterministic terminal-state reduction and semantic screen capture.

The lab deliberately delegates VT parsing to :mod:`pyte`.  Warren owns only the
domain-specific lifecycle reducer and the small scenario vocabulary that connects
typed events to the real renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

import pyte

from agent.activity import ActivityState, reduce_activity
from agent.events import RunEvent
from agent.terminal.renderer import ColorMode, TerminalRenderer


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
        terminal_type: str | None = None,
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
        selected_terminal = terminal_type or ("xterm-256color" if tty else "dumb")
        self.renderer = renderer_factory(
            stdout=self.stdout,
            stderr=self.stderr,
            color=color,
            animation=tty,
            width=width,
            clock=self.clock,
            terminal_type=selected_terminal,
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
