"""Rich terminal rendering with a stable, control-safe non-TTY transcript."""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from io import TextIOBase
from typing import IO, Iterator, Literal, TextIO

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, ProgressColumn, SpinnerColumn, Task, TaskID, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from agent.events import (
    AnalysisCompleted,
    CandidateSelected,
    LlmCallCompleted,
    LlmCallPurpose,
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
from agent.models import AnalysisOutput
from agent.service import RunResult, TickerRunResult

ColorMode = Literal["auto", "always", "never"]
ColorSystem = Literal["auto", "standard", "256", "truecolor", "windows"]

NAVY_BRAND = "#4A9CFF"
NAVY_STRONG = "#93C5FD"
NAVY_MUTED = "#6B8FB3"
NAVY_BORDER = "#294766"
NAVY_ACTIVITY = "#7CB7E8"

WARREN_THEME = Theme(
    {
        "warren.brand": f"bold {NAVY_BRAND}",
        "warren.strong": f"bold {NAVY_STRONG}",
        "warren.muted": f"dim {NAVY_MUTED}",
        "warren.border": NAVY_BORDER,
        "warren.activity": NAVY_ACTIVITY,
        "warren.success": f"bold {NAVY_STRONG}",
        "warren.warning": "#BFA66A",
        "warren.error": "bold #C9828E",
    }
)

_TOOL_LABELS = {
    "get_quote": "Market quote",
    "get_fundamentals": "Company fundamentals",
    "get_growth_metrics": "Growth metrics",
    "get_news": "Company news",
    "read_filing": "Regulatory filing",
    "get_valuation_multiples": "Valuation multiples",
    "get_valuation_history": "Valuation history",
    "get_quality_metrics": "Quality metrics",
    "get_financial_strength": "Financial strength",
    "estimate_intrinsic_value": "Intrinsic value",
    "model_dirt_scenarios": "DIRT scenarios",
    "get_capital_allocation": "Capital allocation",
    "get_key_persons": "Key people",
    "get_insider_activity": "Insider activity",
    "get_peer_comparison": "Peer comparison",
    "get_adverse_media": "Adverse media",
    "get_forensic_evidence": "Forensic evidence",
    "screen_watchlists": "Watchlist screening",
    "get_holding_context": "Portfolio context",
    "screen_universe": "Market screening",
}

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|token|authorization|secret)\b"
    r"(\s*[:=]\s*)(\S+)"
)
_ANTHROPIC_KEY = re.compile(r"\bsk-ant-[A-Za-z0-9_-]+\b")


def sanitize_terminal_text(value: object) -> str:
    """Return printable text with terminal control sequences made inert.

    Newlines are retained for long-form results. Tabs and every other Unicode
    control/format character are replaced with spaces, including ESC, C1 controls,
    bidi overrides, and zero-width formatting characters.
    """

    text = str(value)
    safe: list[str] = []
    for char in text:
        if char == "\n":
            safe.append(char)
            continue
        if unicodedata.category(char) in {"Cc", "Cf", "Cs"}:
            safe.append(" ")
        else:
            safe.append(char)
    sanitized = "".join(safe)
    sanitized = _SECRET_ASSIGNMENT.sub(r"\1\2[redacted]", sanitized)
    return _ANTHROPIC_KEY.sub("[redacted]", sanitized)


def _tool_label(tool_name: str) -> str:
    sanitized = sanitize_terminal_text(tool_name)
    return _TOOL_LABELS.get(sanitized, sanitized.replace("_", " ").strip().title())


def _tool_succeeded(status: str) -> bool:
    return status.casefold() in {"ok", "success"}


def _format_duration(latency_ms: int) -> str:
    if latency_ms < 1000:
        return f"{latency_ms}ms"
    if latency_ms >= 60_000:
        minutes, remainder_ms = divmod(latency_ms, 60_000)
        return f"{minutes}m {remainder_ms / 1000:.1f}s"
    return f"{latency_ms / 1000:.1f}s"


def _is_tty(stream: IO[str]) -> bool:
    isatty = getattr(stream, "isatty", None)
    return bool(callable(isatty) and isatty())


def _console(
    stream: TextIO,
    *,
    color: ColorMode,
    width: int | None,
) -> Console:
    no_color = bool(os.environ.get("NO_COLOR"))
    terminal = _is_tty(stream)
    force_terminal: bool | None = None
    if color == "never" or no_color or (color == "auto" and not terminal):
        color_system: ColorSystem | None = None
        force_terminal = False
    elif color == "always":
        color_system = "truecolor"
        force_terminal = True
    else:
        color_system = "auto"
    return Console(
        file=stream,
        force_terminal=force_terminal,
        color_system=color_system,
        no_color=color_system is None,
        theme=WARREN_THEME,
        width=width,
        highlight=False,
        markup=False,
        soft_wrap=False,
    )


@dataclass(slots=True)
class _ProgressState:
    message: str = "Preparing analysis…"
    model_wait: _ModelWait | None = None


@dataclass(frozen=True, slots=True)
class _ModelWait:
    ticker: str
    purpose: LlmCallPurpose
    tool_count: int


def _model_wait_message(wait: _ModelWait, elapsed: float, width: int) -> str:
    """Return honest, width-aware model activity without inventing progress."""

    if elapsed >= 45:
        if width < 48:
            return f"{wait.ticker} · waiting · Ctrl-C"
        return f"Still waiting for model · {wait.ticker} · Ctrl-C to cancel"
    if elapsed >= 15:
        if width < 48:
            return f"Waiting on model · {wait.ticker}"
        return f"Waiting for model response · {wait.ticker}"
    action = {
        "planning": "Planning research",
        "synthesis": "Synthesizing analysis",
        "finalizing": "Finalizing analysis",
        "validation": "Validating analysis",
    }[wait.purpose]
    if width < 48:
        return f"{action} · {wait.ticker}"
    tools = f" · {wait.tool_count} tools complete" if wait.tool_count else ""
    return f"{action} · {wait.ticker}{tools}"


class _ActivityColumn(ProgressColumn):
    def __init__(self, state: _ProgressState, width: int) -> None:
        super().__init__()
        self._state = state
        self._width = width

    def render(self, task: Task) -> Text:
        if self._state.model_wait is None:
            message = self._state.message
        else:
            message = _model_wait_message(
                self._state.model_wait,
                task.elapsed or 0.0,
                self._width,
            )
        return Text(message, style="warren.activity", no_wrap=True, overflow="ellipsis")


class TerminalRenderer:
    """Display-safe event sink and final-result renderer.

    Durable progress events are diagnostics and go to stderr. Final analyses and
    comparisons are results and go to stdout, which keeps piping predictable.
    """

    def __init__(
        self,
        *,
        stdout: TextIO = sys.stdout,
        stderr: TextIO = sys.stderr,
        color: ColorMode = "auto",
        animation: bool = True,
        show_cost: bool = True,
        width: int | None = None,
    ) -> None:
        self.stdout = _console(stdout, color=color, width=width)
        self.stderr = _console(stderr, color=color, width=width)
        self.show_cost = show_cost
        self._interactive_results_enabled = animation
        self._state = _ProgressState()
        self._evidence_by_ticker: dict[str, list[str]] = {}
        self._live_enabled = (
            animation and _is_tty(stderr) and os.environ.get("TERM", "").casefold() != "dumb"
        )
        self._live: Live | None = None
        self._progress: Progress | None = None
        self._progress_task_id: TaskID | None = None
        self._plain_activity_started = False
        self._last_research_tool_count = 0

    @property
    def narrow(self) -> bool:
        return self.stdout.width < 72

    @property
    def supports_result_binder(self) -> bool:
        """Whether stdout can safely host a temporary interactive result view."""

        return (
            self._interactive_results_enabled
            and _is_tty(self.stdout.file)
            and os.environ.get("TERM", "").casefold() != "dumb"
        )

    def __enter__(self) -> TerminalRenderer:
        self.start_activity(self._state.message)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.stop_live()

    def stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        self._progress = None
        self._progress_task_id = None
        self._plain_activity_started = False

    @contextmanager
    def activity(
        self,
        message: str,
        *,
        announce: bool = False,
    ) -> Iterator[TerminalRenderer]:
        """Show immediate activity and guarantee terminal cleanup on every exit path."""

        self.start_activity(message, announce=announce)
        try:
            yield self
        finally:
            self.stop_live()

    def start_activity(self, message: str, *, announce: bool = False) -> None:
        self._state.message = sanitize_terminal_text(message)
        self._state.model_wait = None
        if announce:
            self.stderr.print(Text(self._state.message, style="warren.activity"))
            self._plain_activity_started = True
        if self._live_enabled:
            if self._live is None:
                self._progress = Progress(
                    SpinnerColumn(style="warren.activity"),
                    _ActivityColumn(self._state, self.stderr.width),
                    TimeElapsedColumn(),
                    console=self.stderr,
                    auto_refresh=False,
                    expand=False,
                )
                self._progress_task_id = self._progress.add_task(
                    self._state.message,
                    total=None,
                )
                self._live = Live(
                    self._progress,
                    console=self.stderr,
                    transient=True,
                    refresh_per_second=10,
                    redirect_stdout=False,
                    redirect_stderr=False,
                )
                self._live.start(refresh=True)
            else:
                self.update_activity(self._state.message)
        elif not self._plain_activity_started:
            self.stderr.print(Text(self._state.message, style="warren.activity"))
            self._plain_activity_started = True

    def update_activity(self, message: str) -> None:
        self._state.message = sanitize_terminal_text(message)
        self._state.model_wait = None
        if self._progress is not None and self._progress_task_id is not None:
            self._progress.update(self._progress_task_id, description=self._state.message)
            if self._live is not None:
                self._live.refresh()
        elif self._plain_activity_started:
            self.stderr.print(Text(self._state.message, style="warren.activity"))
        else:
            self.start_activity(self._state.message)

    def _start_model_activity(self, event: LlmCallStarted) -> None:
        ticker = sanitize_terminal_text(event.ticker or "request")
        self._state.model_wait = _ModelWait(ticker, event.purpose, event.tool_count)
        if event.tool_count > self._last_research_tool_count:
            self._render_research_transition(event.tool_count)
            self._last_research_tool_count = event.tool_count
        if self._progress is not None and self._progress_task_id is not None:
            self._progress.reset(self._progress_task_id, start=True)
            if self._live is not None:
                self._live.refresh()
        elif self._plain_activity_started:
            self.stderr.print(
                Text(
                    _model_wait_message(self._state.model_wait, 0.0, self.stderr.width),
                    style="warren.activity",
                )
            )
        else:
            wait = self._state.model_wait
            self.start_activity(_model_wait_message(wait, 0.0, self.stderr.width))
            self._state.model_wait = wait
            if self._live is not None:
                self._live.refresh()

    def _render_research_transition(self, tool_count: int) -> None:
        line = Text()
        line.append("✓ ", style="warren.success")
        line.append("Research pass", style="warren.success")
        line.append(f"  ·  {tool_count} tools gathered", style="warren.muted")
        self.stderr.print(line)

    def welcome(self) -> None:
        line = Text()
        line.append("Warren", style="warren.strong")
        line.append("  ·  stock analysis  ·  ", style="warren.muted")
        line.append("/help", style="warren.brand")
        line.append(" for commands", style="warren.muted")
        self.stdout.print(line)

    def render_help(self) -> None:
        self.stdout.print(Text("Commands", style="warren.strong"))
        table = Table.grid(padding=(0, 2), expand=False)
        table.add_column(style="warren.brand", no_wrap=True)
        table.add_column()
        for command, description in (
            ("Analyze", "Analyze AAPL · Compare COST with WMT · Review my portfolio"),
            ("Research", "/discover · /gem-hunt"),
            ("Runs", "/history [ticker] · /show RUN_ID · /trace [RUN_ID]"),
            ("Holdings", "/portfolio · /watchlist"),
            ("Session", "/new · /persona [default|dirt] · /budget [USD]"),
            ("Reference", "/tools · /help · /quit"),
        ):
            table.add_row(command, description)
        self.stdout.print(table)

    def show_stopping(self) -> None:
        """Make cancellation visible even while a provider call is still blocking."""

        self.stop_live()
        self.stderr.print(
            Text(
                "■ Stopping… press Ctrl-C again to stop immediately",
                style="warren.warning",
            )
        )

    def emit(self, event: RunEvent) -> None:
        if isinstance(event, RunStarted):
            self._evidence_by_ticker.clear()
            self._last_research_tool_count = 0
        elif (
            isinstance(event, ToolCallCompleted)
            and event.ticker is not None
            and _tool_succeeded(event.status)
        ):
            evidence = self._evidence_by_ticker.setdefault(event.ticker, [])
            if event.tool_name not in evidence:
                evidence.append(event.tool_name)
        if isinstance(event, LlmCallStarted):
            self._start_model_activity(event)
            return
        if isinstance(event, LlmCallCompleted):
            self._render_model_completion(event)
            ticker = sanitize_terminal_text(event.ticker) if event.ticker else "the request"
            self.update_activity(f"Analyzing {ticker}…")
            return
        if isinstance(event, ToolCallCompleted):
            self._render_tool_completion(event)
            ticker = sanitize_terminal_text(event.ticker) if event.ticker else "the request"
            self.update_activity(f"Analyzing {ticker}…")
            return
        line = self._event_line(event)
        if line is None:
            return
        message = f"{line[0]}: {line[1]}"
        if self._live is not None:
            self.update_activity(message)
        else:
            self.stderr.print(Text(message, style="warren.activity"))

    def evidence_for(self, ticker: str) -> tuple[str, ...]:
        """Return the safe tool-name evidence index captured for the latest run."""

        return tuple(self._evidence_by_ticker.get(ticker, ()))

    def _render_tool_completion(self, event: ToolCallCompleted) -> None:
        success = _tool_succeeded(event.status)
        glyph = "✓" if success else "✗"
        style = "warren.success" if success else "warren.error"
        label = _tool_label(event.tool_name)
        line = Text()
        line.append(glyph + " ", style=style)
        line.append(label, style=style)
        if event.cached:
            line.append("  ·  cached", style="warren.muted")
        if event.latency_ms is not None:
            line.append(f"  ·  {_format_duration(event.latency_ms)}", style="warren.muted")
        if event.retry_count:
            line.append(f"  ·  {event.retry_count} retries", style="warren.muted")
        if event.error_summary:
            line.append(
                "  ·  " + sanitize_terminal_text(event.error_summary),
                style="warren.error",
            )
        self.stderr.print(line)

    def _render_model_completion(self, event: LlmCallCompleted) -> None:
        label = {
            "planning": "Research plan",
            "synthesis": "Model synthesis",
            "finalizing": "Final analysis",
            "validation": "Analysis validation",
        }[event.purpose]
        line = Text()
        line.append("✓ ", style="warren.success")
        line.append(label, style="warren.success")
        line.append(f"  ·  {event.output_tokens:,} tokens", style="warren.muted")
        if event.latency_ms is not None:
            line.append(f"  ·  {_format_duration(event.latency_ms)}", style="warren.muted")
        if event.cache_read_tokens or event.cache_creation_tokens:
            line.append(
                f"  ·  cache-read={event.cache_read_tokens:,}"
                f" cache-write={event.cache_creation_tokens:,}",
                style="warren.muted",
            )
        if self.show_cost:
            line.append(f"  ·  ${event.cost_usd:.4f}", style="warren.muted")
        self.stderr.print(line)

    def _event_line(self, event: RunEvent) -> tuple[str, str] | None:
        if isinstance(event, RunStarted):
            tickers = ", ".join(sanitize_terminal_text(t) for t in event.tickers)
            mode = sanitize_terminal_text(event.mode or "analysis").replace("_", " ")
            return "Analyzing", tickers or mode
        if isinstance(event, ScreeningStarted):
            total = "?" if event.total is None else str(event.total)
            return "Screening", f"{total} securities"
        if isinstance(event, ScreeningProgress):
            ticker = f" {sanitize_terminal_text(event.ticker)}" if event.ticker else ""
            return "Screening", f"{event.completed}/{event.total}{ticker}"
        if isinstance(event, CandidateSelected):
            rank = f"#{event.rank} " if event.rank is not None else ""
            return "Selected", f"{rank}{sanitize_terminal_text(event.ticker)}"
        if isinstance(event, TickerStarted):
            return "Analyzing", sanitize_terminal_text(event.ticker)
        if isinstance(event, ToolCallStarted):
            return "Using", _tool_label(event.tool_name)
        if isinstance(event, AnalysisCompleted):
            recommendation = sanitize_terminal_text(event.recommendation or "unknown")
            confidence = "?" if event.confidence is None else f"{event.confidence:.0%}"
            return (
                "Analysis",
                f"{sanitize_terminal_text(event.ticker)} {recommendation} {confidence}",
            )
        if isinstance(event, RunCancelled):
            return "Stopping", "run cancelled safely"
        if isinstance(event, RunFailed):
            return (
                "Failed",
                f"{sanitize_terminal_text(event.run_id)} failed: "
                f"{sanitize_terminal_text(event.error_msg or 'unknown error')}",
            )
        if isinstance(event, RunCompleted):
            cost = f", ${event.total_cost_usd:.4f}" if self.show_cost else ""
            duration = (
                f", {event.duration_seconds:.1f}s" if event.duration_seconds is not None else ""
            )
            return (
                "Done",
                f"{sanitize_terminal_text(event.run_id)} "
                f"{sanitize_terminal_text(event.status)}{cost}{duration}",
            )
        return None

    def render_result(self, result: RunResult, *, compare: bool = False) -> None:
        """Render final structured results deterministically to stdout."""

        self.stop_live()
        ticker_results = result.ticker_results
        if compare or len(ticker_results) > 1:
            self.render_comparison(ticker_results)
        else:
            for ticker_result in ticker_results:
                if ticker_result.analysis is not None:
                    self.render_analysis(ticker_result.analysis)
                else:
                    self._render_ticker_error(ticker_result)
        if result.status not in {"success", "cancelled"} and result.error_msg:
            self.diagnostic(result.error_msg, error=result.status == "failed")
        self._render_run_footer(result)

    def render_compact_result(self, result: RunResult) -> None:
        """Leave a durable one-line recommendation before opening the binder."""

        self.stop_live()
        analysis = result.analyses[0]
        self.stdout.print(
            Text(
                f"{sanitize_terminal_text(analysis.ticker)} | "
                f"{sanitize_terminal_text(analysis.recommendation).upper()} | "
                f"confidence {analysis.confidence:.0%} | "
                f"Run {sanitize_terminal_text(result.run_id)}"
            )
        )

    def render_analysis(self, analysis: AnalysisOutput) -> None:
        ticker = sanitize_terminal_text(analysis.ticker)
        recommendation = sanitize_terminal_text(analysis.recommendation).upper()
        confidence = f"{analysis.confidence:.0%}"
        thesis = sanitize_terminal_text(analysis.thesis)
        risks = [sanitize_terminal_text(risk) for risk in analysis.key_risks]
        quality = [sanitize_terminal_text(note) for note in analysis.data_quality_notes]
        lynch_pros = [sanitize_terminal_text(item) for item in analysis.lynch_signals.pros]
        lynch_cons = [sanitize_terminal_text(item) for item in analysis.lynch_signals.cons]
        buffett_pros = [sanitize_terminal_text(item) for item in analysis.buffett_signals.pros]
        buffett_cons = [sanitize_terminal_text(item) for item in analysis.buffett_signals.cons]
        decision = analysis.dirt_decision

        if self.narrow or not _is_tty(self.stdout.file):
            self.stdout.print(Text(f"{ticker} | {recommendation} | confidence {confidence}"))
            self.stdout.print(Text(f"Thesis: {thesis}"))
            if analysis.termination_reason != "success":
                self.stdout.print(
                    Text(f"Termination: {sanitize_terminal_text(analysis.termination_reason)}")
                )
            self.stdout.print(
                Text(
                    "Lynch: + "
                    + ("; ".join(lynch_pros) or "none")
                    + " | - "
                    + ("; ".join(lynch_cons) or "none")
                )
            )
            self.stdout.print(
                Text(
                    "Buffett: + "
                    + ("; ".join(buffett_pros) or "none")
                    + " | - "
                    + ("; ".join(buffett_cons) or "none")
                )
            )
            self.stdout.print(Text("Risks: " + "; ".join(risks)))
            if decision is not None:
                self.stdout.print(
                    Text(
                        "DIRT: "
                        f"{sanitize_terminal_text(decision.outcome).upper()} | "
                        f"weighted IRR {decision.probability_weighted_irr:.1%} | "
                        f"entry {decision.currency} {decision.required_entry_price:,.2f}"
                    )
                )
            if quality:
                self.stdout.print(Text("Data quality: " + "; ".join(quality)))
            return

        body: list[Text] = [
            Text(thesis),
            Text(),
            Text("Lynch signals", style="warren.strong"),
            Text("+ " + ("; ".join(lynch_pros) or "none")),
            Text("- " + ("; ".join(lynch_cons) or "none")),
            Text(),
            Text("Buffett signals", style="warren.strong"),
            Text("+ " + ("; ".join(buffett_pros) or "none")),
            Text("- " + ("; ".join(buffett_cons) or "none")),
            Text(),
            Text("Key risks", style="warren.strong"),
        ]
        if analysis.termination_reason != "success":
            body[1:1] = [
                Text(),
                Text(
                    "Termination: " + sanitize_terminal_text(analysis.termination_reason),
                    style="warren.warning",
                ),
            ]
        body.extend(Text(f"• {risk}") for risk in risks)
        if decision is not None:
            body.extend(
                [
                    Text(),
                    Text("DIRT decision", style="warren.strong"),
                    Text(
                        f"{sanitize_terminal_text(decision.outcome).upper()} · "
                        f"weighted IRR {decision.probability_weighted_irr:.1%} · "
                        f"entry {decision.currency} {decision.required_entry_price:,.2f}"
                    ),
                ]
            )
        if quality:
            body.extend([Text(), Text("Data quality", style="warren.strong")])
            body.extend(Text(f"• {note}") for note in quality)
        title = Text(f"{ticker}  {recommendation}  {confidence}", style="warren.strong")
        self.stdout.print(Panel(Group(*body), title=title, border_style="warren.border"))

    def render_comparison(self, ticker_results: Sequence[TickerRunResult]) -> None:
        """Render every requested ticker in order, including failed positions."""

        if self.narrow or not _is_tty(self.stdout.file):
            for ticker_result in ticker_results:
                if ticker_result.analysis is not None:
                    self.render_analysis(ticker_result.analysis)
                else:
                    self._render_ticker_error(ticker_result)
            return

        table = Table(
            title=Text("Comparison", style="warren.strong"),
            show_lines=False,
            expand=True,
            border_style="warren.border",
            header_style="warren.brand",
        )
        table.add_column("Ticker", no_wrap=True, style="warren.strong")
        table.add_column("Call", no_wrap=True, style="warren.brand")
        table.add_column("Confidence", justify="right", no_wrap=True, style="warren.muted")
        table.add_column("Thesis")
        table.add_column("Key risks")
        for ticker_result in ticker_results:
            analysis = ticker_result.analysis
            if analysis is None:
                table.add_row(
                    sanitize_terminal_text(ticker_result.ticker),
                    "ERROR",
                    "—",
                    sanitize_terminal_text(ticker_result.error or "analysis unavailable"),
                    "—",
                )
            else:
                thesis = sanitize_terminal_text(analysis.thesis)
                if analysis.termination_reason != "success":
                    thesis += "\nTermination: " + sanitize_terminal_text(
                        analysis.termination_reason
                    )
                table.add_row(
                    sanitize_terminal_text(analysis.ticker),
                    sanitize_terminal_text(analysis.recommendation).upper(),
                    f"{analysis.confidence:.0%}",
                    thesis,
                    "\n".join(sanitize_terminal_text(risk) for risk in analysis.key_risks),
                )
        self.stdout.print(table)

    def _render_ticker_error(self, ticker_result: TickerRunResult) -> None:
        ticker = sanitize_terminal_text(ticker_result.ticker)
        message = sanitize_terminal_text(ticker_result.error or "analysis unavailable")
        self.stdout.print(Text(f"{ticker} | ERROR | {message}", style="warren.error"))

    def _render_run_footer(self, result: RunResult) -> None:
        if result.status == "cancelled":
            footer = (
                f"■ Interrupted · {result.total_tool_calls} tools · "
                f"{result.duration_seconds:.1f}s · ${result.total_cost_usd:.4f} · "
                f"Run {sanitize_terminal_text(result.run_id)}"
            )
            self.stdout.print(Text(footer, style="warren.warning"))
            return
        footer = (
            f"Run {sanitize_terminal_text(result.run_id)} | "
            f"{sanitize_terminal_text(result.status)} | "
            f"cost ${result.total_cost_usd:.4f} | "
            f"tokens {result.total_input_tokens} in/{result.total_output_tokens} out | "
            f"tools {result.total_tool_calls} | {result.duration_seconds:.1f}s"
        )
        self.stdout.print(Text(footer, style="warren.muted"))

    def notice(self, message: object) -> None:
        self.stdout.print(Text(sanitize_terminal_text(message)))

    def diagnostic(self, message: object, *, error: bool = False) -> None:
        style = "warren.error" if error else "warren.warning"
        self.stderr.print(Text(sanitize_terminal_text(message), style=style))


def is_text_stream(value: object) -> bool:
    """Testing/type helper for objects accepted by Rich Console."""

    return isinstance(value, TextIOBase) or hasattr(value, "write")
