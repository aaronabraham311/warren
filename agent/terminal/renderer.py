"""Rich terminal rendering with a stable, control-safe non-TTY transcript."""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from io import TextIOBase
from typing import IO, Literal, TextIO

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent.events import (
    AnalysisCompleted,
    CandidateSelected,
    LlmCallCompleted,
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


def _is_tty(stream: IO[str]) -> bool:
    isatty = getattr(stream, "isatty", None)
    return bool(callable(isatty) and isatty())


def _console(
    stream: TextIO,
    *,
    color: ColorMode,
    width: int | None,
) -> Console:
    no_color = "NO_COLOR" in os.environ
    terminal = _is_tty(stream)
    force_terminal = color == "always" and not no_color
    if color == "never" or no_color or (color == "auto" and not terminal):
        color_system: ColorSystem | None = None
        force_terminal = False
    else:
        color_system = "auto"
    return Console(
        file=stream,
        force_terminal=force_terminal,
        color_system=color_system,
        no_color=color_system is None,
        width=width,
        highlight=False,
        markup=False,
        soft_wrap=False,
    )


@dataclass(slots=True)
class _ProgressState:
    phase: str = "ready"
    detail: str = ""


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
        self._state = _ProgressState()
        self._live_enabled = animation and _is_tty(stderr)
        self._live: Live | None = None

    @property
    def narrow(self) -> bool:
        return self.stdout.width < 72

    def __enter__(self) -> TerminalRenderer:
        if self._live_enabled and self._live is None:
            self._live = Live(
                self._progress_renderable(),
                console=self.stderr,
                transient=True,
                refresh_per_second=8,
            )
            self._live.start(refresh=True)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.stop_live()

    def stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def emit(self, event: RunEvent) -> None:
        line = self._event_line(event)
        if line is None:
            return
        self._state.phase, self._state.detail = line
        if self._live is not None:
            self._live.update(self._progress_renderable(), refresh=True)
        else:
            self.stderr.print(Text(f"{line[0]}: {line[1]}"))

    def _progress_renderable(self) -> Text:
        return Text(f"{self._state.phase}: {self._state.detail}", style="cyan")

    def _event_line(self, event: RunEvent) -> tuple[str, str] | None:
        if isinstance(event, RunStarted):
            mode = sanitize_terminal_text(event.mode or "run")
            tickers = ", ".join(sanitize_terminal_text(t) for t in event.tickers)
            return (
                "run",
                f"{sanitize_terminal_text(event.run_id)} started {mode}"
                + (f" [{tickers}]" if tickers else ""),
            )
        if isinstance(event, ScreeningStarted):
            total = "?" if event.total is None else str(event.total)
            return "screen", f"screening {total} securities"
        if isinstance(event, ScreeningProgress):
            ticker = f" {sanitize_terminal_text(event.ticker)}" if event.ticker else ""
            return "screen", f"{event.completed}/{event.total}{ticker}"
        if isinstance(event, CandidateSelected):
            rank = f"#{event.rank} " if event.rank is not None else ""
            return "candidate", f"{rank}{sanitize_terminal_text(event.ticker)}"
        if isinstance(event, TickerStarted):
            return "analyze", sanitize_terminal_text(event.ticker)
        if isinstance(event, ToolCallStarted):
            return "tool", f"starting {sanitize_terminal_text(event.tool_name)}"
        if isinstance(event, ToolCallCompleted):
            cached = " cached" if event.cached else ""
            latency = f" {event.latency_ms}ms" if event.latency_ms is not None else ""
            retries = f" retries={event.retry_count}" if event.retry_count else ""
            error = (
                f" error={sanitize_terminal_text(event.error_summary)}"
                if event.error_summary
                else ""
            )
            return (
                "tool",
                f"{sanitize_terminal_text(event.tool_name)} {sanitize_terminal_text(event.status)}"
                f"{cached}{latency}{retries}{error}",
            )
        if isinstance(event, LlmCallCompleted):
            cost = f" ${event.cost_usd:.4f}" if self.show_cost else ""
            cache = ""
            if event.cache_read_tokens or event.cache_creation_tokens:
                cache = (
                    f" cache-read={event.cache_read_tokens}"
                    f" cache-write={event.cache_creation_tokens}"
                )
            return "model", f"{event.input_tokens} in / {event.output_tokens} out{cache}{cost}"
        if isinstance(event, AnalysisCompleted):
            recommendation = sanitize_terminal_text(event.recommendation or "unknown")
            confidence = "?" if event.confidence is None else f"{event.confidence:.0%}"
            return (
                "analysis",
                f"{sanitize_terminal_text(event.ticker)} {recommendation} {confidence}",
            )
        if isinstance(event, RunCancelled):
            return "run", f"{sanitize_terminal_text(event.run_id)} cancelled safely"
        if isinstance(event, RunFailed):
            return (
                "run",
                f"{sanitize_terminal_text(event.run_id)} failed: "
                f"{sanitize_terminal_text(event.error_msg or 'unknown error')}",
            )
        if isinstance(event, RunCompleted):
            cost = f", ${event.total_cost_usd:.4f}" if self.show_cost else ""
            duration = (
                f", {event.duration_seconds:.1f}s" if event.duration_seconds is not None else ""
            )
            return (
                "run",
                f"{sanitize_terminal_text(event.run_id)} "
                f"{sanitize_terminal_text(event.status)}{cost}{duration}",
            )

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
        if result.status != "success" and result.error_msg:
            self.diagnostic(result.error_msg, error=result.status == "failed")
        self._render_run_footer(result)

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
            Text("Lynch signals", style="bold"),
            Text("+ " + ("; ".join(lynch_pros) or "none")),
            Text("- " + ("; ".join(lynch_cons) or "none")),
            Text(),
            Text("Buffett signals", style="bold"),
            Text("+ " + ("; ".join(buffett_pros) or "none")),
            Text("- " + ("; ".join(buffett_cons) or "none")),
            Text(),
            Text("Key risks", style="bold"),
        ]
        body.extend(Text(f"• {risk}") for risk in risks)
        if decision is not None:
            body.extend(
                [
                    Text(),
                    Text("DIRT decision", style="bold"),
                    Text(
                        f"{sanitize_terminal_text(decision.outcome).upper()} · "
                        f"weighted IRR {decision.probability_weighted_irr:.1%} · "
                        f"entry {decision.currency} {decision.required_entry_price:,.2f}"
                    ),
                ]
            )
        if quality:
            body.extend([Text(), Text("Data quality", style="bold")])
            body.extend(Text(f"• {note}") for note in quality)
        title = Text(f"{ticker}  {recommendation}  {confidence}")
        self.stdout.print(Panel(Group(*body), title=title, border_style="blue"))

    def render_comparison(self, ticker_results: Sequence[TickerRunResult]) -> None:
        """Render every requested ticker in order, including failed positions."""

        if self.narrow or not _is_tty(self.stdout.file):
            for ticker_result in ticker_results:
                if ticker_result.analysis is not None:
                    self.render_analysis(ticker_result.analysis)
                else:
                    self._render_ticker_error(ticker_result)
            return

        table = Table(title="Comparison", show_lines=True, expand=True)
        table.add_column("Ticker", no_wrap=True)
        table.add_column("Call", no_wrap=True)
        table.add_column("Confidence", justify="right", no_wrap=True)
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
                table.add_row(
                    sanitize_terminal_text(analysis.ticker),
                    sanitize_terminal_text(analysis.recommendation).upper(),
                    f"{analysis.confidence:.0%}",
                    sanitize_terminal_text(analysis.thesis),
                    "\n".join(sanitize_terminal_text(risk) for risk in analysis.key_risks),
                )
        self.stdout.print(table)

    def _render_ticker_error(self, ticker_result: TickerRunResult) -> None:
        ticker = sanitize_terminal_text(ticker_result.ticker)
        message = sanitize_terminal_text(ticker_result.error or "analysis unavailable")
        self.stdout.print(Text(f"{ticker} | ERROR | {message}", style="red"))

    def _render_run_footer(self, result: RunResult) -> None:
        footer = (
            f"Run {sanitize_terminal_text(result.run_id)} | "
            f"{sanitize_terminal_text(result.status)} | "
            f"cost ${result.total_cost_usd:.4f} | "
            f"tokens {result.total_input_tokens} in/{result.total_output_tokens} out | "
            f"tools {result.total_tool_calls} | {result.duration_seconds:.1f}s"
        )
        self.stdout.print(Text(footer, style="dim"))

    def notice(self, message: object) -> None:
        self.stdout.print(Text(sanitize_terminal_text(message)))

    def diagnostic(self, message: object, *, error: bool = False) -> None:
        style = "red" if error else "yellow"
        self.stderr.print(Text(sanitize_terminal_text(message), style=style))


def is_text_stream(value: object) -> bool:
    """Testing/type helper for objects accepted by Rich Console."""

    return isinstance(value, TextIOBase) or hasattr(value, "write")
