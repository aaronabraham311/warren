"""Prompt-toolkit REPL for Warren's shared run service."""

from __future__ import annotations

import os
import signal
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Iterator, Protocol, TextIO, cast

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from agent.cancellation import CancellationToken
from agent.chat import RecentContext
from agent.models import AnalysisOutput
from agent.requests import FollowUpKind, RunnableRequest, StoredResultFollowUp
from agent.service import RunMode, RunRequest, RunResult, TickerRunResult, execute_run
from agent.terminal.commands import (
    BudgetCommand,
    CommandError,
    DiscoverCommand,
    GemHuntCommand,
    HelpCommand,
    HistoryCommand,
    PersonaCommand,
    PortfolioCommand,
    ShowCommand,
    ToolsCommand,
    TraceCommand,
    WatchlistCommand,
)
from agent.terminal.completion import WarrenCompleter
from agent.terminal.renderer import NAVY_BRAND, NAVY_MUTED, ColorMode, TerminalRenderer
from agent.terminal.settings import TerminalSettings, save_settings

if TYPE_CHECKING:
    from agent.terminal.queries import (
        HistoryEntry,
        PortfolioEntry,
        RunDetail,
        TraceResult,
        WatchlistEntry,
    )


class RequestParser(Protocol):
    def __call__(self, text: str, recent: object | None = None) -> object: ...


class RunExecutor(Protocol):
    def __call__(
        self,
        request: RunRequest,
        *,
        event_sink: TerminalRenderer,
        cancellation: CancellationToken,
    ) -> RunResult: ...


class PromptReader(Protocol):
    def __call__(self, prompt: str) -> str: ...


class CommandParser(Protocol):
    def __call__(self, text: str) -> object: ...


class CommandHandler(Protocol):
    def __call__(self, command: object, app: TerminalApp) -> bool: ...


class _PromptToolkitReader:
    def __init__(self, history_file: Path | None) -> None:
        history = FileHistory(str(history_file)) if history_file is not None else None
        no_color = bool(os.environ.get("NO_COLOR"))
        self._session: PromptSession[str] = PromptSession(
            history=history,
            completer=WarrenCompleter(),
            style=Style.from_dict(
                {
                    "prompt.brand": "bold" if no_color else f"{NAVY_MUTED} bold",
                    "prompt.chevron": "bold" if no_color else f"{NAVY_BRAND} bold",
                }
            ),
        )

    def __call__(self, prompt: str) -> str:
        del prompt
        return self._session.prompt(
            FormattedText(
                [
                    ("class:prompt.brand", "warren "),
                    ("class:prompt.chevron", "› "),
                ]
            )
        )


class _StreamReader:
    """Stable line reader for piped input; emits no interactive prompt bytes."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def __call__(self, prompt: str) -> str:
        del prompt
        line = self._stream.readline()
        if line == "":
            raise EOFError
        return line.rstrip("\r\n")


def _default_parse_request(text: str, recent: object | None = None) -> object:
    from agent.requests import RecentContext as ParserRecentContext
    from agent.requests import parse_request

    parser_context = recent if isinstance(recent, ParserRecentContext) else None
    return parse_request(text, recent=parser_context)


def _default_parse_command(text: str) -> object:
    from agent.terminal.commands import parse_command

    return parse_command(text)


def _call_zero_arg(value: object, name: str) -> object | None:
    method = getattr(value, name, None)
    if not callable(method):
        return None
    return cast(Callable[[], object], method)()


def _to_run_request(
    parsed: object,
    *,
    max_cost_usd: float,
    default_persona: str,
) -> RunRequest | None:
    if isinstance(parsed, RunRequest):
        return parsed
    if isinstance(parsed, RunnableRequest):
        persona = "dirt" if default_persona == "dirt" else "default"
        return parsed.to_run_request(max_cost_usd=max_cost_usd, default_persona=persona)
    for method_name in ("to_run_request", "to_service_request"):
        converted = _call_zero_arg(parsed, method_name)
        if isinstance(converted, RunRequest):
            return converted
    for attribute in ("run_request", "service_request", "request"):
        converted = getattr(parsed, attribute, None)
        if isinstance(converted, RunRequest):
            return converted
    return None


def _message_from(parsed: object) -> str:
    for attribute in ("message", "prompt", "reason", "detail"):
        value = getattr(parsed, attribute, None)
        if isinstance(value, str) and value:
            return value
    return type(parsed).__name__


def _analysis_list(parsed: object) -> list[AnalysisOutput] | None:
    value = getattr(parsed, "analyses", None)
    if isinstance(value, (list, tuple)) and all(isinstance(item, AnalysisOutput) for item in value):
        return list(value)
    analysis = getattr(parsed, "analysis", None)
    if isinstance(analysis, AnalysisOutput):
        return [analysis]
    return None


def _list_history(ticker: str | None) -> tuple[HistoryEntry, ...]:
    from agent.terminal.queries import list_history

    return tuple(list_history(ticker))


def _get_run(run_id: str) -> RunDetail | None:
    from agent.terminal.queries import get_run

    return get_run(run_id)


def _get_trace(run_id: str | None) -> TraceResult | None:
    from agent.terminal.queries import get_trace

    return get_trace(run_id)


def _list_portfolio() -> tuple[PortfolioEntry, ...]:
    from agent.terminal.queries import list_portfolio

    return tuple(list_portfolio())


def _list_watchlist() -> tuple[WatchlistEntry, ...]:
    from agent.terminal.queries import list_watchlist

    return tuple(list_watchlist())


def _display(value: object | None, fallback: str = "-") -> str:
    return fallback if value is None else str(value)


def _joined(values: object, fallback: str = "-") -> str:
    if isinstance(values, (list, tuple)):
        rendered = ", ".join(str(item) for item in values)
        return rendered or fallback
    return fallback


@contextmanager
def _cancel_on_sigint(
    token: CancellationToken,
    on_cancel: Callable[[], None] | None = None,
) -> Iterator[None]:
    """Turn the first SIGINT into cancellation and let the second interrupt unwind."""

    previous_handler = signal.getsignal(signal.SIGINT)
    interrupt_count = 0

    def handle_sigint(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        nonlocal interrupt_count
        interrupt_count += 1
        if interrupt_count == 1:
            token.cancel()
            if on_cancel is not None:
                on_cancel()
            return
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_handler)


class TerminalApp:
    """Stateful REPL shell with injectable parsing, execution, and input."""

    def __init__(
        self,
        *,
        renderer: TerminalRenderer,
        reader: PromptReader,
        parse_request: RequestParser = _default_parse_request,
        executor: RunExecutor = execute_run,
        parse_command: CommandParser | None = None,
        command_handler: CommandHandler | None = None,
        recent: RecentContext | None = None,
        default_persona: str = "default",
        max_cost_usd: float = 1.25,
        settings: TerminalSettings | None = None,
        settings_saver: Callable[[TerminalSettings], object] = save_settings,
        api_key_available: Callable[[], bool] | None = None,
    ) -> None:
        self.renderer = renderer
        self.reader = reader
        self.parse_request = parse_request
        self.executor = executor
        self.parse_command = parse_command
        self.command_handler = command_handler
        self.recent = recent if recent is not None else RecentContext()
        self.default_persona = default_persona
        self.max_cost_usd = max_cost_usd
        self.settings = settings or TerminalSettings(
            persona="dirt" if default_persona == "dirt" else "default",
            max_cost_usd=max_cost_usd,
        )
        self.settings_saver = settings_saver
        self.api_key_available = api_key_available or (
            lambda: bool(os.environ.get("ANTHROPIC_API_KEY"))
        )
        self._interrupt_armed = False

    def run(self) -> int:
        self.renderer.welcome()
        while True:
            try:
                raw = self.reader("warren> ")
            except EOFError:
                return 0
            except KeyboardInterrupt:
                if self._interrupt_armed:
                    return 130
                self._interrupt_armed = True
                self.renderer.diagnostic("Interrupted. Press Ctrl-C again to exit.")
                continue

            text = raw.strip()
            if not text:
                continue
            if text.lower() in {"/quit", "/exit"}:
                return 0
            if text.lower() == "/new":
                self.recent.clear()
                self.renderer.notice("Started a new context.")
                self._interrupt_armed = False
                continue

            try:
                if text.startswith("/"):
                    self._handle_command(text)
                else:
                    self._handle_request(text)
                self._interrupt_armed = False
            except KeyboardInterrupt:
                # The service normally converts the first interrupt into a cancelled
                # result. This fallback covers injected/legacy executors.
                self._interrupt_armed = True
                self.renderer.diagnostic("Run interrupted safely. Press Ctrl-C again to exit.")
            except Exception as exc:
                self.renderer.diagnostic(
                    f"Unexpected {type(exc).__name__}; the operation was not completed.",
                    error=True,
                )

    def _handle_command(self, text: str) -> None:
        parser = self.parse_command if self.parse_command is not None else _default_parse_command
        command = parser(text)
        if self.command_handler is None:
            self._handle_builtin_command(command)
            return
        should_exit = self.command_handler(command, self)
        if should_exit:
            raise EOFError

    def _handle_builtin_command(self, command: object) -> None:
        if isinstance(command, CommandError):
            message = command.message
            if command.usage:
                message += f" Usage: {command.usage}"
            self.renderer.diagnostic(message)
        elif isinstance(command, HelpCommand):
            self.renderer.render_help()
        elif isinstance(command, PersonaCommand):
            if command.persona is None:
                self.renderer.notice(f"Persona: {self.default_persona}")
            else:
                candidate = self.settings.model_copy(update={"persona": command.persona})
                self.settings_saver(candidate)
                self.settings = candidate
                self.default_persona = command.persona
                self.renderer.notice(f"Persona set to {command.persona}.")
        elif isinstance(command, BudgetCommand):
            if command.max_cost_usd is None:
                self.renderer.notice(f"Budget: ${self.max_cost_usd:.2f}")
            else:
                candidate = self.settings.model_copy(update={"max_cost_usd": command.max_cost_usd})
                self.settings_saver(candidate)
                self.settings = candidate
                self.max_cost_usd = command.max_cost_usd
                self.renderer.notice(f"Budget set to ${command.max_cost_usd:.2f}.")
        elif isinstance(command, HistoryCommand):
            self._render_history(command.ticker)
        elif isinstance(command, ShowCommand):
            self._render_stored_run(command.run_id)
        elif isinstance(command, TraceCommand):
            self._render_trace(command.run_id)
        elif isinstance(command, PortfolioCommand):
            self._render_portfolio()
        elif isinstance(command, WatchlistCommand):
            self._render_watchlist()
        elif isinstance(command, DiscoverCommand):
            self._execute_service(
                "/discover",
                RunRequest(
                    mode=RunMode.DISCOVERY,
                    persona="dirt" if self.default_persona == "dirt" else "default",
                    max_cost_usd=self.max_cost_usd,
                ),
            )
        elif isinstance(command, GemHuntCommand):
            self._execute_service(
                "/gem-hunt",
                RunRequest(mode=RunMode.GEM_HUNT, persona="dirt", max_cost_usd=self.max_cost_usd),
            )
        elif isinstance(command, ToolsCommand):
            from agent.tools import TOOL_REGISTRY

            categories = {
                "Market & fundamentals": (
                    "get_quote",
                    "get_fundamentals",
                    "get_growth_metrics",
                    "get_news",
                    "read_filing",
                ),
                "Valuation & quality": (
                    "get_valuation_multiples",
                    "get_valuation_history",
                    "get_quality_metrics",
                    "get_financial_strength",
                    "estimate_intrinsic_value",
                    "model_dirt_scenarios",
                ),
                "Management & competition": (
                    "get_capital_allocation",
                    "get_key_persons",
                    "get_insider_activity",
                    "get_peer_comparison",
                ),
                "Risk & evidence": (
                    "get_adverse_media",
                    "get_forensic_evidence",
                    "screen_watchlists",
                ),
                "Portfolio & discovery": ("get_holding_context", "screen_universe"),
            }
            for category, names in categories.items():
                available = [name for name in names if name in TOOL_REGISTRY]
                if available:
                    self.renderer.notice(f"{category}: " + ", ".join(available))
        else:
            self.renderer.diagnostic(f"Unsupported command: {type(command).__name__}.")

    def _render_history(self, ticker: str | None) -> None:
        entries = _list_history(ticker)
        if not entries:
            qualifier = f" for {ticker}" if ticker else ""
            self.renderer.notice(f"No stored runs{qualifier}.")
            return
        self.renderer.notice("Recent runs:")
        for entry in entries:
            tickers = _joined(getattr(entry, "tickers", ()))
            started = _display(getattr(entry, "started_at", None))
            cost = getattr(entry, "total_cost_usd", None)
            cost_text = "-" if cost is None else f"${float(cost):.4f}"
            self.renderer.notice(
                f"{_display(getattr(entry, 'run_id', None))} | "
                f"{_display(getattr(entry, 'status', None))} | {started} | "
                f"{tickers} | {cost_text}"
            )

    def _render_stored_run(self, run_id: str) -> None:
        detail = _get_run(run_id)
        if detail is None:
            self.renderer.notice(f"Run {run_id} was not found.")
            return
        self.renderer.notice(
            f"Run {_display(getattr(detail, 'run_id', run_id))} | "
            f"{_display(getattr(detail, 'status', None))} | "
            f"started {_display(getattr(detail, 'started_at', None))} | "
            f"completed {_display(getattr(detail, 'completed_at', None))}"
        )
        self.renderer.notice(
            "Usage: "
            f"${float(getattr(detail, 'total_cost_usd', 0.0) or 0.0):.4f} | "
            f"{int(getattr(detail, 'total_input_tokens', 0) or 0)} in/"
            f"{int(getattr(detail, 'total_output_tokens', 0) or 0)} out | "
            f"{int(getattr(detail, 'num_tool_calls', 0) or 0)} tools"
        )
        error_msg = getattr(detail, "error_msg", None)
        if error_msg:
            self.renderer.diagnostic(f"Run error: {error_msg}", error=True)
        ticker_results = getattr(detail, "ticker_results", ())
        if isinstance(ticker_results, (list, tuple)) and ticker_results:
            for ticker_result in ticker_results:
                analysis = getattr(ticker_result, "analysis", None)
                if analysis is None:
                    self.renderer.notice(
                        f"{_display(getattr(ticker_result, 'ticker', None))} | ERROR | "
                        f"{_display(getattr(ticker_result, 'error', None), 'analysis unavailable')}"
                    )
                else:
                    self._render_stored_analysis(analysis)
        else:
            analyses = getattr(detail, "analyses", ())
            if isinstance(analyses, (list, tuple)):
                for analysis in analyses:
                    self._render_stored_analysis(analysis)

    def _render_stored_analysis(self, analysis: object) -> None:
        recommendation = _display(getattr(analysis, "recommendation", None)).upper()
        confidence = getattr(analysis, "confidence", None)
        confidence_text = "-" if confidence is None else f"{float(confidence):.0%}"
        self.renderer.notice(
            f"{_display(getattr(analysis, 'ticker', None))} | "
            f"{recommendation} | {confidence_text} | "
            f"{_display(getattr(analysis, 'thesis', None))}"
        )
        termination = getattr(analysis, "termination_reason", None)
        if termination and termination != "success":
            self.renderer.notice(f"Termination: {_display(termination)}")
        for label, attribute in (("Lynch", "lynch_signals"), ("Buffett", "buffett_signals")):
            signals = getattr(analysis, attribute, {})
            if isinstance(signals, dict):
                self.renderer.notice(
                    f"{label}: + "
                    + _joined(signals.get("pros", ()), "none")
                    + " | - "
                    + _joined(signals.get("cons", ()), "none")
                )
        self.renderer.notice("Risks: " + _joined(getattr(analysis, "key_risks", ()), "none"))
        quality = getattr(analysis, "data_quality_notes", ())
        if quality:
            self.renderer.notice("Data quality: " + _joined(quality))
        dirt_decision = getattr(analysis, "dirt_decision", None)
        if isinstance(dirt_decision, dict):
            outcome = _display(dirt_decision.get("outcome"), "unknown").upper()
            weighted_irr = dirt_decision.get("probability_weighted_irr")
            irr_text = (
                "-" if not isinstance(weighted_irr, (int, float)) else f"{float(weighted_irr):.1%}"
            )
            self.renderer.notice(f"DIRT: {outcome} | weighted IRR {irr_text}")

    def _render_trace(self, run_id: str | None) -> None:
        trace = _get_trace(run_id)
        if trace is None:
            label = f" {run_id}" if run_id else ""
            self.renderer.notice(f"No trace found for run{label}.")
            return
        self.renderer.notice(f"Trace {_display(getattr(trace, 'run_id', run_id))}:")
        events = getattr(trace, "events", ())
        if isinstance(events, (list, tuple)):
            for event in events:
                timestamp = _display(getattr(event, "timestamp", None))
                event_name = _display(getattr(event, "event", None))
                summary = _display(getattr(event, "summary", None), "")
                ticker = _display(getattr(event, "ticker", None), "")
                details = " | ".join(part for part in (ticker, summary) if part)
                self.renderer.notice(
                    f"{timestamp} | {event_name}" + (f" | {details}" if details else "")
                )
        warnings = getattr(trace, "warnings", ())
        if isinstance(warnings, (list, tuple)):
            for warning in warnings:
                self.renderer.diagnostic(
                    "Trace warning: " + _display(getattr(warning, "message", warning))
                )

    def _render_portfolio(self) -> None:
        holdings = _list_portfolio()
        if not holdings:
            self.renderer.notice("No stored portfolio holdings.")
            return
        self.renderer.notice("Portfolio snapshot:")
        for holding in holdings:
            price = getattr(holding, "current_price", None)
            price_text = "-" if price is None else f"${float(price):,.2f}"
            self.renderer.notice(
                f"{_display(getattr(holding, 'ticker', None))} | "
                f"{float(getattr(holding, 'shares', 0.0) or 0.0):g} shares | "
                f"cost ${float(getattr(holding, 'cost_basis', 0.0) or 0.0):,.2f} | "
                f"snapshot {price_text}"
            )

    def _render_watchlist(self) -> None:
        entries = _list_watchlist()
        if not entries:
            self.renderer.notice("Watchlist is empty.")
            return
        self.renderer.notice("Watchlist:")
        for entry in entries:
            notes = _display(getattr(entry, "notes", None), "")
            suffix = f" | {notes}" if notes else ""
            self.renderer.notice(f"{_display(getattr(entry, 'ticker', None))}{suffix}")

    def _handle_request(self, text: str) -> None:
        parsed = self.parse_request(text, recent=self.recent.parser_context())
        request = _to_run_request(
            parsed,
            max_cost_usd=self.max_cost_usd,
            default_persona=self.default_persona,
        )
        if request is not None:
            self._execute_service(text, request, compare="compare" in text.casefold())
            return

        if isinstance(parsed, StoredResultFollowUp):
            self._render_follow_up(parsed)
            return

        analyses = _analysis_list(parsed)
        if analyses is not None:
            if len(analyses) > 1:
                self.renderer.render_comparison(
                    [
                        TickerRunResult(
                            ticker=analysis.ticker,
                            analysis_type=analysis.analysis_type,
                            analysis=analysis,
                        )
                        for analysis in analyses
                    ]
                )
            elif analyses:
                self.renderer.render_analysis(analyses[0])
            self.recent.record(text, tickers=tuple(item.ticker for item in analyses))
            return

        kind = type(parsed).__name__
        message = _message_from(parsed)
        if kind == "Clarification":
            self.renderer.notice(message)
        elif kind in {"Unsupported", "StoredResultFollowUp"}:
            self.renderer.diagnostic(message)
        else:
            self.renderer.diagnostic(f"Unsupported request: {message}")

    def _execute_service(
        self,
        text: str,
        request: RunRequest,
        *,
        compare: bool = False,
    ) -> None:
        if not self.api_key_available():
            self.renderer.diagnostic(
                "ANTHROPIC_API_KEY is not configured; no run was started.",
                error=True,
            )
            return
        token = CancellationToken()
        with self.renderer.activity("Preparing analysis…"):
            with _cancel_on_sigint(
                token,
                on_cancel=self.renderer.show_stopping,
            ):
                result = self.executor(
                    request,
                    event_sink=self.renderer,
                    cancellation=token,
                )
        self.renderer.render_result(
            result,
            compare=compare or len(request.tickers) > 1,
        )
        self.recent.record_result(text, result)
        if result.status == "cancelled":
            self._interrupt_armed = True

    def _render_follow_up(self, follow_up: StoredResultFollowUp) -> None:
        analysis = self.recent.select_ticker(follow_up.ticker)
        if analysis is None:
            self.renderer.diagnostic("No completed analysis is available for that ticker.")
            return
        if follow_up.kind is FollowUpKind.SELECT_TICKER:
            self.renderer.render_analysis(analysis)
        elif follow_up.kind is FollowUpKind.RISKS:
            self.renderer.notice("Key risks: " + "; ".join(analysis.key_risks))
        elif follow_up.kind is FollowUpKind.DATA_QUALITY:
            notes = analysis.data_quality_notes or ["No data-quality notes."]
            self.renderer.notice("Data quality: " + "; ".join(notes))
        elif follow_up.kind is FollowUpKind.LYNCH:
            self.renderer.notice(
                "Lynch pros: "
                + "; ".join(analysis.lynch_signals.pros)
                + "\nLynch cons: "
                + "; ".join(analysis.lynch_signals.cons)
            )
        elif follow_up.kind is FollowUpKind.BUFFETT:
            self.renderer.notice(
                "Buffett pros: "
                + "; ".join(analysis.buffett_signals.pros)
                + "\nBuffett cons: "
                + "; ".join(analysis.buffett_signals.cons)
            )
        elif follow_up.kind is FollowUpKind.EVIDENCE:
            evidence = analysis.dirt_signals
            refs = [] if evidence is None else evidence.forensic_evidence_ids
            tools = self.renderer.evidence_for(analysis.ticker)
            parts: list[str] = []
            if tools:
                parts.append("tools: " + ", ".join(tools))
            if refs:
                parts.append("references: " + ", ".join(refs))
            self.renderer.notice(
                "Evidence: " + (" | ".join(parts) if parts else "see the run trace")
            )
        elif follow_up.kind is FollowUpKind.WHY:
            self.renderer.notice(analysis.thesis)


def _setting(settings: object, name: str, default: object) -> object:
    value = getattr(settings, name, default)
    return default if value is None else value


def _float_setting(settings: object, name: str, default: float) -> float:
    value = _setting(settings, name, default)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _load_terminal_settings() -> tuple[object | None, Path | None]:
    try:
        from agent.terminal.settings import history_path, load_settings

        settings = load_settings()
        return settings, history_path()
    except (ImportError, OSError, ValueError):
        return None, None


def create_app(
    *,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    reader: PromptReader | None = None,
    parse_request: RequestParser = _default_parse_request,
    executor: RunExecutor = execute_run,
    parse_command: CommandParser | None = None,
    command_handler: CommandHandler | None = None,
    settings: object | None = None,
    settings_saver: Callable[[TerminalSettings], object] = save_settings,
    api_key_available: Callable[[], bool] | None = None,
    width: int | None = None,
) -> TerminalApp:
    loaded_settings, history_file = (
        _load_terminal_settings() if settings is None else (settings, None)
    )
    active_settings = loaded_settings if settings is None else settings
    color_value = str(_setting(active_settings, "color", "auto"))
    color: ColorMode = (
        cast(ColorMode, color_value) if color_value in {"auto", "always", "never"} else "auto"
    )
    renderer = TerminalRenderer(
        stdout=stdout,
        stderr=stderr,
        color=color,
        animation=bool(_setting(active_settings, "animation", True)),
        show_cost=bool(_setting(active_settings, "show_cost", True)),
        width=width,
    )
    prompt_reader = reader
    if prompt_reader is None:
        prompt_reader = (
            _PromptToolkitReader(history_file)
            if getattr(stdin, "isatty", lambda: False)()
            else _StreamReader(stdin)
        )
    typed_settings = active_settings if isinstance(active_settings, TerminalSettings) else None
    return TerminalApp(
        renderer=renderer,
        reader=prompt_reader,
        parse_request=parse_request,
        executor=executor,
        parse_command=parse_command,
        command_handler=command_handler,
        default_persona=str(_setting(active_settings, "persona", "default")),
        max_cost_usd=_float_setting(active_settings, "max_cost_usd", 1.25),
        settings=typed_settings,
        settings_saver=settings_saver,
        api_key_available=api_key_available,
    )


def _recover_interrupted_runs() -> int:
    from storage.engine import migrate
    from storage.recovery import reconcile_orphans

    migrate(quiet=True)
    log_dir = Path(os.environ.get("WARREN_LOGS_DIR", "logs/runs"))
    return reconcile_orphans(log_dir)


def main() -> int:
    app = create_app()
    try:
        with app.renderer.activity("Starting Warren…", announce=True):
            recovered = _recover_interrupted_runs()
    except Exception:
        # A missing/unmigrated database must not prevent offline help and local
        # read-only commands from starting. Service runs migrate before writing.
        app.renderer.diagnostic("Startup recovery scan was unavailable.")
    else:
        if recovered:
            app.renderer.notice(f"Recovered {recovered} interrupted run(s).")
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
