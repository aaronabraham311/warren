"""Prompt-toolkit REPL for Warren's shared run service."""

from __future__ import annotations

import os
import signal
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Iterator, Protocol, TextIO, cast

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

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
    PersonaCommand,
    PortfolioCommand,
    ToolsCommand,
)
from agent.terminal.completion import WarrenCompleter
from agent.terminal.renderer import ColorMode, TerminalRenderer
from agent.terminal.settings import TerminalSettings, save_settings


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
        self._session: PromptSession[str] = PromptSession(
            history=history,
            completer=WarrenCompleter(),
        )

    def __call__(self, prompt: str) -> str:
        return self._session.prompt(prompt)


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


@contextmanager
def _cancel_on_sigint(token: CancellationToken) -> Iterator[None]:
    """Turn the first SIGINT into cancellation and let the second interrupt unwind."""

    previous_handler = signal.getsignal(signal.SIGINT)
    interrupt_count = 0

    def handle_sigint(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        nonlocal interrupt_count
        interrupt_count += 1
        if interrupt_count == 1:
            token.cancel()
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
        self.renderer.notice("Warren interactive terminal. Type /quit to exit.")
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
            self.renderer.notice(
                "Commands: /help /new /history [ticker] /show RUN_ID /trace [RUN_ID] "
                "/portfolio /watchlist /discover /gem-hunt /persona [default|dirt] "
                "/budget [USD] /tools /quit"
            )
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
        elif isinstance(command, PortfolioCommand):
            self._execute_service(
                "/portfolio",
                RunRequest(
                    mode=RunMode.PORTFOLIO,
                    persona="dirt" if self.default_persona == "dirt" else "default",
                    max_cost_usd=self.max_cost_usd,
                ),
            )
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

            self.renderer.notice("Agent tools: " + ", ".join(sorted(TOOL_REGISTRY)))
        else:
            self.renderer.diagnostic(
                f"{type(command).__name__} will be available with the stored-data commands."
            )

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
        with self.renderer:
            with _cancel_on_sigint(token):
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
            self.renderer.notice(
                "Evidence references: " + (", ".join(refs) if refs else "see the run trace")
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


def main() -> int:
    return create_app().run()


if __name__ == "__main__":
    raise SystemExit(main())
