from __future__ import annotations

import signal
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.cancellation import CancellationToken
from agent.events import RunStarted
from agent.models import AnalysisOutput, LynchBuffettSignals
from agent.requests import RecentContext as ParserRecentContext
from agent.service import RunMode, RunRequest, RunResult, TickerRunResult
from agent.terminal import app as terminal_app
from agent.terminal.app import RunExecutor, create_app
from agent.terminal.completion import WarrenCompleter
from agent.terminal.renderer import TerminalRenderer
from agent.terminal.settings import TerminalSettings


def _analysis(ticker: str = "AAPL") -> AnalysisOutput:
    return AnalysisOutput(
        ticker=ticker,
        analysis_type="holding",
        recommendation="hold",
        confidence=0.7,
        thesis=f"{ticker} has a sufficiently detailed thesis for terminal testing.",
        lynch_signals=LynchBuffettSignals(pros=["growth"], cons=["price"]),
        buffett_signals=LynchBuffettSignals(pros=["moat"], cons=["valuation"]),
        key_risks=["execution"],
    )


def _result(ticker: str = "AAPL", *, status: str = "success") -> RunResult:
    analysis = _analysis(ticker)
    return RunResult(
        run_id="run-app",
        status=status,  # type: ignore[arg-type]
        ticker_results=(TickerRunResult(ticker, "holding", analysis=analysis),),
        total_cost_usd=0.02,
        total_input_tokens=10,
        total_output_tokens=5,
        total_tool_calls=1,
        duration_seconds=0.5,
        error_msg="cancelled" if status == "cancelled" else None,
    )


class _Executor(RunExecutor):
    def __init__(self, result: RunResult | None = None) -> None:
        self.result = result or _result()
        self.requests: list[RunRequest] = []

    def __call__(
        self,
        request: RunRequest,
        *,
        event_sink: TerminalRenderer,
        cancellation: CancellationToken,
    ) -> RunResult:
        del cancellation
        self.requests.append(request)
        event_sink.emit(RunStarted("run-app", request.mode.value, tuple(request.tickers)))
        return self.result


def test_piped_repl_analyze_follow_up_and_quit_uses_settings() -> None:
    stdout = StringIO()
    stderr = StringIO()
    executor = _Executor()
    seen_recent: list[object | None] = []

    def parse(text: str, recent: object | None = None) -> object:
        from agent.requests import parse_request

        seen_recent.append(recent)
        return parse_request(
            text,
            recent=recent if isinstance(recent, ParserRecentContext) else None,
        )

    app = create_app(
        stdin=StringIO("Analyze AAPL\nshow risks\n/quit\n"),
        stdout=stdout,
        stderr=stderr,
        parse_request=parse,
        executor=executor,
        settings=TerminalSettings(persona="dirt", max_cost_usd=2.5, color="never"),
        api_key_available=lambda: True,
    )
    assert app.run() == 0

    assert executor.requests[0].persona == "dirt"
    assert executor.requests[0].max_cost_usd == 2.5
    assert seen_recent[0] is None
    assert isinstance(seen_recent[1], ParserRecentContext)
    assert seen_recent[1].tickers == ("AAPL",)
    assert "Key risks: execution" in stdout.getvalue()
    assert "warren>" not in stdout.getvalue()
    assert "\x1b" not in stdout.getvalue() + stderr.getvalue()


def test_missing_api_key_does_not_start_executor() -> None:
    executor = _Executor()
    stderr = StringIO()
    app = create_app(
        stdin=StringIO("Analyze AAPL\n/quit\n"),
        stdout=StringIO(),
        stderr=stderr,
        executor=executor,
        settings=TerminalSettings(color="never"),
        api_key_available=lambda: False,
    )
    assert app.run() == 0
    assert executor.requests == []
    assert "no run was started" in stderr.getvalue()


def test_help_persona_and_budget_work_and_persist_without_service() -> None:
    saved: list[TerminalSettings] = []
    stdout = StringIO()
    executor = _Executor()
    app = create_app(
        stdin=StringIO("/help\n/persona dirt\n/budget 3.5\n/quit\n"),
        stdout=stdout,
        stderr=StringIO(),
        executor=executor,
        settings=TerminalSettings(color="never"),
        settings_saver=lambda settings: saved.append(settings),
        api_key_available=lambda: True,
    )
    assert app.run() == 0
    assert executor.requests == []
    assert saved[-1].persona == "dirt"
    assert saved[-1].max_cost_usd == 3.5
    assert "Commands:" in stdout.getvalue()


def test_stored_data_and_snapshot_commands_work_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _Executor()
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(
        terminal_app,
        "_list_history",
        lambda ticker: (
            SimpleNamespace(
                run_id="run-old",
                status="success",
                started_at="2026-08-05T10:00:00Z",
                tickers=(ticker or "AAPL",),
                total_cost_usd=0.12,
            ),
        ),
    )
    monkeypatch.setattr(
        terminal_app,
        "_get_run",
        lambda run_id: SimpleNamespace(
            run_id=run_id,
            status="success",
            started_at="start",
            completed_at="end",
            total_cost_usd=0.12,
            total_input_tokens=100,
            total_output_tokens=25,
            num_tool_calls=3,
            analyses=(
                SimpleNamespace(
                    ticker="AAPL",
                    recommendation="hold",
                    confidence=0.75,
                    thesis="Stored thesis",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        terminal_app,
        "_get_trace",
        lambda run_id: SimpleNamespace(
            run_id=run_id or "run-old",
            events=(SimpleNamespace(timestamp="10:00", event="run_started", summary="AAPL"),),
            warnings=(SimpleNamespace(message="ignored torn final line"),),
        ),
    )
    monkeypatch.setattr(
        terminal_app,
        "_list_portfolio",
        lambda: (SimpleNamespace(ticker="AAPL", shares=10, cost_basis=150, current_price=201),),
    )
    monkeypatch.setattr(
        terminal_app,
        "_list_watchlist",
        lambda: (SimpleNamespace(ticker="COST", notes="compounder"),),
    )

    app = create_app(
        stdin=StringIO("/history AAPL\n/show run-old\n/trace\n/portfolio\n/watchlist\n/quit\n"),
        stdout=stdout,
        stderr=stderr,
        executor=executor,
        settings=TerminalSettings(color="never"),
        api_key_available=lambda: False,
    )

    assert app.run() == 0
    assert executor.requests == []
    output = stdout.getvalue()
    assert "run-old | success" in output
    assert "AAPL | HOLD | 75% | Stored thesis" in output
    assert "run_started | AAPL" in output
    assert "AAPL | 10 shares | cost $150.00 | snapshot $201.00" in output
    assert "COST | compounder" in output
    assert "ignored torn final line" in stderr.getvalue()


def test_empty_stored_data_commands_report_clear_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(terminal_app, "_list_history", lambda ticker: ())
    monkeypatch.setattr(terminal_app, "_get_run", lambda run_id: None)
    monkeypatch.setattr(terminal_app, "_get_trace", lambda run_id: None)
    monkeypatch.setattr(terminal_app, "_list_portfolio", lambda: ())
    monkeypatch.setattr(terminal_app, "_list_watchlist", lambda: ())
    stdout = StringIO()
    app = create_app(
        stdin=StringIO("/history\n/show absent\n/trace absent\n/portfolio\n/watchlist\n/quit\n"),
        stdout=stdout,
        stderr=StringIO(),
        settings=TerminalSettings(color="never"),
        api_key_available=lambda: False,
    )

    assert app.run() == 0
    output = stdout.getvalue()
    assert "No stored runs." in output
    assert "Run absent was not found." in output
    assert "No trace found for run absent." in output
    assert "No stored portfolio holdings." in output
    assert "Watchlist is empty." in output


class _InterruptReader:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        del prompt
        self.calls += 1
        raise KeyboardInterrupt


def test_second_prompt_interrupt_exits_after_safe_first_interrupt() -> None:
    stderr = StringIO()
    app = create_app(
        reader=_InterruptReader(),
        stdout=StringIO(),
        stderr=stderr,
        settings=TerminalSettings(color="never"),
    )
    assert app.run() == 130
    assert "Press Ctrl-C again" in stderr.getvalue()


def test_unexpected_exception_does_not_echo_secret() -> None:
    def broken(text: str, recent: object | None = None) -> object:
        del text, recent
        raise RuntimeError("api_key=top-secret")

    stderr = StringIO()
    app = create_app(
        stdin=StringIO("hello\n/quit\n"),
        stdout=StringIO(),
        stderr=stderr,
        parse_request=broken,
        settings=TerminalSettings(color="never"),
    )
    assert app.run() == 0
    assert "top-secret" not in stderr.getvalue()
    assert "Unexpected RuntimeError" in stderr.getvalue()


def test_explicit_settings_saver_can_target_temp_state(tmp_path: Path) -> None:
    saved_paths: list[Path] = []

    def saver(settings: TerminalSettings) -> object:
        from agent.terminal.settings import save_settings

        path = save_settings(settings, tmp_path)
        saved_paths.append(path)
        return path

    app = create_app(
        stdin=StringIO("/persona dirt\n/quit\n"),
        stdout=StringIO(),
        stderr=StringIO(),
        settings=TerminalSettings(color="never"),
        settings_saver=saver,
    )
    assert app.run() == 0
    assert saved_paths == [tmp_path / "settings.json"]


def test_prompt_toolkit_reader_installs_local_completer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Session:
        def prompt(self, prompt: str) -> str:
            return prompt

    def make_session(**kwargs: object) -> Session:
        captured.update(kwargs)
        return Session()

    monkeypatch.setattr(terminal_app, "PromptSession", make_session)
    reader = terminal_app._PromptToolkitReader(None)

    assert reader("warren> ") == "warren> "
    assert isinstance(captured["completer"], WarrenCompleter)


@pytest.mark.parametrize(
    ("command", "candidate_persona", "candidate_budget"),
    [
        ("/persona dirt", "dirt", 1.25),
        ("/budget 3.50", "default", 3.5),
    ],
)
def test_failed_settings_save_does_not_change_in_memory_state(
    command: str,
    candidate_persona: str,
    candidate_budget: float,
) -> None:
    candidates: list[TerminalSettings] = []

    def fail_save(candidate: TerminalSettings) -> object:
        candidates.append(candidate)
        raise OSError("disk unavailable")

    app = create_app(
        stdin=StringIO(f"{command}\n/quit\n"),
        stdout=StringIO(),
        stderr=StringIO(),
        settings=TerminalSettings(persona="default", max_cost_usd=1.25, color="never"),
        settings_saver=fail_save,
    )

    assert app.run() == 0
    assert candidates == [
        TerminalSettings(
            persona=candidate_persona,
            max_cost_usd=candidate_budget,
            color="never",
        )
    ]
    assert app.default_persona == "default"
    assert app.max_cost_usd == 1.25
    assert app.settings.persona == "default"
    assert app.settings.max_cost_usd == 1.25


def test_first_run_sigint_cancels_and_handler_is_always_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_handler = object()
    installed: list[object] = []

    monkeypatch.setattr("agent.terminal.app.signal.getsignal", lambda _sig: previous_handler)

    def set_handler(_sig: int, handler: object) -> object:
        installed.append(handler)
        return previous_handler

    monkeypatch.setattr("agent.terminal.app.signal.signal", set_handler)

    def executor(
        request: RunRequest,
        *,
        event_sink: TerminalRenderer,
        cancellation: CancellationToken,
    ) -> RunResult:
        del request, event_sink
        handler = installed[-1]
        assert callable(handler)
        handler(signal.SIGINT, None)
        assert cancellation.is_cancelled
        return _result(status="cancelled")

    app = create_app(
        stdin=StringIO(),
        stdout=StringIO(),
        stderr=StringIO(),
        executor=executor,
        settings=TerminalSettings(color="never"),
        api_key_available=lambda: True,
    )
    app._execute_service("Analyze AAPL", RunRequest(mode=RunMode.TICKERS, tickers=("AAPL",)))

    assert installed[-1] is previous_handler


def test_second_run_sigint_unwinds_immediately_and_restores_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_handler = object()
    installed: list[object] = []

    monkeypatch.setattr("agent.terminal.app.signal.getsignal", lambda _sig: previous_handler)

    def set_handler(_sig: int, handler: object) -> object:
        installed.append(handler)
        return previous_handler

    monkeypatch.setattr("agent.terminal.app.signal.signal", set_handler)

    def executor(
        request: RunRequest,
        *,
        event_sink: TerminalRenderer,
        cancellation: CancellationToken,
    ) -> RunResult:
        del request, event_sink
        handler = installed[-1]
        assert callable(handler)
        handler(signal.SIGINT, None)
        assert cancellation.is_cancelled
        handler(signal.SIGINT, None)
        raise AssertionError("the second handler invocation must raise")

    app = create_app(
        stdin=StringIO(),
        stdout=StringIO(),
        stderr=StringIO(),
        executor=executor,
        settings=TerminalSettings(color="never"),
        api_key_available=lambda: True,
    )

    with pytest.raises(KeyboardInterrupt):
        app._execute_service("Analyze AAPL", RunRequest(mode=RunMode.TICKERS, tickers=("AAPL",)))
    assert installed[-1] is previous_handler


def test_startup_recovery_uses_configured_log_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[Path] = []
    migrated: list[bool] = []

    def reconcile(log_dir: Path) -> int:
        seen.append(log_dir)
        return 2

    monkeypatch.setenv("WARREN_LOGS_DIR", str(tmp_path / "run-traces"))
    monkeypatch.setattr("storage.engine.migrate", lambda: migrated.append(True))
    monkeypatch.setattr("storage.recovery.reconcile_orphans", reconcile)

    assert terminal_app._recover_interrupted_runs() == 2
    assert migrated == [True]
    assert seen == [tmp_path / "run-traces"]


def test_main_reports_recovered_runs_before_entering_repl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notices: list[object] = []
    fake_app = SimpleNamespace(
        renderer=SimpleNamespace(notice=notices.append, diagnostic=lambda message: None),
        run=lambda: 0,
    )
    monkeypatch.setattr(terminal_app, "create_app", lambda: fake_app)
    monkeypatch.setattr(terminal_app, "_recover_interrupted_runs", lambda: 3)

    assert terminal_app.main() == 0
    assert notices == ["Recovered 3 interrupted run(s)."]
