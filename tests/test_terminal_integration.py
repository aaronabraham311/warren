"""Public-boundary integration tests for the interactive Warren terminal."""

from __future__ import annotations

import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest

from agent.cancellation import CancellationToken
from agent.models import AnalysisOutput, LynchBuffettSignals
from agent.service import RunRequest, RunResult, TickerRunResult
from agent.terminal.app import create_app
from agent.terminal.renderer import TerminalRenderer
from agent.terminal.settings import TerminalSettings


def _analysis(ticker: str) -> AnalysisOutput:
    return AnalysisOutput(
        ticker=ticker,
        analysis_type="holding",
        recommendation="hold",
        confidence=0.72,
        thesis=f"{ticker} has a sufficiently detailed thesis for an integration test.",
        lynch_signals=LynchBuffettSignals(pros=["growth"], cons=["price"]),
        buffett_signals=LynchBuffettSignals(pros=["moat"], cons=["valuation"]),
        key_risks=["execution"],
    )


class _RecordingExecutor:
    def __init__(self, result: RunResult) -> None:
        self.result = result
        self.requests: list[RunRequest] = []

    def __call__(
        self,
        request: RunRequest,
        *,
        event_sink: TerminalRenderer,
        cancellation: CancellationToken,
    ) -> RunResult:
        del event_sink, cancellation
        self.requests.append(request)
        return self.result


@pytest.mark.skipif(os.name == "nt", reason="PTY lifecycle is POSIX-specific")
def test_real_pty_keeps_model_activity_after_durable_tool_output() -> None:
    import pty
    import re
    import select
    from textwrap import dedent

    child = dedent(
        """
        import sys
        import time
        from agent.events import LlmCallStarted, ToolCallCompleted
        from agent.terminal.renderer import TerminalRenderer

        renderer = TerminalRenderer(
            stdout=sys.stdout,
            stderr=sys.stderr,
            color="auto",
            animation=True,
            width=80,
        )
        with renderer.activity("Preparing analysis…"):
            renderer.emit(
                ToolCallCompleted("run-1", "AMD", "get_quote", "ok", False, 178, 0)
            )
            renderer.emit(
                LlmCallStarted("run-1", "AMD", "sonnet", "synthesis", 4, 7)
            )
            time.sleep(0.3)
        """
    )
    master_fd, slave_fd = pty.openpty()
    environment = os.environ.copy()
    environment.pop("NO_COLOR", None)
    environment["TERM"] = "xterm-256color"
    process = subprocess.Popen(
        [sys.executable, "-c", child],
        cwd=Path(__file__).parents[1],
        env=environment,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)
    chunks: list[bytes] = []
    try:
        while True:
            ready, _, _ = select.select([master_fd], [], [], 2.0)
            if ready:
                try:
                    chunk = os.read(master_fd, 16_384)
                except OSError:
                    break
                if chunk:
                    chunks.append(chunk)
                else:
                    break
            if process.poll() is not None and not ready:
                break
        assert process.wait(timeout=5) == 0
    finally:
        os.close(master_fd)
        if process.poll() is None:
            process.kill()

    transcript = b"".join(chunks).decode(errors="replace")
    plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", transcript)
    assert "✓ Market quote" in plain
    assert "Research pass" in plain
    assert "Synthesizing analysis · AMD · 7 tools complete" in plain
    assert "\x1b[?25l" in transcript
    assert "\x1b[?25h" in transcript


@pytest.mark.skipif(os.name == "nt", reason="PTY lifecycle is POSIX-specific")
def test_real_pty_binder_restores_alternate_screen_and_cursor() -> None:
    import pty
    import select
    import time
    from textwrap import dedent

    child = dedent(
        """
        from agent.terminal.binder import BinderBlock, BinderDocument, BinderPage, ResultBinder

        pages = tuple(
            BinderPage(label, label, (BinderBlock(label, "A long but safe report section."),))
            for label in ("Summary", "Thesis", "Signals", "Risks", "Evidence")
        )
        ResultBinder().run(BinderDocument("AMD", "HOLD", "68%", pages))
        print("prompt-restored")
        """
    )
    master_fd, slave_fd = pty.openpty()
    environment = os.environ.copy()
    environment.pop("NO_COLOR", None)
    environment["TERM"] = "xterm-256color"
    process = subprocess.Popen(
        [sys.executable, "-c", child],
        cwd=Path(__file__).parents[1],
        env=environment,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)
    chunks: list[bytes] = []
    try:
        time.sleep(0.2)
        os.write(master_fd, b"2jq")
        while True:
            ready, _, _ = select.select([master_fd], [], [], 2.0)
            if ready:
                try:
                    chunk = os.read(master_fd, 16_384)
                except OSError:
                    break
                if chunk:
                    chunks.append(chunk)
                else:
                    break
            if process.poll() is not None and not ready:
                break
        assert process.wait(timeout=5) == 0
    finally:
        os.close(master_fd)
        if process.poll() is None:
            process.kill()

    transcript = b"".join(chunks).decode(errors="replace")
    assert "Summary" in transcript
    assert "Thesis" in transcript
    assert "prompt-restored" in transcript
    assert "\x1b[?1049h" in transcript
    assert "\x1b[?1049l" in transcript
    assert "\x1b[?25l" in transcript
    assert "\x1b[?25h" in transcript


def test_piped_compare_preserves_order_settings_partial_failure_and_plain_text() -> None:
    result = RunResult(
        run_id="run-integration",
        status="success",
        ticker_results=(
            TickerRunResult("MSFT", "holding", analysis=_analysis("MSFT")),
            TickerRunResult("AAPL", "holding", error="upstream data unavailable"),
            TickerRunResult("GOOG", "holding", analysis=_analysis("GOOG")),
        ),
        total_cost_usd=0.125,
        total_input_tokens=120,
        total_output_tokens=30,
        total_tool_calls=4,
        duration_seconds=2.5,
        error_msg=None,
    )
    executor = _RecordingExecutor(result)
    stdout = StringIO()
    stderr = StringIO()

    app = create_app(
        stdin=StringIO("Compare MSFT, AAPL, and GOOG\n/quit\n"),
        stdout=stdout,
        stderr=stderr,
        executor=executor,
        settings=TerminalSettings(persona="dirt", max_cost_usd=2.75, color="auto"),
        api_key_available=lambda: True,
        width=60,
    )

    assert app.run() == 0
    assert len(executor.requests) == 1
    request = executor.requests[0]
    assert request.tickers == ["MSFT", "AAPL", "GOOG"]
    assert request.persona == "dirt"
    assert request.max_cost_usd == 2.75

    output = stdout.getvalue()
    diagnostics = stderr.getvalue()
    assert output.index("MSFT | HOLD") < output.index("AAPL | ERROR")
    assert output.index("AAPL | ERROR") < output.index("GOOG | HOLD")
    assert "upstream data unavailable" in output
    assert "Run run-integration | success | cost $0.1250" in output
    assert "warren>" not in output + diagnostics
    assert "\x1b" not in output + diagnostics


def test_missing_api_key_rejects_piped_analysis_without_runtime_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    executor = _RecordingExecutor(
        RunResult(
            run_id="must-not-run",
            status="success",
            ticker_results=(),
            total_cost_usd=0.0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_tool_calls=0,
            duration_seconds=0.0,
            error_msg=None,
        )
    )
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
    assert not (tmp_path / ".warren").exists()
    assert not (tmp_path / "logs").exists()
    assert not (tmp_path / "warren.db").exists()


def test_console_help_from_temp_directory_creates_no_runtime_state(tmp_path: Path) -> None:
    executable = Path(sys.executable).with_name("warren")
    if not executable.exists():
        pytest.skip("the warren console script is not installed in the current environment")
    environment = os.environ.copy()
    environment["WARREN_STATE_DIR"] = str(tmp_path / ".warren")
    environment["WARREN_LOGS_DIR"] = str(tmp_path / "logs" / "runs")
    environment["WARREN_DB"] = str(tmp_path / "warren.db")

    completed = subprocess.run(
        [str(executable), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Interactive Warren stock-analysis terminal" in completed.stdout
    assert not (tmp_path / ".warren").exists()
    assert not (tmp_path / "logs").exists()
    assert not (tmp_path / "warren.db").exists()


def test_console_startup_is_generic_and_hides_migration_details(tmp_path: Path) -> None:
    executable = Path(sys.executable).with_name("warren")
    if not executable.exists():
        pytest.skip("the warren console script is not installed in the current environment")
    environment = os.environ.copy()
    environment["WARREN_STATE_DIR"] = str(tmp_path / ".warren")
    environment["WARREN_LOGS_DIR"] = str(tmp_path / "logs" / "runs")
    environment["WARREN_DB"] = str(tmp_path / "warren.db")
    environment["NO_COLOR"] = "1"

    completed = subprocess.run(
        [str(executable)],
        cwd=Path(__file__).parents[1],
        env=environment,
        input="/quit\n",
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    transcript = completed.stdout + completed.stderr
    assert completed.returncode == 0, transcript
    assert "Starting Warren…" in transcript
    assert "Warren  ·  stock analysis" in transcript
    assert "alembic.runtime.migration" not in transcript
    assert "Running upgrade" not in transcript
