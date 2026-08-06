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
