from __future__ import annotations

from io import StringIO

from agent.events import LlmCallCompleted, RunCompleted, RunFailed, ToolCallCompleted
from agent.models import AnalysisOutput, LynchBuffettSignals
from agent.service import RunResult, TickerRunResult
from agent.terminal.renderer import TerminalRenderer, sanitize_terminal_text


def _analysis(ticker: str, thesis: str | None = None) -> AnalysisOutput:
    return AnalysisOutput(
        ticker=ticker,
        analysis_type="holding",
        recommendation="hold",
        confidence=0.72,
        thesis=thesis or f"{ticker} has a sufficiently detailed durable investment thesis.",
        lynch_signals=LynchBuffettSignals(pros=["growth"], cons=["price"]),
        buffett_signals=LynchBuffettSignals(pros=["moat"], cons=["valuation"]),
        key_risks=["execution"],
    )


def _result(*analyses: AnalysisOutput) -> RunResult:
    return RunResult(
        run_id="run-123",
        status="success",
        ticker_results=tuple(
            TickerRunResult(item.ticker, "holding", analysis=item) for item in analyses
        ),
        total_cost_usd=0.1234,
        total_input_tokens=120,
        total_output_tokens=30,
        total_tool_calls=4,
        duration_seconds=2.5,
        error_msg=None,
    )


def test_non_tty_comparison_preserves_request_order_and_prints_footer() -> None:
    stdout = StringIO()
    stderr = StringIO()
    renderer = TerminalRenderer(stdout=stdout, stderr=stderr, color="auto", width=100)

    renderer.render_result(_result(_analysis("MSFT"), _analysis("AAPL")), compare=True)

    text = stdout.getvalue()
    assert text.index("MSFT | HOLD") < text.index("AAPL | HOLD")
    assert "Lynch: + growth | - price" in text
    assert "Buffett: + moat | - valuation" in text
    assert "Run run-123 | success | cost $0.1234" in text
    assert "tokens 120 in/30 out | tools 4 | 2.5s" in text
    assert "\x1b" not in text
    assert stderr.getvalue() == ""


def test_partial_comparison_preserves_failed_ticker_position() -> None:
    stdout = StringIO()
    stderr = StringIO()
    renderer = TerminalRenderer(stdout=stdout, stderr=stderr, color="never", width=100)
    result = RunResult(
        run_id="partial-run",
        status="success",
        ticker_results=(
            TickerRunResult("MSFT", "holding", analysis=_analysis("MSFT")),
            TickerRunResult("AAPL", "holding", error="upstream unavailable"),
            TickerRunResult("GOOG", "holding", analysis=_analysis("GOOG")),
        ),
        total_cost_usd=0.1,
        total_input_tokens=10,
        total_output_tokens=5,
        total_tool_calls=2,
        duration_seconds=1.0,
        error_msg=None,
    )

    renderer.render_result(result, compare=True)

    text = stdout.getvalue()
    assert text.index("MSFT | HOLD") < text.index("AAPL | ERROR")
    assert text.index("AAPL | ERROR") < text.index("GOOG | HOLD")
    assert "upstream unavailable" in text
    assert stderr.getvalue() == ""


def test_control_sequences_and_secret_values_are_inert() -> None:
    unsafe = "bad\x1b[31m\u202esecret api_key=abc123 sk-ant-private"
    safe = sanitize_terminal_text(unsafe)
    assert "\x1b" not in safe
    assert "\u202e" not in safe
    assert "abc123" not in safe
    assert "sk-ant-private" not in safe
    assert safe.count("[redacted]") == 2


def test_progress_diagnostics_include_cache_retry_error_duration_and_run_id() -> None:
    stdout = StringIO()
    stderr = StringIO()
    renderer = TerminalRenderer(stdout=stdout, stderr=stderr, color="never")
    renderer.emit(
        LlmCallCompleted(
            "run-1",
            "AAPL",
            "model",
            9,
            100,
            20,
            0.01,
            cache_read_tokens=70,
            cache_creation_tokens=5,
        )
    )
    renderer.emit(
        ToolCallCompleted(
            "run-1",
            "AAPL",
            "get_quote",
            "error",
            False,
            11,
            2,
            error_summary="token=secret-value failed",
        )
    )
    renderer.emit(RunCompleted("run-1", "success", 0.01, 1.25))

    diagnostic = stderr.getvalue()
    assert "cache-read=70 cache-write=5" in diagnostic
    assert "retries=2" in diagnostic
    assert "secret-value" not in diagnostic
    assert "run-1 success, $0.0100, 1.2s" in diagnostic
    assert stdout.getvalue() == ""


def test_no_color_overrides_forced_color(monkeypatch: object) -> None:
    from pytest import MonkeyPatch

    assert isinstance(monkeypatch, MonkeyPatch)
    monkeypatch.setenv("NO_COLOR", "1")
    stderr = StringIO()
    renderer = TerminalRenderer(stdout=StringIO(), stderr=stderr, color="always")
    renderer.emit(RunFailed("r", "boom"))
    assert "\x1b" not in stderr.getvalue()
