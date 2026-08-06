from __future__ import annotations

from io import StringIO

from agent.events import (
    LlmCallCompleted,
    RunCompleted,
    RunFailed,
    RunStarted,
    ToolCallCompleted,
    ToolCallStarted,
)
from agent.models import AnalysisOutput, LynchBuffettSignals
from agent.service import RunResult, TickerRunResult
from agent.terminal.renderer import TerminalRenderer, sanitize_terminal_text


class _TTYBuffer(StringIO):
    def isatty(self) -> bool:
        return True


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
    assert "2 retries" in diagnostic
    assert "✗ Market quote  ·  11ms" in diagnostic
    assert "secret-value" not in diagnostic
    assert "run-1 success, $0.0100, 1.2s" in diagnostic
    assert stdout.getvalue() == ""


def test_plain_activity_is_immediate_and_completed_tools_persist_once() -> None:
    stderr = StringIO()
    renderer = TerminalRenderer(stdout=StringIO(), stderr=stderr, color="never")

    with renderer.activity("Preparing analysis…"):
        assert stderr.getvalue() == "Preparing analysis…\n"
        renderer.emit(ToolCallStarted("run-1", "AAPL", "read_filing"))
        renderer.emit(ToolCallCompleted("run-1", "AAPL", "read_filing", "success", False, 1420, 0))

    transcript = stderr.getvalue()
    assert "Using: Regulatory filing" in transcript
    assert transcript.count("✓ Regulatory filing  ·  1.4s") == 1
    assert "Analyzing AAPL…" in transcript


def test_forced_color_uses_the_shared_navy_theme(monkeypatch: object) -> None:
    from pytest import MonkeyPatch

    assert isinstance(monkeypatch, MonkeyPatch)
    monkeypatch.delenv("NO_COLOR", raising=False)
    stdout = StringIO()
    renderer = TerminalRenderer(stdout=stdout, stderr=StringIO(), color="always")

    renderer.welcome()

    rendered = stdout.getvalue()
    assert "Warren" in rendered
    assert "\x1b[" in rendered
    assert "/help" in rendered


def test_tty_activity_always_stops_live_renderer(monkeypatch: object) -> None:
    from pytest import MonkeyPatch

    assert isinstance(monkeypatch, MonkeyPatch)
    monkeypatch.delenv("NO_COLOR", raising=False)
    stderr = _TTYBuffer()
    renderer = TerminalRenderer(
        stdout=_TTYBuffer(),
        stderr=stderr,
        color="always",
        animation=True,
    )

    with renderer.activity("Preparing analysis…"):
        assert bool(renderer._live)
        renderer.update_activity("Using Market quote")

    assert renderer._live is None
    assert renderer._progress is None
    assert "Preparing analysis" not in stderr.getvalue()


def test_renderer_builds_safe_latest_run_evidence_index() -> None:
    renderer = TerminalRenderer(stdout=StringIO(), stderr=StringIO(), color="never")
    renderer.emit(RunStarted("run-1", "tickers", ("AAPL",)))
    renderer.emit(ToolCallCompleted("run-1", "AAPL", "get_quote", "success", False, 5, 0))
    renderer.emit(ToolCallCompleted("run-1", "AAPL", "get_quote", "success", True, 1, 0))
    renderer.emit(ToolCallCompleted("run-1", "AAPL", "get_news", "error", False, 5, 0))

    assert renderer.evidence_for("AAPL") == ("get_quote",)
    renderer.emit(RunStarted("run-2", "tickers", ("MSFT",)))
    assert renderer.evidence_for("AAPL") == ()


def test_no_color_overrides_forced_color(monkeypatch: object) -> None:
    from pytest import MonkeyPatch

    assert isinstance(monkeypatch, MonkeyPatch)
    monkeypatch.setenv("NO_COLOR", "1")
    stderr = StringIO()
    renderer = TerminalRenderer(stdout=StringIO(), stderr=stderr, color="always")
    renderer.emit(RunFailed("r", "boom"))
    assert "\x1b" not in stderr.getvalue()


def test_non_normal_termination_reason_is_visible() -> None:
    stdout = StringIO()
    analysis = _analysis("AAPL").model_copy(update={"termination_reason": "iteration_capped"})
    TerminalRenderer(stdout=stdout, stderr=StringIO(), color="never").render_analysis(analysis)

    assert "Termination: iteration_capped" in stdout.getvalue()


def test_cancelled_result_has_one_durable_interruption_summary() -> None:
    stdout = StringIO()
    stderr = StringIO()
    result = RunResult(
        run_id="cancelled-run",
        status="cancelled",
        ticker_results=(),
        total_cost_usd=0.0987,
        total_input_tokens=5,
        total_output_tokens=443,
        total_tool_calls=7,
        duration_seconds=99.9,
        error_msg="cancelled by user",
    )

    TerminalRenderer(stdout=stdout, stderr=stderr, color="never").render_result(result)

    assert "■ Interrupted · 7 tools · 99.9s · $0.0987 · Run cancelled-run" in stdout.getvalue()
    assert "cancelled by user" not in stderr.getvalue()
