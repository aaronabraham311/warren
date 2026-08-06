from __future__ import annotations

import sys
from types import ModuleType

from agent.chat import RecentContext
from agent.models import AnalysisOutput, LynchBuffettSignals
from agent.service import RunResult, TickerRunResult


def _analysis(ticker: str) -> AnalysisOutput:
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


def _partial_result() -> RunResult:
    return RunResult(
        run_id="run-context",
        status="success",
        ticker_results=(
            TickerRunResult("MSFT", "holding", error="unavailable"),
            TickerRunResult("AAPL", "holding", analysis=_analysis("AAPL")),
            TickerRunResult("GOOG", "holding", analysis=_analysis("GOOG")),
        ),
        total_cost_usd=0.1,
        total_input_tokens=10,
        total_output_tokens=5,
        total_tool_calls=2,
        duration_seconds=1.0,
        error_msg=None,
    )


def test_recent_context_defaults_to_first_success_and_keeps_failed_ticker_unselected() -> None:
    recent = RecentContext()
    recent.record_result("compare", _partial_result())

    parser_context = recent.parser_context()
    assert parser_context is not None
    assert parser_context.tickers == ("MSFT", "AAPL", "GOOG")
    assert parser_context.selected_ticker == "AAPL"
    default_analysis = recent.select_ticker()
    assert default_analysis is not None and default_analysis.ticker == "AAPL"

    assert recent.select_ticker("MSFT") is None
    parser_context = recent.parser_context()
    assert parser_context is not None and parser_context.selected_ticker == "AAPL"
    explicit_analysis = recent.select_ticker("GOOG")
    assert explicit_analysis is not None and explicit_analysis.ticker == "GOOG"


def test_main_loads_dotenv_before_terminal_app(monkeypatch: object) -> None:
    from pytest import MonkeyPatch

    assert isinstance(monkeypatch, MonkeyPatch)
    calls: list[str] = []

    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda: calls.append("dotenv"))
    fake_app = ModuleType("agent.terminal.app")

    def terminal_main() -> int:
        calls.append("app")
        return 7

    fake_app.main = terminal_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent.terminal.app", fake_app)

    from agent.chat import main

    assert main([]) == 7
    assert calls == ["dotenv", "app"]
