"""Tests for the eval fixture recorder.

The on-disk format, the path layout, and replay itself are ``eval.tool_fixtures``' job and
are tested in ``test_eval_tool_fixtures.py``. What is tested here is the recorder's own
contract: it drives the real tools, overwrites in place, records data-source errors rather
than dropping them, and leaves the golden set fully covered.
"""

import json
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.tools import TOOL_REGISTRY
from agent.tools.base import ToolResult, ToolResultError, ToolResultOk
from data_sources.yfinance_client import PriceData, ValuationHistory
from eval.fixtures.recorder import RECORDED_CALLS, REGIONAL_RECORDED_CALLS, record_ticker
from eval.golden_set import EvalExample, load_all_examples
from eval.tool_fixtures import FIXTURES_DIR, tool_fixture_path


def _ok(price: float) -> ToolResultOk:
    return ToolResultOk(
        data=PriceData(
            ticker="AAPL",
            current_price=price,
            previous_close=188.0,
            day_change_pct=1.33,
            volume=50_000_000,
            as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
            data_age_hours=1,
        )
    )


def _valuation_history_ok() -> ToolResultOk:
    return ToolResultOk(
        data=ValuationHistory(
            ticker="AAPL",
            as_of=date(2026, 8, 5),
            years_covered=4,
            current_pe=30.0,
            current_pb=8.0,
            pe_percentile=25.0,
            pb_percentile=0.0,
            pb_min=8.0,
            pb_vs_10y_low=1.0,
            fiscal_years=[2025, 2024, 2023, 2022],
            pe_series=[30.0, 32.0, 28.0, 35.0],
            pb_series=[8.0, 9.0, 8.5, 10.0],
            data_age_hours=24,
        )
    )


@contextmanager
def _stub_every_tool(
    return_value: ToolResult | None = None, side_effect: Exception | None = None
) -> Iterator[None]:
    """Stub ``run`` on every registered tool.

    Each tool overrides ``run``, so patching ``Tool.run`` on the base class would silently
    leave the real implementations in place — and the recorder would hit the network.
    """
    with ExitStack() as stack:
        for tool in TOOL_REGISTRY.values():
            stub = MagicMock(return_value=return_value, side_effect=side_effect)
            stack.enter_context(patch.object(type(tool), "run", stub))
        yield


def test_record_ticker_overwrites_rather_than_duplicating(tmp_path: Path) -> None:
    quote_input = TOOL_REGISTRY["get_quote"].input_schema(ticker="AAPL").model_dump(mode="json")

    with _stub_every_tool(return_value=_ok(190.5)):
        record_ticker("AAPL", tmp_path)
    with _stub_every_tool(return_value=_ok(200.0)):
        record_ticker("AAPL", tmp_path)

    quote_dir = (tmp_path / "AAPL" / "tools" / "get_quote").iterdir()
    assert len(list(quote_dir)) == 1, "a re-record overwrites its fixture, never duplicates it"

    path = tool_fixture_path("AAPL", "get_quote", quote_input, tmp_path)
    assert '"current_price": 200.0' in path.read_text()


def test_record_ticker_persists_tool_errors(tmp_path: Path) -> None:
    """A data source that is genuinely unavailable replays as that error, not as a hole."""
    error = ToolResultError(error_code="not_found", message="delisted", retryable=False)

    with _stub_every_tool(return_value=error):
        summary = record_ticker("ZZZZ", tmp_path)

    assert summary.ok == 0
    assert len(summary.errors) == len(RECORDED_CALLS)
    assert not summary.failures
    quote_input = TOOL_REGISTRY["get_quote"].input_schema(ticker="ZZZZ").model_dump(mode="json")
    assert tool_fixture_path("ZZZZ", "get_quote", quote_input, tmp_path).exists()


def test_record_ticker_survives_a_raising_tool(tmp_path: Path) -> None:
    """An exception is a bug, not data: report it and leave the old fixture in place."""
    with _stub_every_tool(side_effect=RuntimeError("boom")):
        summary = record_ticker("AAPL", tmp_path)

    assert summary.ok == 0
    assert len(summary.failures) == len(RECORDED_CALLS)
    assert not (tmp_path / "AAPL").exists()


def test_recorder_includes_valuation_history() -> None:
    assert "get_valuation_history" in {call.tool for call in RECORDED_CALLS}


def test_regional_recorder_includes_forensic_evidence_without_polluting_us_set() -> None:
    assert {call.tool for call in REGIONAL_RECORDED_CALLS} == {"get_forensic_evidence"}
    assert "get_forensic_evidence" not in {call.tool for call in RECORDED_CALLS}


def test_record_ticker_can_target_one_tool(tmp_path: Path) -> None:
    call = next(call for call in RECORDED_CALLS if call.tool == "get_valuation_history")
    with _stub_every_tool(return_value=_valuation_history_ok()):
        summary = record_ticker("AAPL", tmp_path, calls=(call,))

    assert summary.ok == 1
    fixture_dir = tmp_path / "AAPL" / "tools" / "get_valuation_history"
    assert fixture_dir.is_dir()
    assert not (tmp_path / "AAPL" / "tools" / "get_quote").exists()
    payload = json.loads(next(fixture_dir.iterdir()).read_text())
    assert isinstance(
        TOOL_REGISTRY["get_valuation_history"].output_schema.model_validate(payload["data"]),
        ValuationHistory,
    )


# Default examples require every recorded call. Regional DIRT examples remain partial overall,
# but G15 separately pins live valuation-history coverage for both current and legacy tickers.
@pytest.mark.parametrize(
    "example",
    [e for e in load_all_examples() if e.persona != "dirt"],
    ids=lambda e: e.ticker,
)
def test_golden_set_fixture_completeness(example: EvalExample) -> None:
    """Every default golden-set ticker has a committed fixture for every recorded call."""
    ticker = example.ticker
    missing = [
        call.tool
        for call in RECORDED_CALLS
        if not tool_fixture_path(
            ticker,
            call.tool,
            TOOL_REGISTRY[call.tool]
            .input_schema.model_validate({"ticker": ticker, **call.input})
            .model_dump(mode="json"),
            FIXTURES_DIR,
        ).exists()
    ]
    assert not missing, f"{ticker} is missing fixtures for: {', '.join(missing)}"


@pytest.mark.parametrize(
    "ticker",
    [
        # Current G12 canonical junior-market examples.
        "ABP.MI",
        "LAB.MC",
        "CHP.WA",
        # Legacy G15 examples remain replayable, including honest not-found errors.
        "DIR.MI",
        "CIRSA.MC",
        "KPL.WA",
    ],
)
def test_regional_valuation_history_fixture_completeness(ticker: str) -> None:
    payload = (
        TOOL_REGISTRY["get_valuation_history"].input_schema(ticker=ticker).model_dump(mode="json")
    )
    assert tool_fixture_path(ticker, "get_valuation_history", payload, FIXTURES_DIR).exists()
