"""Tests for the eval fixture recorder.

The on-disk format, the path layout, and replay itself are ``eval.tool_fixtures``' job and
are tested in ``test_eval_tool_fixtures.py``. What is tested here is the recorder's own
contract: it drives the real tools, overwrites in place, records data-source errors rather
than dropping them, and leaves the golden set fully covered.
"""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.tools import TOOL_REGISTRY
from agent.tools.base import ToolResult, ToolResultError, ToolResultOk
from data_sources.yfinance_client import PriceData
from eval.fixtures.recorder import RECORDED_CALLS, record_ticker
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


# DIRT/deep-value gems (persona="dirt") carry small *synthetic* partial fixtures authored by
# hand — their international tickers do not record cleanly from live APIs, and the live replay
# they exercise is never run in CI. Completeness is asserted only for the default examples,
# which are recorded from live data via the recorder.
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
