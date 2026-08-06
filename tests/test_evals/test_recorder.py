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

from agent.budget import Budget, RunContext
from agent.tools import TOOL_REGISTRY
from agent.tools.base import ToolResult, ToolResultError, ToolResultOk
from data_sources.edgar_client import FilingSection
from data_sources.yfinance_client import PriceData, ValuationHistory
from eval.fixture_evidence import (
    CURATED_FILING_CONCEPTS,
    FILING_CALLS,
    NEWS_WINDOWS,
    validate_fixture_result,
)
from eval.fixtures.recorder import (
    CORE_RECORDED_CALLS,
    RECORDED_CALLS,
    REGIONAL_RECORDED_CALLS,
    RecordedCall,
    mandatory_evidence_calls,
    record_ticker,
)
from eval.golden_set import EvalExample, load_all_examples
from eval.tool_fixtures import FIXTURES_DIR, FixtureToolRunner, tool_fixture_path
from storage.logger import RunLogger


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


def test_recorder_covers_explicit_news_windows_without_aliasing() -> None:
    calls = [call for call in RECORDED_CALLS if call.tool == "get_news"]
    assert [call.input["days"] for call in calls] == list(NEWS_WINDOWS)


def test_recorder_covers_every_supported_filing_section_form_pair() -> None:
    assert ("10-K", "business") in FILING_CALLS
    assert ("10-K", "risk_factors") in FILING_CALLS
    assert ("10-K", "mdna") in FILING_CALLS
    assert ("10-K", "financial_statements") in FILING_CALLS
    assert ("10-K", "executive_summary") in FILING_CALLS
    assert ("10-Q", "financial_statements") in FILING_CALLS
    assert ("10-Q", "mdna") in FILING_CALLS
    assert ("10-Q", "risk_factors") in FILING_CALLS
    assert ("10-Q", "executive_summary") in FILING_CALLS
    assert ("8-K", "executive_summary") in FILING_CALLS
    assert ("DEF 14A", "compensation") in FILING_CALLS
    assert ("DEF 14A", "related_party") in FILING_CALLS
    assert ("DEF 14A", "executive_summary") in FILING_CALLS


def test_mandatory_evidence_calls_are_ticker_scoped() -> None:
    assert mandatory_evidence_calls("sbux") == (
        RecordedCall("read_filing", {"filing_type": "10-K", "section": "mdna"}),
    )
    assert mandatory_evidence_calls("AAPL") == ()


def _filing(text: str, *, section: str = "mdna") -> FilingSection:
    return FilingSection(
        ticker="SBUX",
        filing_type="10-K",
        section=section,
        fiscal_year=2025,
        filing_date=datetime(2025, 11, 14, tzinfo=timezone.utc).date(),
        text=text,
        word_count=len(text.split()),
        truncated=False,
        edgar_url="https://www.sec.gov/example",
    )


def test_filing_validator_rejects_toc_fragment() -> None:
    tool = TOOL_REGISTRY["read_filing"]
    tool_input = tool.input_schema.model_validate(
        {"ticker": "SBUX", "filing_type": "10-K", "section": "mdna"}
    )
    fragment = "Item 7 of this Report. Table of Contents"

    with pytest.raises(ValueError, match="unusable filing evidence"):
        validate_fixture_result("SBUX", tool, tool_input, ToolResultOk(data=_filing(fragment)))


def test_filing_validator_requires_curated_evidence_concept() -> None:
    tool = TOOL_REGISTRY["read_filing"]
    tool_input = tool.input_schema.model_validate(
        {"ticker": "SBUX", "filing_type": "10-K", "section": "mdna"}
    )
    text = " ".join(["Coffee stores generated healthy revenue and margins."] * 20)

    with pytest.raises(ValueError, match="missing curated concept"):
        validate_fixture_result("SBUX", tool, tool_input, ToolResultOk(data=_filing(text)))


def test_filing_validator_accepts_substantive_curated_evidence() -> None:
    tool = TOOL_REGISTRY["read_filing"]
    tool_input = tool.input_schema.model_validate(
        {"ticker": "SBUX", "filing_type": "10-K", "section": "mdna"}
    )
    text = " ".join(
        [
            "Comparable sales reflected lower traffic while management invested in store "
            "throughput, loyalty, labor, and the customer experience."
        ]
        * 10
    )

    validate_fixture_result("SBUX", tool, tool_input, ToolResultOk(data=_filing(text)))


def test_fixture_validator_rejects_wrong_success_schema() -> None:
    tool = TOOL_REGISTRY["get_news"]
    tool_input = tool.input_schema.model_validate({"ticker": "AAPL", "days": 7})
    with pytest.raises(Exception, match="validation error"):
        validate_fixture_result("AAPL", tool, tool_input, _ok(190.5))


def test_invalid_filing_does_not_overwrite_existing_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call = RecordedCall("read_filing", {"filing_type": "10-K", "section": "mdna"})
    monkeypatch.setattr("eval.fixtures.recorder.RECORDED_CALLS", (call,))
    tool = TOOL_REGISTRY["read_filing"]
    tool_input = tool.input_schema.model_validate({"ticker": "SBUX", **call.input})
    path = tool_fixture_path("SBUX", "read_filing", tool_input.model_dump(mode="json"), tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("previous-good-fixture")
    fragment = ToolResultOk(data=_filing("Item 7 of this Report. Table of Contents"))

    with patch.object(type(tool), "run", return_value=fragment):
        summary = record_ticker("SBUX", tmp_path)

    assert summary.failures
    assert path.read_text() == "previous-good-fixture"


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
        for call in CORE_RECORDED_CALLS
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


def test_curated_filing_expectations_have_usable_evidence(tmp_path: Path) -> None:
    """Every qualitative filing expectation has a valid committed replay source."""
    ctx = RunContext(
        run_id="fixture-evidence-test",
        budget=Budget(),
        logger=RunLogger("fixture-evidence-test", tmp_path),
    )
    failures: list[str] = []
    tool = TOOL_REGISTRY["read_filing"]
    for ticker, filing_type, section in CURATED_FILING_CONCEPTS:
        runner = FixtureToolRunner(ticker)
        tool_input = tool.input_schema.model_validate(
            {"ticker": ticker, "filing_type": filing_type, "section": section}
        )
        result = runner.run(tool, tool_input, ctx)
        if isinstance(result, ToolResultError):
            failures.append(f"{ticker}/{filing_type}/{section}: {result.message}")

    assert not failures, "\n".join(failures)
