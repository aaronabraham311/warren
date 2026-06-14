"""Unit tests for the agent tools (Tech Spec §5).

Tools are tested at the data-source-client boundary: each tool module imports a
client getter (``yfinance_client`` / ``finnhub_client`` / ``edgar_client``), which
these tests monkeypatch to return fakes. No network is hit (the autouse
``_no_live_network`` guard enforces this). Each test asserts the tool returns a
``ToolResult`` — never raises — per the error-as-data contract.
"""

import tempfile
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from agent.budget import Budget, RunContext
from agent.tools import TOOL_DEFINITIONS, TOOL_REGISTRY
from agent.tools.base import Tool, ToolResultError, ToolResultOk
from agent.tools.filings import ReadFilingInput, ReadFilingTool
from agent.tools.fundamentals import GetFundamentalsInput, GetFundamentalsTool
from agent.tools.growth import GetGrowthMetricsInput, GetGrowthMetricsTool
from agent.tools.holdings import GetHoldingContextInput, GetHoldingContextTool, HoldingContext
from agent.tools.news import GetNewsInput, GetNewsTool, NewsResult
from agent.tools.quote import GetQuoteInput, GetQuoteTool
from agent.tools.screen import ScreenResult, ScreenUniverseInput, ScreenUniverseTool
from data_sources.edgar_client import FilingSection
from data_sources.errors import DataSourceError
from data_sources.finnhub_client import FinnhubFinancials, NewsItem
from data_sources.yfinance_client import FundamentalsData, PriceData
from storage.logger import RunLogger

# ── Helpers ───────────────────────────────────────────────────────────────────


def _ctx() -> RunContext:
    logger = RunLogger("run-tools-test", Path(tempfile.mkdtemp()))
    return RunContext(run_id="run-tools-test", budget=Budget(), logger=logger)


def _price(ticker: str = "AAPL") -> PriceData:
    return PriceData(
        ticker=ticker,
        current_price=182.5,
        previous_close=180.0,
        day_change_pct=1.39,
        volume=55_000_000,
        as_of=datetime.now(timezone.utc),
        data_age_hours=0,
    )


def _fundamentals(
    ticker: str = "AAPL", *, pe: float | None = 28.5, age: int = 1
) -> FundamentalsData:
    return FundamentalsData(
        ticker=ticker,
        as_of=date.today(),
        pe_ratio=pe,
        pb_ratio=45.2,
        roe_pct=147.0,
        debt_to_equity=150.5,
        fcf_ttm_usd=90_000_000_000,
        operating_margin_pct=30.0,
        net_margin_pct=25.0,
        data_age_hours=age,
        source="yfinance",
    )


class _FakeYF:
    def __init__(
        self,
        *,
        price: PriceData | DataSourceError | Exception | None = None,
        fundamentals: Callable[[str], FundamentalsData | DataSourceError] | None = None,
    ) -> None:
        self._price = price
        self._fundamentals = fundamentals
        self.fundamentals_calls = 0

    def get_price(self, ticker: str) -> PriceData | DataSourceError:
        if isinstance(self._price, Exception):
            raise self._price
        assert self._price is not None
        return self._price

    def get_fundamentals(self, ticker: str) -> FundamentalsData | DataSourceError:
        self.fundamentals_calls += 1
        assert self._fundamentals is not None
        return self._fundamentals(ticker)


class _FakeFinnhub:
    def __init__(
        self,
        *,
        financials: FinnhubFinancials | None = None,
        news: list[NewsItem] | DataSourceError | None = None,
    ) -> None:
        self._financials = financials
        self._news = news
        self.financials_calls = 0

    def get_basic_financials(self, ticker: str) -> FinnhubFinancials | DataSourceError:
        self.financials_calls += 1
        assert self._financials is not None
        return self._financials

    def get_news(self, ticker: str, days: int = 7) -> list[NewsItem] | DataSourceError:
        assert self._news is not None
        return self._news


# ── Registry / definitions (AC #1, AC #3 offline portion) ─────────────────────


def test_registry_has_all_seven_tools() -> None:
    assert set(TOOL_REGISTRY) == {
        "get_quote",
        "get_fundamentals",
        "get_growth_metrics",
        "read_filing",
        "get_news",
        "screen_universe",
        "get_holding_context",
    }
    assert all(isinstance(t, Tool) for t in TOOL_REGISTRY.values())


def test_tool_definitions_are_well_formed_json_schema() -> None:
    assert len(TOOL_DEFINITIONS) == len(TOOL_REGISTRY)
    for definition in TOOL_DEFINITIONS:
        assert isinstance(definition["name"], str)
        assert isinstance(definition["description"], str)
        schema = definition["input_schema"]
        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert "properties" in schema


# ── get_quote ─────────────────────────────────────────────────────────────────


def test_get_quote_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.tools.quote.yfinance_client", lambda: _FakeYF(price=_price()))
    result = GetQuoteTool().run(GetQuoteInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, PriceData)
    assert result.data.current_price == pytest.approx(182.5)


def test_get_quote_maps_data_source_error(monkeypatch: pytest.MonkeyPatch) -> None:
    err = DataSourceError(error_code="network", message="boom")
    monkeypatch.setattr("agent.tools.quote.yfinance_client", lambda: _FakeYF(price=err))
    result = GetQuoteTool().run(GetQuoteInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultError)
    assert result.error_code == "network"
    assert result.retryable is True


def test_get_quote_catches_unexpected_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agent.tools.quote.yfinance_client",
        lambda: _FakeYF(price=RuntimeError("kaboom")),
    )
    result = GetQuoteTool().run(GetQuoteInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultError)
    assert result.error_code == "unknown"


# ── get_fundamentals fallback (AC #4) ─────────────────────────────────────────


def test_get_fundamentals_fresh_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    yf = _FakeYF(fundamentals=lambda t: _fundamentals(t, age=1))
    fh = _FakeFinnhub(
        financials=FinnhubFinancials(
            ticker="AAPL", as_of=date.today(), pe_ratio=20.0, pb_ratio=3.0, roe_pct=15.0
        )
    )
    monkeypatch.setattr("agent.tools.fundamentals.yfinance_client", lambda: yf)
    monkeypatch.setattr("agent.tools.fundamentals.finnhub_client", lambda: fh)
    result = GetFundamentalsTool().run(GetFundamentalsInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, FundamentalsData)
    assert result.data.source == "yfinance"
    assert fh.financials_calls == 0


def test_get_fundamentals_stale_falls_back_to_finnhub(monkeypatch: pytest.MonkeyPatch) -> None:
    yf = _FakeYF(fundamentals=lambda t: _fundamentals(t, age=1000))  # stale
    fh = _FakeFinnhub(
        financials=FinnhubFinancials(
            ticker="AAPL", as_of=date.today(), pe_ratio=20.0, pb_ratio=3.0, roe_pct=15.0
        )
    )
    monkeypatch.setattr("agent.tools.fundamentals.yfinance_client", lambda: yf)
    monkeypatch.setattr("agent.tools.fundamentals.finnhub_client", lambda: fh)
    result = GetFundamentalsTool().run(GetFundamentalsInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, FundamentalsData)
    assert result.data.source == "finnhub"
    assert fh.financials_calls == 1


def test_get_fundamentals_stale_without_finnhub_returns_yfinance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yf = _FakeYF(fundamentals=lambda t: _fundamentals(t, age=1000))
    monkeypatch.setattr("agent.tools.fundamentals.yfinance_client", lambda: yf)
    monkeypatch.setattr("agent.tools.fundamentals.finnhub_client", lambda: None)
    result = GetFundamentalsTool().run(GetFundamentalsInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, FundamentalsData)
    assert result.data.source == "yfinance"


# ── get_growth_metrics ────────────────────────────────────────────────────────


def test_get_growth_metrics_maps_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    class _YF:
        def get_growth_metrics(self, ticker: str) -> DataSourceError:
            return DataSourceError(error_code="not_found", message="nope")

    monkeypatch.setattr("agent.tools.growth.yfinance_client", lambda: _YF())
    result = GetGrowthMetricsTool().run(GetGrowthMetricsInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultError)
    assert result.error_code == "not_found"
    assert result.retryable is False


# ── read_filing ───────────────────────────────────────────────────────────────


def test_read_filing_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    section = FilingSection(
        ticker="AAPL",
        filing_type="10-K",
        section="business",
        fiscal_year=2023,
        filing_date=date.today(),
        text="We make phones.",
        word_count=3,
        truncated=False,
        edgar_url="https://sec.gov/x",
    )

    class _Edgar:
        def get_filing_section(self, *a: object, **k: object) -> FilingSection:
            return section

    monkeypatch.setattr("agent.tools.filings.edgar_client", lambda: _Edgar())
    result = ReadFilingTool().run(
        ReadFilingInput(ticker="AAPL", filing_type="10-K", section="business"), _ctx()
    )
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, FilingSection)


def test_read_filing_maps_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Edgar:
        def get_filing_section(self, *a: object, **k: object) -> DataSourceError:
            return DataSourceError(error_code="not_found", message="no filing")

    monkeypatch.setattr("agent.tools.filings.edgar_client", lambda: _Edgar())
    result = ReadFilingTool().run(
        ReadFilingInput(ticker="AAPL", filing_type="10-K", section="business"), _ctx()
    )
    assert isinstance(result, ToolResultError)
    assert result.error_code == "not_found"


# ── get_news ──────────────────────────────────────────────────────────────────


def test_get_news_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    item = NewsItem(
        headline="Up", summary="s", source="x", datetime=datetime.now(timezone.utc), url="https://x"
    )
    monkeypatch.setattr("agent.tools.news.finnhub_client", lambda: _FakeFinnhub(news=[item]))
    result = GetNewsTool().run(GetNewsInput(ticker="AAPL", days=7), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, NewsResult)
    assert len(result.data.items) == 1


def test_get_news_without_key_returns_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.tools.news.finnhub_client", lambda: None)
    result = GetNewsTool().run(GetNewsInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultError)
    assert result.error_code == "not_found"


# ── screen_universe (AC #5) ───────────────────────────────────────────────────


def test_screen_universe_filters_by_pe(monkeypatch: pytest.MonkeyPatch) -> None:
    pe_by_ticker = {"AAPL": 12.0, "MSFT": 30.0, "GOOG": 14.0}
    yf = _FakeYF(fundamentals=lambda t: _fundamentals(t, pe=pe_by_ticker[t]))
    monkeypatch.setattr("agent.tools.screen._universe", lambda: ["AAPL", "MSFT", "GOOG"])
    monkeypatch.setattr("agent.tools.screen.yfinance_client", lambda: yf)
    result = ScreenUniverseTool().run(ScreenUniverseInput(criteria={"pe_ratio_max": 15}), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ScreenResult)
    assert result.data.tickers == ["AAPL", "GOOG"]


def test_screen_universe_excludes_missing_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    yf = _FakeYF(fundamentals=lambda t: _fundamentals(t, pe=None))
    monkeypatch.setattr("agent.tools.screen._universe", lambda: ["AAPL"])
    monkeypatch.setattr("agent.tools.screen.yfinance_client", lambda: yf)
    result = ScreenUniverseTool().run(ScreenUniverseInput(criteria={"pe_ratio_max": 15}), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ScreenResult)
    assert result.data.tickers == []


# ── get_holding_context ───────────────────────────────────────────────────────


def test_get_holding_context_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    # AAPL is in the committed data/portfolio.csv (10 shares @ 150.00).
    monkeypatch.setattr("agent.tools.holdings.yfinance_client", lambda: _FakeYF(price=_price()))
    result = GetHoldingContextTool().run(GetHoldingContextInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, HoldingContext)
    assert result.data.shares == pytest.approx(10.0)
    assert result.data.cost_basis == pytest.approx(150.0)
    # (182.5 - 150) / 150 * 100 ≈ 21.67
    assert result.data.unrealized_pnl_pct == pytest.approx(21.67, rel=1e-3)


def test_get_holding_context_not_in_portfolio() -> None:
    result = GetHoldingContextTool().run(GetHoldingContextInput(ticker="ZZZZ"), _ctx())
    assert isinstance(result, ToolResultError)
    assert result.error_code == "not_found"
