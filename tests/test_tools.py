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
from typing import Literal

import pytest

from agent.budget import Budget, RunContext
from agent.tools import TOOL_DEFINITIONS, TOOL_REGISTRY
from agent.tools.base import Tool, ToolResultError, ToolResultOk
from agent.tools.capital_allocation import (
    GetCapitalAllocationInput,
    GetCapitalAllocationTool,
)
from agent.tools.filings import ReadFilingInput, ReadFilingTool
from agent.tools.financial_strength import GetFinancialStrengthInput, GetFinancialStrengthTool
from agent.tools.fundamentals import (
    _STALE_FUNDAMENTALS_H,
    GetFundamentalsInput,
    GetFundamentalsTool,
)
from agent.tools.growth import GetGrowthMetricsInput, GetGrowthMetricsTool
from agent.tools.holdings import GetHoldingContextInput, GetHoldingContextTool, HoldingContext
from agent.tools.insider import GetInsiderActivityInput, GetInsiderActivityTool, InsiderActivity
from agent.tools.intrinsic_value import (
    DCFAssumptions,
    EstimateIntrinsicValueInput,
    EstimateIntrinsicValueTool,
    IntrinsicValue,
    _intrinsic_equity_value,
    _reverse_dcf_growth,
)
from agent.tools.news import GetNewsInput, GetNewsTool, NewsResult
from agent.tools.peers import GetPeerComparisonInput, GetPeerComparisonTool, PeerComparison
from agent.tools.persons import GetKeyPersonsInput, GetKeyPersonsTool, KeyPersonsData
from agent.tools.quality import GetQualityMetricsInput, GetQualityMetricsTool
from agent.tools.quote import GetQuoteInput, GetQuoteTool
from agent.tools.screen import ScreenResult, ScreenUniverseInput, ScreenUniverseTool
from agent.tools.valuation import GetValuationMultiplesInput, GetValuationMultiplesTool
from agent.tools.valuation_history import (
    GetValuationHistoryInput,
    GetValuationHistoryTool,
)
from data_sources.edgar_client import FilingSection, SC13Holder
from data_sources.errors import DataSourceError
from data_sources.filing_models import SourceSystem, TranslationStatus
from data_sources.finnhub_client import FinnhubFinancials, FinnhubInsiderTransaction, NewsItem
from data_sources.yfinance_client import (
    BalanceSheetRow,
    CapitalAllocation,
    CashFlowRow,
    FinancialsHistory,
    FinancialStrengthData,
    FundamentalsData,
    IncomeStatementRow,
    InstitutionalHolderRecord,
    KeyPersonsRaw,
    OfficerRecord,
    OwnershipData,
    PiotroskySignals,
    PriceData,
    QualityData,
    ValuationData,
    ValuationHistory,
)
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
    ticker: str = "AAPL",
    *,
    pe: float | None = 28.5,
    age: int = 1,
    gross_margin: float | None = 40.0,
    sector: str | None = None,
    analyst_count: int | None = None,
) -> FundamentalsData:
    return FundamentalsData(
        ticker=ticker,
        as_of=date.today(),
        pe_ratio=pe,
        pb_ratio=45.2,
        roe_pct=147.0,
        debt_to_equity=150.5,
        fcf_ttm_usd=90_000_000_000,
        gross_margin_pct=gross_margin,
        operating_margin_pct=30.0,
        net_margin_pct=25.0,
        sector=sector,
        analyst_count=analyst_count,
        data_age_hours=age,
        source="yfinance",
    )


class _FakeYF:
    def __init__(
        self,
        *,
        price: PriceData | DataSourceError | Exception | None = None,
        fundamentals: Callable[[str], FundamentalsData | DataSourceError] | None = None,
        valuation: (
            Callable[[str], ValuationData | DataSourceError]
            | ValuationData
            | DataSourceError
            | None
        ) = None,
        quality: (
            Callable[[str], QualityData | DataSourceError] | QualityData | DataSourceError | None
        ) = None,
        ownership: OwnershipData | DataSourceError | None = None,
        financial_strength: FinancialStrengthData | DataSourceError | None = None,
        financials: FinancialsHistory | DataSourceError | None = None,
        capital_allocation: CapitalAllocation | DataSourceError | None = None,
        key_persons: KeyPersonsRaw | DataSourceError | None = None,
        russell2000: list[str] | DataSourceError | None = None,
        valuation_history: ValuationHistory | DataSourceError | Exception | None = None,
    ) -> None:
        self._valuation_history = valuation_history
        self._price = price
        self._fundamentals = fundamentals
        self._valuation = valuation
        self._quality = quality
        self._ownership = ownership
        self._financial_strength = financial_strength
        self._financials = financials
        self._capital_allocation = capital_allocation
        self._key_persons = key_persons
        self._russell2000 = russell2000
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

    def get_valuation_multiples(self, ticker: str) -> ValuationData | DataSourceError:
        assert self._valuation is not None
        if callable(self._valuation):
            return self._valuation(ticker)
        return self._valuation

    def get_quality_metrics(self, ticker: str) -> QualityData | DataSourceError:
        assert self._quality is not None
        if callable(self._quality):
            return self._quality(ticker)
        return self._quality

    def get_ownership(self, ticker: str) -> OwnershipData | DataSourceError:
        if self._ownership is None:
            return OwnershipData(
                ticker=ticker, as_of=date.today(), insider_pct=None, institutional_pct=None
            )
        return self._ownership

    def get_financial_strength(self, ticker: str) -> FinancialStrengthData | DataSourceError:
        assert self._financial_strength is not None
        return self._financial_strength

    def get_financials(self, ticker: str) -> FinancialsHistory | DataSourceError:
        assert self._financials is not None
        return self._financials

    def get_capital_allocation(self, ticker: str) -> CapitalAllocation | DataSourceError:
        assert self._capital_allocation is not None
        return self._capital_allocation

    def get_key_persons(self, ticker: str) -> KeyPersonsRaw | DataSourceError:
        assert self._key_persons is not None
        return self._key_persons

    def get_russell2000_tickers(self) -> list[str] | DataSourceError:
        if self._russell2000 is None:
            return []
        return self._russell2000

    def get_valuation_history(self, ticker: str) -> ValuationHistory | DataSourceError:
        if isinstance(self._valuation_history, Exception):
            raise self._valuation_history
        assert self._valuation_history is not None
        return self._valuation_history


class _FakeFinnhub:
    def __init__(
        self,
        *,
        financials: FinnhubFinancials | None = None,
        news: list[NewsItem] | DataSourceError | None = None,
        insider_txns: list[FinnhubInsiderTransaction] | DataSourceError | None = None,
    ) -> None:
        self._financials = financials
        self._news = news
        self._insider_txns = insider_txns
        self.financials_calls = 0

    def get_basic_financials(self, ticker: str) -> FinnhubFinancials | DataSourceError:
        self.financials_calls += 1
        assert self._financials is not None
        return self._financials

    def get_news(self, ticker: str, days: int = 7) -> list[NewsItem] | DataSourceError:
        assert self._news is not None
        return self._news

    def get_insider_transactions(
        self, ticker: str, days: int = 90
    ) -> list[FinnhubInsiderTransaction] | DataSourceError:
        assert self._insider_txns is not None
        return self._insider_txns


# ── Registry / definitions (AC #1, AC #3 offline portion) ─────────────────────


def test_registry_has_all_eighteen_tools() -> None:
    assert set(TOOL_REGISTRY) == {
        "get_quote",
        "get_fundamentals",
        "get_growth_metrics",
        "read_filing",
        "get_news",
        "screen_universe",
        "get_holding_context",
        "get_valuation_multiples",
        "get_valuation_history",
        "get_quality_metrics",
        "get_insider_activity",
        "get_peer_comparison",
        "get_financial_strength",
        "estimate_intrinsic_value",
        "get_capital_allocation",
        "get_key_persons",
        "get_adverse_media",
        "screen_watchlists",
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


def test_get_fundamentals_fiscal_normal_age_stays_yfinance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # yfinance fundamentals age off lastFiscalYearEnd, so a current fetch is still
    # ~months old (e.g. ~10mo since the last annual filing). That is NOT stale — the
    # tool must keep yfinance's full 9-field payload, not downgrade to Finnhub's basics.
    yf = _FakeYF(fundamentals=lambda t: _fundamentals(t, age=7000))  # ~10 months
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
    # the six fields the Finnhub fallback would null out must survive
    assert result.data.debt_to_equity is not None
    assert result.data.fcf_ttm_usd is not None
    assert result.data.gross_margin_pct is not None
    assert result.data.operating_margin_pct is not None
    assert result.data.net_margin_pct is not None


def test_get_fundamentals_stale_falls_back_to_finnhub(monkeypatch: pytest.MonkeyPatch) -> None:
    yf = _FakeYF(fundamentals=lambda t: _fundamentals(t, age=_STALE_FUNDAMENTALS_H + 1))  # stale
    fh = _FakeFinnhub(
        financials=FinnhubFinancials(
            ticker="AAPL",
            as_of=date.today(),
            pe_ratio=20.0,
            pb_ratio=3.0,
            roe_pct=15.0,
            debt_to_equity=187.0,
            gross_margin_pct=46.2,
            operating_margin_pct=30.1,
            net_margin_pct=25.3,
        )
    )
    monkeypatch.setattr("agent.tools.fundamentals.yfinance_client", lambda: yf)
    monkeypatch.setattr("agent.tools.fundamentals.finnhub_client", lambda: fh)
    result = GetFundamentalsTool().run(GetFundamentalsInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, FundamentalsData)
    assert result.data.source == "finnhub"
    assert fh.financials_calls == 1
    # Widened projection: margins and debt-to-equity survive the fallback (no longer nulled).
    assert result.data.debt_to_equity == pytest.approx(187.0)
    assert result.data.gross_margin_pct == pytest.approx(46.2)
    assert result.data.operating_margin_pct == pytest.approx(30.1)
    assert result.data.net_margin_pct == pytest.approx(25.3)
    # Finnhub's basics endpoint supplies neither — these stay None on the fallback path.
    assert result.data.fcf_ttm_usd is None
    assert result.data.sector is None


def test_get_fundamentals_stale_without_finnhub_returns_yfinance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yf = _FakeYF(fundamentals=lambda t: _fundamentals(t, age=_STALE_FUNDAMENTALS_H + 1))
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


def _filing_section(text: str = "We make phones.", section: str = "business") -> FilingSection:
    return FilingSection(
        ticker="AAPL",
        filing_type="10-K",
        section=section,
        fiscal_year=2023,
        filing_date=date.today(),
        text=text,
        word_count=len(text.split()),
        truncated=False,
        edgar_url="https://sec.gov/x",
    )


class _FakeEdgar:
    def __init__(
        self,
        result: FilingSection | DataSourceError | None = None,
        sc13: list[SC13Holder] | DataSourceError | None = None,
    ) -> None:
        self._result = result
        self._sc13 = sc13

    def get_filing_section(self, *a: object, **k: object) -> FilingSection | DataSourceError:
        assert self._result is not None
        return self._result

    def get_sc13_holders(self, ticker: str) -> list[SC13Holder] | DataSourceError:
        if self._sc13 is None:
            return []
        return self._sc13


def test_read_filing_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    section = _filing_section().model_copy(update={"source_language": "en"})
    monkeypatch.setattr("agent.tools.filings.edgar_client", lambda: _FakeEdgar(section))
    monkeypatch.setattr(
        "agent.tools.filings.yfinance_client",
        lambda: _FakeYF(fundamentals=lambda t: DataSourceError("not_found", "no data")),
    )
    result = ReadFilingTool().run(
        ReadFilingInput(ticker="AAPL", filing_type="10-K", section="business"), _ctx()
    )
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, FilingSection)
    assert result.data.source_language == "en"


def test_read_filing_maps_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agent.tools.filings.edgar_client",
        lambda: _FakeEdgar(DataSourceError(error_code="not_found", message="no filing")),
    )
    monkeypatch.setattr(
        "agent.tools.filings.yfinance_client",
        lambda: _FakeYF(fundamentals=lambda t: DataSourceError("not_found", "no data")),
    )
    result = ReadFilingTool().run(
        ReadFilingInput(ticker="AAPL", filing_type="10-K", section="business"), _ctx()
    )
    assert isinstance(result, ToolResultError)
    assert result.error_code == "not_found"


def test_read_filing_propagates_source_stage_and_retryability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent.tools.filings.edgar_client",
        lambda: _FakeEdgar(
            DataSourceError(
                error_code="rate_limit",
                message="slow down",
                stage="download",
                source="edgar",
            )
        ),
    )
    result = ReadFilingTool().run(
        ReadFilingInput(ticker="AAPL", filing_type="10-K", section="business"), _ctx()
    )

    assert isinstance(result, ToolResultError)
    assert result.error_code == "rate_limit"
    assert result.retryable is True
    assert result.stage == "download"
    assert result.source == "edgar"


def test_read_filing_preserves_non_retryable_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent.tools.filings.edgar_client",
        lambda: _FakeEdgar(
            DataSourceError(
                error_code="parse",
                message="bad document",
                stage="extract",
                source="edgar",
            )
        ),
    )
    result = ReadFilingTool().run(
        ReadFilingInput(ticker="AAPL", filing_type="10-K", section="business"), _ctx()
    )

    assert isinstance(result, ToolResultError)
    assert result.error_code == "parse"
    assert result.retryable is False
    assert result.stage == "extract"


def test_read_filing_non_us_ticker_degrades_to_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    # A non-US name (Milan .MI) has no SEC/EDGAR presence — the client returns not_found
    # as data, and the tool maps it to a clean ToolResultError rather than raising.
    monkeypatch.setattr(
        "agent.tools.filings.edgar_client",
        lambda: _FakeEdgar(DataSourceError(error_code="not_found", message="unknown ticker")),
    )
    monkeypatch.setattr(
        "agent.tools.filings.yfinance_client",
        lambda: _FakeYF(fundamentals=lambda t: DataSourceError("not_found", "no data")),
    )
    result = ReadFilingTool().run(
        ReadFilingInput(ticker="DIR.MI", filing_type="10-K", section="business"), _ctx()
    )
    assert isinstance(result, ToolResultError)
    assert result.error_code == "not_found"


def test_read_filing_translate_plumbed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.tools.filings.edgar_client", lambda: _FakeEdgar(_filing_section()))
    monkeypatch.setattr(
        "agent.tools.filings.yfinance_client",
        lambda: _FakeYF(fundamentals=lambda t: DataSourceError("not_found", "no data")),
    )
    result = ReadFilingTool().run(
        ReadFilingInput(
            ticker="AAPL",
            filing_type="10-K",
            section="business",
            translate=True,
            source_language="ja",
        ),
        _ctx(),
    )
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, FilingSection)
    assert result.data.translate is True
    assert result.data.source_language == "ja"
    assert result.data.translation_status is TranslationStatus.FAILED


def test_read_filing_grounded_language_wins_over_caller_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    section = _filing_section().model_copy(update={"source_language": "en"})
    monkeypatch.setattr("agent.tools.filings.edgar_client", lambda: _FakeEdgar(section))
    monkeypatch.setattr(
        "agent.tools.filings.yfinance_client",
        lambda: _FakeYF(fundamentals=lambda t: DataSourceError("not_found", "no data")),
    )

    result = ReadFilingTool().run(
        ReadFilingInput(
            ticker="AAPL",
            filing_type="10-K",
            section="business",
            translate=True,
            source_language="ja",
        ),
        _ctx(),
    )

    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, FilingSection)
    assert result.data.source_language == "en"
    assert result.data.output_language == "en"
    assert result.data.translation_status is TranslationStatus.NOT_NEEDED


def test_read_filing_uses_source_neutral_stored_regional_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    section = _filing_section(section="full_document").model_copy(
        update={
            "ticker": "DIR.MI",
            "filing_type": "annual",
            "source_language": "it",
            "source_system": SourceSystem.BORSA_ITALIANA,
        }
    )
    monkeypatch.setattr("agent.tools.filings.stored_filing_client", lambda: _FakeEdgar(section))
    monkeypatch.setattr(
        "agent.tools.filings.edgar_client",
        lambda: (_ for _ in ()).throw(AssertionError("EDGAR must not be called")),
    )
    monkeypatch.setattr(
        "agent.tools.filings.yfinance_client",
        lambda: _FakeYF(fundamentals=lambda t: DataSourceError("not_found", "no data")),
    )

    result = ReadFilingTool().run(
        ReadFilingInput(ticker="DIR.MI", filing_type="annual", section="full_document"),
        _ctx(),
    )

    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, FilingSection)
    assert result.data.source_system is SourceSystem.BORSA_ITALIANA


def test_read_filing_aggregator_discrepancy_note_populated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Filing says gross margin 20%, yfinance says 43.8% — diff > 5pp → note set
    section = _filing_section(
        text="Our gross margin was 20.0% for the fiscal year ended September 2023.",
        section="mdna",
    )
    monkeypatch.setattr("agent.tools.filings.edgar_client", lambda: _FakeEdgar(section))
    monkeypatch.setattr(
        "agent.tools.filings.yfinance_client",
        lambda: _FakeYF(fundamentals=lambda t: _fundamentals(t, gross_margin=43.8)),
    )
    result = ReadFilingTool().run(
        ReadFilingInput(ticker="AAPL", filing_type="10-K", section="mdna"), _ctx()
    )
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, FilingSection)
    assert result.data.aggregator_discrepancy_note is not None
    assert "gross margin" in result.data.aggregator_discrepancy_note.lower()
    assert "20.0" in result.data.aggregator_discrepancy_note
    assert "43.8" in result.data.aggregator_discrepancy_note


def test_read_filing_no_discrepancy_when_margins_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Filing says 43.5%, yfinance says 43.8% — diff < 5pp → note is None
    section = _filing_section(
        text="Gross margin of 43.5% was achieved in fiscal 2023.",
        section="mdna",
    )
    monkeypatch.setattr("agent.tools.filings.edgar_client", lambda: _FakeEdgar(section))
    monkeypatch.setattr(
        "agent.tools.filings.yfinance_client",
        lambda: _FakeYF(fundamentals=lambda t: _fundamentals(t, gross_margin=43.8)),
    )
    result = ReadFilingTool().run(
        ReadFilingInput(ticker="AAPL", filing_type="10-K", section="mdna"), _ctx()
    )
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, FilingSection)
    assert result.data.aggregator_discrepancy_note is None


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


def test_screen_universe_unknown_key_rejected() -> None:
    with pytest.raises(ValueError, match="totally_unknown_key"):
        ScreenUniverseInput(criteria={"totally_unknown_key": 999, "pe_ratio_max": 15})


def test_screen_universe_max_analyst_coverage_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = {"UNCOV": 1, "COVERED": 25, "ZERO": 0}
    yf = _FakeYF(fundamentals=lambda t: _fundamentals(t, analyst_count=counts[t]))
    monkeypatch.setattr("agent.tools.screen._universe", lambda: ["UNCOV", "COVERED", "ZERO"])
    monkeypatch.setattr("agent.tools.screen.yfinance_client", lambda: yf)
    result = ScreenUniverseTool().run(
        ScreenUniverseInput(criteria={"max_analyst_coverage": 3}), _ctx()
    )
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ScreenResult)
    # 1 and 0 are <= 3 and pass; 25 is filtered OUT.
    assert result.data.tickers == ["UNCOV", "ZERO"]


def test_screen_universe_max_analyst_coverage_none_excludes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unknown coverage (None) must NOT silently pass a coverage gate.
    yf = _FakeYF(fundamentals=lambda t: _fundamentals(t, analyst_count=None))
    monkeypatch.setattr("agent.tools.screen._universe", lambda: ["UNKNOWN"])
    monkeypatch.setattr("agent.tools.screen.yfinance_client", lambda: yf)
    result = ScreenUniverseTool().run(
        ScreenUniverseInput(criteria={"max_analyst_coverage": 3}), _ctx()
    )
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ScreenResult)
    assert result.data.tickers == []


def test_screen_universe_require_zero_analyst_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = {"ZERO": 0, "ONE": 1, "UNKNOWN": None}
    yf = _FakeYF(fundamentals=lambda t: _fundamentals(t, analyst_count=counts[t]))
    monkeypatch.setattr("agent.tools.screen._universe", lambda: ["ZERO", "ONE", "UNKNOWN"])
    monkeypatch.setattr("agent.tools.screen.yfinance_client", lambda: yf)
    result = ScreenUniverseTool().run(
        ScreenUniverseInput(criteria={"require_zero_analyst_coverage": 1}), _ctx()
    )
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ScreenResult)
    # Only a demonstrably-zero-coverage name passes; 1 analyst and unknown both fail.
    assert result.data.tickers == ["ZERO"]


def test_screen_universe_require_zero_analyst_coverage_zero_threshold_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # threshold 0 disables the gate (mirrors require_net_cash=0).
    yf = _FakeYF(fundamentals=lambda t: _fundamentals(t, analyst_count=42))
    monkeypatch.setattr("agent.tools.screen._universe", lambda: ["COVERED"])
    monkeypatch.setattr("agent.tools.screen.yfinance_client", lambda: yf)
    result = ScreenUniverseTool().run(
        ScreenUniverseInput(criteria={"require_zero_analyst_coverage": 0}), _ctx()
    )
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ScreenResult)
    assert result.data.tickers == ["COVERED"]


def _valuation_for_screen(
    ticker: str = "AAPL",
    *,
    ev_to_ebit: float | None = 8.0,
    price_to_ncav: float | None = 0.7,
    market_cap_usd: int | None = 500_000_000,
    net_cash_positive: bool = True,
    net_cash_usd: int | None = 50_000_000,
) -> ValuationData:
    return ValuationData(
        ticker=ticker,
        as_of=date.today(),
        enterprise_value=None,
        ev_to_ebit=ev_to_ebit,
        ev_to_ebitda=None,
        acquirers_multiple=ev_to_ebit,
        fcf_yield=None,
        earnings_yield=None,
        market_cap_usd=market_cap_usd,
        ncav=None,
        ncav_to_market_cap=None,
        is_net_net=False,
        price_to_ncav=price_to_ncav,
        net_cash_usd=net_cash_usd,
        net_cash_positive=net_cash_positive,
        p_tangible_book=None,
        dividend_yield_pct=None,
        data_age_hours=0,
    )


def _quality_for_screen(
    ticker: str = "AAPL",
    *,
    consecutive_profit_years: int | None = 7,
) -> QualityData:
    return QualityData(
        ticker=ticker,
        as_of=date.today(),
        roic_pct=None,
        roic_series=[],
        roic_mean=None,
        roa_pct=None,
        gross_margin_pct=None,
        gross_margin_series=[],
        gross_margin_stdev=None,
        cash_conversion_ttm=None,
        cash_conversion_series=[],
        consecutive_profit_years=consecutive_profit_years,
        ncav_trend=None,
        data_age_hours=0,
    )


def test_screen_universe_max_ev_ebit_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    yf = _FakeYF(
        fundamentals=lambda t: _fundamentals(t),
        valuation=_valuation_for_screen(ev_to_ebit=8.0),
    )
    monkeypatch.setattr("agent.tools.screen._universe", lambda: ["AAPL"])
    monkeypatch.setattr("agent.tools.screen.yfinance_client", lambda: yf)
    result = ScreenUniverseTool().run(ScreenUniverseInput(criteria={"max_ev_ebit": 10}), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ScreenResult)
    assert result.data.tickers == ["AAPL"]


def test_screen_universe_max_ev_ebit_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    yf = _FakeYF(
        fundamentals=lambda t: _fundamentals(t),
        valuation=_valuation_for_screen(ev_to_ebit=15.0),
    )
    monkeypatch.setattr("agent.tools.screen._universe", lambda: ["AAPL"])
    monkeypatch.setattr("agent.tools.screen.yfinance_client", lambda: yf)
    result = ScreenUniverseTool().run(ScreenUniverseInput(criteria={"max_ev_ebit": 10}), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ScreenResult)
    assert result.data.tickers == []


def test_screen_universe_max_ev_ebit_none_excludes(monkeypatch: pytest.MonkeyPatch) -> None:
    yf = _FakeYF(
        fundamentals=lambda t: _fundamentals(t),
        valuation=_valuation_for_screen(ev_to_ebit=None),
    )
    monkeypatch.setattr("agent.tools.screen._universe", lambda: ["AAPL"])
    monkeypatch.setattr("agent.tools.screen.yfinance_client", lambda: yf)
    result = ScreenUniverseTool().run(ScreenUniverseInput(criteria={"max_ev_ebit": 10}), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ScreenResult)
    assert result.data.tickers == []


def test_screen_universe_max_price_to_ncav(monkeypatch: pytest.MonkeyPatch) -> None:
    pass_v = _valuation_for_screen(price_to_ncav=0.7)
    fail_v = _valuation_for_screen(price_to_ncav=1.5)
    valuation_by = {"AAPL": pass_v, "MSFT": fail_v}
    yf = _FakeYF(
        fundamentals=lambda t: _fundamentals(t),
        valuation=lambda t: valuation_by[t],
    )
    monkeypatch.setattr("agent.tools.screen._universe", lambda: ["AAPL", "MSFT"])
    monkeypatch.setattr("agent.tools.screen.yfinance_client", lambda: yf)
    result = ScreenUniverseTool().run(
        ScreenUniverseInput(criteria={"max_price_to_ncav": 1.0}), _ctx()
    )
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ScreenResult)
    assert result.data.tickers == ["AAPL"]


def test_screen_universe_max_market_cap_usd(monkeypatch: pytest.MonkeyPatch) -> None:
    small = _valuation_for_screen(market_cap_usd=300_000_000)
    large = _valuation_for_screen(market_cap_usd=5_000_000_000)
    valuation_by = {"AAPL": small, "MSFT": large}
    yf = _FakeYF(
        fundamentals=lambda t: _fundamentals(t),
        valuation=lambda t: valuation_by[t],
    )
    monkeypatch.setattr("agent.tools.screen._universe", lambda: ["AAPL", "MSFT"])
    monkeypatch.setattr("agent.tools.screen.yfinance_client", lambda: yf)
    result = ScreenUniverseTool().run(
        ScreenUniverseInput(criteria={"max_market_cap_usd": 1_000_000_000}), _ctx()
    )
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ScreenResult)
    assert result.data.tickers == ["AAPL"]


def test_screen_universe_min_market_cap_and_dividend(monkeypatch: pytest.MonkeyPatch) -> None:
    passing = _valuation_for_screen(market_cap_usd=50_000_000)
    passing.dividend_yield_pct = 2.0
    shell = _valuation_for_screen(market_cap_usd=2_000_000)
    shell.dividend_yield_pct = 2.0
    no_dividend = _valuation_for_screen(market_cap_usd=50_000_000)
    no_dividend.dividend_yield_pct = 0.0
    valuation_by = {"PASS": passing, "SHELL": shell, "NODIV": no_dividend}
    yf = _FakeYF(
        fundamentals=lambda t: _fundamentals(t),
        valuation=lambda t: valuation_by[t],
    )
    monkeypatch.setattr("agent.tools.screen._universe", lambda: list(valuation_by))
    monkeypatch.setattr("agent.tools.screen.yfinance_client", lambda: yf)
    result = ScreenUniverseTool().run(
        ScreenUniverseInput(
            criteria={"min_market_cap_usd": 5_400_000, "min_dividend_yield_pct": 1.0}
        ),
        _ctx(),
    )
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ScreenResult)
    assert result.data.tickers == ["PASS"]


def test_screen_universe_require_net_cash_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    yf = _FakeYF(
        fundamentals=lambda t: _fundamentals(t),
        valuation=_valuation_for_screen(net_cash_positive=True),
    )
    monkeypatch.setattr("agent.tools.screen._universe", lambda: ["AAPL"])
    monkeypatch.setattr("agent.tools.screen.yfinance_client", lambda: yf)
    result = ScreenUniverseTool().run(ScreenUniverseInput(criteria={"require_net_cash": 1}), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ScreenResult)
    assert result.data.tickers == ["AAPL"]


def test_screen_universe_require_net_cash_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    yf = _FakeYF(
        fundamentals=lambda t: _fundamentals(t),
        valuation=_valuation_for_screen(net_cash_positive=False),
    )
    monkeypatch.setattr("agent.tools.screen._universe", lambda: ["AAPL"])
    monkeypatch.setattr("agent.tools.screen.yfinance_client", lambda: yf)
    result = ScreenUniverseTool().run(ScreenUniverseInput(criteria={"require_net_cash": 1}), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ScreenResult)
    assert result.data.tickers == []


def test_screen_universe_require_net_cash_zero_threshold_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # threshold=0 means "don't require" — net-debt company still passes
    yf = _FakeYF(
        fundamentals=lambda t: _fundamentals(t),
        valuation=_valuation_for_screen(net_cash_positive=False),
    )
    monkeypatch.setattr("agent.tools.screen._universe", lambda: ["AAPL"])
    monkeypatch.setattr("agent.tools.screen.yfinance_client", lambda: yf)
    result = ScreenUniverseTool().run(ScreenUniverseInput(criteria={"require_net_cash": 0}), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ScreenResult)
    assert result.data.tickers == ["AAPL"]


def test_screen_universe_min_consecutive_profit_years_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yf = _FakeYF(
        fundamentals=lambda t: _fundamentals(t),
        quality=_quality_for_screen(consecutive_profit_years=7),
    )
    monkeypatch.setattr("agent.tools.screen._universe", lambda: ["AAPL"])
    monkeypatch.setattr("agent.tools.screen.yfinance_client", lambda: yf)
    result = ScreenUniverseTool().run(
        ScreenUniverseInput(criteria={"min_consecutive_profit_years": 5}), _ctx()
    )
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ScreenResult)
    assert result.data.tickers == ["AAPL"]


def test_screen_universe_min_consecutive_profit_years_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yf = _FakeYF(
        fundamentals=lambda t: _fundamentals(t),
        quality=_quality_for_screen(consecutive_profit_years=3),
    )
    monkeypatch.setattr("agent.tools.screen._universe", lambda: ["AAPL"])
    monkeypatch.setattr("agent.tools.screen.yfinance_client", lambda: yf)
    result = ScreenUniverseTool().run(
        ScreenUniverseInput(criteria={"min_consecutive_profit_years": 5}), _ctx()
    )
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ScreenResult)
    assert result.data.tickers == []


def test_screen_universe_combined_dirt_criteria(monkeypatch: pytest.MonkeyPatch) -> None:
    # AAPL passes all; MSFT fails max_ev_ebit
    val_by = {
        "AAPL": _valuation_for_screen(ev_to_ebit=7.0, net_cash_positive=True),
        "MSFT": _valuation_for_screen(ev_to_ebit=20.0, net_cash_positive=True),
    }
    qual_by = {
        "AAPL": _quality_for_screen(consecutive_profit_years=8),
        "MSFT": _quality_for_screen(consecutive_profit_years=8),
    }
    yf = _FakeYF(
        fundamentals=lambda t: _fundamentals(t),
        valuation=lambda t: val_by[t],
        quality=lambda t: qual_by[t],
    )
    monkeypatch.setattr("agent.tools.screen._universe", lambda: ["AAPL", "MSFT"])
    monkeypatch.setattr("agent.tools.screen.yfinance_client", lambda: yf)
    result = ScreenUniverseTool().run(
        ScreenUniverseInput(
            criteria={
                "max_ev_ebit": 10,
                "require_net_cash": 1,
                "min_consecutive_profit_years": 5,
            }
        ),
        _ctx(),
    )
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ScreenResult)
    assert result.data.tickers == ["AAPL"]


def test_screen_universe_limitation_note_always_present(monkeypatch: pytest.MonkeyPatch) -> None:
    yf = _FakeYF(fundamentals=lambda t: _fundamentals(t, pe=10.0))
    monkeypatch.setattr("agent.tools.screen._universe", lambda: ["AAPL"])
    monkeypatch.setattr("agent.tools.screen.yfinance_client", lambda: yf)
    result = ScreenUniverseTool().run(ScreenUniverseInput(criteria={"pe_ratio_max": 15}), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ScreenResult)
    note = result.data.universe_limitation_note
    assert "US-only" in note
    assert "Russell 2000" in note
    assert "300M" in note or "$300" in note


def test_screen_universe_russell2000_tickers_merged(monkeypatch: pytest.MonkeyPatch) -> None:
    # get_russell2000_tickers() returns R2000 tickers; _universe() merges them
    # with portfolio/watchlist and deduplicates. Verify R2000-only tickers pass filters.
    pe_by_ticker = {"AAPL": 12.0, "BE": 8.0, "CRDO": 25.0}
    yf = _FakeYF(
        fundamentals=lambda t: _fundamentals(t, pe=pe_by_ticker[t]),
        russell2000=["BE", "CRDO"],
    )
    # Patch _universe to simulate portfolio containing AAPL plus R2000 results
    monkeypatch.setattr("agent.tools.screen._universe", lambda: ["AAPL", "BE", "CRDO"])
    monkeypatch.setattr("agent.tools.screen.yfinance_client", lambda: yf)
    result = ScreenUniverseTool().run(ScreenUniverseInput(criteria={"pe_ratio_max": 15}), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ScreenResult)
    assert "AAPL" in result.data.tickers
    assert "BE" in result.data.tickers  # R2000 ticker that passes
    assert "CRDO" not in result.data.tickers  # R2000 ticker that fails


def test_screen_universe_r2k_network_error_falls_back_to_static(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # When get_russell2000_tickers() returns DataSourceError, _universe() still
    # returns portfolio/watchlist tickers — screen degrades gracefully.
    yf = _FakeYF(
        fundamentals=lambda t: _fundamentals(t, pe=10.0),
        russell2000=DataSourceError(error_code="network", message="timeout"),
    )
    monkeypatch.setattr("agent.tools.screen._universe", lambda: ["AAPL"])
    monkeypatch.setattr("agent.tools.screen.yfinance_client", lambda: yf)
    result = ScreenUniverseTool().run(ScreenUniverseInput(criteria={"pe_ratio_max": 15}), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ScreenResult)
    assert result.data.tickers == ["AAPL"]


def test_dirt_persona_note_diverges_from_screen_tool_note() -> None:
    # G9 globalized the DIRT persona's universe note (US small-caps + Milan/Madrid/Warsaw)
    # while the screen_universe TOOL deliberately stays US-only (Russell 2000). The two
    # surfaces are now intentionally decoupled: the screen tool's US-only note must NOT be a
    # substring of the globalized DIRT prompt, and the DIRT prompt names the non-US exchanges.
    from agent.persona import DIRT_SYSTEM_PROMPT
    from agent.tools.screen import _UNIVERSE_LIMITATION_NOTE

    assert _UNIVERSE_LIMITATION_NOTE not in DIRT_SYSTEM_PROMPT
    assert "US-only" in _UNIVERSE_LIMITATION_NOTE
    assert "US-only" not in DIRT_SYSTEM_PROMPT
    assert "Euronext Growth Milan" in DIRT_SYSTEM_PROMPT


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


# ── get_valuation_multiples ───────────────────────────────────────────────────


def _valuation(
    ticker: str = "AAPL",
    *,
    ev: int | None = 3_000_000_000_000,
    op_income: float | None = 120_000_000_000.0,
    ebitda: float | None = 150_000_000_000.0,
    fcf: float | None = 110_000_000_000.0,
    mkt_cap: int | None = 2_800_000_000_000,
    curr_assets: int | None = 150_000_000_000,
    total_liab: int | None = 300_000_000_000,
    tangible_bv: float | None = 60_000_000_000.0,
    div_yield: float | None = 0.5,
) -> ValuationData:
    ev_f = float(ev) if ev is not None else None
    ev_to_ebit = round(ev_f / op_income, 2) if ev_f and op_income and op_income > 0 else None
    ev_to_ebitda = round(ev_f / ebitda, 2) if ev_f and ebitda and ebitda > 0 else None
    fcf_yield = round(float(fcf) / ev_f * 100, 4) if fcf and ev_f and ev_f > 0 else None
    earnings_yield = round(op_income / ev_f * 100, 4) if op_income and ev_f and ev_f > 0 else None
    ncav: int | None = None
    if curr_assets is not None and total_liab is not None:
        ncav = curr_assets - total_liab
    ncav_to_mkt_cap: float | None = None
    is_net_net = False
    if ncav is not None and mkt_cap and mkt_cap > 0:
        ncav_to_mkt_cap = round(ncav / mkt_cap, 4)
        is_net_net = ncav > mkt_cap
    p_tangible_book: float | None = None
    if mkt_cap and tangible_bv and tangible_bv > 0:
        p_tangible_book = round(mkt_cap / tangible_bv, 2)
    price_to_ncav_val: float | None = None
    if ncav is not None and ncav > 0 and mkt_cap and mkt_cap > 0:
        price_to_ncav_val = round(float(mkt_cap) / ncav, 4)
    return ValuationData(
        ticker=ticker,
        as_of=date.today(),
        enterprise_value=ev,
        ev_to_ebit=ev_to_ebit,
        ev_to_ebitda=ev_to_ebitda,
        acquirers_multiple=ev_to_ebit,
        fcf_yield=fcf_yield,
        earnings_yield=earnings_yield,
        market_cap_usd=mkt_cap,
        ncav=ncav,
        ncav_to_market_cap=ncav_to_mkt_cap,
        is_net_net=is_net_net,
        price_to_ncav=price_to_ncav_val,
        net_cash_usd=None,
        net_cash_positive=False,
        p_tangible_book=p_tangible_book,
        dividend_yield_pct=div_yield,
        data_age_hours=0,
    )


def test_get_valuation_multiples_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    v = _valuation()
    monkeypatch.setattr("agent.tools.valuation.yfinance_client", lambda: _FakeYF(valuation=v))
    result = GetValuationMultiplesTool().run(GetValuationMultiplesInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ValuationData)
    assert result.data.ev_to_ebit == pytest.approx(25.0)
    assert result.data.ev_to_ebitda == pytest.approx(20.0)
    assert result.data.acquirers_multiple == result.data.ev_to_ebit
    assert result.data.earnings_yield == pytest.approx(4.0)
    assert result.data.is_net_net is False


def test_get_valuation_multiples_net_net_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    # NCAV = 500B - 100B = 400B > market cap 300B → net-net
    v = _valuation(
        mkt_cap=300_000_000_000,
        curr_assets=500_000_000_000,
        total_liab=100_000_000_000,
    )
    monkeypatch.setattr("agent.tools.valuation.yfinance_client", lambda: _FakeYF(valuation=v))
    result = GetValuationMultiplesTool().run(GetValuationMultiplesInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ValuationData)
    assert result.data.ncav == 400_000_000_000
    assert result.data.is_net_net is True
    assert result.data.ncav_to_market_cap == pytest.approx(400 / 300, rel=1e-4)


def test_get_valuation_multiples_none_when_ev_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    v = _valuation(ev=None)
    monkeypatch.setattr("agent.tools.valuation.yfinance_client", lambda: _FakeYF(valuation=v))
    result = GetValuationMultiplesTool().run(GetValuationMultiplesInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ValuationData)
    assert result.data.ev_to_ebit is None
    assert result.data.ev_to_ebitda is None
    assert result.data.acquirers_multiple is None
    assert result.data.fcf_yield is None
    assert result.data.earnings_yield is None


def test_get_valuation_multiples_maps_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    err = DataSourceError(error_code="not_found", message="no data")
    monkeypatch.setattr("agent.tools.valuation.yfinance_client", lambda: _FakeYF(valuation=err))
    result = GetValuationMultiplesTool().run(GetValuationMultiplesInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultError)
    assert result.error_code == "not_found"
    assert result.retryable is False


# ── get_quality_metrics ───────────────────────────────────────────────────────


def _quality(ticker: str = "AAPL") -> QualityData:
    return QualityData(
        ticker=ticker,
        as_of=date.today(),
        roic_pct=41.2,
        roic_series=[41.2, 39.8, 38.5, 30.1],
        roic_mean=37.4,
        roa_pct=27.5,
        gross_margin_pct=44.1,
        gross_margin_series=[44.1, 43.3, 41.8, 38.2],
        gross_margin_stdev=2.4,
        cash_conversion_ttm=1.03,
        cash_conversion_series=[1.03, 1.12, 0.98, 1.28],
        consecutive_profit_years=5,
        ncav_trend="growing",
        data_age_hours=2000,
    )


def test_get_quality_metrics_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    q = _quality()
    monkeypatch.setattr("agent.tools.quality.yfinance_client", lambda: _FakeYF(quality=q))
    result = GetQualityMetricsTool().run(GetQualityMetricsInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, QualityData)
    assert result.data.roic_pct == pytest.approx(41.2)
    assert result.data.gross_margin_stdev == pytest.approx(2.4)
    assert result.data.cash_conversion_ttm == pytest.approx(1.03)


def test_get_quality_metrics_maps_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    err = DataSourceError(error_code="not_found", message="No data for ZZZZZ")
    monkeypatch.setattr("agent.tools.quality.yfinance_client", lambda: _FakeYF(quality=err))
    result = GetQualityMetricsTool().run(GetQualityMetricsInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultError)
    assert result.error_code == "not_found"


def test_get_quality_metrics_catches_unexpected_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        def get_quality_metrics(self, ticker: str) -> QualityData:
            raise RuntimeError("unexpected boom")

    monkeypatch.setattr("agent.tools.quality.yfinance_client", lambda: _Boom())
    result = GetQualityMetricsTool().run(GetQualityMetricsInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultError)
    assert result.error_code == "unknown"


# ── get_insider_activity ──────────────────────────────────────────────────────


def _txn(
    name: str = "Tim Cook",
    txn_type: str = "buy",
    shares: int = 1000,
    value: float | None = 150_000.0,
    txn_date: date | None = None,
) -> FinnhubInsiderTransaction:
    from typing import Literal

    t: Literal["buy", "sell", "other"] = txn_type  # type: ignore[assignment]
    return FinnhubInsiderTransaction(
        name=name,
        transaction_type=t,
        shares=shares,
        value=value,
        transaction_date=txn_date or date.today(),
    )


def _ownership(ticker: str = "AAPL", insider: float = 0.07, inst: float = 59.5) -> OwnershipData:
    return OwnershipData(
        ticker=ticker, as_of=date.today(), insider_pct=insider, institutional_pct=inst
    )


def test_get_insider_activity_net_buy_bullish(monkeypatch: pytest.MonkeyPatch) -> None:
    txns = [_txn(shares=2000, txn_type="buy"), _txn(shares=500, txn_type="sell")]
    fh = _FakeFinnhub(insider_txns=txns)
    yf = _FakeYF(ownership=_ownership())
    monkeypatch.setattr("agent.tools.insider.finnhub_client", lambda: fh)
    monkeypatch.setattr("agent.tools.insider.yfinance_client", lambda: yf)
    result = GetInsiderActivityTool().run(GetInsiderActivityInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, InsiderActivity)
    assert result.data.insider_sentiment == "bullish"
    assert result.data.net_shares_bought == 1500
    assert result.data.insider_ownership_pct == pytest.approx(0.07)
    assert result.data.institutional_ownership_pct == pytest.approx(59.5)
    assert len(result.data.transactions) == 2


def test_get_insider_activity_net_sell_bearish(monkeypatch: pytest.MonkeyPatch) -> None:
    txns = [_txn(shares=100, txn_type="buy"), _txn(shares=5000, txn_type="sell")]
    monkeypatch.setattr(
        "agent.tools.insider.finnhub_client", lambda: _FakeFinnhub(insider_txns=txns)
    )
    monkeypatch.setattr("agent.tools.insider.yfinance_client", lambda: _FakeYF())
    result = GetInsiderActivityTool().run(GetInsiderActivityInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, InsiderActivity)
    assert result.data.insider_sentiment == "bearish"
    assert result.data.net_shares_bought == -4900


def test_get_insider_activity_only_other_txns_insufficient(monkeypatch: pytest.MonkeyPatch) -> None:
    txns = [_txn(txn_type="other", shares=500)]
    monkeypatch.setattr(
        "agent.tools.insider.finnhub_client", lambda: _FakeFinnhub(insider_txns=txns)
    )
    monkeypatch.setattr("agent.tools.insider.yfinance_client", lambda: _FakeYF())
    result = GetInsiderActivityTool().run(GetInsiderActivityInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, InsiderActivity)
    assert result.data.insider_sentiment == "insufficient_data"
    assert result.data.net_shares_bought == 0


def test_get_insider_activity_no_finnhub_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.tools.insider.finnhub_client", lambda: None)
    result = GetInsiderActivityTool().run(GetInsiderActivityInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultError)
    assert result.error_code == "not_found"
    assert result.retryable is False


def test_get_insider_activity_maps_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    err = DataSourceError(error_code="network", message="connection refused")
    monkeypatch.setattr(
        "agent.tools.insider.finnhub_client",
        lambda: _FakeFinnhub(insider_txns=err),
    )
    monkeypatch.setattr("agent.tools.insider.yfinance_client", lambda: _FakeYF())
    result = GetInsiderActivityTool().run(GetInsiderActivityInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultError)
    assert result.error_code == "network"
    assert result.retryable is True


def test_get_insider_activity_ownership_none_when_yf_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    txns = [_txn(shares=1000, txn_type="buy")]
    monkeypatch.setattr(
        "agent.tools.insider.finnhub_client", lambda: _FakeFinnhub(insider_txns=txns)
    )
    monkeypatch.setattr(
        "agent.tools.insider.yfinance_client",
        lambda: _FakeYF(ownership=DataSourceError(error_code="not_found", message="gone")),
    )
    result = GetInsiderActivityTool().run(GetInsiderActivityInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, InsiderActivity)
    assert result.data.insider_ownership_pct is None
    assert result.data.institutional_ownership_pct is None
    assert result.data.insider_sentiment == "bullish"


# ── get_peer_comparison ───────────────────────────────────────────────────────


def _peer_valuation(
    ticker: str, *, ev_to_ebit: float | None = 20.0, fcf_yield: float | None = 5.0
) -> ValuationData:
    return ValuationData(
        ticker=ticker,
        as_of=date.today(),
        enterprise_value=1_000_000_000,
        ev_to_ebit=ev_to_ebit,
        ev_to_ebitda=None,
        acquirers_multiple=ev_to_ebit,
        fcf_yield=fcf_yield,
        earnings_yield=None,
        market_cap_usd=None,
        ncav=None,
        ncav_to_market_cap=None,
        is_net_net=False,
        price_to_ncav=None,
        net_cash_usd=None,
        net_cash_positive=False,
        p_tangible_book=None,
        dividend_yield_pct=None,
        data_age_hours=0,
    )


# ── get_financial_strength ────────────────────────────────────────────────────


def _financial_strength(
    ticker: str = "AAPL",
    *,
    f_score: int | None = 7,
    z_score: float | None = 3.5,
    z_zone: Literal["distress", "grey", "safe"] | None = "safe",
    interest_coverage: float | None = 12.0,
    current_ratio: float | None = 1.5,
    net_debt_to_ebitda: float | None = 0.8,
) -> FinancialStrengthData:
    signals = PiotroskySignals(
        roa_positive=True,
        op_cf_positive=True,
        roa_improved=True,
        accruals_negative=True,
        leverage_decreased=True,
        current_ratio_improved=True,
        no_dilution=True,
        gross_margin_improved=False,
        asset_turnover_improved=False,
    )
    return FinancialStrengthData(
        ticker=ticker,
        as_of=date.today(),
        f_score=f_score,
        f_signals=signals,
        z_score=z_score,
        z_zone=z_zone,
        interest_coverage=interest_coverage,
        current_ratio=current_ratio,
        net_debt_to_ebitda=net_debt_to_ebitda,
        data_age_hours=0,
    )


def test_get_peer_comparison_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # AAPL: pe=10, ev_to_ebit=15, gross_margin=50
    # MSFT: pe=20, ev_to_ebit=25, gross_margin=60
    # GOOG: pe=15, ev_to_ebit=20, gross_margin=40
    fund_map = {
        "AAPL": _fundamentals("AAPL", pe=10.0, gross_margin=50.0),
        "MSFT": _fundamentals("MSFT", pe=20.0, gross_margin=60.0),
        "GOOG": _fundamentals("GOOG", pe=15.0, gross_margin=40.0),
    }
    val_map = {
        "AAPL": _peer_valuation("AAPL", ev_to_ebit=15.0, fcf_yield=6.0),
        "MSFT": _peer_valuation("MSFT", ev_to_ebit=25.0, fcf_yield=4.0),
        "GOOG": _peer_valuation("GOOG", ev_to_ebit=20.0, fcf_yield=5.0),
    }
    yf = _FakeYF(
        fundamentals=lambda t: fund_map[t],
        valuation=lambda t: val_map[t],
    )
    monkeypatch.setattr("agent.tools.peers.yfinance_client", lambda: yf)
    result = GetPeerComparisonTool().run(
        GetPeerComparisonInput(ticker="AAPL", peers=["MSFT", "GOOG"]), _ctx()
    )
    assert isinstance(result, ToolResultOk)
    pc = result.data
    assert isinstance(pc, PeerComparison)
    assert pc.ticker == "AAPL"
    assert set(pc.peers) == {"MSFT", "GOOG"}
    assert len(pc.all_metrics) == 3
    assert pc.all_metrics[0].ticker == "AAPL"

    # P/E: lower is better; sorted [AAPL=10, GOOG=15, MSFT=20] → AAPL rank=1
    pe_summary = pc.summary["pe_ratio"]
    assert pe_summary.ticker_rank == 1
    assert pe_summary.ticker_percentile == pytest.approx(100.0)
    assert pe_summary.peer_median == pytest.approx(17.5)  # median of [20, 15]

    # EV/EBIT: lower is better; sorted [AAPL=15, GOOG=20, MSFT=25] → AAPL rank=1
    ev_summary = pc.summary["ev_to_ebit"]
    assert ev_summary.ticker_rank == 1
    assert ev_summary.ticker_percentile == pytest.approx(100.0)

    # gross_margin: higher better; AAPL=50, MSFT=60, GOOG=40 → AAPL rank=2
    gm_summary = pc.summary["gross_margin_pct"]
    assert gm_summary.ticker_rank == 2
    assert gm_summary.peer_median == pytest.approx(50.0)  # median of [60, 40]


def test_get_peer_comparison_explicit_peers_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    fund_map = {
        "AAPL": _fundamentals("AAPL", pe=10.0),
        "IBM": _fundamentals("IBM", pe=12.0),
        "DELL": _fundamentals("DELL", pe=8.0),
    }
    yf = _FakeYF(
        fundamentals=lambda t: fund_map[t],
        valuation=lambda t: _peer_valuation(t),
    )
    monkeypatch.setattr("agent.tools.peers.yfinance_client", lambda: yf)
    result = GetPeerComparisonTool().run(
        GetPeerComparisonInput(ticker="AAPL", peers=["IBM", "DELL"]), _ctx()
    )
    assert isinstance(result, ToolResultOk)
    pc = result.data
    assert isinstance(pc, PeerComparison)
    assert set(pc.peers) == {"IBM", "DELL"}


def test_get_peer_comparison_failed_peer_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    not_found = DataSourceError(error_code="not_found", message="no data")
    fund_map: dict[str, FundamentalsData | DataSourceError] = {
        "AAPL": _fundamentals("AAPL", pe=10.0),
        "MSFT": _fundamentals("MSFT", pe=20.0),
        "GOOG": not_found,
        "META": _fundamentals("META", pe=15.0),
    }

    def _val(t: str) -> ValuationData | DataSourceError:
        return not_found if t == "GOOG" else _peer_valuation(t)

    yf = _FakeYF(
        fundamentals=lambda t: fund_map[t],
        valuation=_val,
    )
    monkeypatch.setattr("agent.tools.peers.yfinance_client", lambda: yf)
    result = GetPeerComparisonTool().run(
        GetPeerComparisonInput(ticker="AAPL", peers=["MSFT", "GOOG", "META"]), _ctx()
    )
    assert isinstance(result, ToolResultOk)
    pc = result.data
    assert isinstance(pc, PeerComparison)
    assert "GOOG" not in pc.peers
    assert "MSFT" in pc.peers
    assert "META" in pc.peers


def test_get_peer_comparison_insufficient_peers_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    not_found = DataSourceError(error_code="not_found", message="no data")
    yf = _FakeYF(
        fundamentals=lambda t: _fundamentals(t) if t == "AAPL" else not_found,
        valuation=lambda t: not_found,
    )
    monkeypatch.setattr("agent.tools.peers.yfinance_client", lambda: yf)
    result = GetPeerComparisonTool().run(
        GetPeerComparisonInput(ticker="AAPL", peers=["MSFT", "GOOG"]), _ctx()
    )
    assert isinstance(result, ToolResultError)
    assert result.error_code == "not_found"


def test_get_peer_comparison_sector_fallback_to_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    fund_map = {
        "AAPL": _fundamentals("AAPL", pe=10.0, sector=None),
        "MSFT": _fundamentals("MSFT", pe=20.0),
        "GOOG": _fundamentals("GOOG", pe=15.0),
    }
    yf = _FakeYF(
        fundamentals=lambda t: fund_map.get(t, _fundamentals(t)),
        valuation=lambda t: _peer_valuation(t),
    )
    monkeypatch.setattr("agent.tools.peers.yfinance_client", lambda: yf)
    monkeypatch.setattr("agent.tools.peers._universe", lambda: ["MSFT", "GOOG"])
    result = GetPeerComparisonTool().run(GetPeerComparisonInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    pc = result.data
    assert isinstance(pc, PeerComparison)
    assert set(pc.peers) == {"MSFT", "GOOG"}


def test_get_financial_strength_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    fs = _financial_strength()
    monkeypatch.setattr(
        "agent.tools.financial_strength.yfinance_client",
        lambda: _FakeYF(financial_strength=fs),
    )
    result = GetFinancialStrengthTool().run(GetFinancialStrengthInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, FinancialStrengthData)
    assert result.data.f_score == 7
    assert result.data.z_zone == "safe"
    assert result.data.interest_coverage == pytest.approx(12.0)
    assert result.data.f_signals.roa_positive is True
    assert result.data.f_signals.gross_margin_improved is False


def test_get_financial_strength_none_when_data_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    fs = _financial_strength(f_score=None, z_score=None, z_zone=None, interest_coverage=None)
    monkeypatch.setattr(
        "agent.tools.financial_strength.yfinance_client",
        lambda: _FakeYF(financial_strength=fs),
    )
    result = GetFinancialStrengthTool().run(GetFinancialStrengthInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, FinancialStrengthData)
    assert result.data.f_score is None
    assert result.data.z_score is None
    assert result.data.z_zone is None
    assert result.data.interest_coverage is None


def test_get_financial_strength_maps_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    err = DataSourceError(error_code="not_found", message="no data")
    monkeypatch.setattr(
        "agent.tools.financial_strength.yfinance_client",
        lambda: _FakeYF(financial_strength=err),
    )
    result = GetFinancialStrengthTool().run(GetFinancialStrengthInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultError)
    assert result.error_code == "not_found"
    assert result.retryable is False


# ── estimate_intrinsic_value ──────────────────────────────────────────────────


def _financials(
    ticker: str = "AAPL",
    *,
    free_cash_flow: int | None = 1000,
    cfo: int | None = None,
    capex: int | None = None,
    shares_outstanding: int | None = 100,
    data_age_hours: int = 0,
) -> FinancialsHistory:
    return FinancialsHistory(
        ticker=ticker,
        as_of=date.today(),
        fiscal_years=[2023],
        income_statement=[
            IncomeStatementRow(
                fiscal_year=2023,
                revenue=10_000,
                gross_profit=None,
                operating_income=None,
                net_income=None,
                ebit=None,
                ebitda=None,
                interest_expense=None,
                pretax_income=None,
                tax_provision=None,
            )
        ],
        balance_sheet=[
            BalanceSheetRow(
                fiscal_year=2023,
                total_assets=None,
                total_liabilities=None,
                current_assets=None,
                current_liabilities=None,
                long_term_debt=None,
                total_debt=None,
                cash_and_equivalents=None,
                retained_earnings=None,
                common_stock=None,
                shares_outstanding=shares_outstanding,
            )
        ],
        cash_flow=[
            CashFlowRow(
                fiscal_year=2023,
                cfo=cfo,
                capex=capex,
                free_cash_flow=free_cash_flow,
                dividends_paid=None,
                buybacks=None,
            )
        ],
        data_age_hours=data_age_hours,
    )


def test_intrinsic_equity_value_deterministic() -> None:
    # base=1000, g==d==0.10, tg=0, N=10. Each discounted projection term = 1000 → 10*1000;
    # discounted terminal value = base*(1.1)^10 / 0.10 / (1.1)^10 = 10000. Total = 20000.
    a = DCFAssumptions(
        growth_rate=0.10, discount_rate=0.10, terminal_growth_rate=0.0, projection_years=10
    )
    assert _intrinsic_equity_value(1000.0, a) == pytest.approx(20_000.0)


def test_intrinsic_equity_value_none_when_discount_le_terminal() -> None:
    a = DCFAssumptions(
        growth_rate=0.05, discount_rate=0.02, terminal_growth_rate=0.025, projection_years=10
    )
    assert _intrinsic_equity_value(1000.0, a) is None


def test_reverse_dcf_recovers_growth() -> None:
    a = DCFAssumptions(
        growth_rate=0.10, discount_rate=0.10, terminal_growth_rate=0.0, projection_years=10
    )
    # Target equity value 20000 with base 1000 implies the growth that produced it: 0.10.
    assert _reverse_dcf_growth(20_000.0, 1000.0, a) == pytest.approx(0.10, abs=1e-3)


def test_reverse_dcf_none_for_nonpositive_base() -> None:
    a = DCFAssumptions(
        growth_rate=0.08, discount_rate=0.10, terminal_growth_rate=0.025, projection_years=10
    )
    assert _reverse_dcf_growth(20_000.0, -5.0, a) is None


def test_estimate_intrinsic_value_deterministic_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yf = _FakeYF(financials=_financials(), price=_price())  # price 182.5
    monkeypatch.setattr("agent.tools.intrinsic_value.yfinance_client", lambda: yf)
    # Override assumptions so the hand-computed value is exact: per-share = 20000/100 = 200.
    result = EstimateIntrinsicValueTool().run(
        EstimateIntrinsicValueInput(
            ticker="AAPL",
            growth_rate=0.10,
            discount_rate=0.10,
            terminal_growth_rate=0.0,
            projection_years=10,
        ),
        _ctx(),
    )
    assert isinstance(result, ToolResultOk)
    iv = result.data
    assert isinstance(iv, IntrinsicValue)
    assert iv.owner_earnings_base == 1000
    assert iv.owner_earnings_source == "free_cash_flow"
    assert iv.shares_outstanding == 100
    assert iv.intrinsic_value_per_share == pytest.approx(200.0)
    assert iv.intrinsic_equity_value == 20_000
    # margin of safety = (200 - 182.5) / 200 = 0.0875
    assert iv.margin_of_safety == pytest.approx(0.0875)
    # assumptions echoed
    assert iv.assumptions.growth_rate == pytest.approx(0.10)
    assert iv.assumptions.projection_years == 10
    # reverse DCF: implied growth that makes value == price*shares (182.5*100 = 18250) < 0.10
    assert iv.reverse_dcf_implied_growth is not None
    assert iv.reverse_dcf_implied_growth < 0.10


def test_estimate_intrinsic_value_uses_conservative_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yf = _FakeYF(financials=_financials(), price=_price())
    monkeypatch.setattr("agent.tools.intrinsic_value.yfinance_client", lambda: yf)
    result = EstimateIntrinsicValueTool().run(EstimateIntrinsicValueInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    iv = result.data
    assert isinstance(iv, IntrinsicValue)
    assert iv.assumptions.growth_rate == pytest.approx(0.08)
    assert iv.assumptions.discount_rate == pytest.approx(0.10)
    assert iv.assumptions.terminal_growth_rate == pytest.approx(0.025)
    assert iv.assumptions.projection_years == 10
    assert iv.intrinsic_value_per_share is not None


def test_estimate_intrinsic_value_cfo_minus_capex_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No reported FCF → fall back to CFO + capex (capex is negative in yfinance).
    fin = _financials(free_cash_flow=None, cfo=1500, capex=-500)
    monkeypatch.setattr(
        "agent.tools.intrinsic_value.yfinance_client",
        lambda: _FakeYF(financials=fin, price=_price()),
    )
    result = EstimateIntrinsicValueTool().run(EstimateIntrinsicValueInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    iv = result.data
    assert isinstance(iv, IntrinsicValue)
    assert iv.owner_earnings_base == 1000
    assert iv.owner_earnings_source == "cfo_minus_capex"


def test_estimate_intrinsic_value_none_when_base_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fin = _financials(free_cash_flow=None, cfo=None, capex=None)
    monkeypatch.setattr(
        "agent.tools.intrinsic_value.yfinance_client",
        lambda: _FakeYF(financials=fin, price=_price()),
    )
    result = EstimateIntrinsicValueTool().run(EstimateIntrinsicValueInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    iv = result.data
    assert isinstance(iv, IntrinsicValue)
    assert iv.owner_earnings_base is None
    assert iv.intrinsic_value_per_share is None
    assert iv.margin_of_safety is None
    assert iv.reverse_dcf_implied_growth is None


def test_estimate_intrinsic_value_maps_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    err = DataSourceError(error_code="not_found", message="no data")
    monkeypatch.setattr(
        "agent.tools.intrinsic_value.yfinance_client",
        lambda: _FakeYF(financials=err),
    )
    result = EstimateIntrinsicValueTool().run(EstimateIntrinsicValueInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultError)
    assert result.error_code == "not_found"
    assert result.retryable is False


def test_estimate_intrinsic_value_catches_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Boom:
        def get_financials(self, ticker: str) -> FinancialsHistory:
            raise RuntimeError("kaboom")

    monkeypatch.setattr("agent.tools.intrinsic_value.yfinance_client", lambda: _Boom())
    result = EstimateIntrinsicValueTool().run(EstimateIntrinsicValueInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultError)
    assert result.error_code == "unknown"


# ── get_capital_allocation ────────────────────────────────────────────────────


def _capital_allocation(ticker: str = "AAPL") -> CapitalAllocation:
    return CapitalAllocation(
        ticker=ticker,
        as_of=date.today(),
        years_covered=4,
        share_count_cagr_pct=-2.89,
        share_count_series=[15550061000, 15943425000, 16426786000, 16976763000],
        buyback_yield_pct=2.7211,
        dividend_yield_pct=0.5272,
        shareholder_yield_pct=3.2483,
        dividend_growth_streak=3,
        payout_ratio_pct=15.4905,
        net_debt_series=[81123000000, 96423000000, 89779000000, 74420000000],
        net_debt_trajectory="levering",
        data_age_hours=0,
    )


def test_get_capital_allocation_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    ca = _capital_allocation()
    monkeypatch.setattr(
        "agent.tools.capital_allocation.yfinance_client",
        lambda: _FakeYF(capital_allocation=ca),
    )
    result = GetCapitalAllocationTool().run(GetCapitalAllocationInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, CapitalAllocation)
    assert result.data.share_count_cagr_pct == pytest.approx(-2.89)
    assert result.data.shareholder_yield_pct == pytest.approx(3.2483)
    assert result.data.dividend_growth_streak == 3
    assert result.data.net_debt_trajectory == "levering"


def test_get_capital_allocation_maps_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    err = DataSourceError(error_code="not_found", message="no data")
    monkeypatch.setattr(
        "agent.tools.capital_allocation.yfinance_client",
        lambda: _FakeYF(capital_allocation=err),
    )
    result = GetCapitalAllocationTool().run(GetCapitalAllocationInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultError)
    assert result.error_code == "not_found"
    assert result.retryable is False


# ── get_key_persons ───────────────────────────────────────────────────────────


def _key_persons_raw(*, controlling: bool = False) -> KeyPersonsRaw:
    return KeyPersonsRaw(
        ticker="AAPL",
        as_of=date.today(),
        officers=[
            OfficerRecord(
                name="Timothy D. Cook",
                title="CEO & Director",
                year_born=1961,
                total_pay_usd=63151817,
            ),
            OfficerRecord(
                name="Luca Maestri",
                title="CFO",
                year_born=1964,
                total_pay_usd=27230396,
            ),
        ],
        institutional_holders=[
            InstitutionalHolderRecord(
                name="Vanguard Group Inc",
                shares=1273985728,
                pct_held=0.35 if controlling else 0.0796,
                value=241148000000,
            ),
            InstitutionalHolderRecord(
                name="BlackRock Inc.",
                shares=1020413756,
                pct_held=0.0637,
                value=193013000000,
            ),
            InstitutionalHolderRecord(
                name="State Street Corporation",
                shares=594031369,
                pct_held=0.0371,
                value=112369000000,
            ),
        ],
        data_age_hours=0,
    )


def test_get_key_persons_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _key_persons_raw()
    monkeypatch.setattr("agent.tools.persons.yfinance_client", lambda: _FakeYF(key_persons=raw))
    monkeypatch.setattr(
        "agent.tools.persons.edgar_client",
        lambda: _FakeEdgar(sc13=[]),
    )
    result = GetKeyPersonsTool().run(GetKeyPersonsInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, KeyPersonsData)
    data = result.data
    assert data.ticker == "AAPL"
    # Officers appear
    names = [p.name for p in data.persons]
    assert "Timothy D. Cook" in names
    assert "Luca Maestri" in names
    # Officers have no ownership_pct
    officers = [p for p in data.persons if p.source == "yfinance_officers"]
    assert all(p.ownership_pct is None for p in officers)
    # Institutional holders appear (yfinance returns all top holders, not filtered by %)
    holders = [p for p in data.persons if p.source == "yfinance_holders"]
    holder_names = [p.name for p in holders]
    assert "Vanguard Group Inc" in holder_names
    assert "BlackRock Inc." in holder_names
    # Holders have ownership_pct set
    assert all(p.ownership_pct is not None for p in holders)
    # No single holder ≥ 30%
    assert data.controlling_holder_identified is False


def test_get_key_persons_controlling_identified(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _key_persons_raw(controlling=True)
    monkeypatch.setattr("agent.tools.persons.yfinance_client", lambda: _FakeYF(key_persons=raw))
    monkeypatch.setattr(
        "agent.tools.persons.edgar_client",
        lambda: _FakeEdgar(sc13=[]),
    )
    result = GetKeyPersonsTool().run(GetKeyPersonsInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, KeyPersonsData)
    assert result.data.controlling_holder_identified is True


def test_get_key_persons_edgar_13d_flags_controlling(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _key_persons_raw()
    sc13 = [SC13Holder(name="ACTIVIST FUND LLC", form_type="SC 13D", filing_date=date.today())]
    monkeypatch.setattr("agent.tools.persons.yfinance_client", lambda: _FakeYF(key_persons=raw))
    monkeypatch.setattr(
        "agent.tools.persons.edgar_client",
        lambda: _FakeEdgar(sc13=sc13),
    )
    result = GetKeyPersonsTool().run(GetKeyPersonsInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, KeyPersonsData)
    # SC 13D filer (active control intent) → controlling_holder_identified
    assert result.data.controlling_holder_identified is True
    # The 13D filer is added to persons list
    names = [p.name for p in result.data.persons]
    assert "ACTIVIST FUND LLC" in names


def test_get_key_persons_yf_error_returns_tool_error(monkeypatch: pytest.MonkeyPatch) -> None:
    err = DataSourceError(error_code="not_found", message="no data")
    monkeypatch.setattr("agent.tools.persons.yfinance_client", lambda: _FakeYF(key_persons=err))
    monkeypatch.setattr(
        "agent.tools.persons.edgar_client",
        lambda: _FakeEdgar(sc13=[]),
    )
    result = GetKeyPersonsTool().run(GetKeyPersonsInput(ticker="AAPL"), _ctx())
    assert isinstance(result, ToolResultError)
    assert result.error_code == "not_found"


@pytest.mark.parametrize(
    "ticker",
    ["DIR.MI", "CIRSA.MC", "KPL.WA", "480S.MC", "4MB.WA", "WAMI28.MI", "AAPL", "BRK.B"],
)
def test_tool_input_schema_accepts_suffix_tickers(ticker: str) -> None:
    # The shared TICKER_PATTERN (data_sources.symbols) backs every tool input's ticker
    # field via agent.tools.base; gem-hunt needs the loop to call tools with exchange
    # suffixes without input-schema validation rejecting them.
    assert GetQuoteInput(ticker=ticker).ticker == ticker


@pytest.mark.parametrize("ticker", ["480S.MC", "4MB.WA", "WAMI28.MI"])
def test_read_filing_input_accepts_g12_alphanumeric_symbols(ticker: str) -> None:
    request = ReadFilingInput(ticker=ticker, filing_type="10-K", section="business")
    assert request.ticker == ticker


def test_read_filing_input_accepts_source_neutral_regional_contract() -> None:
    request = ReadFilingInput(ticker="DIR.MI", filing_type="annual", section="full_document")
    assert request.filing_type == "annual"
    assert request.section == "full_document"


def test_tool_input_schema_rejects_garbage_ticker() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GetQuoteInput(ticker="not-a-ticker")


# ── get_valuation_history (G8) ────────────────────────────────────────────────


def _valuation_history(ticker: str = "KPL.WA") -> ValuationHistory:
    return ValuationHistory(
        ticker=ticker,
        as_of=date.today(),
        years_covered=4,
        current_pe=6.0,
        current_pb=1.0,
        pe_percentile=0.0,
        pb_percentile=0.0,
        pb_min=1.0,
        pb_vs_10y_low=1.0,
        pe_series=[6.0, 8.0, 10.0, 12.0],
        pb_series=[1.0, 2.0, 3.0, 4.0],
        fiscal_years=[2024, 2023, 2022, 2021],
        data_age_hours=100,
    )


def test_get_valuation_history_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agent.tools.valuation_history.yfinance_client",
        lambda: _FakeYF(valuation_history=_valuation_history()),
    )
    result = GetValuationHistoryTool().run(GetValuationHistoryInput(ticker="KPL.WA"), _ctx())
    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ValuationHistory)
    assert result.data.pb_percentile == 0.0
    assert result.data.pb_vs_10y_low == 1.0


def test_get_valuation_history_description_disclaims_legacy_ten_year_name() -> None:
    tool = TOOL_REGISTRY["get_valuation_history"]
    assert "NOT evidence of a 10-year low" in tool.description
    assert "years_covered" in tool.description


def test_get_valuation_history_maps_data_source_error(monkeypatch: pytest.MonkeyPatch) -> None:
    err = DataSourceError(error_code="not_found", message="no data")
    monkeypatch.setattr(
        "agent.tools.valuation_history.yfinance_client",
        lambda: _FakeYF(valuation_history=err),
    )
    result = GetValuationHistoryTool().run(GetValuationHistoryInput(ticker="KPL.WA"), _ctx())
    assert isinstance(result, ToolResultError)
    assert result.error_code == "not_found"


def test_get_valuation_history_catches_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent.tools.valuation_history.yfinance_client",
        lambda: _FakeYF(valuation_history=RuntimeError("boom")),
    )
    result = GetValuationHistoryTool().run(GetValuationHistoryInput(ticker="KPL.WA"), _ctx())
    assert isinstance(result, ToolResultError)
    assert result.error_code == "unknown"
