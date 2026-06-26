"""Unit tests for YFinanceClient.

All yfinance network calls are monkeypatched — no live network is hit.
The SQLite cache uses an in-memory connection via the yf_conn fixture.
Fixture data is loaded from eval/fixtures/AAPL/yfinance/ where appropriate
to tie the tests to the same recorded responses used by the eval harness.
"""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from data_sources.cache import CacheStore, make_key
from data_sources.errors import DataSourceError
from data_sources.yfinance_client import (
    CapitalAllocation,
    FinancialsHistory,
    FinancialStrengthData,
    FundamentalsData,
    GrowthData,
    PriceData,
    QualityData,
    ValuationData,
    YFinanceClient,
)
from eval.fixtures import load_fixture

# ── Helpers ───────────────────────────────────────────────────────────────────


def _no_sleep(seconds: float) -> None:
    pass


def _make_client(conn: sqlite3.Connection, **kwargs: object) -> YFinanceClient:
    return YFinanceClient(conn, _sleep=_no_sleep, **kwargs)  # type: ignore[arg-type]


def _mock_fast_info(fixture: dict[str, object]) -> MagicMock:
    fi = MagicMock()
    fi.last_price = fixture.get("last_price")
    fi.previous_close = fixture.get("previous_close")
    fi.three_month_average_volume = fixture.get("three_month_average_volume")
    return fi


def _mock_ticker_for_price(fixture: dict[str, object]) -> MagicMock:
    t = MagicMock()
    t.fast_info = _mock_fast_info(fixture)
    return t


def _mock_ticker_for_fundamentals(fixture: dict[str, object]) -> MagicMock:
    t = MagicMock()
    t.info = fixture
    return t


def _mock_ticker_for_growth(fixture: dict[str, object]) -> MagicMock:
    """Build a ticker mock whose .financials behaves like a pandas DataFrame subset."""
    t = MagicMock()
    # info subset used by _fetch_growth_metrics
    t.info = {
        "pegRatio": fixture.get("pegRatio"),
        "lastFiscalYearEnd": fixture.get("lastFiscalYearEnd"),
        "regularMarketPrice": fixture.get("regularMarketPrice"),
        # pad with dummy keys so len(info) > 5
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
    }
    # Simulate financials DataFrame .index / .loc / .dropna / .values
    revenue = fixture.get("Total Revenue", [])
    net_income = fixture.get("Net Income", [])

    def make_series(values: object) -> MagicMock:
        s = MagicMock()
        s.dropna.return_value = s
        s.values = list(values) if isinstance(values, list) else []
        return s

    fin = MagicMock()
    # Support `metric in fin.index`
    fin.index = list(
        (["Total Revenue"] if revenue else []) + (["Net Income"] if net_income else [])
    )

    def _getitem(self: object, k: str) -> MagicMock:
        return make_series(revenue if k == "Total Revenue" else net_income)

    fin.loc.__getitem__ = _getitem
    t.financials = fin
    return t


def _expire_cache(conn: sqlite3.Connection, key: str) -> None:
    conn.execute("UPDATE cache SET expires_at = '2000-01-01T00:00:00+00:00' WHERE key = ?", (key,))
    conn.commit()


# ── get_price: normal response ────────────────────────────────────────────────


def test_get_price_returns_valid_data(yf_conn: sqlite3.Connection) -> None:
    fixture = load_fixture("AAPL", "yfinance", "get_price")
    client = _make_client(yf_conn)

    with patch(
        "data_sources.yfinance_client.yf.Ticker",
        return_value=_mock_ticker_for_price(fixture),
    ):
        result = client.get_price("AAPL")

    assert isinstance(result, PriceData)
    assert result.ticker == "AAPL"
    assert result.current_price == pytest.approx(182.5)
    assert result.previous_close == pytest.approx(180.0)
    assert result.day_change_pct == pytest.approx(1.39, rel=1e-2)
    assert result.volume == 55_000_000
    assert result.data_age_hours == 0
    assert result.source == "yfinance"


# ── get_price: cache hit ──────────────────────────────────────────────────────


def test_get_price_caches_result(yf_conn: sqlite3.Connection) -> None:
    fixture = load_fixture("AAPL", "yfinance", "get_price")
    client = _make_client(yf_conn)
    call_count = [0]

    def side_effect(ticker: str) -> MagicMock:
        call_count[0] += 1
        return _mock_ticker_for_price(fixture)

    with patch("data_sources.yfinance_client.yf.Ticker", side_effect=side_effect):
        result1 = client.get_price("AAPL")
        result2 = client.get_price("AAPL")

    assert call_count[0] == 1, "yf.Ticker should be called only once (second is cache hit)"
    assert isinstance(result1, PriceData)
    assert isinstance(result2, PriceData)
    assert result2.current_price == result1.current_price


# ── get_price: expired cache hits network again ───────────────────────────────


def test_get_price_expired_cache_hits_network_again(yf_conn: sqlite3.Connection) -> None:
    fixture = load_fixture("AAPL", "yfinance", "get_price")
    client = _make_client(yf_conn)
    call_count = [0]

    def side_effect(ticker: str) -> MagicMock:
        call_count[0] += 1
        return _mock_ticker_for_price(fixture)

    with patch("data_sources.yfinance_client.yf.Ticker", side_effect=side_effect):
        client.get_price("AAPL")
        _expire_cache(yf_conn, make_key("yf_price", "AAPL"))
        client.get_price("AAPL")

    assert call_count[0] == 2, "expired cache should trigger a second network call"


# ── get_price: network error ──────────────────────────────────────────────────


def test_get_price_returns_error_on_network_failure(yf_conn: sqlite3.Connection) -> None:
    client = _make_client(yf_conn)

    with patch(
        "data_sources.yfinance_client.yf.Ticker",
        side_effect=OSError("connection refused"),
    ):
        result = client.get_price("AAPL")

    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"
    assert "connection refused" in result.message


# ── get_fundamentals: normal response ─────────────────────────────────────────


def test_get_fundamentals_returns_valid_data(yf_conn: sqlite3.Connection) -> None:
    fixture = load_fixture("AAPL", "yfinance", "get_fundamentals")
    client = _make_client(yf_conn)

    with patch(
        "data_sources.yfinance_client.yf.Ticker",
        return_value=_mock_ticker_for_fundamentals(fixture),
    ):
        result = client.get_fundamentals("AAPL")

    assert isinstance(result, FundamentalsData)
    assert result.ticker == "AAPL"
    assert result.pe_ratio == pytest.approx(28.5)
    assert result.pb_ratio == pytest.approx(45.2)
    assert result.roe_pct is not None and result.roe_pct > 0
    assert result.debt_to_equity == pytest.approx(150.5)
    assert result.fcf_ttm_usd == 90_000_000_000
    assert result.operating_margin_pct is not None
    assert result.net_margin_pct is not None
    assert result.data_age_hours >= 0
    assert result.source == "yfinance"


# ── get_fundamentals: cache hit ───────────────────────────────────────────────


def test_get_fundamentals_caches_result(yf_conn: sqlite3.Connection) -> None:
    fixture = load_fixture("AAPL", "yfinance", "get_fundamentals")
    client = _make_client(yf_conn)
    call_count = [0]

    def side_effect(ticker: str) -> MagicMock:
        call_count[0] += 1
        return _mock_ticker_for_fundamentals(fixture)

    with patch("data_sources.yfinance_client.yf.Ticker", side_effect=side_effect):
        result1 = client.get_fundamentals("AAPL")
        result2 = client.get_fundamentals("AAPL")

    assert call_count[0] == 1, "second call must be a cache hit"
    assert isinstance(result1, FundamentalsData)
    assert isinstance(result2, FundamentalsData)
    assert result2.data_age_hours == result1.data_age_hours


# ── get_fundamentals: expired cache hits network again ────────────────────────


def test_get_fundamentals_expired_cache_hits_network_again(yf_conn: sqlite3.Connection) -> None:
    fixture = load_fixture("AAPL", "yfinance", "get_fundamentals")
    client = _make_client(yf_conn)
    call_count = [0]

    def side_effect(ticker: str) -> MagicMock:
        call_count[0] += 1
        return _mock_ticker_for_fundamentals(fixture)

    with patch("data_sources.yfinance_client.yf.Ticker", side_effect=side_effect):
        client.get_fundamentals("AAPL")
        _expire_cache(yf_conn, make_key("yf_fundamentals", "AAPL"))
        client.get_fundamentals("AAPL")

    assert call_count[0] == 2, "expired cache should trigger a second network call"


# ── get_fundamentals: network error ───────────────────────────────────────────


def test_get_fundamentals_returns_error_on_network_failure(yf_conn: sqlite3.Connection) -> None:
    client = _make_client(yf_conn)

    with patch(
        "data_sources.yfinance_client.yf.Ticker",
        side_effect=OSError("timeout"),
    ):
        result = client.get_fundamentals("AAPL")

    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"


# ── get_fundamentals: invalid / delisted ticker ────────────────────────────────


def test_get_fundamentals_returns_not_found_for_invalid_ticker(yf_conn: sqlite3.Connection) -> None:
    client = _make_client(yf_conn)

    # yfinance returns a near-empty dict for unknown tickers — recorded as an
    # error fixture under eval/fixtures/INVALID/.
    invalid_info = load_fixture("INVALID", "yfinance", "get_fundamentals", name="error_not_found")

    def side_effect(ticker: str) -> MagicMock:
        t = MagicMock()
        t.info = invalid_info
        return t

    with patch("data_sources.yfinance_client.yf.Ticker", side_effect=side_effect):
        result = client.get_fundamentals("ZZZZZ")

    assert isinstance(result, DataSourceError)
    assert result.error_code == "not_found"


def test_get_fundamentals_returns_not_found_for_empty_info(yf_conn: sqlite3.Connection) -> None:
    client = _make_client(yf_conn)

    def side_effect(ticker: str) -> MagicMock:
        t = MagicMock()
        t.info = {}
        return t

    with patch("data_sources.yfinance_client.yf.Ticker", side_effect=side_effect):
        result = client.get_fundamentals("ZZZZZ")

    assert isinstance(result, DataSourceError)
    assert result.error_code == "not_found"


# ── get_fundamentals: 429 → retry → success ───────────────────────────────────


def test_get_fundamentals_rate_limit_retry(yf_conn: sqlite3.Connection) -> None:
    fixture = load_fixture("AAPL", "yfinance", "get_fundamentals")
    sleep_calls: list[float] = []
    client = YFinanceClient(yf_conn, _sleep=lambda s: sleep_calls.append(s))
    call_count = [0]

    def side_effect(ticker: str) -> MagicMock:
        call_count[0] += 1
        if call_count[0] == 1:
            raise Exception("Too Many Requests")
        return _mock_ticker_for_fundamentals(fixture)

    with patch("data_sources.yfinance_client.yf.Ticker", side_effect=side_effect):
        result = client.get_fundamentals("AAPL")

    assert isinstance(result, FundamentalsData), "should succeed after one retry"
    assert call_count[0] == 2
    # The backoff sleep (2**0 = 1s) should have fired between attempts
    assert any(s >= 1.0 for s in sleep_calls), f"expected backoff sleep, got: {sleep_calls}"


# ── get_fundamentals: exhausted retries → error ───────────────────────────────


def test_get_fundamentals_max_retries_returns_error(yf_conn: sqlite3.Connection) -> None:
    client = _make_client(yf_conn)

    with patch(
        "data_sources.yfinance_client.yf.Ticker",
        side_effect=OSError("flaky"),
    ):
        result = client.get_fundamentals("AAPL")

    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"


# ── get_growth_metrics: normal response ───────────────────────────────────────


def test_get_growth_metrics_returns_valid_data(yf_conn: sqlite3.Connection) -> None:
    fixture = load_fixture("AAPL", "yfinance", "get_growth_metrics")
    client = _make_client(yf_conn)

    with patch(
        "data_sources.yfinance_client.yf.Ticker",
        return_value=_mock_ticker_for_growth(fixture),
    ):
        result = client.get_growth_metrics("AAPL")

    assert isinstance(result, GrowthData)
    assert result.ticker == "AAPL"
    assert result.peg_ratio == pytest.approx(2.3)
    # 3-year CAGR from [394328e6, 383285e6, 365817e6, 274515e6]
    assert result.revenue_cagr_3y is not None
    assert result.earnings_cagr_3y is not None
    assert result.source == "yfinance"


def test_get_growth_metrics_caches_result(yf_conn: sqlite3.Connection) -> None:
    fixture = load_fixture("AAPL", "yfinance", "get_growth_metrics")
    client = _make_client(yf_conn)
    call_count = [0]

    def side_effect(ticker: str) -> MagicMock:
        call_count[0] += 1
        return _mock_ticker_for_growth(fixture)

    with patch("data_sources.yfinance_client.yf.Ticker", side_effect=side_effect):
        client.get_growth_metrics("AAPL")
        client.get_growth_metrics("AAPL")

    assert call_count[0] == 1


# ── get_growth_metrics: network error ─────────────────────────────────────────


def test_get_growth_metrics_returns_error_on_network_failure(yf_conn: sqlite3.Connection) -> None:
    client = _make_client(yf_conn)

    with patch(
        "data_sources.yfinance_client.yf.Ticker",
        side_effect=RuntimeError("dns failure"),
    ):
        result = client.get_growth_metrics("AAPL")

    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"


# ── get_growth_metrics: missing financials gracefully handled ─────────────────


def test_get_growth_metrics_missing_financials_returns_none_cagr(
    yf_conn: sqlite3.Connection,
) -> None:
    client = _make_client(yf_conn)

    def side_effect(ticker: str) -> MagicMock:
        t = MagicMock()
        t.info = {
            "pegRatio": 2.1,
            "lastFiscalYearEnd": 1696032000,
            "regularMarketPrice": 182.5,
            "a": 1,
            "b": 2,
            "c": 3,
            "d": 4,
        }
        # financials with empty index — no revenue/earnings rows
        fin = MagicMock()
        fin.index = []
        t.financials = fin
        return t

    with patch("data_sources.yfinance_client.yf.Ticker", side_effect=side_effect):
        result = client.get_growth_metrics("AAPL")

    assert isinstance(result, GrowthData)
    assert result.revenue_cagr_3y is None
    assert result.earnings_cagr_3y is None
    assert result.peg_ratio == pytest.approx(2.1)


# ── get_quality_metrics helpers ──────────────────────────────────────────────


def _make_df_mock(data: dict[str, list[float]]) -> MagicMock:
    """Build a DataFrame-like mock from metric → values list."""
    df = MagicMock()
    df.index = list(data.keys())

    def _getitem(self: object, metric: str) -> MagicMock:
        s = MagicMock()
        s.dropna.return_value = s
        s.values = data.get(metric, [])
        return s

    df.loc.__getitem__ = _getitem
    return df


def _mock_ticker_for_quality(fixture: dict[str, object]) -> MagicMock:
    t = MagicMock()
    inc = fixture.get("income_statement", {})
    bs = fixture.get("balance_sheet", {})
    cf = fixture.get("cashflow", {})
    t.info = {
        "lastFiscalYearEnd": fixture.get("lastFiscalYearEnd"),
        "regularMarketPrice": fixture.get("regularMarketPrice"),
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
        "e": 5,
        "f": 6,
    }
    t.financials = _make_df_mock(inc if isinstance(inc, dict) else {})
    t.balance_sheet = _make_df_mock(bs if isinstance(bs, dict) else {})
    t.cashflow = _make_df_mock(cf if isinstance(cf, dict) else {})
    return t


# ── get_quality_metrics: normal response ──────────────────────────────────────


def test_get_quality_metrics_returns_valid_data(yf_conn: sqlite3.Connection) -> None:
    fixture = load_fixture("AAPL", "yfinance", "get_quality_metrics")
    client = _make_client(yf_conn)

    with patch(
        "data_sources.yfinance_client.yf.Ticker",
        return_value=_mock_ticker_for_quality(fixture),
    ):
        result = client.get_quality_metrics("AAPL")

    assert isinstance(result, QualityData)
    assert result.ticker == "AAPL"
    assert result.source == "yfinance"
    assert result.roic_pct is not None and result.roic_pct > 0
    assert len(result.roic_series) == 4
    assert result.roic_mean is not None
    assert result.roa_pct is not None and result.roa_pct > 0
    assert result.gross_margin_pct is not None
    assert result.gross_margin_pct == pytest.approx(44.13, rel=0.01)
    assert len(result.gross_margin_series) == 4
    assert result.gross_margin_stdev is not None
    assert result.cash_conversion_ttm is not None and result.cash_conversion_ttm > 0
    assert len(result.cash_conversion_series) == 4


def test_get_quality_metrics_caches_result(yf_conn: sqlite3.Connection) -> None:
    fixture = load_fixture("AAPL", "yfinance", "get_quality_metrics")
    client = _make_client(yf_conn)
    call_count = [0]

    def side_effect(ticker: str) -> MagicMock:
        call_count[0] += 1
        return _mock_ticker_for_quality(fixture)

    with patch("data_sources.yfinance_client.yf.Ticker", side_effect=side_effect):
        client.get_quality_metrics("AAPL")
        client.get_quality_metrics("AAPL")

    assert call_count[0] == 1


def test_get_quality_metrics_returns_error_on_network_failure(
    yf_conn: sqlite3.Connection,
) -> None:
    client = _make_client(yf_conn)

    with patch(
        "data_sources.yfinance_client.yf.Ticker",
        side_effect=OSError("timeout"),
    ):
        result = client.get_quality_metrics("AAPL")

    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"


def test_get_quality_metrics_missing_statements_returns_empty_series(
    yf_conn: sqlite3.Connection,
) -> None:
    client = _make_client(yf_conn)

    def side_effect(ticker: str) -> MagicMock:
        t = MagicMock()
        t.info = {
            "regularMarketPrice": 182.5,
            "a": 1,
            "b": 2,
            "c": 3,
            "d": 4,
            "e": 5,
            "f": 6,
        }
        empty = _make_df_mock({})
        t.financials = empty
        t.balance_sheet = empty
        t.cashflow = empty
        return t

    with patch("data_sources.yfinance_client.yf.Ticker", side_effect=side_effect):
        result = client.get_quality_metrics("AAPL")

    assert isinstance(result, QualityData)
    assert result.roic_pct is None
    assert result.roic_series == []
    assert result.roic_mean is None
    assert result.roa_pct is None
    assert result.gross_margin_pct is None
    assert result.gross_margin_stdev is None
    assert result.cash_conversion_ttm is None


def test_get_quality_metrics_not_found(yf_conn: sqlite3.Connection) -> None:
    client = _make_client(yf_conn)

    def side_effect(ticker: str) -> MagicMock:
        t = MagicMock()
        t.info = {}
        return t

    with patch("data_sources.yfinance_client.yf.Ticker", side_effect=side_effect):
        result = client.get_quality_metrics("ZZZZZ")

    assert isinstance(result, DataSourceError)
    assert result.error_code == "not_found"


# ── get_financials / get_financial_strength helpers ──────────────────────────


def _make_stmt_df_mock(statement: dict[str, object]) -> MagicMock:
    """DataFrame-like mock with year columns, metric index, and positional values."""
    from datetime import date as _date

    years_raw = statement.get("fiscal_years", [])
    years = years_raw if isinstance(years_raw, list) else []
    metrics = {k: v for k, v in statement.items() if k != "fiscal_years"}
    df = MagicMock()
    df.columns = [_date(int(y), 12, 31) for y in years if isinstance(y, int)]
    df.index = list(metrics.keys())

    def _getitem(self: object, metric: str) -> MagicMock:
        s = MagicMock()
        s.values = metrics.get(metric, [])
        return s

    df.loc.__getitem__ = _getitem
    return df


def _mock_ticker_for_financials(fixture: dict[str, object]) -> MagicMock:
    t = MagicMock()
    t.info = {
        "lastFiscalYearEnd": fixture.get("lastFiscalYearEnd"),
        "regularMarketPrice": fixture.get("regularMarketPrice"),
        "marketCap": fixture.get("marketCap"),
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
    }
    inc = fixture.get("income_statement", {})
    bs = fixture.get("balance_sheet", {})
    cf = fixture.get("cashflow", {})
    t.financials = _make_stmt_df_mock(inc if isinstance(inc, dict) else {})
    t.balance_sheet = _make_stmt_df_mock(bs if isinstance(bs, dict) else {})
    t.cashflow = _make_stmt_df_mock(cf if isinstance(cf, dict) else {})
    return t


# ── get_financials: normal response ───────────────────────────────────────────


def test_get_financials_returns_valid_data(yf_conn: sqlite3.Connection) -> None:
    fixture = load_fixture("AAPL", "yfinance", "get_financials")
    client = _make_client(yf_conn)

    with patch(
        "data_sources.yfinance_client.yf.Ticker",
        return_value=_mock_ticker_for_financials(fixture),
    ):
        result = client.get_financials("AAPL")

    assert isinstance(result, FinancialsHistory)
    assert result.ticker == "AAPL"
    assert result.source == "yfinance"
    # ≥3 years across all three statements (AC #1).
    assert result.fiscal_years == [2023, 2022, 2021, 2020]
    assert len(result.income_statement) == 4
    assert len(result.balance_sheet) == 4
    assert len(result.cash_flow) == 4
    # Year-aligned, newest-first; key line items the consumer tools need.
    inc0 = result.income_statement[0]
    assert inc0.fiscal_year == 2023
    assert inc0.revenue == 383285000000
    assert inc0.gross_profit == 169148000000
    assert inc0.operating_income == 114301000000
    assert inc0.net_income == 96995000000
    bs0 = result.balance_sheet[0]
    assert bs0.total_assets == 352583000000
    assert bs0.current_assets is not None and bs0.current_liabilities is not None
    assert bs0.long_term_debt == 95281000000
    assert bs0.shares_outstanding == 15550061000  # from "Share Issued"
    cf0 = result.cash_flow[0]
    assert cf0.cfo == 110543000000
    assert cf0.capex == -10959000000
    assert cf0.dividends_paid == -15025000000
    assert cf0.buybacks == -77550000000


def test_get_financials_caches_result(yf_conn: sqlite3.Connection) -> None:
    fixture = load_fixture("AAPL", "yfinance", "get_financials")
    client = _make_client(yf_conn)
    call_count = [0]

    def side_effect(ticker: str) -> MagicMock:
        call_count[0] += 1
        return _mock_ticker_for_financials(fixture)

    with patch("data_sources.yfinance_client.yf.Ticker", side_effect=side_effect):
        client.get_financials("AAPL")
        client.get_financials("AAPL")

    assert call_count[0] == 1, "second call must be a cache hit"


def test_get_financials_network_error(yf_conn: sqlite3.Connection) -> None:
    client = _make_client(yf_conn)

    with patch("data_sources.yfinance_client.yf.Ticker", side_effect=OSError("timeout")):
        result = client.get_financials("AAPL")

    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"


def test_get_financials_not_found_for_empty_info(yf_conn: sqlite3.Connection) -> None:
    client = _make_client(yf_conn)

    def side_effect(ticker: str) -> MagicMock:
        t = MagicMock()
        t.info = {}
        return t

    with patch("data_sources.yfinance_client.yf.Ticker", side_effect=side_effect):
        result = client.get_financials("ZZZZZ")

    assert isinstance(result, DataSourceError)
    assert result.error_code == "not_found"


def test_get_financials_missing_statements_returns_empty_rows(yf_conn: sqlite3.Connection) -> None:
    client = _make_client(yf_conn)

    def side_effect(ticker: str) -> MagicMock:
        t = MagicMock()
        t.info = {"regularMarketPrice": 182.5, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        empty = _make_stmt_df_mock({})
        t.financials = empty
        t.balance_sheet = empty
        t.cashflow = empty
        return t

    with patch("data_sources.yfinance_client.yf.Ticker", side_effect=side_effect):
        result = client.get_financials("AAPL")

    assert isinstance(result, FinancialsHistory)
    assert result.income_statement == []
    assert result.balance_sheet == []
    assert result.cash_flow == []


# ── get_financial_strength: client-level, computed off the shared history ──────


def test_get_financial_strength_computes_from_financials(yf_conn: sqlite3.Connection) -> None:
    fixture = load_fixture("AAPL", "yfinance", "get_financials")
    client = _make_client(yf_conn)

    with patch(
        "data_sources.yfinance_client.yf.Ticker",
        return_value=_mock_ticker_for_financials(fixture),
    ):
        result = client.get_financial_strength("AAPL")

    assert isinstance(result, FinancialStrengthData)
    assert result.ticker == "AAPL"
    # AAPL: profitable, low leverage → high F-score, safe Z-zone.
    assert result.f_score is not None and result.f_score >= 5
    assert result.z_score is not None and result.z_zone == "safe"
    # Profitability signals fire (positive ROA and operating cash flow).
    assert result.f_signals.roa_positive is True
    assert result.f_signals.op_cf_positive is True
    # no_dilution keys off the balance-sheet "Common Stock" line, which rose
    # (64.8B → 73.8B), so the signal is False — preserved legacy behaviour.
    assert result.f_signals.no_dilution is False
    assert result.current_ratio is not None
    assert result.interest_coverage is not None and result.interest_coverage > 0


# ── get_capital_allocation: client-level, computed off the shared history ──────


def test_get_capital_allocation_computes_from_financials(yf_conn: sqlite3.Connection) -> None:
    fixture = load_fixture("AAPL", "yfinance", "get_financials")
    client = _make_client(yf_conn)

    with patch(
        "data_sources.yfinance_client.yf.Ticker",
        return_value=_mock_ticker_for_financials(fixture),
    ):
        result = client.get_capital_allocation("AAPL")

    assert isinstance(result, CapitalAllocation)
    assert result.ticker == "AAPL"
    assert result.years_covered == 4
    # Share-count fell 16.98B → 15.55B over 4 years → net buybacks (negative CAGR). (AC #1)
    assert result.share_count_series == [15550061000, 15943425000, 16426786000, 16976763000]
    assert result.share_count_cagr_pct is not None and result.share_count_cagr_pct < 0
    # Yields off the most recent year against a 2.85T market cap. (AC #1)
    assert result.buyback_yield_pct == pytest.approx(2.7211, abs=1e-3)
    assert result.dividend_yield_pct == pytest.approx(0.5272, abs=1e-3)
    assert result.shareholder_yield_pct == pytest.approx(3.2483, abs=1e-3)
    # Dividend rose every available year, payout off FY2023 net income. (AC #2)
    assert result.dividend_growth_streak == 3
    assert result.payout_ratio_pct == pytest.approx(15.4905, abs=1e-3)
    # Net debt = total_debt - cash, newest-first.
    assert result.net_debt_series == [81123000000, 96423000000, 89779000000, 74420000000]
    assert result.net_debt_trajectory in {"delevering", "levering", "stable"}


def test_get_capital_allocation_caches_result(yf_conn: sqlite3.Connection) -> None:
    fixture = load_fixture("AAPL", "yfinance", "get_financials")
    client = _make_client(yf_conn)
    call_count = [0]

    def side_effect(ticker: str) -> MagicMock:
        call_count[0] += 1
        return _mock_ticker_for_financials(fixture)

    with patch("data_sources.yfinance_client.yf.Ticker", side_effect=side_effect):
        client.get_capital_allocation("AAPL")
        client.get_capital_allocation("AAPL")

    assert call_count[0] == 1, "second call must be a cache hit"


def test_get_capital_allocation_not_found_for_empty_info(yf_conn: sqlite3.Connection) -> None:
    client = _make_client(yf_conn)

    def side_effect(ticker: str) -> MagicMock:
        t = MagicMock()
        t.info = {}
        return t

    with patch("data_sources.yfinance_client.yf.Ticker", side_effect=side_effect):
        result = client.get_capital_allocation("ZZZZZ")

    assert isinstance(result, DataSourceError)
    assert result.error_code == "not_found"


# ── CacheStore unit tests ─────────────────────────────────────────────────────


def test_cache_miss_returns_none(yf_conn: sqlite3.Connection) -> None:
    cache = CacheStore(yf_conn)
    assert cache.get("nonexistent-key") is None


def test_cache_set_and_get(yf_conn: sqlite3.Connection) -> None:
    cache = CacheStore(yf_conn)
    cache.set("k", '{"x": 1}', ttl_hours=1.0)
    assert cache.get("k") == '{"x": 1}'


def test_cache_expired_entry_returns_none(yf_conn: sqlite3.Connection) -> None:
    cache = CacheStore(yf_conn)
    cache.set("k", '{"x": 1}', ttl_hours=-1.0)  # already expired
    assert cache.get("k") is None


def test_cache_expired_entry_can_be_replaced(yf_conn: sqlite3.Connection) -> None:
    cache = CacheStore(yf_conn)
    cache.set("k", "old", ttl_hours=-1.0)
    assert cache.get("k") is None
    cache.set("k", "new", ttl_hours=1.0)
    assert cache.get("k") == "new"


# ── load_fixture utility ──────────────────────────────────────────────────────


def test_load_fixture_returns_dict() -> None:
    fixture = load_fixture("AAPL", "yfinance", "get_price")
    assert isinstance(fixture, dict)
    assert "last_price" in fixture
    assert fixture["last_price"] == pytest.approx(182.5)


def test_load_fixture_fundamentals() -> None:
    fixture = load_fixture("AAPL", "yfinance", "get_fundamentals")
    assert "trailingPE" in fixture
    assert "regularMarketPrice" in fixture


def test_load_fixture_growth_metrics() -> None:
    fixture = load_fixture("AAPL", "yfinance", "get_growth_metrics")
    assert "pegRatio" in fixture
    assert "Total Revenue" in fixture


# ── get_valuation_multiples helpers ──────────────────────────────────────────


def _mock_ticker_for_valuation(fixture: dict[str, object]) -> MagicMock:
    t = MagicMock()
    t.info = fixture
    return t


# ── get_valuation_multiples: net-cash-positive (AAPL) ────────────────────────


def test_get_valuation_multiples_net_cash_positive(yf_conn: sqlite3.Connection) -> None:
    fixture = load_fixture("AAPL", "yfinance", "get_valuation_multiples")
    client = _make_client(yf_conn)

    with patch(
        "data_sources.yfinance_client.yf.Ticker",
        return_value=_mock_ticker_for_valuation(fixture),
    ):
        result = client.get_valuation_multiples("AAPL")

    assert isinstance(result, ValuationData)
    assert result.ticker == "AAPL"
    assert result.source == "yfinance"
    # net cash: totalCash=162B - totalDebt=111B = +51B
    assert result.net_cash_usd == 51_000_000_000
    assert result.net_cash_positive is True
    # AAPL ncav = currentAssets - totalLiab = 143.6B - 290.4B < 0 → price_to_ncav is None
    assert result.ncav is not None and result.ncav < 0
    assert result.price_to_ncav is None
    # existing fields still present
    assert result.ev_to_ebit is not None
    assert result.ncav_to_market_cap is not None


# ── get_valuation_multiples: net-debt (GM) ────────────────────────────────────


def test_get_valuation_multiples_net_debt(yf_conn: sqlite3.Connection) -> None:
    fixture = load_fixture("GM", "yfinance", "get_valuation_multiples")
    client = _make_client(yf_conn)

    with patch(
        "data_sources.yfinance_client.yf.Ticker",
        return_value=_mock_ticker_for_valuation(fixture),
    ):
        result = client.get_valuation_multiples("GM")

    assert isinstance(result, ValuationData)
    assert result.ticker == "GM"
    # net cash: totalCash=27B - totalDebt=130B = -103B
    assert result.net_cash_usd == -103_000_000_000
    assert result.net_cash_positive is False


# ── get_valuation_multiples: price_to_ncav consistency ───────────────────────


def test_get_valuation_multiples_price_to_ncav_consistency(yf_conn: sqlite3.Connection) -> None:
    # Synthetic small-cap with positive NCAV: currentAssets=100M, totalLiab=40M, mktCap=45M
    client = _make_client(yf_conn)
    info: dict[str, object] = {
        "regularMarketPrice": 9.0,
        "enterpriseValue": 50_000_000,
        "ebitda": 8_000_000,
        "operatingIncome": 6_000_000,
        "freeCashflow": 4_000_000,
        "marketCap": 45_000_000,
        "currentAssets": 100_000_000,
        "totalLiab": 40_000_000,
        "tangibleBookValue": 60_000_000,
        "dividendYield": None,
        "lastFiscalYearEnd": 1696032000,
        "totalCash": 30_000_000,
        "totalDebt": 10_000_000,
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
    }

    with patch(
        "data_sources.yfinance_client.yf.Ticker",
        return_value=_mock_ticker_for_valuation(info),
    ):
        result = client.get_valuation_multiples("XSMALL")

    assert isinstance(result, ValuationData)
    ncav = 100_000_000 - 40_000_000  # 60M
    mkt_cap = 45_000_000
    assert result.ncav == ncav
    assert result.ncav_to_market_cap == pytest.approx(ncav / mkt_cap, rel=1e-4)
    assert result.price_to_ncav == pytest.approx(mkt_cap / ncav, rel=1e-4)
    # price_to_ncav * ncav_to_market_cap ≈ 1.0
    assert result.price_to_ncav is not None and result.ncav_to_market_cap is not None
    assert result.price_to_ncav * result.ncav_to_market_cap == pytest.approx(1.0, rel=1e-4)
    # net cash: 30M - 10M = +20M
    assert result.net_cash_usd == 20_000_000
    assert result.net_cash_positive is True


# ── get_valuation_multiples: missing cash fields → graceful None ──────────────


def test_get_valuation_multiples_missing_cash_fields(yf_conn: sqlite3.Connection) -> None:
    client = _make_client(yf_conn)
    info: dict[str, object] = {
        "regularMarketPrice": 50.0,
        "marketCap": 1_000_000_000,
        "currentAssets": 500_000_000,
        "totalLiab": 300_000_000,
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
    }

    with patch(
        "data_sources.yfinance_client.yf.Ticker",
        return_value=_mock_ticker_for_valuation(info),
    ):
        result = client.get_valuation_multiples("NOCASH")

    assert isinstance(result, ValuationData)
    assert result.net_cash_usd is None
    assert result.net_cash_positive is False
