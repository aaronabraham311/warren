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
from data_sources.yfinance_client import (
    DataSourceError,
    FundamentalsData,
    GrowthData,
    PriceData,
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
        "a": 1, "b": 2, "c": 3, "d": 4,
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
    conn.execute(
        "UPDATE cache SET expires_at = '2000-01-01T00:00:00+00:00' WHERE key = ?", (key,)
    )
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

    def side_effect(ticker: str) -> MagicMock:
        t = MagicMock()
        # yfinance returns a near-empty dict for unknown tickers
        t.info = {"regularMarketPrice": None, "currentPrice": None, "quoteType": "NONE"}
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
            "a": 1, "b": 2, "c": 3, "d": 4,
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
