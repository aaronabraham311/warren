"""Unit tests for FinnhubClient.

All Finnhub network calls are mocked — no live network, no API key needed (the
autouse socket guard enforces this). ``data_sources.finnhub_client.finnhub.Client``
is patched so the constructed ``self.client`` is a MagicMock with stubbed
``company_news`` / ``company_basic_financials``. Their return payloads come from the
recorded fixtures under ``eval/fixtures/AAPL/finnhub/`` via the ``finnhub_fixture``
conftest fixture. The SQLite cache uses the in-memory ``finnhub_conn`` fixture.
"""

import sqlite3
from unittest.mock import MagicMock, patch

import finnhub
import pytest

from data_sources.cache import make_key
from data_sources.errors import DataSourceError
from data_sources.finnhub_client import (
    FinnhubClient,
    FinnhubFinancials,
    FinnhubInsiderTransaction,
    NewsItem,
    RateLimiter,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _no_sleep(seconds: float) -> None:
    pass


def _api_exc(status_code: int, message: str = "error") -> finnhub.FinnhubAPIException:
    resp = MagicMock()
    resp.json.return_value = {"error": message}
    resp.status_code = status_code
    return finnhub.FinnhubAPIException(resp)


def _make_client(
    conn: sqlite3.Connection, mock_finnhub: MagicMock, _sleep: object = _no_sleep
) -> FinnhubClient:
    with patch("data_sources.finnhub_client.finnhub.Client", return_value=mock_finnhub):
        return FinnhubClient(conn, api_key="test-key", _sleep=_sleep)  # type: ignore[arg-type]


def _expire_cache(conn: sqlite3.Connection, key: str) -> None:
    conn.execute("UPDATE cache SET expires_at = '2000-01-01T00:00:00+00:00' WHERE key = ?", (key,))
    conn.commit()


# ── Construction ──────────────────────────────────────────────────────────────


def test_empty_api_key_raises_environment_error(finnhub_conn: sqlite3.Connection) -> None:
    with pytest.raises(EnvironmentError):
        FinnhubClient(finnhub_conn, api_key="")


# ── get_news ──────────────────────────────────────────────────────────────────


def test_get_news_returns_sorted_items(
    finnhub_conn: sqlite3.Connection, finnhub_fixture: dict[str, object]
) -> None:
    mock = MagicMock()
    mock.company_news.return_value = finnhub_fixture["news"]
    client = _make_client(finnhub_conn, mock)

    result = client.get_news("AAPL", days=7)

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(n, NewsItem) for n in result)
    # Sorted by datetime descending — newest first.
    assert result[0].headline == "Newer headline"
    assert result[1].headline == "Older headline"
    assert result[0].datetime > result[1].datetime
    # Free tier has no per-article sentiment.
    assert result[0].sentiment is None


def test_get_news_caches_result(
    finnhub_conn: sqlite3.Connection, finnhub_fixture: dict[str, object]
) -> None:
    mock = MagicMock()
    mock.company_news.return_value = finnhub_fixture["news"]
    client = _make_client(finnhub_conn, mock)

    client.get_news("AAPL", days=7)
    client.get_news("AAPL", days=7)

    assert mock.company_news.call_count == 1, "second call must be a cache hit"


def test_get_news_expired_cache_refetches(
    finnhub_conn: sqlite3.Connection, finnhub_fixture: dict[str, object]
) -> None:
    mock = MagicMock()
    mock.company_news.return_value = finnhub_fixture["news"]
    client = _make_client(finnhub_conn, mock)

    client.get_news("AAPL", days=7)
    _expire_cache(finnhub_conn, make_key("finnhub_news", "AAPL", "7"))
    client.get_news("AAPL", days=7)

    assert mock.company_news.call_count == 2, "expired cache should trigger a refetch"


def test_get_news_empty_list_is_cached_and_returned(finnhub_conn: sqlite3.Connection) -> None:
    mock = MagicMock()
    mock.company_news.return_value = []
    client = _make_client(finnhub_conn, mock)

    result = client.get_news("AAPL")
    assert result == []


def test_get_news_network_error(finnhub_conn: sqlite3.Connection) -> None:
    mock = MagicMock()
    mock.company_news.side_effect = _api_exc(500, "server error")
    client = _make_client(finnhub_conn, mock)

    result = client.get_news("AAPL")
    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"


# ── get_basic_financials ──────────────────────────────────────────────────────


def test_get_basic_financials_maps_fields(
    finnhub_conn: sqlite3.Connection, finnhub_fixture: dict[str, object]
) -> None:
    mock = MagicMock()
    mock.company_basic_financials.return_value = finnhub_fixture["financials"]
    client = _make_client(finnhub_conn, mock)

    result = client.get_basic_financials("AAPL")

    assert isinstance(result, FinnhubFinancials)
    assert result.ticker == "AAPL"
    assert result.pe_ratio == pytest.approx(28.5)
    assert result.pb_ratio == pytest.approx(45.2)
    assert result.roe_pct == pytest.approx(150.0)
    assert result.source == "finnhub"


def test_get_basic_financials_caches_result(
    finnhub_conn: sqlite3.Connection, finnhub_fixture: dict[str, object]
) -> None:
    mock = MagicMock()
    mock.company_basic_financials.return_value = finnhub_fixture["financials"]
    client = _make_client(finnhub_conn, mock)

    client.get_basic_financials("AAPL")
    client.get_basic_financials("AAPL")

    assert mock.company_basic_financials.call_count == 1


def test_get_basic_financials_expired_cache_refetches(
    finnhub_conn: sqlite3.Connection, finnhub_fixture: dict[str, object]
) -> None:
    mock = MagicMock()
    mock.company_basic_financials.return_value = finnhub_fixture["financials"]
    client = _make_client(finnhub_conn, mock)

    client.get_basic_financials("AAPL")
    _expire_cache(finnhub_conn, make_key("finnhub_financials", "AAPL"))
    client.get_basic_financials("AAPL")

    assert mock.company_basic_financials.call_count == 2


def test_get_basic_financials_not_found_for_empty_metric(finnhub_conn: sqlite3.Connection) -> None:
    mock = MagicMock()
    mock.company_basic_financials.return_value = {"metric": {}}
    client = _make_client(finnhub_conn, mock)

    result = client.get_basic_financials("ZZZZZ")
    assert isinstance(result, DataSourceError)
    assert result.error_code == "not_found"


def test_get_basic_financials_falls_back_to_pb_annual(finnhub_conn: sqlite3.Connection) -> None:
    mock = MagicMock()
    mock.company_basic_financials.return_value = {
        "metric": {"peTTM": 10.0, "pbAnnual": 3.3, "roeTTM": 20.0}
    }
    client = _make_client(finnhub_conn, mock)

    result = client.get_basic_financials("AAPL")
    assert isinstance(result, FinnhubFinancials)
    assert result.pb_ratio == pytest.approx(3.3)


# ── get_insider_transactions ──────────────────────────────────────────────────


def test_get_insider_transactions_parses_buy_sell_rows(
    finnhub_conn: sqlite3.Connection, finnhub_fixture: dict[str, object]
) -> None:
    mock = MagicMock()
    mock.stock_insider_transactions.return_value = finnhub_fixture["insider"]
    client = _make_client(finnhub_conn, mock)

    result = client.get_insider_transactions("AAPL", days=90)

    assert isinstance(result, list)
    assert all(isinstance(t, FinnhubInsiderTransaction) for t in result)
    by_type = {t.transaction_type for t in result}
    assert "buy" in by_type and "sell" in by_type  # codes P and S mapped
    # Code "A" (award) is neither a buy nor a sell.
    assert any(t.transaction_type == "other" for t in result)
    sells = [t for t in result if t.transaction_type == "sell"]
    assert sells[0].shares > 0  # stored as absolute share count
    assert sells[0].value is not None


def test_get_insider_transactions_caches_result(
    finnhub_conn: sqlite3.Connection, finnhub_fixture: dict[str, object]
) -> None:
    mock = MagicMock()
    mock.stock_insider_transactions.return_value = finnhub_fixture["insider"]
    client = _make_client(finnhub_conn, mock)

    client.get_insider_transactions("AAPL", days=90)
    client.get_insider_transactions("AAPL", days=90)

    assert mock.stock_insider_transactions.call_count == 1, "second call must be a cache hit"


def test_get_insider_transactions_empty_data_returns_empty_list(
    finnhub_conn: sqlite3.Connection,
) -> None:
    mock = MagicMock()
    mock.stock_insider_transactions.return_value = {"symbol": "AAPL", "data": []}
    client = _make_client(finnhub_conn, mock)

    result = client.get_insider_transactions("AAPL")
    assert result == []


def test_get_insider_transactions_network_error(finnhub_conn: sqlite3.Connection) -> None:
    mock = MagicMock()
    mock.stock_insider_transactions.side_effect = _api_exc(500, "server error")
    client = _make_client(finnhub_conn, mock)

    result = client.get_insider_transactions("AAPL")
    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"


# ── 429 retry / backoff ───────────────────────────────────────────────────────


def test_429_retries_then_succeeds(
    finnhub_conn: sqlite3.Connection, finnhub_fixture: dict[str, object]
) -> None:
    sleep_calls: list[float] = []
    mock = MagicMock()
    call_count = [0]
    financials = finnhub_fixture["financials"]

    def side_effect(symbol: str, metric: str) -> object:
        call_count[0] += 1
        if call_count[0] == 1:
            raise _api_exc(429, "rate limited")
        return financials

    mock.company_basic_financials.side_effect = side_effect
    client = _make_client(finnhub_conn, mock, _sleep=lambda s: sleep_calls.append(s))

    result = client.get_basic_financials("AAPL")

    assert isinstance(result, FinnhubFinancials)
    assert call_count[0] == 2
    assert any(s >= 1.0 for s in sleep_calls), f"expected backoff sleep, got: {sleep_calls}"


def test_429_exhausted_returns_network_error(finnhub_conn: sqlite3.Connection) -> None:
    mock = MagicMock()
    mock.company_basic_financials.side_effect = _api_exc(429, "rate limited")
    client = _make_client(finnhub_conn, mock)

    result = client.get_basic_financials("AAPL")
    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"
    assert mock.company_basic_financials.call_count == 3


# ── RateLimiter (sliding window) ──────────────────────────────────────────────


def test_rate_limiter_blocks_61st_call_without_raising() -> None:
    sleep_calls: list[float] = []
    # Frozen clock: all 61 calls land in the same window.
    limiter = RateLimiter(
        max_calls=60,
        period=60.0,
        _sleep=lambda s: sleep_calls.append(s),
        _monotonic=lambda: 1000.0,
    )

    # 61 acquisitions in under 60s must not raise; the 61st must block (sleep).
    for _ in range(61):
        limiter.acquire()

    assert len(sleep_calls) == 1, "exactly the 61st call should have blocked"
    assert sleep_calls[0] == pytest.approx(60.0)


def test_rate_limiter_allows_calls_after_window_passes() -> None:
    sleep_calls: list[float] = []
    now = [1000.0]
    limiter = RateLimiter(
        max_calls=2,
        period=10.0,
        _sleep=lambda s: sleep_calls.append(s),
        _monotonic=lambda: now[0],
    )

    limiter.acquire()
    limiter.acquire()
    # Advance the clock past the window — the old calls should age out.
    now[0] = 1011.0
    limiter.acquire()

    assert sleep_calls == [], "no blocking needed once the window has passed"
