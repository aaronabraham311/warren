"""Finnhub free-tier client — news + fundamentals fallback.

Two roles in the Warren harness:
  1. News retrieval (``get_news``) for event detection and sentiment context.
  2. Fundamentals fallback (``get_basic_financials``) when yfinance returns stale
     data — the field names mirror ``FundamentalsData`` so the agent can compare
     sources and reason about discrepancies. Every model carries a ``source`` field.

Free tier caps at 60 req/min; ``RateLimiter`` enforces that ceiling internally so
callers never have to. Responses are cached to SQLite (1h news, 24h financials) and
429s are retried with exponential backoff. Like the other data-source clients, public
methods return ``DataSourceError`` on failure rather than raising — the one exception
is construction-time validation of the API key.
"""

import sqlite3
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from typing import Literal, TypeVar

import finnhub
from pydantic import BaseModel, TypeAdapter

from data_sources.cache import CacheStore, make_key
from data_sources.errors import DataSourceError
from data_sources.symbols import to_finnhub_symbol

_T = TypeVar("_T")


# ── Pydantic output schemas ───────────────────────────────────────────────────


class NewsItem(BaseModel):
    headline: str
    summary: str
    source: str
    datetime: datetime
    url: str
    # Finnhub's free-tier company-news endpoint carries no per-article sentiment,
    # so this is always None today — the field exists for forward compatibility.
    sentiment: float | None = None


class FinnhubFinancials(BaseModel):
    ticker: str
    as_of: date
    pe_ratio: float | None
    pb_ratio: float | None
    roe_pct: float | None
    debt_to_equity: float | None = None
    gross_margin_pct: float | None = None
    operating_margin_pct: float | None = None
    net_margin_pct: float | None = None
    # Field names deliberately mirror FundamentalsData (yfinance) so comparison
    # logic can treat the two as structurally equivalent.
    source: Literal["finnhub"] = "finnhub"


class FinnhubInsiderTransaction(BaseModel):
    name: str
    transaction_type: Literal["buy", "sell", "other"]
    shares: int
    value: float | None
    transaction_date: date


_NEWS_ADAPTER = TypeAdapter(list[NewsItem])
_INSIDER_ADAPTER = TypeAdapter(list[FinnhubInsiderTransaction])


# ── Internal sentinels (mapped to DataSourceError before returning) ───────────


class _NotFoundError(Exception):
    pass


# ── Field conversion helpers (keep mypy strict-clean; no Any in returns) ──────


def _as_float(v: object) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def _de_to_pct(v: object) -> float | None:
    """Finnhub reports debt-to-equity as a raw ratio (e.g. 1.87); yfinance's
    ``debtToEquity`` is percent-form (e.g. 187.0). Scale by 100 so the projected
    ``FundamentalsData.debt_to_equity`` keeps a single, consistent convention."""
    f = _as_float(v)
    return f * 100 if f is not None else None


def _as_str(v: object) -> str:
    return v if isinstance(v, str) else ""


def _epoch_to_dt(v: object) -> datetime:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return datetime.fromtimestamp(float(v), tz=timezone.utc)
    return datetime.now(timezone.utc)


# ── Rate limiter ──────────────────────────────────────────────────────────────


class RateLimiter:
    """Sliding-window limiter: never more than ``max_calls`` within any
    ``period``-second span, regardless of when within the window they land.

    The clock and sleep are injectable so tests can exercise the ceiling without
    actually waiting on the wall clock.
    """

    def __init__(
        self,
        max_calls: int,
        period: float,
        _sleep: Callable[[float], None] = time.sleep,
        _monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max = max_calls
        self._period = period
        self._sleep = _sleep
        self._monotonic = _monotonic
        self._calls: list[float] = []

    def acquire(self) -> None:
        now = self._monotonic()
        # Drop timestamps that have aged out of the window.
        self._calls = [t for t in self._calls if now - t < self._period]
        if len(self._calls) >= self._max:
            # Window full — block until the oldest call ages out, then re-evict.
            sleep_for = self._period - (now - self._calls[0])
            if sleep_for > 0:
                self._sleep(sleep_for)
            now = self._monotonic()
            self._calls = [t for t in self._calls if now - t < self._period]
        self._calls.append(self._monotonic())


# ── FinnhubClient ─────────────────────────────────────────────────────────────


class FinnhubClient:
    def __init__(
        self,
        db_conn: sqlite3.Connection,
        api_key: str,
        *,
        cache_ttl_news_h: float = 1.0,
        cache_ttl_financials_h: float = 24.0,
        _sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise EnvironmentError(
                "FINNHUB_API_KEY is required. Set it in the environment before "
                "constructing FinnhubClient."
            )
        self.client = finnhub.Client(api_key=api_key)
        self._cache = CacheStore(db_conn)
        self._ttl_news = cache_ttl_news_h
        self._ttl_financials = cache_ttl_financials_h
        self._sleep = _sleep
        self._rate_limiter = RateLimiter(max_calls=60, period=60.0, _sleep=_sleep)

    # ── Retry / backoff ────────────────────────────────────────────────────

    def _with_retry(self, fn: Callable[[], _T], max_attempts: int = 3) -> _T:
        for attempt in range(max_attempts):
            try:
                self._rate_limiter.acquire()
                return fn()
            except finnhub.FinnhubAPIException as exc:
                # Only 429 (rate limit) is retryable; back off and try again.
                if exc.status_code == 429 and attempt < max_attempts - 1:
                    self._sleep(2**attempt)
                    continue
                raise
        raise RuntimeError("unreachable")

    # ── get_news ───────────────────────────────────────────────────────────

    def get_news(self, ticker: str, days: int = 7) -> list[NewsItem] | DataSourceError:
        symbol = to_finnhub_symbol(ticker)
        key = make_key("finnhub_news", symbol, str(days))
        cached = self._cache.get(key)
        if cached is not None:
            return list(_NEWS_ADAPTER.validate_json(cached))

        to_date = date.today()
        from_date = to_date - timedelta(days=days)
        try:
            raw = self._with_retry(
                lambda: self.client.company_news(
                    symbol,
                    _from=from_date.isoformat(),
                    to=to_date.isoformat(),
                )
            )
            items = self._parse_news(raw)
        except finnhub.FinnhubRequestException as exc:
            return DataSourceError(error_code="network", message=str(exc))
        except finnhub.FinnhubAPIException as exc:
            return DataSourceError(error_code="network", message=str(exc))
        except (KeyError, ValueError, TypeError) as exc:
            return DataSourceError(error_code="parse", message=str(exc))

        # Newest first.
        items.sort(key=lambda n: n.datetime, reverse=True)
        self._cache.set(key, _NEWS_ADAPTER.dump_json(items).decode(), self._ttl_news)
        return items

    @staticmethod
    def _parse_news(raw: object) -> list[NewsItem]:
        if not isinstance(raw, list):
            raise ValueError("unexpected company_news shape (expected a list)")
        items: list[NewsItem] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            items.append(
                NewsItem(
                    headline=_as_str(entry.get("headline")),
                    summary=_as_str(entry.get("summary")),
                    source=_as_str(entry.get("source")),
                    datetime=_epoch_to_dt(entry.get("datetime")),
                    url=_as_str(entry.get("url")),
                    sentiment=None,
                )
            )
        return items

    # ── get_basic_financials ───────────────────────────────────────────────

    def get_basic_financials(self, ticker: str) -> FinnhubFinancials | DataSourceError:
        symbol = to_finnhub_symbol(ticker)
        key = make_key("finnhub_financials", symbol)
        cached = self._cache.get(key)
        if cached is not None:
            return FinnhubFinancials.model_validate_json(cached)

        try:
            raw = self._with_retry(lambda: self.client.company_basic_financials(symbol, "all"))
            result = self._parse_financials(symbol, raw)
        except _NotFoundError:
            return DataSourceError(
                error_code="not_found", message=f"No Finnhub financials for {symbol}"
            )
        except finnhub.FinnhubRequestException as exc:
            return DataSourceError(error_code="network", message=str(exc))
        except finnhub.FinnhubAPIException as exc:
            return DataSourceError(error_code="network", message=str(exc))
        except (KeyError, ValueError, TypeError) as exc:
            return DataSourceError(error_code="parse", message=str(exc))

        self._cache.set(key, result.model_dump_json(), self._ttl_financials)
        return result

    @staticmethod
    def _parse_financials(ticker: str, raw: object) -> FinnhubFinancials:
        if not isinstance(raw, dict):
            raise ValueError("unexpected company_basic_financials shape (expected a dict)")
        metric = raw.get("metric")
        # Finnhub returns {"metric": null} (or an empty dict) for unknown symbols.
        if not isinstance(metric, dict) or not metric:
            raise _NotFoundError(ticker)
        return FinnhubFinancials(
            ticker=ticker.upper(),
            as_of=date.today(),
            pe_ratio=_as_float(metric.get("peTTM")),
            pb_ratio=_as_float(metric.get("pbQuarterly") or metric.get("pbAnnual")),
            roe_pct=_as_float(metric.get("roeTTM")),
            # Finnhub margins are already percentages — map directly (no x100).
            gross_margin_pct=_as_float(
                metric.get("grossMarginTTM") or metric.get("grossMarginAnnual")
            ),
            operating_margin_pct=_as_float(
                metric.get("operatingMarginTTM") or metric.get("operatingMarginAnnual")
            ),
            net_margin_pct=_as_float(
                metric.get("netProfitMarginTTM") or metric.get("netProfitMarginAnnual")
            ),
            debt_to_equity=_de_to_pct(
                metric.get("totalDebt/totalEquityQuarterly")
                or metric.get("totalDebt/totalEquityAnnual")
            ),
            source="finnhub",
        )

    # ── get_insider_transactions ───────────────────────────────────────────

    def get_insider_transactions(
        self, ticker: str, days: int = 90
    ) -> list[FinnhubInsiderTransaction] | DataSourceError:
        symbol = to_finnhub_symbol(ticker)
        key = make_key("finnhub_insider", symbol, str(days))
        cached = self._cache.get(key)
        if cached is not None:
            return list(_INSIDER_ADAPTER.validate_json(cached))

        to_date = date.today()
        from_date = to_date - timedelta(days=days)
        try:
            raw = self._with_retry(
                lambda: self.client.stock_insider_transactions(
                    symbol,
                    from_date.isoformat(),
                    to_date.isoformat(),
                )
            )
            items = self._parse_insider_transactions(raw)
        except finnhub.FinnhubRequestException as exc:
            return DataSourceError(error_code="network", message=str(exc))
        except finnhub.FinnhubAPIException as exc:
            return DataSourceError(error_code="network", message=str(exc))
        except (KeyError, ValueError, TypeError) as exc:
            return DataSourceError(error_code="parse", message=str(exc))

        self._cache.set(key, _INSIDER_ADAPTER.dump_json(items).decode(), self._ttl_financials)
        return items

    _TRANSACTION_CODE_MAP: dict[str, Literal["buy", "sell", "other"]] = {
        "P": "buy",
        "S": "sell",
    }

    @classmethod
    def _parse_insider_transactions(cls, raw: object) -> list[FinnhubInsiderTransaction]:
        if not isinstance(raw, dict):
            raise ValueError("unexpected stock_insider_transactions shape (expected a dict)")
        data = raw.get("data")
        if data is None:
            return []
        if not isinstance(data, list):
            raise ValueError("unexpected 'data' field (expected a list)")
        items: list[FinnhubInsiderTransaction] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            code = entry.get("transactionCode")
            txn_type: Literal["buy", "sell", "other"] = cls._TRANSACTION_CODE_MAP.get(
                code if isinstance(code, str) else "", "other"
            )
            raw_date = entry.get("transactionDate")
            try:
                txn_date = date.fromisoformat(_as_str(raw_date)) if raw_date else date.today()
            except ValueError:
                txn_date = date.today()
            shares_raw = entry.get("share")
            shares = abs(int(float(shares_raw))) if isinstance(shares_raw, (int, float)) else 0
            items.append(
                FinnhubInsiderTransaction(
                    name=_as_str(entry.get("name")),
                    transaction_type=txn_type,
                    shares=shares,
                    value=_as_float(entry.get("value")),
                    transaction_date=txn_date,
                )
            )
        return items
