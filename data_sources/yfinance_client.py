import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Literal, TypeVar

import yfinance as yf
from pydantic import BaseModel

from data_sources.cache import CacheStore, make_key
from data_sources.errors import DataSourceError

_T = TypeVar("_T")


# ── Legacy dataclass + function kept for backward compatibility ───────────────


@dataclass
class QuoteData:
    ticker: str
    price: float | None
    previous_close: float | None
    day_change_pct: float | None
    volume: int | None


def get_quote(ticker: str) -> QuoteData:
    info = yf.Ticker(ticker).fast_info
    price = info.last_price
    prev_close = info.previous_close
    volume = info.three_month_average_volume

    day_change_pct = (
        round((price - prev_close) / prev_close * 100, 2)
        if prev_close and prev_close != 0
        else None
    )

    return QuoteData(
        ticker=ticker.upper(),
        price=round(price, 2) if price else None,
        previous_close=round(prev_close, 2) if prev_close else None,
        day_change_pct=day_change_pct,
        volume=int(volume) if volume else None,
    )


# ── Pydantic output schemas ───────────────────────────────────────────────────


class PriceData(BaseModel):
    ticker: str
    current_price: float | None
    previous_close: float | None
    day_change_pct: float | None
    volume: int | None
    as_of: datetime
    data_age_hours: int
    source: Literal["yfinance"] = "yfinance"


class FundamentalsData(BaseModel):
    ticker: str
    as_of: date
    pe_ratio: float | None
    pb_ratio: float | None
    roe_pct: float | None
    debt_to_equity: float | None
    fcf_ttm_usd: int | None
    operating_margin_pct: float | None
    net_margin_pct: float | None
    data_age_hours: int
    source: Literal["yfinance", "finnhub"]


class GrowthData(BaseModel):
    ticker: str
    as_of: date
    revenue_cagr_3y: float | None
    revenue_cagr_5y: float | None = None
    earnings_cagr_3y: float | None
    peg_ratio: float | None
    data_age_hours: int
    source: Literal["yfinance"] = "yfinance"


# ── Internal sentinels ────────────────────────────────────────────────────────


class _NotFoundError(Exception):
    pass


# ── Field conversion helpers (avoid Any in annotations) ──────────────────────


def _as_float(v: object) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _as_pct(v: object) -> float | None:
    if isinstance(v, (int, float)):
        return round(float(v) * 100, 4)
    return None


def _as_int(v: object) -> int | None:
    if isinstance(v, (int, float)):
        return int(float(v))
    return None


def _fiscal_age_hours(info: dict[str, object]) -> int:
    fiscal_end = info.get("lastFiscalYearEnd")
    if isinstance(fiscal_end, (int, float)):
        return int((datetime.now(timezone.utc).timestamp() - float(fiscal_end)) / 3600)
    return 0


# ── YFinanceClient ────────────────────────────────────────────────────────────


class YFinanceClient:
    def __init__(
        self,
        db_conn: sqlite3.Connection,
        cache_ttl_prices_h: float = 1.0,
        cache_ttl_fundamentals_h: float = 24.0,
        _sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._cache = CacheStore(db_conn)
        self._ttl_prices = cache_ttl_prices_h
        self._ttl_fundamentals = cache_ttl_fundamentals_h
        self._sleep = _sleep
        self._last_call: float = 0.0

    # ── Throttle + retry ──────────────────────────────────────────────────

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        wait = max(0.0, 0.1 - elapsed)
        if wait > 0:
            self._sleep(wait)
        self._last_call = time.monotonic()

    def _fetch_with_retry(
        self,
        fn: Callable[[], _T],
        max_attempts: int = 3,
        no_retry: tuple[type[Exception], ...] = (),
    ) -> _T:
        for attempt in range(max_attempts):
            try:
                self._throttle()
                return fn()
            except Exception as exc:
                if isinstance(exc, no_retry) or attempt == max_attempts - 1:
                    raise
                self._sleep(2**attempt)
        raise RuntimeError("unreachable")

    # ── get_price ─────────────────────────────────────────────────────────

    def get_price(self, ticker: str) -> PriceData | DataSourceError:
        key = make_key("yf_price", ticker)
        cached = self._cache.get(key)
        if cached is not None:
            return PriceData.model_validate_json(cached)
        try:
            result = self._fetch_with_retry(
                lambda: self._fetch_price(ticker),
                no_retry=(_NotFoundError,),
            )
        except _NotFoundError:
            return DataSourceError(error_code="not_found", message=f"No price data for {ticker}")
        except Exception as exc:
            return DataSourceError(error_code="network", message=str(exc))
        self._cache.set(key, result.model_dump_json(), self._ttl_prices)
        return result

    def _fetch_price(self, ticker: str) -> PriceData:
        t = yf.Ticker(ticker)
        fi = t.fast_info
        price = getattr(fi, "last_price", None)
        prev_close = getattr(fi, "previous_close", None)
        volume = getattr(fi, "three_month_average_volume", None)
        if price is None:
            raise _NotFoundError(ticker)
        p = float(price)
        pc = float(prev_close) if prev_close is not None else None
        day_change_pct = round((p - pc) / pc * 100, 2) if pc and pc != 0.0 else None
        return PriceData(
            ticker=ticker.upper(),
            current_price=round(p, 2),
            previous_close=round(pc, 2) if pc is not None else None,
            day_change_pct=day_change_pct,
            volume=int(float(volume)) if volume is not None else None,
            as_of=datetime.now(timezone.utc),
            data_age_hours=0,
        )

    # ── get_fundamentals ──────────────────────────────────────────────────

    def get_fundamentals(self, ticker: str) -> FundamentalsData | DataSourceError:
        key = make_key("yf_fundamentals", ticker)
        cached = self._cache.get(key)
        if cached is not None:
            return FundamentalsData.model_validate_json(cached)
        try:
            result = self._fetch_with_retry(
                lambda: self._fetch_fundamentals(ticker),
                no_retry=(_NotFoundError,),
            )
        except _NotFoundError:
            return DataSourceError(error_code="not_found", message=f"No data for {ticker}")
        except Exception as exc:
            return DataSourceError(error_code="network", message=str(exc))
        self._cache.set(key, result.model_dump_json(), self._ttl_fundamentals)
        return result

    def _fetch_fundamentals(self, ticker: str) -> FundamentalsData:
        t = yf.Ticker(ticker)
        info: dict[str, object] = t.info
        if not isinstance(info, dict) or len(info) <= 5:
            raise _NotFoundError(ticker)
        if info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            raise _NotFoundError(ticker)
        return FundamentalsData(
            ticker=ticker.upper(),
            as_of=date.today(),
            pe_ratio=_as_float(info.get("trailingPE")),
            pb_ratio=_as_float(info.get("priceToBook")),
            roe_pct=_as_pct(info.get("returnOnEquity")),
            debt_to_equity=_as_float(info.get("debtToEquity")),
            fcf_ttm_usd=_as_int(info.get("freeCashflow")),
            operating_margin_pct=_as_pct(info.get("operatingMargins")),
            net_margin_pct=_as_pct(info.get("profitMargins")),
            data_age_hours=_fiscal_age_hours(info),
            source="yfinance",
        )

    # ── get_growth_metrics ────────────────────────────────────────────────

    def get_growth_metrics(self, ticker: str) -> GrowthData | DataSourceError:
        key = make_key("yf_growth", ticker)
        cached = self._cache.get(key)
        if cached is not None:
            return GrowthData.model_validate_json(cached)
        try:
            result = self._fetch_with_retry(
                lambda: self._fetch_growth_metrics(ticker),
                no_retry=(_NotFoundError,),
            )
        except _NotFoundError:
            return DataSourceError(error_code="not_found", message=f"No data for {ticker}")
        except Exception as exc:
            return DataSourceError(error_code="network", message=str(exc))
        self._cache.set(key, result.model_dump_json(), self._ttl_fundamentals)
        return result

    def _fetch_growth_metrics(self, ticker: str) -> GrowthData:
        t = yf.Ticker(ticker)
        info: dict[str, object] = t.info
        if not isinstance(info, dict) or len(info) <= 5:
            raise _NotFoundError(ticker)
        return GrowthData(
            ticker=ticker.upper(),
            as_of=date.today(),
            revenue_cagr_3y=self._compute_cagr(t, "Total Revenue", max_years=3),
            revenue_cagr_5y=self._compute_cagr(t, "Total Revenue", max_years=5),
            earnings_cagr_3y=self._compute_cagr(t, "Net Income", max_years=3),
            peg_ratio=_as_float(info.get("pegRatio")),
            data_age_hours=_fiscal_age_hours(info),
        )

    def _compute_cagr(
        self, ticker_obj: object, metric: str, max_years: int | None = None
    ) -> float | None:
        try:
            fin = ticker_obj.financials  # type: ignore[attr-defined]
            if metric not in fin.index:
                return None
            series = fin.loc[metric].dropna()
            values: list[float] = [float(v) for v in series.values if float(v) > 0]
            if len(values) < 2:
                return None
            # Values are newest-first; cap the window to (max_years + 1) data points.
            if max_years is not None:
                values = values[: max_years + 1]
            n_years = len(values) - 1
            return round(float((values[0] / values[-1]) ** (1.0 / n_years) - 1.0), 4)
        except Exception:
            return None
