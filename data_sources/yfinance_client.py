import json
import math
import sqlite3
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Literal, TypeVar

import requests
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
    gross_margin_pct: float | None
    operating_margin_pct: float | None
    net_margin_pct: float | None
    sector: str | None
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


class ValuationData(BaseModel):
    ticker: str
    as_of: date
    enterprise_value: int | None
    ev_to_ebit: float | None
    ev_to_ebitda: float | None
    acquirers_multiple: float | None
    fcf_yield: float | None
    earnings_yield: float | None
    market_cap_usd: int | None
    ncav: int | None
    ncav_to_market_cap: float | None
    is_net_net: bool
    price_to_ncav: float | None
    net_cash_usd: int | None
    net_cash_positive: bool
    p_tangible_book: float | None
    dividend_yield_pct: float | None
    data_age_hours: int
    source: Literal["yfinance"] = "yfinance"


class QualityData(BaseModel):
    ticker: str
    as_of: date
    roic_pct: float | None
    roic_series: list[float | None]
    roic_mean: float | None
    roa_pct: float | None
    gross_margin_pct: float | None
    gross_margin_series: list[float | None]
    gross_margin_stdev: float | None
    cash_conversion_ttm: float | None
    cash_conversion_series: list[float | None]
    consecutive_profit_years: int | None
    ncav_trend: Literal["growing", "stable", "declining"] | None
    data_age_hours: int
    source: Literal["yfinance"] = "yfinance"


class OwnershipData(BaseModel):
    ticker: str
    as_of: date
    insider_pct: float | None
    institutional_pct: float | None
    source: Literal["yfinance"] = "yfinance"


class OfficerRecord(BaseModel):
    name: str
    title: str
    year_born: int | None
    total_pay_usd: int | None


class InstitutionalHolderRecord(BaseModel):
    name: str
    shares: int | None
    pct_held: float | None  # decimal fraction of float outstanding (e.g. 0.0796 = 7.96%)
    value: int | None


class KeyPersonsRaw(BaseModel):
    ticker: str
    as_of: date
    officers: list[OfficerRecord]
    institutional_holders: list[InstitutionalHolderRecord]
    data_age_hours: int
    source: Literal["yfinance"] = "yfinance"


# ── Multi-year financial-statement history (the data foundation the value tools share) ──
#
# Rows are newest-first and year-aligned: index 0 is the most recent fiscal year. A
# null line item (NaN in the upstream frame, or a metric absent for that year) is None
# in the row, so the year positions stay aligned across statements. Consumers that want
# the legacy null-compacted series call ``YFinanceClient._series(rows, field)``.


class IncomeStatementRow(BaseModel):
    fiscal_year: int
    revenue: int | None
    gross_profit: int | None
    operating_income: int | None
    net_income: int | None
    ebit: int | None
    ebitda: int | None
    interest_expense: int | None
    pretax_income: int | None
    tax_provision: int | None


class BalanceSheetRow(BaseModel):
    fiscal_year: int
    total_assets: int | None
    total_liabilities: int | None
    current_assets: int | None
    current_liabilities: int | None
    long_term_debt: int | None
    total_debt: int | None
    cash_and_equivalents: int | None
    retained_earnings: int | None
    common_stock: int | None
    shares_outstanding: int | None


class CashFlowRow(BaseModel):
    fiscal_year: int
    cfo: int | None
    capex: int | None
    free_cash_flow: int | None
    dividends_paid: int | None
    buybacks: int | None


class FinancialsHistory(BaseModel):
    ticker: str
    as_of: date
    fiscal_years: list[int]
    income_statement: list[IncomeStatementRow]
    balance_sheet: list[BalanceSheetRow]
    cash_flow: list[CashFlowRow]
    data_age_hours: int
    source: Literal["yfinance"] = "yfinance"


# yfinance line-item name → row field. The first matching name in a tuple wins (fallbacks).
_INCOME_METRICS: dict[str, tuple[str, ...]] = {
    "revenue": ("Total Revenue",),
    "gross_profit": ("Gross Profit",),
    "operating_income": ("Operating Income",),
    "net_income": ("Net Income",),
    "ebit": ("EBIT",),
    "ebitda": ("EBITDA",),
    "interest_expense": ("Interest Expense",),
    "pretax_income": ("Pretax Income",),
    "tax_provision": ("Tax Provision",),
}
_BALANCE_METRICS: dict[str, tuple[str, ...]] = {
    "total_assets": ("Total Assets",),
    "total_liabilities": ("Total Liabilities Net Minority Interest",),
    "current_assets": ("Current Assets",),
    "current_liabilities": ("Current Liabilities",),
    "long_term_debt": ("Long Term Debt",),
    "total_debt": ("Total Debt",),
    "cash_and_equivalents": ("Cash And Cash Equivalents",),
    "retained_earnings": ("Retained Earnings",),
    "common_stock": ("Common Stock",),
    "shares_outstanding": ("Share Issued", "Ordinary Shares Number"),
}
_CASHFLOW_METRICS: dict[str, tuple[str, ...]] = {
    "cfo": ("Operating Cash Flow",),
    "capex": ("Capital Expenditure",),
    "free_cash_flow": ("Free Cash Flow",),
    "dividends_paid": ("Cash Dividends Paid",),
    "buybacks": ("Repurchase Of Capital Stock",),
}

_MAX_FIN_YEARS = 5


# ── Internal sentinels ────────────────────────────────────────────────────────


class PiotroskySignals(BaseModel):
    roa_positive: bool | None
    op_cf_positive: bool | None
    roa_improved: bool | None
    accruals_negative: bool | None
    leverage_decreased: bool | None
    current_ratio_improved: bool | None
    no_dilution: bool | None
    gross_margin_improved: bool | None
    asset_turnover_improved: bool | None


class FinancialStrengthData(BaseModel):
    ticker: str
    as_of: date
    f_score: int | None
    f_signals: PiotroskySignals
    z_score: float | None
    z_zone: Literal["distress", "grey", "safe"] | None
    interest_coverage: float | None
    current_ratio: float | None
    net_debt_to_ebitda: float | None
    data_age_hours: int
    source: Literal["yfinance"] = "yfinance"


class CapitalAllocation(BaseModel):
    ticker: str
    as_of: date
    years_covered: int
    share_count_cagr_pct: float | None  # negative = net buybacks, positive = dilution
    share_count_series: list[int | None]  # shares outstanding, newest-first
    buyback_yield_pct: float | None  # |buybacks| / market cap
    dividend_yield_pct: float | None  # |dividends paid| / market cap
    shareholder_yield_pct: float | None  # buyback + dividend yield
    dividend_growth_streak: int | None  # consecutive years the dividend grew
    payout_ratio_pct: float | None  # |dividends paid| / net income
    net_debt_series: list[int | None]  # total_debt - cash, newest-first
    net_debt_trajectory: Literal["delevering", "levering", "stable"] | None
    data_age_hours: int
    source: Literal["yfinance"] = "yfinance"


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


_TTL_KEY_PERSONS_H = 720.0  # 30 days — persons change rarely
_TTL_RUSSELL2000_H = 168.0  # 7 days — index rebalances quarterly

# Vanguard VTWO ETF holdings API — returns the full Russell 2000 constituent list
# as JSON with pagination. No auth required. TTL kept at 7 days so the constituent
# list stays fresh between quarterly rebalances without hitting the network each run.
_VTWO_HOLDINGS_URL = (
    "https://investor.vanguard.com/investment-products/etfs/profile/api/"
    "VTWO/portfolio-holding/stock"
)
_VTWO_PAGE_SIZE = 500


def _as_int_safe(v: object) -> int | None:
    """Like ``_as_int`` but rejects bools and NaN (statement frames carry NaN gaps)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isnan(f):
            return None
        return int(f)
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
        sector_raw = info.get("sector")
        sector = str(sector_raw) if isinstance(sector_raw, str) else None
        return FundamentalsData(
            ticker=ticker.upper(),
            as_of=date.today(),
            pe_ratio=_as_float(info.get("trailingPE")),
            pb_ratio=_as_float(info.get("priceToBook")),
            roe_pct=_as_pct(info.get("returnOnEquity")),
            debt_to_equity=_as_float(info.get("debtToEquity")),
            fcf_ttm_usd=_as_int(info.get("freeCashflow")),
            gross_margin_pct=_as_pct(info.get("grossMargins")),
            operating_margin_pct=_as_pct(info.get("operatingMargins")),
            net_margin_pct=_as_pct(info.get("profitMargins")),
            sector=sector,
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
        hist = self._build_financials(t, ticker)
        revenue = self._series(hist.income_statement, "revenue")
        net_income = self._series(hist.income_statement, "net_income")
        return GrowthData(
            ticker=ticker.upper(),
            as_of=date.today(),
            revenue_cagr_3y=self._cagr(revenue, max_years=3),
            revenue_cagr_5y=self._cagr(revenue, max_years=5),
            earnings_cagr_3y=self._cagr(net_income, max_years=3),
            peg_ratio=_as_float(info.get("pegRatio")),
            data_age_hours=_fiscal_age_hours(info),
        )

    @staticmethod
    def _cagr(series: list[float], max_years: int | None = None) -> float | None:
        # Newest-first positive values only, capped to (max_years + 1) data points.
        values = [v for v in series if v > 0]
        if len(values) < 2:
            return None
        if max_years is not None:
            values = values[: max_years + 1]
        n_years = len(values) - 1
        return round(float((values[0] / values[-1]) ** (1.0 / n_years) - 1.0), 4)

    # ── get_valuation_multiples ───────────────────────────────────────────────

    def get_valuation_multiples(self, ticker: str) -> ValuationData | DataSourceError:
        key = make_key("yf_valuation", ticker)
        cached = self._cache.get(key)
        if cached is not None:
            return ValuationData.model_validate_json(cached)
        try:
            result = self._fetch_with_retry(
                lambda: self._fetch_valuation_multiples(ticker),
                no_retry=(_NotFoundError,),
            )
        except _NotFoundError:
            return DataSourceError(error_code="not_found", message=f"No data for {ticker}")
        except Exception as exc:
            return DataSourceError(error_code="network", message=str(exc))
        self._cache.set(key, result.model_dump_json(), self._ttl_fundamentals)
        return result

    def _fetch_valuation_multiples(self, ticker: str) -> ValuationData:
        info: dict[str, object] = yf.Ticker(ticker).info
        if not isinstance(info, dict) or len(info) <= 5:
            raise _NotFoundError(ticker)
        if info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            raise _NotFoundError(ticker)

        ev = _as_int(info.get("enterpriseValue"))
        ebitda = _as_float(info.get("ebitda"))
        op_income = _as_float(info.get("operatingIncome"))
        fcf = _as_float(info.get("freeCashflow"))
        mkt_cap = _as_int(info.get("marketCap"))
        curr_assets_raw = info.get("currentAssets") or info.get("totalCurrentAssets")
        curr_assets = _as_int(curr_assets_raw)
        total_liab = _as_int(info.get("totalLiab"))
        tangible_bv = _as_float(info.get("tangibleBookValue"))

        ev_f = float(ev) if ev is not None else None
        ev_to_ebit = round(ev_f / op_income, 2) if ev_f and op_income and op_income > 0 else None
        ev_to_ebitda = round(ev_f / ebitda, 2) if ev_f and ebitda and ebitda > 0 else None
        fcf_yield = round(float(fcf) / ev_f * 100, 4) if fcf and ev_f and ev_f > 0 else None
        earnings_yield = (
            round(op_income / ev_f * 100, 4) if op_income and ev_f and ev_f > 0 else None
        )

        ncav: int | None = None
        if curr_assets is not None and total_liab is not None:
            ncav = curr_assets - total_liab

        ncav_to_mkt_cap: float | None = None
        is_net_net = False
        if ncav is not None and mkt_cap and mkt_cap > 0:
            ncav_to_mkt_cap = round(ncav / mkt_cap, 4)
            is_net_net = ncav > mkt_cap

        price_to_ncav: float | None = None
        if ncav is not None and ncav > 0 and mkt_cap and mkt_cap > 0:
            price_to_ncav = round(float(mkt_cap) / ncav, 4)

        total_cash = _as_int(info.get("totalCash"))
        total_debt_info = _as_int(info.get("totalDebt"))
        net_cash_usd: int | None = None
        if total_cash is not None and total_debt_info is not None:
            net_cash_usd = total_cash - total_debt_info
        net_cash_positive = net_cash_usd is not None and net_cash_usd > 0

        p_tangible_book: float | None = None
        if mkt_cap and tangible_bv and tangible_bv > 0:
            p_tangible_book = round(mkt_cap / tangible_bv, 2)

        return ValuationData(
            ticker=ticker.upper(),
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
            price_to_ncav=price_to_ncav,
            net_cash_usd=net_cash_usd,
            net_cash_positive=net_cash_positive,
            p_tangible_book=p_tangible_book,
            dividend_yield_pct=_as_pct(info.get("dividendYield")),
            data_age_hours=_fiscal_age_hours(info),
        )

    # ── get_quality_metrics ───────────────────────────────────────────────────

    def get_quality_metrics(self, ticker: str) -> QualityData | DataSourceError:
        key = make_key("yf_quality", ticker)
        cached = self._cache.get(key)
        if cached is not None:
            return QualityData.model_validate_json(cached)
        try:
            result = self._fetch_with_retry(
                lambda: self._fetch_quality_metrics(ticker),
                no_retry=(_NotFoundError,),
            )
        except _NotFoundError:
            return DataSourceError(error_code="not_found", message=f"No data for {ticker}")
        except Exception as exc:
            return DataSourceError(error_code="network", message=str(exc))
        self._cache.set(key, result.model_dump_json(), self._ttl_fundamentals)
        return result

    # ── get_financial_strength ────────────────────────────────────────────────

    def get_financial_strength(self, ticker: str) -> FinancialStrengthData | DataSourceError:
        key = make_key("yf_financial_strength", ticker)
        cached = self._cache.get(key)
        if cached is not None:
            return FinancialStrengthData.model_validate_json(cached)
        try:
            result = self._fetch_with_retry(
                lambda: self._fetch_financial_strength(ticker),
                no_retry=(_NotFoundError,),
            )
        except _NotFoundError:
            return DataSourceError(error_code="not_found", message=f"No data for {ticker}")
        except Exception as exc:
            return DataSourceError(error_code="network", message=str(exc))
        self._cache.set(key, result.model_dump_json(), self._ttl_fundamentals)
        return result

    # ── get_financials (multi-year statement history) ─────────────────────────

    def get_financials(self, ticker: str) -> FinancialsHistory | DataSourceError:
        key = make_key("yf_financials", ticker)
        cached = self._cache.get(key)
        if cached is not None:
            return FinancialsHistory.model_validate_json(cached)
        try:
            result = self._fetch_with_retry(
                lambda: self._fetch_financials(ticker),
                no_retry=(_NotFoundError,),
            )
        except _NotFoundError:
            return DataSourceError(error_code="not_found", message=f"No data for {ticker}")
        except Exception as exc:
            return DataSourceError(error_code="network", message=str(exc))
        self._cache.set(key, result.model_dump_json(), self._ttl_fundamentals)
        return result

    def _fetch_financials(self, ticker: str) -> FinancialsHistory:
        t = yf.Ticker(ticker)
        info: dict[str, object] = t.info
        if not isinstance(info, dict) or len(info) <= 5:
            raise _NotFoundError(ticker)
        if info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            raise _NotFoundError(ticker)
        # Missing statements yield empty row lists rather than an error (mirrors
        # get_quality_metrics); only an unknown ticker (empty info) is not_found.
        return self._build_financials(t, ticker)

    def _fetch_quality_metrics(self, ticker: str) -> QualityData:
        t = yf.Ticker(ticker)
        info: dict[str, object] = t.info
        if not isinstance(info, dict) or len(info) <= 5:
            raise _NotFoundError(ticker)
        if info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            raise _NotFoundError(ticker)

        hist = self._build_financials(t, ticker)
        roic_series = self._compute_roic_series(hist)
        roic_vals = [v for v in roic_series if v is not None]
        roic_mean = round(sum(roic_vals) / len(roic_vals), 4) if roic_vals else None

        gm_series = self._compute_gross_margin_series(hist)
        gm_vals = [v for v in gm_series if v is not None]
        gm_stdev = round(statistics.stdev(gm_vals), 4) if len(gm_vals) >= 2 else None

        cc_series = self._compute_cash_conversion_series(hist)
        roa_pct = self._compute_roa(hist)

        return QualityData(
            ticker=ticker.upper(),
            as_of=date.today(),
            roic_pct=roic_series[0] if roic_series else None,
            roic_series=roic_series,
            roic_mean=roic_mean,
            roa_pct=roa_pct,
            gross_margin_pct=gm_series[0] if gm_series else None,
            gross_margin_series=gm_series,
            gross_margin_stdev=gm_stdev,
            cash_conversion_ttm=cc_series[0] if cc_series else None,
            cash_conversion_series=cc_series,
            consecutive_profit_years=self._compute_consecutive_profit_years(hist),
            ncav_trend=self._compute_ncav_trend(hist),
            data_age_hours=_fiscal_age_hours(info),
        )

    def _fetch_financial_strength(self, ticker: str) -> FinancialStrengthData:
        t = yf.Ticker(ticker)
        info: dict[str, object] = t.info
        if not isinstance(info, dict) or len(info) <= 5:
            raise _NotFoundError(ticker)
        if info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            raise _NotFoundError(ticker)

        hist = self._build_financials(t, ticker)
        inc, bs, cf = hist.income_statement, hist.balance_sheet, hist.cash_flow
        net_incomes = self._series(inc, "net_income")
        op_cfs = self._series(cf, "cfo")
        total_assets = self._series(bs, "total_assets")
        current_assets = self._series(bs, "current_assets")
        current_liabs = self._series(bs, "current_liabilities")
        lt_debt = self._series(bs, "long_term_debt")
        gross_profits = self._series(inc, "gross_profit")
        revenues = self._series(inc, "revenue")
        retained_earnings = self._series(bs, "retained_earnings")
        total_liabs = self._series(bs, "total_liabilities")
        ebit_series = self._series(inc, "ebit")
        interest_exp_series = self._series(inc, "interest_expense")
        total_debt = self._series(bs, "total_debt")
        cash_series = self._series(bs, "cash_and_equivalents")
        ebitda_series = self._series(inc, "ebitda")
        shares_series = self._series(bs, "common_stock")

        signals = self._compute_piotroski(
            net_incomes,
            op_cfs,
            total_assets,
            current_assets,
            current_liabs,
            lt_debt,
            gross_profits,
            revenues,
            shares_series,
        )
        f_score: int | None = None
        if any(v is not None for v in signals.model_dump().values()):
            f_score = sum(1 for v in signals.model_dump().values() if v is True)

        z_score: float | None = None
        z_zone: Literal["distress", "grey", "safe"] | None = None
        mkt_cap = _as_float(info.get("marketCap"))
        if (
            len(current_assets) >= 1
            and len(current_liabs) >= 1
            and len(total_assets) >= 1
            and len(retained_earnings) >= 1
            and len(ebit_series) >= 1
            and len(total_liabs) >= 1
            and len(revenues) >= 1
            and mkt_cap is not None
        ):
            ta = total_assets[0]
            if ta > 0:
                x1 = (current_assets[0] - current_liabs[0]) / ta
                x2 = retained_earnings[0] / ta
                x3 = ebit_series[0] / ta
                x4 = mkt_cap / total_liabs[0] if total_liabs[0] != 0 else 0.0
                x5 = revenues[0] / ta
                z = round(1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5, 4)
                z_score = z
                z_zone = "distress" if z < 1.81 else ("grey" if z <= 2.99 else "safe")

        interest_coverage: float | None = None
        if ebit_series and interest_exp_series and interest_exp_series[0] != 0:
            # yfinance reports interest expense as negative
            ie = abs(interest_exp_series[0])
            if ie > 0:
                interest_coverage = round(ebit_series[0] / ie, 2)

        current_ratio: float | None = None
        if current_assets and current_liabs and current_liabs[0] != 0:
            current_ratio = round(current_assets[0] / current_liabs[0], 4)

        net_debt_to_ebitda: float | None = None
        if total_debt and cash_series and ebitda_series and ebitda_series[0] != 0:
            net_debt = total_debt[0] - cash_series[0]
            net_debt_to_ebitda = round(net_debt / abs(ebitda_series[0]), 4)

        return FinancialStrengthData(
            ticker=ticker.upper(),
            as_of=date.today(),
            f_score=f_score,
            f_signals=signals,
            z_score=z_score,
            z_zone=z_zone,
            interest_coverage=interest_coverage,
            current_ratio=current_ratio,
            net_debt_to_ebitda=net_debt_to_ebitda,
            data_age_hours=_fiscal_age_hours(info),
        )

    # ── get_capital_allocation (management quality) ───────────────────────────

    def get_capital_allocation(self, ticker: str) -> CapitalAllocation | DataSourceError:
        key = make_key("yf_capital_allocation", ticker)
        cached = self._cache.get(key)
        if cached is not None:
            return CapitalAllocation.model_validate_json(cached)
        try:
            result = self._fetch_with_retry(
                lambda: self._fetch_capital_allocation(ticker),
                no_retry=(_NotFoundError,),
            )
        except _NotFoundError:
            return DataSourceError(error_code="not_found", message=f"No data for {ticker}")
        except Exception as exc:
            return DataSourceError(error_code="network", message=str(exc))
        self._cache.set(key, result.model_dump_json(), self._ttl_fundamentals)
        return result

    def _fetch_capital_allocation(self, ticker: str) -> CapitalAllocation:
        t = yf.Ticker(ticker)
        info: dict[str, object] = t.info
        if not isinstance(info, dict) or len(info) <= 5:
            raise _NotFoundError(ticker)
        if info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            raise _NotFoundError(ticker)

        hist = self._build_financials(t, ticker)
        bs, cf, inc = hist.balance_sheet, hist.cash_flow, hist.income_statement
        mkt_cap = _as_float(info.get("marketCap"))

        # Share-count trend (newest-first). Negative CAGR = net buybacks.
        share_count_series = [r.shares_outstanding for r in bs]
        shares = [float(s) for s in share_count_series if s is not None]
        share_count_cagr_pct: float | None = None
        if len(shares) >= 2 and shares[0] > 0 and shares[-1] > 0:
            n = len(shares) - 1
            share_count_cagr_pct = round(((shares[0] / shares[-1]) ** (1 / n) - 1) * 100, 4)

        # Yields off the most recent year (cash-flow magnitudes / market cap).
        buybacks = self._series(cf, "buybacks")
        dividends = self._series(cf, "dividends_paid")
        buyback_yield_pct: float | None = None
        dividend_yield_pct: float | None = None
        if mkt_cap is not None and mkt_cap > 0:
            if buybacks:
                buyback_yield_pct = round(abs(buybacks[0]) / mkt_cap * 100, 4)
            if dividends:
                dividend_yield_pct = round(abs(dividends[0]) / mkt_cap * 100, 4)
        shareholder_yield_pct: float | None = None
        if buyback_yield_pct is not None or dividend_yield_pct is not None:
            shareholder_yield_pct = round(
                (buyback_yield_pct or 0.0) + (dividend_yield_pct or 0.0), 4
            )

        # Dividend growth streak: consecutive newest-first years the payout grew.
        dividend_growth_streak: int | None = None
        if dividends:
            mags = [abs(d) for d in dividends]
            streak = 0
            for i in range(len(mags) - 1):
                if mags[i] > mags[i + 1]:
                    streak += 1
                else:
                    break
            dividend_growth_streak = streak

        # Payout ratio off the most recent year.
        net_incomes = self._series(inc, "net_income")
        payout_ratio_pct: float | None = None
        if dividends and net_incomes and net_incomes[0] > 0:
            payout_ratio_pct = round(abs(dividends[0]) / net_incomes[0] * 100, 4)

        # Net-debt trajectory: per-year (total_debt - cash), window change vs oldest.
        net_debt_series = [
            (r.total_debt - r.cash_and_equivalents)
            if r.total_debt is not None and r.cash_and_equivalents is not None
            else None
            for r in bs
        ]
        nd = [v for v in net_debt_series if v is not None]
        net_debt_trajectory: Literal["delevering", "levering", "stable"] | None = None
        if len(nd) >= 2 and nd[-1] != 0:
            change = (nd[0] - nd[-1]) / abs(nd[-1])
            net_debt_trajectory = (
                "delevering" if change < -0.05 else ("levering" if change > 0.05 else "stable")
            )

        return CapitalAllocation(
            ticker=ticker.upper(),
            as_of=date.today(),
            years_covered=len(bs),
            share_count_cagr_pct=share_count_cagr_pct,
            share_count_series=share_count_series,
            buyback_yield_pct=buyback_yield_pct,
            dividend_yield_pct=dividend_yield_pct,
            shareholder_yield_pct=shareholder_yield_pct,
            dividend_growth_streak=dividend_growth_streak,
            payout_ratio_pct=payout_ratio_pct,
            net_debt_series=net_debt_series,
            net_debt_trajectory=net_debt_trajectory,
            data_age_hours=_fiscal_age_hours(info),
        )

    def _compute_piotroski(
        self,
        net_incomes: list[float],
        op_cfs: list[float],
        total_assets: list[float],
        current_assets: list[float],
        current_liabs: list[float],
        lt_debt: list[float],
        gross_profits: list[float],
        revenues: list[float],
        shares: list[float],
    ) -> PiotroskySignals:
        def _roa(idx: int) -> float | None:
            if idx < len(net_incomes) and idx < len(total_assets) and total_assets[idx] != 0:
                return net_incomes[idx] / total_assets[idx]
            return None

        roa0 = _roa(0)
        roa1 = _roa(1)
        ta0 = total_assets[0] if total_assets else None

        f1: bool | None = (roa0 > 0) if roa0 is not None else None
        f2: bool | None = (op_cfs[0] > 0) if op_cfs else None
        f3: bool | None = (roa0 > roa1) if (roa0 is not None and roa1 is not None) else None
        f4: bool | None = None
        if roa0 is not None and op_cfs and ta0 is not None and ta0 != 0:
            f4 = (op_cfs[0] / ta0) > roa0

        f5: bool | None = None
        if (
            len(lt_debt) >= 2
            and len(total_assets) >= 2
            and total_assets[0] != 0
            and total_assets[1] != 0
        ):
            f5 = (lt_debt[0] / total_assets[0]) < (lt_debt[1] / total_assets[1])

        f6: bool | None = None
        if (
            len(current_assets) >= 2
            and len(current_liabs) >= 2
            and current_liabs[0] != 0
            and current_liabs[1] != 0
        ):
            cr0 = current_assets[0] / current_liabs[0]
            cr1 = current_assets[1] / current_liabs[1]
            f6 = cr0 > cr1

        f7: bool | None = (shares[0] <= shares[1]) if len(shares) >= 2 else None

        f8: bool | None = None
        if len(gross_profits) >= 2 and len(revenues) >= 2 and revenues[0] != 0 and revenues[1] != 0:  # noqa: E501
            f8 = (gross_profits[0] / revenues[0]) > (gross_profits[1] / revenues[1])

        f9: bool | None = None
        if (
            len(revenues) >= 2
            and len(total_assets) >= 2
            and total_assets[0] != 0
            and total_assets[1] != 0
        ):
            f9 = (revenues[0] / total_assets[0]) > (revenues[1] / total_assets[1])

        return PiotroskySignals(
            roa_positive=f1,
            op_cf_positive=f2,
            roa_improved=f3,
            accruals_negative=f4,
            leverage_decreased=f5,
            current_ratio_improved=f6,
            no_dilution=f7,
            gross_margin_improved=f8,
            asset_turnover_improved=f9,
        )

    # ── Financial-statement history (shared foundation) ───────────────────────

    def _build_financials(self, ticker_obj: object, ticker: str) -> FinancialsHistory:
        """Read the three statement frames once and build a typed, year-aligned history."""
        inc_df = self._safe_attr(ticker_obj, "financials")
        bs_df = self._safe_attr(ticker_obj, "balance_sheet")
        cf_df = self._safe_attr(ticker_obj, "cashflow")

        inc_cells = self._statement_cells(inc_df, _INCOME_METRICS)
        bs_cells = self._statement_cells(bs_df, _BALANCE_METRICS)
        cf_cells = self._statement_cells(cf_df, _CASHFLOW_METRICS)

        income = [
            IncomeStatementRow(fiscal_year=y, **cells)
            for y, cells in self._rows_from_cells(inc_df, _INCOME_METRICS, inc_cells)
        ]
        balance = [
            BalanceSheetRow(fiscal_year=y, **cells)
            for y, cells in self._rows_from_cells(bs_df, _BALANCE_METRICS, bs_cells)
        ]
        cash = [
            CashFlowRow(fiscal_year=y, **cells)
            for y, cells in self._rows_from_cells(cf_df, _CASHFLOW_METRICS, cf_cells)
        ]
        info = self._safe_info(ticker_obj)
        return FinancialsHistory(
            ticker=ticker.upper(),
            as_of=date.today(),
            fiscal_years=[r.fiscal_year for r in income],
            income_statement=income,
            balance_sheet=balance,
            cash_flow=cash,
            data_age_hours=_fiscal_age_hours(info),
        )

    @staticmethod
    def _safe_attr(ticker_obj: object, attr: str) -> object:
        try:
            return getattr(ticker_obj, attr)
        except Exception:
            return None

    @staticmethod
    def _safe_info(ticker_obj: object) -> dict[str, object]:
        info = getattr(ticker_obj, "info", None)
        return info if isinstance(info, dict) else {}

    def _statement_cells(
        self, df: object, metrics: dict[str, tuple[str, ...]]
    ) -> dict[str, list[int | None]]:
        """For each row field, the positional (newest-first) values, NaN/missing → None."""
        return {field: self._raw_series(df, names) for field, names in metrics.items()}

    @staticmethod
    def _raw_series(df: object, names: tuple[str, ...]) -> list[int | None]:
        try:
            index = df.index  # type: ignore[attr-defined]
            for name in names:
                if name in index:
                    return [_as_int_safe(v) for v in df.loc[name].values]  # type: ignore[attr-defined]
        except Exception:
            return []
        return []

    def _rows_from_cells(
        self,
        df: object,
        metrics: dict[str, tuple[str, ...]],
        cells: dict[str, list[int | None]],
    ) -> list[tuple[int, dict[str, int | None]]]:
        years = self._fiscal_years_of(df)
        n = max([len(v) for v in cells.values()] + [len(years)], default=0)
        n = min(n, _MAX_FIN_YEARS)
        rows: list[tuple[int, dict[str, int | None]]] = []
        for i in range(n):
            cell = {
                field: (cells[field][i] if i < len(cells[field]) else None) for field in metrics
            }
            rows.append((years[i] if i < len(years) else 0, cell))
        return rows

    @staticmethod
    def _fiscal_years_of(df: object) -> list[int]:
        try:
            cols = list(df.columns)  # type: ignore[attr-defined]
        except Exception:
            return []
        years: list[int] = []
        for c in cols:
            y = getattr(c, "year", None)
            years.append(int(y) if isinstance(y, int) else 0)
        return years

    @staticmethod
    def _series(rows: Sequence[BaseModel], field: str) -> list[float]:
        """Null-compacted, newest-first float series for one row field.

        Equivalent to the legacy ``df.loc[metric].dropna()`` extraction: dropping the
        None cells reproduces ``dropna`` because the rows are positionally aligned.
        """
        out: list[float] = []
        for row in rows:
            v = getattr(row, field)
            if v is not None:
                out.append(float(v))
        return out

    def _compute_roic_series(self, hist: FinancialsHistory) -> list[float | None]:
        op_incomes = self._series(hist.income_statement, "operating_income")
        tax_provisions = self._series(hist.income_statement, "tax_provision")
        pretax_incomes = self._series(hist.income_statement, "pretax_income")
        total_assets = self._series(hist.balance_sheet, "total_assets")
        current_liabs = self._series(hist.balance_sheet, "current_liabilities")

        n = min(len(op_incomes), len(total_assets), len(current_liabs))
        if n == 0:
            return []

        result: list[float | None] = []
        for i in range(n):
            op_inc = op_incomes[i]
            ta = total_assets[i]
            cl = current_liabs[i]
            invested_capital = ta - cl
            if invested_capital <= 0:
                result.append(None)
                continue
            if i < len(tax_provisions) and i < len(pretax_incomes) and pretax_incomes[i] != 0:
                tax_rate = max(0.0, min(0.5, tax_provisions[i] / pretax_incomes[i]))
            else:
                tax_rate = 0.21
            nopat = op_inc * (1.0 - tax_rate)
            result.append(round(nopat / invested_capital * 100.0, 4))
        return result

    def _compute_gross_margin_series(self, hist: FinancialsHistory) -> list[float | None]:
        gross_profits = self._series(hist.income_statement, "gross_profit")
        revenues = self._series(hist.income_statement, "revenue")
        n = min(len(gross_profits), len(revenues))
        if n == 0:
            return []
        result: list[float | None] = []
        for i in range(n):
            if revenues[i] == 0:
                result.append(None)
            else:
                result.append(round(gross_profits[i] / revenues[i] * 100.0, 4))
        return result

    def _compute_cash_conversion_series(self, hist: FinancialsHistory) -> list[float | None]:
        fcf_series = self._series(hist.cash_flow, "free_cash_flow")
        if not fcf_series:
            op_cfs = self._series(hist.cash_flow, "cfo")
            capex = self._series(hist.cash_flow, "capex")
            n_cf = min(len(op_cfs), len(capex))
            fcf_series = [op_cfs[i] + capex[i] for i in range(n_cf)]
        net_incomes = self._series(hist.income_statement, "net_income")
        n = min(len(fcf_series), len(net_incomes))
        if n == 0:
            return []
        result: list[float | None] = []
        for i in range(n):
            if net_incomes[i] == 0:
                result.append(None)
            else:
                result.append(round(fcf_series[i] / net_incomes[i], 4))
        return result

    def _compute_roa(self, hist: FinancialsHistory) -> float | None:
        net_incomes = self._series(hist.income_statement, "net_income")
        total_assets = self._series(hist.balance_sheet, "total_assets")
        if not net_incomes or not total_assets or total_assets[0] == 0:
            return None
        return round(net_incomes[0] / total_assets[0] * 100.0, 4)

    def _compute_consecutive_profit_years(self, hist: FinancialsHistory) -> int | None:
        """Count consecutive years (newest-first) with positive operating income."""
        if not hist.income_statement:
            return None
        count = 0
        for row in hist.income_statement:
            if row.operating_income is not None and row.operating_income > 0:
                count += 1
            else:
                break
        return count

    def _compute_ncav_trend(
        self, hist: FinancialsHistory
    ) -> Literal["growing", "stable", "declining"] | None:
        """Classify YoY NCAV direction from ≥3 years; None when history too short."""
        ncavs: list[float] = []
        for row in hist.balance_sheet:
            if row.current_assets is not None and row.total_liabilities is not None:
                ncavs.append(float(row.current_assets) - float(row.total_liabilities))
        if len(ncavs) < 3:
            return None
        deltas = [ncavs[i] - ncavs[i + 1] for i in range(len(ncavs) - 1)]
        positives = sum(1 for d in deltas if d > 0)
        negatives = sum(1 for d in deltas if d < 0)
        if positives > negatives:
            return "growing"
        if negatives > positives:
            return "declining"
        return "stable"

    # ── get_russell2000_tickers ───────────────────────────────────────────────

    def get_russell2000_tickers(self) -> list[str] | DataSourceError:
        """Return the current Russell 2000 constituent tickers, cached for 7 days.

        Fetches from the Vanguard VTWO ETF holdings API (paginated). Falls back
        to a DataSourceError on network failure so callers can degrade gracefully
        (portfolio + watchlist still form the fallback universe).
        """
        key = make_key("yf_russell2000_tickers")
        cached = self._cache.get(key)
        if cached is not None:
            tickers: list[str] = json.loads(cached)
            return tickers
        try:
            tickers = self._fetch_russell2000_tickers()
        except Exception as exc:
            return DataSourceError(error_code="network", message=str(exc))
        self._cache.set(key, json.dumps(tickers), _TTL_RUSSELL2000_H)
        return tickers

    def _fetch_russell2000_tickers(self) -> list[str]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        }
        tickers: list[str] = []
        start = 1
        while True:
            resp = requests.get(
                _VTWO_HOLDINGS_URL,
                params={"start": start, "count": _VTWO_PAGE_SIZE},
                headers=headers,
                timeout=20,
            )
            resp.raise_for_status()
            data: dict[str, object] = resp.json()
            fund = data.get("fund")
            if not isinstance(fund, dict):
                break
            entities = fund.get("entity")
            if not isinstance(entities, list) or not entities:
                break
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                ticker = str(entity.get("ticker", "")).strip().upper()
                if ticker and ticker not in tickers:
                    tickers.append(ticker)
            total = data.get("size")
            start += _VTWO_PAGE_SIZE
            if not isinstance(total, int) or start > total:
                break
        if not tickers:
            raise RuntimeError("Vanguard VTWO API returned no tickers")
        return tickers

    # ── get_ownership ─────────────────────────────────────────────────────

    def get_ownership(self, ticker: str) -> OwnershipData | DataSourceError:
        key = make_key("yf_ownership", ticker)
        cached = self._cache.get(key)
        if cached is not None:
            return OwnershipData.model_validate_json(cached)
        try:
            result = self._fetch_with_retry(
                lambda: self._fetch_ownership(ticker),
                no_retry=(_NotFoundError,),
            )
        except _NotFoundError:
            return DataSourceError(error_code="not_found", message=f"No data for {ticker}")
        except Exception as exc:
            return DataSourceError(error_code="network", message=str(exc))
        self._cache.set(key, result.model_dump_json(), self._ttl_fundamentals)
        return result

    def _fetch_ownership(self, ticker: str) -> OwnershipData:
        info: dict[str, object] = yf.Ticker(ticker).info
        if not isinstance(info, dict) or len(info) <= 5:
            raise _NotFoundError(ticker)
        return OwnershipData(
            ticker=ticker.upper(),
            as_of=date.today(),
            insider_pct=_as_pct(info.get("heldPercentInsiders")),
            institutional_pct=_as_pct(info.get("heldPercentInstitutions")),
        )

    # ── get_key_persons ───────────────────────────────────────────────────────

    def get_key_persons(self, ticker: str) -> KeyPersonsRaw | DataSourceError:
        key = make_key("yf_key_persons", ticker)
        cached = self._cache.get(key)
        if cached is not None:
            return KeyPersonsRaw.model_validate_json(cached)
        try:
            result = self._fetch_with_retry(
                lambda: self._fetch_key_persons(ticker),
                no_retry=(_NotFoundError,),
            )
        except _NotFoundError:
            return DataSourceError(error_code="not_found", message=f"No data for {ticker}")
        except Exception as exc:
            return DataSourceError(error_code="network", message=str(exc))
        self._cache.set(key, result.model_dump_json(), _TTL_KEY_PERSONS_H)
        return result

    def _fetch_key_persons(self, ticker: str) -> KeyPersonsRaw:
        t = yf.Ticker(ticker)
        info: dict[str, object] = t.info
        if not isinstance(info, dict) or len(info) <= 5:
            raise _NotFoundError(ticker)
        if info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            raise _NotFoundError(ticker)

        officers: list[OfficerRecord] = []
        officers_raw = info.get("companyOfficers")
        if isinstance(officers_raw, list):
            for off in officers_raw:
                if not isinstance(off, dict):
                    continue
                name = off.get("name")
                title = off.get("title")
                if not isinstance(name, str) or not isinstance(title, str):
                    continue
                total_pay_raw = off.get("totalPay")
                total_pay: int | None = None
                if isinstance(total_pay_raw, dict):
                    total_pay = _as_int(total_pay_raw.get("raw"))
                elif isinstance(total_pay_raw, (int, float)):
                    total_pay = _as_int(total_pay_raw)
                officers.append(
                    OfficerRecord(
                        name=name,
                        title=title,
                        year_born=_as_int(off.get("yearBorn")),
                        total_pay_usd=total_pay,
                    )
                )

        institutional: list[InstitutionalHolderRecord] = []
        try:
            ih_df = self._safe_attr(t, "institutional_holders")
            if ih_df is not None:
                for _, row in ih_df.iterrows():  # type: ignore[attr-defined]
                    name_val = row.get("Holder")
                    if not isinstance(name_val, str):
                        continue
                    pct_raw = row.get("% Out")
                    institutional.append(
                        InstitutionalHolderRecord(
                            name=name_val,
                            shares=_as_int(row.get("Shares")),
                            pct_held=_as_float(pct_raw),
                            value=_as_int(row.get("Value")),
                        )
                    )
        except Exception:
            pass

        return KeyPersonsRaw(
            ticker=ticker.upper(),
            as_of=date.today(),
            officers=officers,
            institutional_holders=institutional,
            data_age_hours=_fiscal_age_hours(info),
        )
