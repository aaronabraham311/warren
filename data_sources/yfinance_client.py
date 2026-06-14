import sqlite3
import statistics
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
    ncav: int | None
    ncav_to_market_cap: float | None
    is_net_net: bool
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
    data_age_hours: int
    source: Literal["yfinance"] = "yfinance"


class OwnershipData(BaseModel):
    ticker: str
    as_of: date
    insider_pct: float | None
    institutional_pct: float | None
    source: Literal["yfinance"] = "yfinance"


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
            ncav=ncav,
            ncav_to_market_cap=ncav_to_mkt_cap,
            is_net_net=is_net_net,
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

    def _fetch_quality_metrics(self, ticker: str) -> QualityData:
        t = yf.Ticker(ticker)
        info: dict[str, object] = t.info
        if not isinstance(info, dict) or len(info) <= 5:
            raise _NotFoundError(ticker)
        if info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            raise _NotFoundError(ticker)

        roic_series = self._compute_roic_series(t)
        roic_vals = [v for v in roic_series if v is not None]
        roic_mean = round(sum(roic_vals) / len(roic_vals), 4) if roic_vals else None

        gm_series = self._compute_gross_margin_series(t)
        gm_vals = [v for v in gm_series if v is not None]
        gm_stdev = round(statistics.stdev(gm_vals), 4) if len(gm_vals) >= 2 else None

        cc_series = self._compute_cash_conversion_series(t)
        roa_pct = self._compute_roa(t)

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
            data_age_hours=_fiscal_age_hours(info),
        )

    def _fetch_financial_strength(self, ticker: str) -> FinancialStrengthData:
        t = yf.Ticker(ticker)
        info: dict[str, object] = t.info
        if not isinstance(info, dict) or len(info) <= 5:
            raise _NotFoundError(ticker)
        if info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            raise _NotFoundError(ticker)

        net_incomes = self._get_fin_series(t, "financials", "Net Income")
        op_cfs = self._get_fin_series(t, "cashflow", "Operating Cash Flow")
        total_assets = self._get_fin_series(t, "balance_sheet", "Total Assets")
        current_assets = self._get_fin_series(t, "balance_sheet", "Current Assets")
        current_liabs = self._get_fin_series(t, "balance_sheet", "Current Liabilities")
        lt_debt = self._get_fin_series(t, "balance_sheet", "Long Term Debt")
        gross_profits = self._get_fin_series(t, "financials", "Gross Profit")
        revenues = self._get_fin_series(t, "financials", "Total Revenue")
        retained_earnings = self._get_fin_series(t, "balance_sheet", "Retained Earnings")
        total_liabs = self._get_fin_series(
            t, "balance_sheet", "Total Liabilities Net Minority Interest"
        )
        ebit_series = self._get_fin_series(t, "financials", "EBIT")
        interest_exp_series = self._get_fin_series(t, "financials", "Interest Expense")
        total_debt = self._get_fin_series(t, "balance_sheet", "Total Debt")
        cash_series = self._get_fin_series(t, "balance_sheet", "Cash And Cash Equivalents")
        ebitda_series = self._get_fin_series(t, "financials", "EBITDA")
        shares_series = self._get_fin_series(t, "balance_sheet", "Common Stock")

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

    def _get_fin_series(self, ticker_obj: object, attr: str, metric: str) -> list[float]:
        try:
            df = getattr(ticker_obj, attr)
            if metric not in df.index:
                return []
            series = df.loc[metric].dropna()
            return [float(v) for v in series.values]
        except Exception:
            return []

    def _compute_roic_series(self, ticker_obj: object) -> list[float | None]:
        op_incomes = self._get_fin_series(ticker_obj, "financials", "Operating Income")
        tax_provisions = self._get_fin_series(ticker_obj, "financials", "Tax Provision")
        pretax_incomes = self._get_fin_series(ticker_obj, "financials", "Pretax Income")
        total_assets = self._get_fin_series(ticker_obj, "balance_sheet", "Total Assets")
        current_liabs = self._get_fin_series(ticker_obj, "balance_sheet", "Current Liabilities")

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

    def _compute_gross_margin_series(self, ticker_obj: object) -> list[float | None]:
        gross_profits = self._get_fin_series(ticker_obj, "financials", "Gross Profit")
        revenues = self._get_fin_series(ticker_obj, "financials", "Total Revenue")
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

    def _compute_cash_conversion_series(self, ticker_obj: object) -> list[float | None]:
        fcf_series = self._get_fin_series(ticker_obj, "cashflow", "Free Cash Flow")
        if not fcf_series:
            op_cfs = self._get_fin_series(ticker_obj, "cashflow", "Operating Cash Flow")
            capex = self._get_fin_series(ticker_obj, "cashflow", "Capital Expenditure")
            n_cf = min(len(op_cfs), len(capex))
            fcf_series = [op_cfs[i] + capex[i] for i in range(n_cf)]
        net_incomes = self._get_fin_series(ticker_obj, "financials", "Net Income")
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

    def _compute_roa(self, ticker_obj: object) -> float | None:
        net_incomes = self._get_fin_series(ticker_obj, "financials", "Net Income")
        total_assets = self._get_fin_series(ticker_obj, "balance_sheet", "Total Assets")
        if not net_incomes or not total_assets or total_assets[0] == 0:
            return None
        return round(net_incomes[0] / total_assets[0] * 100.0, 4)

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
