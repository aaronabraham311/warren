"""Tests for agent.screening — deterministic, data-grounded quantitative screen."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.screening import (
    DEFAULT_SCREEN_CRITERIA,
    GEM_HUNT_SCREEN_CRITERIA,
    ScreeningResult,
    _closability_checks,
    _deep_value_score,
    _fundamentals_checks,
    _growth_checks,
    run_screening_pass,
    screen_ticker,
    screen_ticker_value,
)
from data_sources.errors import DataSourceError
from data_sources.yfinance_client import (
    FundamentalsData,
    GrowthData,
    QualityData,
    ValuationData,
)
from storage.logger import RunLogger

# ---------------------------------------------------------------------------
# Builders + fake client
# ---------------------------------------------------------------------------

# Default builders produce a name that clears all five criteria comfortably.


def _fundamentals(
    ticker: str = "AAPL",
    pe: float | None = 20.0,
    roe_pct: float | None = 20.0,
    de: float | None = 50.0,
) -> FundamentalsData:
    return FundamentalsData(
        ticker=ticker,
        as_of=date.today(),
        pe_ratio=pe,
        pb_ratio=None,
        roe_pct=roe_pct,
        debt_to_equity=de,
        fcf_ttm_usd=None,
        gross_margin_pct=None,
        operating_margin_pct=None,
        net_margin_pct=None,
        sector=None,
        data_age_hours=1,
        source="yfinance",
    )


def _growth(
    ticker: str = "AAPL",
    peg: float | None = 1.0,
    rev_cagr: float | None = 0.10,
) -> GrowthData:
    return GrowthData(
        ticker=ticker,
        as_of=date.today(),
        revenue_cagr_3y=rev_cagr,
        earnings_cagr_3y=None,
        peg_ratio=peg,
        data_age_hours=1,
    )


class _FakeYF:
    """Stand-in for YFinanceClient that serves canned per-ticker payloads."""

    def __init__(
        self,
        fundamentals: dict[str, FundamentalsData | DataSourceError],
        growth: dict[str, GrowthData | DataSourceError],
    ) -> None:
        self._f = fundamentals
        self._g = growth
        self.growth_calls: list[str] = []

    def get_fundamentals(self, ticker: str) -> FundamentalsData | DataSourceError:
        return self._f[ticker]

    def get_growth_metrics(self, ticker: str) -> GrowthData | DataSourceError:
        self.growth_calls.append(ticker)
        return self._g[ticker]


def _error() -> DataSourceError:
    return DataSourceError(error_code="not_found", message="no data")


# ---------------------------------------------------------------------------
# Criterion checks — units and positivity guards
# ---------------------------------------------------------------------------


def test_fundamentals_checks_all_pass() -> None:
    assert _fundamentals_checks(_fundamentals(), DEFAULT_SCREEN_CRITERIA) == [True, True, True]


def test_pe_boundary_and_negative_guard() -> None:
    c = DEFAULT_SCREEN_CRITERIA
    assert _fundamentals_checks(_fundamentals(pe=30.0), c)[0] is True  # boundary inclusive
    assert _fundamentals_checks(_fundamentals(pe=30.01), c)[0] is False
    assert _fundamentals_checks(_fundamentals(pe=-5.0), c)[0] is False  # loss-maker


def test_roe_percentage_vs_fraction_units() -> None:
    """roe_pct is a percentage (12.0); criteria roe_min is a fraction (0.12)."""
    c = DEFAULT_SCREEN_CRITERIA
    assert _fundamentals_checks(_fundamentals(roe_pct=12.0), c)[1] is True  # exactly 12%
    assert _fundamentals_checks(_fundamentals(roe_pct=11.99), c)[1] is False


def test_debt_to_equity_percentage_units_and_negative_guard() -> None:
    """yfinance D/E is a percentage (100.0 == 1.0x); de_max is a ratio."""
    c = DEFAULT_SCREEN_CRITERIA
    assert _fundamentals_checks(_fundamentals(de=100.0), c)[2] is True  # 1.0x boundary
    assert _fundamentals_checks(_fundamentals(de=100.01), c)[2] is False
    assert _fundamentals_checks(_fundamentals(de=-10.0), c)[2] is False  # negative equity


def test_missing_fundamentals_metrics_are_omitted() -> None:
    checks = _fundamentals_checks(_fundamentals(pe=None, roe_pct=None), DEFAULT_SCREEN_CRITERIA)
    assert checks == [True]  # only D/E present
    assert _fundamentals_checks(None, DEFAULT_SCREEN_CRITERIA) == []


def test_growth_checks_units_and_guards() -> None:
    c = DEFAULT_SCREEN_CRITERIA
    assert _growth_checks(_growth(peg=1.5, rev_cagr=0.05), c) == [True, True]  # boundaries
    assert _growth_checks(_growth(peg=1.6), c)[0] is False
    assert _growth_checks(_growth(peg=-1.0), c)[0] is False  # negative PEG
    assert _growth_checks(_growth(rev_cagr=0.04), c)[1] is False
    assert _growth_checks(None, c) == []


# ---------------------------------------------------------------------------
# screen_ticker — pass logic, missing-data policy, short-circuit
# ---------------------------------------------------------------------------


def test_screen_ticker_all_pass() -> None:
    yf = _FakeYF({"AAPL": _fundamentals()}, {"AAPL": _growth()})
    assert screen_ticker("AAPL", yf, DEFAULT_SCREEN_CRITERIA) is True  # type: ignore[arg-type]


def test_fundamentals_failure_short_circuits_growth_fetch() -> None:
    """A failed fundamentals metric means the ticker can't pass — skip the heavy growth call."""
    yf = _FakeYF({"AAPL": _fundamentals(pe=45.0)}, {"AAPL": _growth()})
    assert screen_ticker("AAPL", yf, DEFAULT_SCREEN_CRITERIA) is False  # type: ignore[arg-type]
    assert yf.growth_calls == []  # growth never fetched


def test_below_min_criteria_present_fails() -> None:
    """Only one metric present (P/E) — not enough evidence to surface the name."""
    yf = _FakeYF(
        {"AAPL": _fundamentals(roe_pct=None, de=None)},
        {"AAPL": _growth(peg=None, rev_cagr=None)},
    )
    assert screen_ticker("AAPL", yf, DEFAULT_SCREEN_CRITERIA) is False  # type: ignore[arg-type]


def test_exactly_three_present_passes() -> None:
    """Three fundamentals present + passing, growth entirely missing → passes."""
    yf = _FakeYF(
        {"AAPL": _fundamentals()},
        {"AAPL": _growth(peg=None, rev_cagr=None)},
    )
    assert screen_ticker("AAPL", yf, DEFAULT_SCREEN_CRITERIA) is True  # type: ignore[arg-type]


def test_datasource_error_treated_as_missing() -> None:
    """A fundamentals fetch error is missing data (no failure) → growth still fetched, but
    two present criteria is below the minimum, so the ticker does not pass."""
    yf = _FakeYF({"AAPL": _error()}, {"AAPL": _growth()})
    assert screen_ticker("AAPL", yf, DEFAULT_SCREEN_CRITERIA) is False  # type: ignore[arg-type]
    assert yf.growth_calls == ["AAPL"]  # no fundamentals failure → growth fetched


# ---------------------------------------------------------------------------
# run_screening_pass
# ---------------------------------------------------------------------------


def test_run_screening_candidates_and_pass_rate() -> None:
    yf = _FakeYF(
        {"A": _fundamentals(), "B": _fundamentals(), "C": _fundamentals(pe=99.0)},
        {"A": _growth(), "B": _growth(), "C": _growth()},
    )
    result = run_screening_pass(["A", "B", "C"], client=yf)  # type: ignore[arg-type]
    assert result.candidates == ["A", "B"]
    assert abs(result.pass_rate - 2 / 3) < 1e-9


def test_run_screening_empty_universe() -> None:
    yf = _FakeYF({}, {})
    result = run_screening_pass([], client=yf)  # type: ignore[arg-type]
    assert result.candidates == []
    assert result.pass_rate == 0.0
    assert isinstance(result, ScreeningResult)


def test_run_screening_uses_singleton_when_no_client() -> None:
    yf = _FakeYF({"AAPL": _fundamentals()}, {"AAPL": _growth()})
    with patch("agent.screening.yfinance_client", return_value=yf):
        result = run_screening_pass(["AAPL"])
    assert result.candidates == ["AAPL"]


def test_phase_events_emitted() -> None:
    yf = _FakeYF({"AAPL": _fundamentals()}, {"AAPL": _growth()})
    logged: list[tuple[str, dict[str, object]]] = []
    mock_logger = MagicMock()
    mock_logger.log.side_effect = lambda event, **kw: logged.append((event, kw))

    run_screening_pass(["AAPL"], logger=mock_logger, client=yf)  # type: ignore[arg-type]

    events = [e for e, _ in logged]
    assert "phase_started" in events
    assert "phase_completed" in events

    started_kw = next(kw for e, kw in logged if e == "phase_started")
    assert started_kw["phase"] == "screening"
    assert started_kw["universe_size"] == 1
    assert started_kw["method"] == "quantitative"

    completed_kw = next(kw for e, kw in logged if e == "phase_completed")
    assert completed_kw["candidates_surfaced"] == ["AAPL"]
    assert "pass_rate" in completed_kw


def test_no_logger_does_not_crash() -> None:
    yf = _FakeYF({"AAPL": _fundamentals(pe=99.0)}, {"AAPL": _growth()})
    result = run_screening_pass(["AAPL"], logger=None, client=yf)  # type: ignore[arg-type]
    assert isinstance(result, ScreeningResult)


def test_screening_accepts_pre_filtered_universe() -> None:
    """Screening must not touch the DB or cooldown state — a filtered list is all it needs."""
    yf = _FakeYF({"MSFT": _fundamentals()}, {"MSFT": _growth()})
    result = run_screening_pass(["MSFT"], client=yf)  # type: ignore[arg-type]
    assert result.candidates == ["MSFT"]


def test_concurrent_screening_matches_sequential() -> None:
    """max_workers>1 must return the SAME candidates in the SAME order as sequential."""
    # A mixed universe (some pass, some fail) in a deliberately non-sorted order.
    universe = ["D", "A", "C", "B", "E"]
    passing = {"A", "C", "E"}
    yf = _FakeYF(
        {t: _fundamentals(pe=20.0 if t in passing else 99.0) for t in universe},
        {t: _growth() for t in universe},
    )

    seq = run_screening_pass(universe, client=yf)  # type: ignore[arg-type]
    conc = run_screening_pass(universe, client=yf, max_workers=4)  # type: ignore[arg-type]

    assert seq.candidates == conc.candidates == ["A", "C", "E"]  # order preserved
    assert seq.pass_rate == conc.pass_rate


# ---------------------------------------------------------------------------
# Deep-value ("gem-hunt") screen — criteria, dividend floor, ranking
# ---------------------------------------------------------------------------


def _value_fundamentals(
    ticker: str = "GEM",
    pb: float | None = 0.8,
    *,
    float_shares: int | None = None,
    avg_volume_3m: int | None = None,
    current_price: float | None = None,
    trading_currency: str | None = None,
) -> FundamentalsData:
    return FundamentalsData(
        ticker=ticker,
        as_of=date.today(),
        pe_ratio=None,
        pb_ratio=pb,
        roe_pct=None,
        debt_to_equity=None,
        fcf_ttm_usd=None,
        gross_margin_pct=None,
        operating_margin_pct=None,
        net_margin_pct=None,
        sector=None,
        float_shares=float_shares,
        avg_volume_3m=avg_volume_3m,
        current_price=current_price,
        trading_currency=trading_currency,
        data_age_hours=1,
        source="yfinance",
    )


def _valuation(
    ticker: str = "GEM",
    ev_ebit: float | None = 6.0,
    price_to_ncav: float | None = 0.9,
    net_cash: bool = True,
    dividend_yield_pct: float | None = 3.0,
    ncav_to_mc: float | None = 0.5,
    market_cap_usd: int | None = 50_000_000,
    market_cap_native: int | None = None,
    currency: str | None = "USD",
) -> ValuationData:
    return ValuationData(
        ticker=ticker,
        as_of=date.today(),
        enterprise_value=None,
        ev_to_ebit=ev_ebit,
        ev_to_ebitda=None,
        acquirers_multiple=None,
        fcf_yield=None,
        earnings_yield=None,
        market_cap_usd=market_cap_usd,
        market_cap_native=market_cap_native,
        currency=currency,
        ncav=None,
        ncav_to_market_cap=ncav_to_mc,
        is_net_net=False,
        price_to_ncav=price_to_ncav,
        net_cash_usd=1 if net_cash else -1,
        net_cash_positive=net_cash,
        p_tangible_book=None,
        dividend_yield_pct=dividend_yield_pct,
        data_age_hours=1,
    )


def _quality(ticker: str = "GEM", profit_years: int | None = 8) -> QualityData:
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
        consecutive_profit_years=profit_years,
        ncav_trend=None,
        data_age_hours=1,
    )


class _FakeValueYF:
    """Stand-in for YFinanceClient serving canned fundamentals/valuation/quality."""

    def __init__(
        self,
        fundamentals: Mapping[str, FundamentalsData | DataSourceError],
        valuation: Mapping[str, ValuationData | DataSourceError],
        quality: Mapping[str, QualityData | DataSourceError],
    ) -> None:
        self._f = fundamentals
        self._v = valuation
        self._q = quality
        self.valuation_calls: list[str] = []
        self.quality_calls: list[str] = []

    def get_fundamentals(self, ticker: str) -> FundamentalsData | DataSourceError:
        return self._f[ticker]

    def get_valuation_multiples(self, ticker: str) -> ValuationData | DataSourceError:
        self.valuation_calls.append(ticker)
        return self._v[ticker]

    def get_quality_metrics(self, ticker: str) -> QualityData | DataSourceError:
        self.quality_calls.append(ticker)
        return self._q[ticker]


def _value_yf(**overrides: object) -> _FakeValueYF:
    """A single 'GEM' name that clears every deep-value gate, with per-model overrides."""
    return _FakeValueYF(
        {"GEM": _value_fundamentals()},
        {"GEM": _valuation(**overrides)},  # type: ignore[arg-type]
        {"GEM": _quality()},
    )


def test_gem_hunt_criteria_are_distinct_from_garp() -> None:
    """The deep-value set uses value keys, not the GARP pe/peg/roe/de/rev_growth gates."""
    assert set(GEM_HUNT_SCREEN_CRITERIA) == {
        "pb_ratio_max",
        "max_ev_ebit",
        "max_price_to_ncav",
        "min_market_cap_usd",
        "max_market_cap_usd",
        "min_dividend_yield_pct",
    }
    assert set(GEM_HUNT_SCREEN_CRITERIA).isdisjoint(DEFAULT_SCREEN_CRITERIA)


def test_value_screen_passes_a_clean_gem() -> None:
    yf = _value_yf()
    out = screen_ticker_value("GEM", yf, GEM_HUNT_SCREEN_CRITERIA)  # type: ignore[arg-type]
    assert out.passed is True
    assert out.score is not None and 0.0 <= out.score <= 1.0


def test_value_screen_filters_high_ev_ebit() -> None:
    """A present-but-failing value gate (EV/EBIT too high) filters the name OUT."""
    yf = _value_yf(ev_ebit=25.0)  # > max_ev_ebit (10.0)
    out = screen_ticker_value("GEM", yf, GEM_HUNT_SCREEN_CRITERIA)  # type: ignore[arg-type]
    assert out.passed is False


def test_value_screen_dividend_floor_filters_zero_dividend() -> None:
    """A zero-dividend name is filtered out when the floor is > 0."""
    yf = _value_yf(dividend_yield_pct=0.0)
    out = screen_ticker_value("GEM", yf, GEM_HUNT_SCREEN_CRITERIA)  # type: ignore[arg-type]
    assert out.passed is False


def test_dividend_yield_pct_scale_is_percent() -> None:
    """ValuationData.dividend_yield_pct is in PERCENT: 3.0 clears a 1.0(%) floor; 0.5 fails."""
    assert GEM_HUNT_SCREEN_CRITERIA["min_dividend_yield_pct"] == 1.0
    above = _value_yf(dividend_yield_pct=3.0)
    below = _value_yf(dividend_yield_pct=0.5)
    assert screen_ticker_value("GEM", above, GEM_HUNT_SCREEN_CRITERIA).passed is True  # type: ignore[arg-type]
    assert screen_ticker_value("GEM", below, GEM_HUNT_SCREEN_CRITERIA).passed is False  # type: ignore[arg-type]


def test_value_screen_net_debt_survives_but_scores_worse() -> None:
    yf = _value_yf(net_cash=False)
    out = screen_ticker_value("GEM", yf, GEM_HUNT_SCREEN_CRITERIA)  # type: ignore[arg-type]
    cash = screen_ticker_value("GEM", _value_yf(), GEM_HUNT_SCREEN_CRITERIA)  # type: ignore[arg-type]
    assert out.passed is True
    assert out.score is not None and cash.score is not None and out.score > cash.score


def test_closability_derives_turnover_float_and_position_cap_without_price_fetch() -> None:
    signals = _closability_checks(
        _value_fundamentals(
            float_shares=1_000_000,
            avg_volume_3m=1_000,
            current_price=5.0,
            trading_currency="EUR",
        ),
        _valuation(
            market_cap_native=50_000_000,
            market_cap_usd=55_000_000,
            currency="EUR",
        ),
    )

    assert signals.daily_turnover_usd == 5_500.0
    assert signals.free_float_pct == 10.0
    assert signals.position_size_cap_usd == 11_000.0
    assert 0.0 <= signals.loss <= 1.0


def test_missing_or_misaligned_trading_currency_keeps_turnover_unknown() -> None:
    missing = _closability_checks(
        _value_fundamentals(avg_volume_3m=1_000, current_price=5.0),
        _valuation(market_cap_native=50_000_000, market_cap_usd=55_000_000, currency=None),
    )
    mismatch = _closability_checks(
        _value_fundamentals(avg_volume_3m=1_000, current_price=5.0, trading_currency="PLN"),
        _valuation(market_cap_native=50_000_000, market_cap_usd=13_000_000, currency="EUR"),
    )
    assert missing.daily_turnover_usd is None
    assert mismatch.daily_turnover_usd is None
    assert "turnover remains unknown" in " ".join(missing.data_quality)


def test_illiquidity_annotates_and_ranks_but_never_excludes() -> None:
    fundamentals = _value_fundamentals(
        float_shares=100_000,
        avg_volume_3m=100,
        current_price=10.0,
        trading_currency="USD",
    )
    yf = _FakeValueYF(
        {"GEM": fundamentals},
        {"GEM": _valuation(market_cap_native=50_000_000)},
        {"GEM": _quality()},
    )

    outcome = screen_ticker_value("GEM", yf, GEM_HUNT_SCREEN_CRITERIA)  # type: ignore[arg-type]

    assert outcome.disposition == "candidate"
    assert outcome.closability is not None
    assert outcome.closability.daily_turnover_usd == 1_000.0
    assert outcome.closability.position_size_cap_usd == 2_000.0


def test_low_float_frou_frou_shape_ranks_below_dispersed_register() -> None:
    tickers = ["FROU", "DISPERSED"]
    fundamentals = {
        "FROU": _value_fundamentals(
            ticker="FROU",
            float_shares=1_113_000,
            avg_volume_3m=10_000,
            current_price=10.0,
            trading_currency="USD",
        ),
        "DISPERSED": _value_fundamentals(
            ticker="DISPERSED",
            float_shares=7_000_000,
            avg_volume_3m=10_000,
            current_price=10.0,
            trading_currency="USD",
        ),
    }
    valuations = {
        ticker: _valuation(
            ticker=ticker,
            market_cap_native=100_000_000,
            market_cap_usd=100_000_000,
        )
        for ticker in tickers
    }
    quality = {ticker: _quality(ticker=ticker) for ticker in tickers}

    result = run_screening_pass(
        tickers,
        criteria=GEM_HUNT_SCREEN_CRITERIA,
        client=_FakeValueYF(fundamentals, valuations, quality),  # type: ignore[arg-type]
        screen_fn=screen_ticker_value,
        rank=True,
    )

    assert result.candidates == ["DISPERSED", "FROU"]
    assert result.scores["DISPERSED"] < result.scores["FROU"]
    assert result.closability["FROU"].free_float_pct == 11.13
    assert result.closability["DISPERSED"].free_float_pct == 70.0


def test_value_screen_short_circuits_on_fundamentals_failure() -> None:
    """A failing P/B skips the valuation + quality fetches."""
    yf = _FakeValueYF(
        {"GEM": _value_fundamentals(pb=5.0)},  # > pb_ratio_max
        {"GEM": _valuation()},
        {"GEM": _quality()},
    )
    out = screen_ticker_value("GEM", yf, GEM_HUNT_SCREEN_CRITERIA)  # type: ignore[arg-type]
    assert out.passed is False
    assert yf.valuation_calls == []
    assert yf.quality_calls == []


def test_deep_value_score_missing_ev_is_bounded() -> None:
    assert 0.0 <= _deep_value_score(_valuation(ev_ebit=None)) <= 1.0
    assert 0.0 <= _deep_value_score(None) <= 1.0


def test_sparse_value_screen_routes_to_deeper_fetch() -> None:
    yf = _FakeValueYF(
        {"GEM": _value_fundamentals(pb=0.8)},
        {"GEM": _valuation(ev_ebit=None, price_to_ncav=None, dividend_yield_pct=None)},
        {"GEM": _quality(profit_years=None)},
    )
    out = screen_ticker_value("GEM", yf, GEM_HUNT_SCREEN_CRITERIA)  # type: ignore[arg-type]
    assert out.disposition == "needs_deeper_fetch"
    assert out.present_metrics == 2  # P/B and market cap; two cap bounds count once


def test_market_cap_hard_band_and_sweet_spot_score() -> None:
    for cap in (2_000_000, 2_000_000_000):
        out = screen_ticker_value(
            "GEM",
            _value_yf(market_cap_usd=cap),  # type: ignore[arg-type]
            GEM_HUNT_SCREEN_CRITERIA,
        )
        assert out.disposition == "rejected"
    fifty = screen_ticker_value(
        "GEM",
        _value_yf(market_cap_usd=50_000_000),  # type: ignore[arg-type]
        GEM_HUNT_SCREEN_CRITERIA,
    )
    four_hundred = screen_ticker_value(
        "GEM",
        _value_yf(market_cap_usd=400_000_000),  # type: ignore[arg-type]
        GEM_HUNT_SCREEN_CRITERIA,
    )
    assert fifty.score is not None and four_hundred.score is not None
    assert fifty.score < four_hundred.score


def test_value_screen_preserves_source_errors() -> None:
    err = DataSourceError(error_code="network", message="temporary")
    yf = _FakeValueYF({"GEM": err}, {"GEM": _valuation()}, {"GEM": _quality()})
    out = screen_ticker_value("GEM", yf, GEM_HUNT_SCREEN_CRITERIA)  # type: ignore[arg-type]
    assert out.disposition == "source_error"
    assert out.error is not None and out.error.retryable is True


def test_value_screen_marks_rate_limits_retryable() -> None:
    err = DataSourceError(error_code="rate_limit", message="slow down")
    yf = _FakeValueYF({"GEM": err}, {"GEM": _valuation()}, {"GEM": _quality()})

    out = screen_ticker_value("GEM", yf, GEM_HUNT_SCREEN_CRITERIA)  # type: ignore[arg-type]

    assert out.disposition == "source_error"
    assert out.error is not None and out.error.retryable is True


def test_sparse_outcome_is_returned_and_written_to_jsonl(tmp_path: Path) -> None:
    yf = _FakeValueYF(
        {"GEM": _value_fundamentals(pb=0.8)},
        {"GEM": _valuation(ev_ebit=None, price_to_ncav=None, dividend_yield_pct=None)},
        {"GEM": _quality(profit_years=None)},
    )
    logger = RunLogger("sparse", tmp_path)
    result = run_screening_pass(
        ["GEM"],
        criteria=GEM_HUNT_SCREEN_CRITERIA,
        client=yf,  # type: ignore[arg-type]
        screen_fn=screen_ticker_value,
        rank=True,
        logger=logger,
    )
    logger.close()
    assert result.needs_deeper_fetch == ["GEM"]
    assert result.surfaced == ["GEM"]
    events = [json.loads(line) for line in (tmp_path / "sparse.jsonl").read_text().splitlines()]
    event = next(e for e in events if e["event"] == "screening_ticker_outcome")
    assert event["ticker"] == "GEM"
    assert event["disposition"] == "needs_deeper_fetch"
    closability_event = next(e for e in events if e["event"] == "screening_closability")
    assert closability_event["ticker"] == "GEM"
    assert closability_event["daily_turnover_usd"] is None
    assert closability_event["closability_loss"] == pytest.approx(0.5)
    assert closability_event["data_quality"]


def test_run_screening_ranks_best_value_first() -> None:
    """rank=True sorts passing candidates by the bounded composite value loss."""
    universe = ["Y", "X", "Z"]  # deliberately non-sorted, non-best-first
    fundamentals = {t: _value_fundamentals(ticker=t) for t in universe}
    valuation = {
        "X": _valuation(ticker="X", ev_ebit=6.0, price_to_ncav=0.3),
        "Y": _valuation(ticker="Y", ev_ebit=6.0, price_to_ncav=1.4),
        "Z": _valuation(ticker="Z", ev_ebit=6.0, price_to_ncav=0.8),
    }
    quality = {t: _quality(ticker=t) for t in universe}
    yf = _FakeValueYF(fundamentals, valuation, quality)

    result = run_screening_pass(
        universe,
        criteria=GEM_HUNT_SCREEN_CRITERIA,
        client=yf,  # type: ignore[arg-type]
        screen_fn=screen_ticker_value,
        rank=True,
    )
    assert result.candidates == ["X", "Z", "Y"]  # best-value-first, not alphabetical
    assert result.candidates[:2] == ["X", "Z"]  # top-N takes the best, not "Y" (alpha-first)
    assert result.scores["X"] < result.scores["Z"] < result.scores["Y"]


def test_rank_false_preserves_universe_order() -> None:
    """Default (rank=False) keeps universe order even on the value path."""
    universe = ["Y", "X", "Z"]
    yf = _FakeValueYF(
        {t: _value_fundamentals(ticker=t) for t in universe},
        {t: _valuation(ticker=t, ev_ebit=float(i + 5)) for i, t in enumerate(universe)},
        {t: _quality(ticker=t) for t in universe},
    )
    result = run_screening_pass(
        universe,
        criteria=GEM_HUNT_SCREEN_CRITERIA,
        client=yf,  # type: ignore[arg-type]
        screen_fn=screen_ticker_value,
    )
    assert result.candidates == ["Y", "X", "Z"]  # universe order, unranked
