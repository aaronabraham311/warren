"""Tests for agent.screening — deterministic, data-grounded quantitative screen."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

from agent.screening import (
    DEFAULT_SCREEN_CRITERIA,
    ScreeningResult,
    _fundamentals_checks,
    _growth_checks,
    run_screening_pass,
    screen_ticker,
)
from data_sources.errors import DataSourceError
from data_sources.yfinance_client import FundamentalsData, GrowthData

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
