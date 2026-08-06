"""Gem-hunt canary — the acceptance test for the deep-value screen + rank (G6/G7).

Unlike the golden-set harness (which grades *analysis* output), this asserts the
*screening* stage does its one job: preserve a current deep-value reference, route
sparse/leveraged names correctly, and reject names whose current facts no longer fit.
The dated profiles below were checked live on 2026-08-05, then frozen for offline replay.

It guards against any regression to ``GEM_HUNT_SCREEN_CRITERIA``,
``screen_ticker_value``, ``_deep_value_score`` or the ranking in
``run_screening_pass`` that would drop a gem out of the surfaced candidates or push
it below the ``_MAX_SCREEN_CANDIDATES`` cutoff. If a gem is filtered out or ranks
below a decoy, this test fails loudly — that is the whole point.

Deterministic and offline: the autouse ``_no_live_network`` guard blocks real calls;
all per-ticker data is synthetic, served by ``_FakeValueYF``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from agent.run import _MAX_SCREEN_CANDIDATES
from agent.screening import (
    GEM_HUNT_SCREEN_CRITERIA,
    run_screening_pass,
    screen_ticker_value,
)
from data_sources.errors import DataSourceError
from data_sources.yfinance_client import (
    FundamentalsData,
    QualityData,
    ValuationData,
)

# Current references and deterministic policy probes — the canary's fixed cast.
CURRENT_GEM = "KPL.WA"
SPARSE_PROBE = "SPARSEPROBE.MI"
LEVERAGED_PROBE = "LEVERAGEDPROBE.MC"
STALE_TICKER = "DIR.MI"
LARGE_CAP_REFERENCE = "CIRSA.MC"
DECOY_PASS = "DECOYPASS.MI"  # clears the gates, but a worse composite score
DECOY_FAIL = "DECOYFAIL.MC"  # violates a value gate (EV/EBIT far above the max)


# ---------------------------------------------------------------------------
# Synthetic per-ticker builders (fake client — no network)
# ---------------------------------------------------------------------------


def _fundamentals(ticker: str, pb: float) -> FundamentalsData:
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
        data_age_hours=1,
        source="yfinance",
    )


def _valuation(
    ticker: str,
    ev_ebit: float | None,
    price_to_ncav: float | None,
    ncav_to_mc: float | None,
    dividend_yield_pct: float | None,
    net_cash: bool = True,
    market_cap_usd: int = 50_000_000,
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


def _quality(ticker: str, profit_years: int) -> QualityData:
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

    def get_fundamentals(self, ticker: str) -> FundamentalsData | DataSourceError:
        return self._f[ticker]

    def get_valuation_multiples(self, ticker: str) -> ValuationData | DataSourceError:
        return self._v[ticker]

    def get_quality_metrics(self, ticker: str) -> QualityData | DataSourceError:
        return self._q[ticker]


# Live KPL/CIRSA observations are rounded only to stable fixture precision. The
# leveraged/sparse probes isolate the two policy regressions named by G13.
_PROFILES: dict[str, dict[str, float | None]] = {
    CURRENT_GEM: {
        "pb": 1.0215,
        "ev_ebit": 3.71,
        "p_ncav": None,
        "ncav_mc": None,
        "div": 6.05,
        "years": 4,
        "mc": 104_842_767,
        "cash": 1,
    },
    LEVERAGED_PROBE: {
        "pb": 0.9,
        "ev_ebit": 5.0,
        "p_ncav": 0.8,
        "ncav_mc": 0.4,
        "div": 3.0,
        "years": 2,
        "mc": 120_000_000,
        "cash": 0,
    },
    SPARSE_PROBE: {
        "pb": 0.5,
        "ev_ebit": None,
        "p_ncav": None,
        "ncav_mc": None,
        "div": None,
        "years": 0,
        "mc": 50_000_000,
        "cash": 1,
    },
    LARGE_CAP_REFERENCE: {
        "pb": 3.5798,
        "ev_ebit": 10.23,
        "p_ncav": None,
        "ncav_mc": None,
        "div": 3.33,
        "years": 4,
        "mc": 2_656_999_155,
        "cash": 0,
    },
    DECOY_PASS: {
        "pb": 1.4,
        "ev_ebit": 9.0,
        "p_ncav": 1.2,
        "ncav_mc": 0.1,
        "div": 1.2,
        "years": 1,
        "mc": 400_000_000,
        "cash": 0,
    },
    DECOY_FAIL: {
        "pb": 1.4,
        "ev_ebit": 25.0,
        "p_ncav": 0.9,
        "ncav_mc": 0.5,
        "div": 2.0,
        "years": 2,
        "mc": 80_000_000,
        "cash": 1,
    },
}


def _build_client() -> _FakeValueYF:
    fundamentals: dict[str, FundamentalsData | DataSourceError] = {}
    valuation: dict[str, ValuationData | DataSourceError] = {}
    quality: dict[str, QualityData | DataSourceError] = {}
    for ticker, p in _PROFILES.items():
        fundamentals[ticker] = _fundamentals(ticker, pb=float(p["pb"] or 0.0))
        valuation[ticker] = _valuation(
            ticker,
            ev_ebit=p["ev_ebit"],
            price_to_ncav=p["p_ncav"],
            ncav_to_mc=p["ncav_mc"],
            dividend_yield_pct=p["div"],
            market_cap_usd=int(p["mc"] or 0),
            net_cash=bool(p["cash"]),
        )
        quality[ticker] = _quality(ticker, profit_years=int(p["years"] or 0))
    missing = DataSourceError(error_code="not_found", message="No data for DIR.MI")
    fundamentals[STALE_TICKER] = missing
    valuation[STALE_TICKER] = missing
    quality[STALE_TICKER] = missing
    return _FakeValueYF(fundamentals, valuation, quality)


# A deliberately non-sorted, non-best-first pool (a decoy sorts alphabetically ahead
# of some gems) so ordering must come from the ranking, not from input/alpha order.
_POOL = [
    DECOY_FAIL,
    CURRENT_GEM,
    DECOY_PASS,
    LARGE_CAP_REFERENCE,
    STALE_TICKER,
    SPARSE_PROBE,
    LEVERAGED_PROBE,
]


def _run() -> object:
    return run_screening_pass(
        _POOL,
        criteria=GEM_HUNT_SCREEN_CRITERIA,
        client=_build_client(),  # type: ignore[arg-type]
        screen_fn=screen_ticker_value,
        rank=True,
    )


def test_current_reference_and_policy_probes_survive() -> None:
    result = _run()
    assert CURRENT_GEM in result.candidates  # type: ignore[attr-defined]
    assert LEVERAGED_PROBE in result.candidates  # type: ignore[attr-defined]
    assert SPARSE_PROBE in result.needs_deeper_fetch  # type: ignore[attr-defined]


def test_runtime_surfaced_top_n_includes_sparse_probe() -> None:
    result = _run()
    assert _MAX_SCREEN_CANDIDATES == 3
    top_n = result.surfaced[:_MAX_SCREEN_CANDIDATES]  # type: ignore[attr-defined]
    assert top_n == [CURRENT_GEM, SPARSE_PROBE, LEVERAGED_PROBE]


def test_stale_and_now_large_cap_references_are_not_silently_treated_as_gems() -> None:
    result = _run()
    assert STALE_TICKER in result.source_errors  # type: ignore[attr-defined]
    assert LARGE_CAP_REFERENCE not in result.surfaced  # type: ignore[attr-defined]


def test_failing_decoy_is_filtered_out() -> None:
    """(c) The gate-violating decoy (EV/EBIT ≫ max) never reaches the candidate list."""
    result = _run()
    assert DECOY_FAIL not in result.candidates  # type: ignore[attr-defined]


def test_passing_decoy_survives_but_ranks_below_top_three() -> None:
    """(d) The passing-but-worse decoy survives screening yet ranks below all gems."""
    result = _run()
    assert DECOY_PASS in result.candidates  # type: ignore[attr-defined]
    # And it lands strictly outside the nightly cutoff.
    assert DECOY_PASS not in result.surfaced[:_MAX_SCREEN_CANDIDATES]  # type: ignore[attr-defined]
