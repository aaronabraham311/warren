"""Gem-hunt canary — the acceptance test for the deep-value screen + rank (G6/G7).

Unlike the golden-set harness (which grades *analysis* output), this asserts the
*screening* stage does its one job: **the known gems get picked up.** It seeds a
candidate pool of the three hand-verified gems (DIR.MI, CIRSA.MC, KPL.WA) plus
decoys, runs the real gem-hunt screen + rank over a fully offline fake client, and
asserts the gems survive screening AND rank in the nightly top-N cutoff.

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

# The three hand-verified gems and the two decoys — the canary's fixed cast.
GEMS = ["DIR.MI", "CIRSA.MC", "KPL.WA"]
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
    ev_ebit: float,
    price_to_ncav: float,
    ncav_to_mc: float,
    dividend_yield_pct: float,
    net_cash: bool = True,
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
        market_cap_usd=None,
        ncav=None,
        ncav_to_market_cap=ncav_to_mc,
        is_net_net=False,
        price_to_ncav=price_to_ncav,
        net_cash_usd=None,
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


# Per-ticker deep-value profiles.
#   The 3 gems: cheap (low EV/EBIT), price-to-NCAV < 1, net cash, ≥ profit &
#   dividend floors → PASS. Composite score = EV/EBIT − NCAV-to-market-cap
#   (lower = cheaper = better rank):
#     DIR.MI    → 4.0 − 0.60 = 3.40  (best)
#     CIRSA.MC  → 5.0 − 0.40 = 4.60
#     KPL.WA    → 6.0 − 0.30 = 5.70
#   DECOY_PASS clears every gate but scores 9.0 − 0.10 = 8.90 → survives, ranks LAST.
#   DECOY_FAIL has EV/EBIT 25.0 (> max_ev_ebit 10.0) → filtered OUT entirely.
# fields: pb, ev_ebit, price_to_ncav, ncav_to_mc, dividend_yield_pct, profit_years
_PROFILES: dict[str, dict[str, float]] = {
    "DIR.MI": {"pb": 0.70, "ev_ebit": 4.0, "p_ncav": 0.70, "ncav_mc": 0.60, "div": 2.5, "years": 7},
    "CIRSA.MC": {"pb": 0.9, "ev_ebit": 5.0, "p_ncav": 0.8, "ncav_mc": 0.4, "div": 3.0, "years": 6},
    "KPL.WA": {"pb": 1.1, "ev_ebit": 6.0, "p_ncav": 0.85, "ncav_mc": 0.3, "div": 1.5, "years": 8},
    DECOY_PASS: {"pb": 1.4, "ev_ebit": 9.0, "p_ncav": 1.2, "ncav_mc": 0.1, "div": 1.2, "years": 5},
    DECOY_FAIL: {"pb": 1.4, "ev_ebit": 25.0, "p_ncav": 0.9, "ncav_mc": 0.5, "div": 2.0, "years": 6},
}


def _build_client() -> _FakeValueYF:
    fundamentals: dict[str, FundamentalsData | DataSourceError] = {}
    valuation: dict[str, ValuationData | DataSourceError] = {}
    quality: dict[str, QualityData | DataSourceError] = {}
    for ticker, p in _PROFILES.items():
        fundamentals[ticker] = _fundamentals(ticker, pb=p["pb"])
        valuation[ticker] = _valuation(
            ticker,
            ev_ebit=p["ev_ebit"],
            price_to_ncav=p["p_ncav"],
            ncav_to_mc=p["ncav_mc"],
            dividend_yield_pct=p["div"],
        )
        quality[ticker] = _quality(ticker, profit_years=int(p["years"]))
    return _FakeValueYF(fundamentals, valuation, quality)


# A deliberately non-sorted, non-best-first pool (a decoy sorts alphabetically ahead
# of some gems) so ordering must come from the ranking, not from input/alpha order.
_POOL = [DECOY_FAIL, "KPL.WA", DECOY_PASS, "CIRSA.MC", "DIR.MI"]


def _run() -> object:
    return run_screening_pass(
        _POOL,
        criteria=GEM_HUNT_SCREEN_CRITERIA,
        client=_build_client(),  # type: ignore[arg-type]
        screen_fn=screen_ticker_value,
        rank=True,
    )


def test_all_three_gems_survive_screening() -> None:
    """(a) Every known gem clears the deep-value gates and is a surfaced candidate."""
    result = _run()
    for gem in GEMS:
        assert gem in result.candidates, f"gem {gem} was filtered out of the screen"  # type: ignore[attr-defined]


def test_top_n_cutoff_is_exactly_the_three_gems() -> None:
    """(b) With the nightly top-N cutoff (=3), the top-ranked names are exactly the gems."""
    result = _run()
    assert _MAX_SCREEN_CANDIDATES == 3
    top_n = result.candidates[:_MAX_SCREEN_CANDIDATES]  # type: ignore[attr-defined]
    # Best-value-first, and exactly the three gems in cheapest→dearest order.
    assert top_n == ["DIR.MI", "CIRSA.MC", "KPL.WA"], f"top-{_MAX_SCREEN_CANDIDATES} was {top_n}"


def test_failing_decoy_is_filtered_out() -> None:
    """(c) The gate-violating decoy (EV/EBIT ≫ max) never reaches the candidate list."""
    result = _run()
    assert DECOY_FAIL not in result.candidates  # type: ignore[attr-defined]


def test_passing_decoy_survives_but_ranks_below_every_gem() -> None:
    """(d) The passing-but-worse decoy survives screening yet ranks below all gems."""
    result = _run()
    assert DECOY_PASS in result.candidates  # type: ignore[attr-defined]
    scores = result.scores  # type: ignore[attr-defined]
    worst_gem_score = max(scores[g] for g in GEMS)
    assert scores[DECOY_PASS] > worst_gem_score
    # And it lands strictly outside the nightly cutoff.
    assert DECOY_PASS not in result.candidates[:_MAX_SCREEN_CANDIDATES]  # type: ignore[attr-defined]
