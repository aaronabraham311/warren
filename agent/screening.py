"""Deterministic quantitative screening pass — a data-grounded filter over the universe.

Fetches real fundamentals and growth metrics for each ticker and applies five
numeric thresholds in code (no LLM). This replaces the earlier Haiku PASS/FAIL
screen, which was asked to judge quantitative criteria *without being given any
numbers* and consequently failed essentially everything (0 candidates / 506).

Grounding the screen in fetched data also removes the per-run Haiku batch cost,
and makes the pass a pure, reproducible function of the source data.

Missing-data policy: only criteria with available data are evaluated. A ticker
passes when nothing present is violated **and** at least ``MIN_CRITERIA_PRESENT``
of the five criteria had data — strict enough not to surface a name on no
evidence, lenient enough to tolerate yfinance's frequent gaps (e.g. a missing
PEG ratio) without wiping out the candidate pool.

Cooldown suppression is the caller's responsibility: filter the universe with
``agent.cooldown.filter_universe_for_cooldown`` *before* passing it here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from agent.tools._clients import yfinance_client
from data_sources.yfinance_client import (
    FundamentalsData,
    GrowthData,
    QualityData,
    ValuationData,
    YFinanceClient,
)
from storage.logger import RunLogger

# Criteria values span floats (thresholds), ints (year counts), and bools (net-cash
# gate). The GARP screen only uses floats; the gem-hunt deep-value screen needs the
# wider set, so the shared plumbing accepts the union. Mapping is used (not dict) so the
# GARP callers' ``dict[str, float]`` stays assignable (Mapping is covariant in its value).
CriteriaValue = float | int | bool
Criteria = Mapping[str, CriteriaValue]

# At least this many criteria must have data for a ticker to be eligible — otherwise a
# name with one lucky metric could sneak through. Shared by the GARP and deep-value passes.
MIN_CRITERIA_PRESENT = 3

# Hard ceiling on how many tickers a single screening pass will fetch. The gem-hunt
# global universe is low-thousands of names; this bounds per-run cost/latency and is a
# no-op for the ~500-name S&P 500 universe. The universe is truncated deterministically
# (it arrives sorted), so the cap never introduces run-to-run nondeterminism.
MAX_UNIVERSE_SIZE = 4000

DEFAULT_SCREEN_CRITERIA: dict[str, float] = {
    "pe_max": 30.0,
    "peg_max": 1.5,
    "roe_min": 0.12,  # fraction: 0.12 == 12%
    "de_max": 1.0,  # ratio: 1.0 == 100% debt-to-equity
    "rev_growth_min": 0.05,  # fraction: 0.05 == 5% 3Y revenue CAGR
}

# Deep-value ("gem-hunt") criteria — distinct from GARP: cheap-quality-on-assets, not
# growth-at-a-reasonable-price. Keys mirror ``agent/tools/screen.py`` (pb_ratio_max,
# max_ev_ebit, max_price_to_ncav, require_net_cash, min_consecutive_profit_years) so the
# nightly screen and the ``screen_universe`` tool can't drift on naming/semantics, PLUS a
# dividend-yield floor. Thresholds live here (the tool takes them from the caller/LLM).
#   pb_ratio_max / max_ev_ebit / max_price_to_ncav : metric ≤ threshold (cheaper = better)
#   require_net_cash            : truthy → balance sheet must be net-cash
#   min_consecutive_profit_years: metric ≥ threshold (quality-of-earnings)
#   min_dividend_yield_pct      : metric ≥ threshold, in PERCENT (see _value_valuation_checks)
GEM_HUNT_SCREEN_CRITERIA: dict[str, CriteriaValue] = {
    "pb_ratio_max": 1.5,
    "max_ev_ebit": 10.0,
    "max_price_to_ncav": 1.5,
    "require_net_cash": True,
    "min_consecutive_profit_years": 5,
    "min_dividend_yield_pct": 1.0,  # percent: 1.0 == 1% (ValuationData.dividend_yield_pct scale)
}

# When a candidate is missing EV/EBIT it can still pass on other gates, but must not
# out-rank names with a known-cheap multiple; a large sentinel sinks it to the bottom.
_MISSING_EV_PENALTY = 1e9


@dataclass
class ScreeningResult:
    candidates: list[str]
    pass_rate: float
    # Deep-value score per candidate (lower = cheaper); empty for the GARP path, which
    # does not score. Populated whenever ranking is requested. G10's canary consumes this.
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class TickerScore:
    """Outcome of screening one ticker: did it pass, and (optionally) its value score."""

    passed: bool
    score: float | None = None


def _fundamentals_checks(fundamentals: FundamentalsData | None, criteria: Criteria) -> list[bool]:
    """Pass/fail for each *present* fundamentals-sourced criterion (P/E, ROE, D/E)."""
    if fundamentals is None:
        return []
    checks: list[bool] = []
    if fundamentals.pe_ratio is not None:
        # A negative P/E (loss-maker) must not pass a value screen, so guard positivity.
        checks.append(0 < fundamentals.pe_ratio <= criteria["pe_max"])
    if fundamentals.roe_pct is not None:
        # roe_pct is a percentage (15.0 == 15%); roe_min is a fraction (0.12 == 12%).
        checks.append(fundamentals.roe_pct >= criteria["roe_min"] * 100)
    if fundamentals.debt_to_equity is not None:
        # yfinance debtToEquity is a percentage (154.0 == 1.54x); de_max is a ratio.
        # Negative equity (negative D/E) is a red flag, so require a non-negative value.
        checks.append(0 <= fundamentals.debt_to_equity <= criteria["de_max"] * 100)
    return checks


def _growth_checks(growth: GrowthData | None, criteria: Criteria) -> list[bool]:
    """Pass/fail for each *present* growth-sourced criterion (PEG, 3Y revenue CAGR)."""
    if growth is None:
        return []
    checks: list[bool] = []
    if growth.peg_ratio is not None:
        # A negative PEG (negative earnings or growth) is not a value signal.
        checks.append(0 < growth.peg_ratio <= criteria["peg_max"])
    if growth.revenue_cagr_3y is not None:
        checks.append(growth.revenue_cagr_3y >= criteria["rev_growth_min"])
    return checks


def screen_ticker(ticker: str, client: YFinanceClient, criteria: Criteria) -> bool:
    """Return True if ``ticker`` clears the GARP screen given fundamentals + growth data."""
    f = client.get_fundamentals(ticker)
    fundamentals = f if isinstance(f, FundamentalsData) else None
    fund_checks = _fundamentals_checks(fundamentals, criteria)

    # A present fundamentals metric that already fails means the ticker can never
    # pass (we require *all* present criteria to pass). Skip the expensive growth
    # fetch — get_growth_metrics pulls full financial statements — in that case.
    if any(not c for c in fund_checks):
        return False

    g = client.get_growth_metrics(ticker)
    growth = g if isinstance(g, GrowthData) else None
    checks = fund_checks + _growth_checks(growth, criteria)

    return len(checks) >= MIN_CRITERIA_PRESENT and all(checks)


# ---------------------------------------------------------------------------
# Deep-value ("gem-hunt") screen — distinct criteria, same present-metric policy
# ---------------------------------------------------------------------------
#
# Each check applies only when (a) its criterion key is present in ``criteria`` AND
# (b) the metric it reads is present on the fetched model — mirroring both
# ``agent/tools/screen.py`` (key-driven) and the GARP checks above (present-metric,
# short-circuit on a present-but-failing metric). Semantics/keys are kept in lockstep
# with ``agent/tools/screen.py``'s _FUNDAMENTALS_/_VALUATION_/_QUALITY_CRITERIA maps.


def _value_fundamentals_checks(
    fundamentals: FundamentalsData | None, criteria: Criteria
) -> list[bool]:
    """Pass/fail for each present fundamentals-sourced value criterion (P/B)."""
    if fundamentals is None:
        return []
    checks: list[bool] = []
    if "pb_ratio_max" in criteria and fundamentals.pb_ratio is not None:
        # A negative P/B (negative book value) is a red flag, not a value signal.
        checks.append(0 < fundamentals.pb_ratio <= criteria["pb_ratio_max"])
    return checks


def _value_valuation_checks(valuation: ValuationData | None, criteria: Criteria) -> list[bool]:
    """Pass/fail for each present valuation-sourced value criterion.

    EV/EBIT, price-to-NCAV, the net-cash boolean gate, and the dividend-yield floor.
    """
    if valuation is None:
        return []
    checks: list[bool] = []
    if "max_ev_ebit" in criteria and valuation.ev_to_ebit is not None:
        # Negative EV/EBIT (negative EBIT) is not a cheapness signal — guard positivity.
        checks.append(0 < valuation.ev_to_ebit <= criteria["max_ev_ebit"])
    if "max_price_to_ncav" in criteria and valuation.price_to_ncav is not None:
        # Negative price-to-NCAV means negative NCAV (no net current asset value); reject.
        checks.append(0 < valuation.price_to_ncav <= criteria["max_price_to_ncav"])
    if criteria.get("require_net_cash"):
        # net_cash_positive is always present (a bool), so this gate always evaluates.
        checks.append(bool(valuation.net_cash_positive))
    if "min_dividend_yield_pct" in criteria and valuation.dividend_yield_pct is not None:
        # Canonical dividend source: ValuationData.dividend_yield_pct — the raw yfinance
        # ``.info["dividendYield"]``, already in PERCENT (AAPL 0.34 → 0.34%). Chosen over
        # CapitalAllocation.dividend_yield_pct (|dividends paid| / market cap, cash-flow-
        # derived) because it's the direct market-observed yield and comes free with the
        # valuation fetch this screen already makes — no extra statement pull. Same units,
        # so the floor is specified in percent. (Scale asserted in tests.)
        checks.append(valuation.dividend_yield_pct >= criteria["min_dividend_yield_pct"])
    return checks


def _value_quality_checks(quality: QualityData | None, criteria: Criteria) -> list[bool]:
    """Pass/fail for the present quality-sourced value criterion (consecutive profit years)."""
    if quality is None:
        return []
    checks: list[bool] = []
    if "min_consecutive_profit_years" in criteria and quality.consecutive_profit_years is not None:
        checks.append(quality.consecutive_profit_years >= criteria["min_consecutive_profit_years"])
    return checks


def _deep_value_score(valuation: ValuationData | None) -> float:
    """Composite deep-value score — **lower is cheaper / better rank**.

    ``score = ev_to_ebit − ncav_to_market_cap``. A lower EV/EBIT (cheaper on operating
    earnings) lowers the score; a higher NCAV-to-market-cap (a deeper discount to net
    current asset value) also lowers it — so both "cheap" directions push toward the top.
    Missing EV/EBIT is penalised to the bottom; missing NCAV discount contributes nothing.
    Callers break score ties deterministically by ticker.
    """
    if valuation is None:
        return _MISSING_EV_PENALTY
    ev_term = valuation.ev_to_ebit if valuation.ev_to_ebit is not None else _MISSING_EV_PENALTY
    ncav = valuation.ncav_to_market_cap
    ncav_discount = ncav if ncav is not None else 0.0
    return ev_term - ncav_discount


def screen_ticker_value(ticker: str, client: YFinanceClient, criteria: Criteria) -> TickerScore:
    """Deep-value screen for one ticker, returning pass/fail plus its value score.

    Fetches fundamentals → valuation → quality, evaluating each stage's present-metric
    checks and short-circuiting when a present metric already fails (as the GARP path
    does). The score is computed from the valuation whenever it was fetched.
    """
    f = client.get_fundamentals(ticker)
    fundamentals = f if isinstance(f, FundamentalsData) else None
    fund_checks = _value_fundamentals_checks(fundamentals, criteria)
    if any(not c for c in fund_checks):
        return TickerScore(passed=False)

    v = client.get_valuation_multiples(ticker)
    valuation = v if isinstance(v, ValuationData) else None
    val_checks = _value_valuation_checks(valuation, criteria)
    score = _deep_value_score(valuation)
    if any(not c for c in val_checks):
        return TickerScore(passed=False, score=score)

    q = client.get_quality_metrics(ticker)
    quality = q if isinstance(q, QualityData) else None
    checks = fund_checks + val_checks + _value_quality_checks(quality, criteria)

    passed = len(checks) >= MIN_CRITERIA_PRESENT and all(checks)
    return TickerScore(passed=passed, score=score)


# A per-ticker screening function: (ticker, client, criteria) → TickerScore. The GARP
# path adapts the boolean ``screen_ticker``; the gem-hunt path uses ``screen_ticker_value``.
ScreenFn = Callable[[str, YFinanceClient, Criteria], TickerScore]


def _garp_screen_fn(ticker: str, client: YFinanceClient, criteria: Criteria) -> TickerScore:
    return TickerScore(passed=screen_ticker(ticker, client, criteria))


def run_screening_pass(
    universe: list[str],
    criteria: Criteria | None = None,
    logger: RunLogger | None = None,
    client: YFinanceClient | None = None,
    max_workers: int | None = None,
    screen_fn: ScreenFn | None = None,
    rank: bool = False,
) -> ScreeningResult:
    """Screen the universe and return tickers that clear all available criteria.

    Args:
        universe: Tickers to screen (cooldown-filtered by the caller). Truncated to
            ``MAX_UNIVERSE_SIZE`` — deterministically, since it arrives sorted.
        criteria: Threshold overrides; defaults to DEFAULT_SCREEN_CRITERIA (GARP).
        logger: Optional RunLogger; emits phase_started / phase_completed events.
        client: YFinanceClient override (defaults to the shared singleton).
        max_workers: When ``None`` (default) the screen runs sequentially — the
            original, deterministic path. When ``> 1`` the per-ticker fetches run on a
            ``ThreadPoolExecutor``; results are still emitted in universe order, so the
            surfaced candidates are identical regardless of worker count.
        screen_fn: Per-ticker screening function; defaults to the boolean GARP screen.
            The gem-hunt path passes ``screen_ticker_value`` (scored deep-value screen).
        rank: When ``False`` (default) candidates are returned in universe order —
            unchanged behaviour. When ``True`` passing candidates are sorted by ascending
            deep-value score (cheaper first), ties broken by ticker.
    """
    effective_criteria: Criteria = criteria if criteria is not None else DEFAULT_SCREEN_CRITERIA
    fn: ScreenFn = screen_fn if screen_fn is not None else _garp_screen_fn
    yf = client if client is not None else yfinance_client()
    universe = universe[:MAX_UNIVERSE_SIZE]

    if logger is not None:
        logger.log(
            "phase_started",
            phase="screening",
            universe_size=len(universe),
            method="quantitative",
        )

    if max_workers is not None and max_workers > 1 and universe:
        # ThreadPoolExecutor.map preserves input order, so zipping back against the
        # (sorted) universe keeps the candidate list deterministic across worker counts.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            outcomes = list(executor.map(lambda t: fn(t, yf, effective_criteria), universe))
    else:
        outcomes = [fn(t, yf, effective_criteria) for t in universe]

    candidates = [t for t, o in zip(universe, outcomes, strict=True) if o.passed]
    scores: dict[str, float] = {
        t: o.score
        for t, o in zip(universe, outcomes, strict=True)
        if o.passed and o.score is not None
    }
    pass_rate = len(candidates) / len(universe) if universe else 0.0

    if rank:
        # Cheaper = better = lower score. None scores (shouldn't occur on the value path)
        # sort last; ties break deterministically by ticker.
        candidates.sort(key=lambda t: (t not in scores, scores.get(t, 0.0), t))

    if logger is not None:
        logger.log(
            "phase_completed",
            phase="screening",
            candidates_surfaced=candidates,
            pass_rate=pass_rate,
        )

    return ScreeningResult(candidates=candidates, pass_rate=pass_rate, scores=scores)
