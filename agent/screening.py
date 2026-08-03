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

from dataclasses import dataclass

from agent.tools._clients import yfinance_client
from data_sources.yfinance_client import FundamentalsData, GrowthData, YFinanceClient
from storage.logger import RunLogger

# At least this many of the five criteria must have data for a ticker to be
# eligible — otherwise a name with one lucky metric could sneak through.
MIN_CRITERIA_PRESENT = 3

DEFAULT_SCREEN_CRITERIA: dict[str, float] = {
    "pe_max": 30.0,
    "peg_max": 1.5,
    "roe_min": 0.12,  # fraction: 0.12 == 12%
    "de_max": 1.0,  # ratio: 1.0 == 100% debt-to-equity
    "rev_growth_min": 0.05,  # fraction: 0.05 == 5% 3Y revenue CAGR
}


@dataclass
class ScreeningResult:
    candidates: list[str]
    pass_rate: float


def _fundamentals_checks(
    fundamentals: FundamentalsData | None, criteria: dict[str, float]
) -> list[bool]:
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


def _growth_checks(growth: GrowthData | None, criteria: dict[str, float]) -> list[bool]:
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


def screen_ticker(ticker: str, client: YFinanceClient, criteria: dict[str, float]) -> bool:
    """Return True if ``ticker`` clears the screen given live fundamentals + growth data."""
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


def run_screening_pass(
    universe: list[str],
    criteria: dict[str, float] | None = None,
    logger: RunLogger | None = None,
    client: YFinanceClient | None = None,
) -> ScreeningResult:
    """Screen the universe and return tickers that clear all available criteria.

    Args:
        universe: Tickers to screen (cooldown-filtered by the caller).
        criteria: Threshold overrides; defaults to DEFAULT_SCREEN_CRITERIA.
        logger: Optional RunLogger; emits phase_started / phase_completed events.
        client: YFinanceClient override (defaults to the shared singleton).
    """
    effective_criteria = criteria if criteria is not None else DEFAULT_SCREEN_CRITERIA
    yf = client if client is not None else yfinance_client()

    if logger is not None:
        logger.log(
            "phase_started",
            phase="screening",
            universe_size=len(universe),
            method="quantitative",
        )

    candidates = [t for t in universe if screen_ticker(t, yf, effective_criteria)]
    pass_rate = len(candidates) / len(universe) if universe else 0.0

    if logger is not None:
        logger.log(
            "phase_completed",
            phase="screening",
            candidates_surfaced=candidates,
            pass_rate=pass_rate,
        )

    return ScreeningResult(candidates=candidates, pass_rate=pass_rate)
