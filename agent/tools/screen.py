import csv
from pathlib import Path

from pydantic import BaseModel, Field

from agent.budget import RunContext
from agent.tools._clients import yfinance_client
from agent.tools.base import Tool, ToolResult, ToolResultError, ToolResultOk
from data_sources.yfinance_client import FundamentalsData, QualityData, ValuationData

# Static CSV files that are always included in the universe.
_STATIC_UNIVERSE_FILES = (Path("data/portfolio.csv"), Path("data/watchlist.csv"))

_UNIVERSE_LIMITATION_NOTE = (
    "DIRT universe: US-only (Russell 2000 + portfolio/watchlist); "
    "aggregator reliability degrades for sub-$300M market-cap names."
)

# Criteria keyed by the data source they require.
# "max" keys: metric must be ≤ threshold. "min" keys: metric must be ≥ threshold.
# roe_min / consecutive_profit_years are in natural units (%, count).

_FUNDAMENTALS_CRITERIA: dict[str, tuple[str, str]] = {
    "pe_ratio_max": ("pe_ratio", "<="),
    "pb_ratio_max": ("pb_ratio", "<="),
    "roe_min": ("roe_pct", ">="),
    "de_max": ("debt_to_equity", "<="),
}

_VALUATION_CRITERIA: dict[str, tuple[str, str]] = {
    "max_ev_ebit": ("ev_to_ebit", "<="),
    "max_price_to_ncav": ("price_to_ncav", "<="),
    "max_market_cap_usd": ("market_cap_usd", "<="),
}

_QUALITY_CRITERIA: dict[str, tuple[str, str]] = {
    "min_consecutive_profit_years": ("consecutive_profit_years", ">="),
}

# Boolean criterion: threshold > 0 → require net_cash_positive == True.
_BOOLEAN_QUALITY_CRITERIA: frozenset[str] = frozenset({"require_net_cash"})

# Unreliable / aspirational: coverage count is typically missing (not zero) for
# truly uncovered names, so this criterion cannot be reliably implemented.
_UNSUPPORTED_CRITERIA: frozenset[str] = frozenset({"require_zero_analyst_coverage"})

_ALL_KNOWN: frozenset[str] = (
    frozenset(_FUNDAMENTALS_CRITERIA)
    | frozenset(_VALUATION_CRITERIA)
    | frozenset(_QUALITY_CRITERIA)
    | _BOOLEAN_QUALITY_CRITERIA
    | _UNSUPPORTED_CRITERIA
)


class ScreenUniverseInput(BaseModel):
    criteria: dict[str, float] = Field(
        description=(
            "Quantitative filters, all of which must pass. Supported keys:\n"
            "  Fundamentals: pe_ratio_max, pb_ratio_max, roe_min (percent), de_max.\n"
            "  Valuation:    max_ev_ebit, max_price_to_ncav, max_market_cap_usd (USD).\n"
            "  Quality:      require_net_cash (1 = require net-cash balance sheet),\n"
            "                min_consecutive_profit_years.\n"
            "  Unsupported (silently ignored — unreliable data): require_zero_analyst_coverage.\n"
            "Unknown keys are silently ignored."
        )
    )


class ScreenResult(BaseModel):
    tickers: list[str]
    criteria: dict[str, float]
    universe_size: int
    universe_limitation_note: str


def _universe() -> list[str]:
    tickers: list[str] = []
    for path in _STATIC_UNIVERSE_FILES:
        if not path.exists():
            continue
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                ticker = (row.get("ticker") or "").strip().upper()
                if ticker and ticker not in tickers:
                    tickers.append(ticker)

    # Russell 2000 tickers — fetched live and cached for 7 days so the
    # constituent list stays current without hitting the network on every run.
    r2k = yfinance_client().get_russell2000_tickers()
    if isinstance(r2k, list):
        for ticker in r2k:
            if ticker and ticker not in tickers:
                tickers.append(ticker)

    return tickers


def _check_numeric(obj: object, attr: str, op: str, threshold: float) -> bool:
    value = getattr(obj, attr, None)
    if not isinstance(value, (int, float)):
        return False
    if op == "<=" and value > threshold:
        return False
    if op == ">=" and value < threshold:
        return False
    return True


def _passes(
    criteria: dict[str, float],
    fundamentals: FundamentalsData | None,
    valuation: ValuationData | None,
    quality: QualityData | None,
) -> bool:
    for key, threshold in criteria.items():
        if key in _FUNDAMENTALS_CRITERIA:
            if fundamentals is None:
                return False
            attr, op = _FUNDAMENTALS_CRITERIA[key]
            if not _check_numeric(fundamentals, attr, op, threshold):
                return False
        elif key in _VALUATION_CRITERIA:
            if valuation is None:
                return False
            attr, op = _VALUATION_CRITERIA[key]
            if not _check_numeric(valuation, attr, op, threshold):
                return False
        elif key in _BOOLEAN_QUALITY_CRITERIA:
            if threshold > 0:
                if valuation is None or not valuation.net_cash_positive:
                    return False
        elif key in _QUALITY_CRITERIA:
            if quality is None:
                return False
            attr, op = _QUALITY_CRITERIA[key]
            if not _check_numeric(quality, attr, op, threshold):
                return False
        # unknown / unsupported keys → silently ignore
    return True


class ScreenUniverseTool(Tool):
    name = "screen_universe"
    description = (
        "Screen the US universe (Russell 2000 + portfolio + watchlist) against "
        "quantitative fundamental and deep-value filters. Returns the tickers that pass every filter. "
        "Supports fundamentals (pe_ratio_max, pb_ratio_max, roe_min, de_max), "
        "valuation (max_ev_ebit, max_price_to_ncav, max_market_cap_usd), and "
        "quality (require_net_cash, min_consecutive_profit_years) criteria. "
        "Note: require_zero_analyst_coverage is not implemented — analyst-coverage "
        "counts are missing rather than zero for truly uncovered names, making this "
        "criterion unreliable; pass it and it will be silently ignored."
    )
    input_schema = ScreenUniverseInput
    output_schema = ScreenResult

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, ScreenUniverseInput)
        criteria = tool_input.criteria

        need_valuation = bool(
            set(criteria) & (frozenset(_VALUATION_CRITERIA) | _BOOLEAN_QUALITY_CRITERIA)
        )
        need_quality = bool(set(criteria) & frozenset(_QUALITY_CRITERIA))

        try:
            universe = _universe()
            passed: list[str] = []
            for ticker in universe:
                fundamentals: FundamentalsData | None = None
                valuation: ValuationData | None = None
                quality: QualityData | None = None

                if set(criteria) & frozenset(_FUNDAMENTALS_CRITERIA):
                    raw = yfinance_client().get_fundamentals(ticker)
                    if isinstance(raw, FundamentalsData):
                        fundamentals = raw

                if need_valuation:
                    raw_v = yfinance_client().get_valuation_multiples(ticker)
                    if isinstance(raw_v, ValuationData):
                        valuation = raw_v

                if need_quality:
                    raw_q = yfinance_client().get_quality_metrics(ticker)
                    if isinstance(raw_q, QualityData):
                        quality = raw_q

                if _passes(criteria, fundamentals, valuation, quality):
                    passed.append(ticker)
        except Exception as exc:
            return ToolResultError(
                error_code="unknown",
                message=f"screen_universe failed: {exc}",
                retryable=False,
            )
        return ToolResultOk(
            data=ScreenResult(
                tickers=passed,
                criteria=criteria,
                universe_size=len(universe),
                universe_limitation_note=_UNIVERSE_LIMITATION_NOTE,
            )
        )
