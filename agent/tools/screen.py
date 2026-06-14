import csv
from pathlib import Path

from pydantic import BaseModel, Field

from agent.budget import RunContext
from agent.tools._clients import yfinance_client
from agent.tools.base import Tool, ToolResult, ToolResultError, ToolResultOk
from data_sources.yfinance_client import FundamentalsData

# Offline universe: the union of tickers in the portfolio and watchlist CSVs.
# (There is no full-market source available without live network access.)
_UNIVERSE_FILES = (Path("data/portfolio.csv"), Path("data/watchlist.csv"))

# criteria key → (FundamentalsData attribute, comparison). "max" keys require the
# metric to be ≤ the threshold; "min" keys require ≥. roe_min is in percent units
# (matching FundamentalsData.roe_pct, e.g. 15.0 not 0.15).
_CRITERIA: dict[str, tuple[str, str]] = {
    "pe_ratio_max": ("pe_ratio", "<="),
    "pb_ratio_max": ("pb_ratio", "<="),
    "roe_min": ("roe_pct", ">="),
    "de_max": ("debt_to_equity", "<="),
}


class ScreenUniverseInput(BaseModel):
    criteria: dict[str, float] = Field(
        description=(
            "Quantitative filters, all of which must pass. Supported keys: "
            "pe_ratio_max, pb_ratio_max, roe_min (percent), de_max."
        )
    )


class ScreenResult(BaseModel):
    tickers: list[str]
    criteria: dict[str, float]
    universe_size: int


def _universe() -> list[str]:
    tickers: list[str] = []
    for path in _UNIVERSE_FILES:
        if not path.exists():
            continue
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                ticker = (row.get("ticker") or "").strip().upper()
                if ticker and ticker not in tickers:
                    tickers.append(ticker)
    return tickers


def _passes(fundamentals: FundamentalsData, criteria: dict[str, float]) -> bool:
    for key, threshold in criteria.items():
        spec = _CRITERIA.get(key)
        if spec is None:
            continue  # unknown criterion — ignore rather than fail the screen
        attr, op = spec
        value = getattr(fundamentals, attr)
        if not isinstance(value, (int, float)):
            return False  # required metric missing for this ticker
        if op == "<=" and value > threshold:
            return False
        if op == ">=" and value < threshold:
            return False
    return True


class ScreenUniverseTool(Tool):
    name = "screen_universe"
    description = (
        "Screen the portfolio+watchlist universe against quantitative fundamental "
        "filters (e.g. {'pe_ratio_max': 20, 'roe_min': 15}). Returns the tickers "
        "that pass every filter."
    )
    input_schema = ScreenUniverseInput
    output_schema = ScreenResult

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, ScreenUniverseInput)
        try:
            universe = _universe()
            passed: list[str] = []
            for ticker in universe:
                fundamentals = yfinance_client().get_fundamentals(ticker)
                if isinstance(fundamentals, FundamentalsData) and _passes(
                    fundamentals, tool_input.criteria
                ):
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
                criteria=tool_input.criteria,
                universe_size=len(universe),
            )
        )
