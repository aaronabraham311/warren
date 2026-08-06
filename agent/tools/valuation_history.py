from pydantic import BaseModel, Field

from agent.budget import RunContext
from agent.tools._clients import yfinance_client
from agent.tools.base import (
    TICKER_PATTERN,
    Tool,
    ToolResult,
    ToolResultError,
    ToolResultOk,
    error_from_data_source,
)
from data_sources.errors import DataSourceError
from data_sources.yfinance_client import ValuationHistory


class GetValuationHistoryInput(BaseModel):
    ticker: str = Field(pattern=TICKER_PATTERN, description="Stock ticker, e.g. AAPL")


class GetValuationHistoryTool(Tool):
    name = "get_valuation_history"
    description = (
        "Valuation-vs-own-listed-history signal — how cheap a stock is versus its OWN "
        "past, the Kino-Polska 'cheapest multiple in its listed life' lens. Builds P/E "
        "and P/B time series from a historical price slice plus per-fiscal-year EPS and "
        "book-value-per-share, then exposes pe_percentile / pb_percentile (0-100 rank of "
        "the current multiple within its own history; LOW = cheap, 0 = cheapest ever) and "
        "the legacy field pb_vs_10y_low (current P/B divided by its historical minimum; "
        "~1.0 = at/near its low). Despite that legacy name, this is NOT evidence of a "
        "10-year low: yfinance normally exposes only ~5 years of statement history. Use "
        "years_covered to state the actual window; with under 2 usable years the percentile "
        "fields are None rather than misleading."
    )
    input_schema = GetValuationHistoryInput
    output_schema = ValuationHistory

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, GetValuationHistoryInput)
        ticker = tool_input.ticker
        try:
            result = yfinance_client().get_valuation_history(ticker)
        except Exception as exc:
            return ToolResultError(
                error_code="unknown",
                message=f"get_valuation_history failed for {ticker}: {exc}",
                retryable=False,
            )
        if isinstance(result, DataSourceError):
            return error_from_data_source(result)
        return ToolResultOk(data=result)
