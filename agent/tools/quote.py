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
from data_sources.yfinance_client import PriceData


class GetQuoteInput(BaseModel):
    ticker: str = Field(pattern=TICKER_PATTERN, description="Stock ticker, e.g. AAPL")


class GetQuoteTool(Tool):
    name = "get_quote"
    description = (
        "Get the current price, previous close, day change percent, and average "
        "volume for a stock ticker."
    )
    input_schema = GetQuoteInput
    output_schema = PriceData

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, GetQuoteInput)
        try:
            result = yfinance_client().get_price(tool_input.ticker)
        except Exception as exc:
            return ToolResultError(
                error_code="unknown",
                message=f"get_quote failed for {tool_input.ticker}: {exc}",
                retryable=False,
            )
        if isinstance(result, DataSourceError):
            return error_from_data_source(result)
        return ToolResultOk(data=result)
