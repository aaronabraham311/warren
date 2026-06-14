from pydantic import BaseModel, Field

from agent.budget import RunContext
from agent.tools._clients import yfinance_client
from agent.tools.base import (
    Tool,
    ToolResult,
    ToolResultError,
    ToolResultOk,
    error_from_data_source,
)
from data_sources.errors import DataSourceError
from data_sources.yfinance_client import ValuationData


class GetValuationMultiplesInput(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z]{1,5}$", description="Stock ticker, e.g. AAPL")


class GetValuationMultiplesTool(Tool):
    name = "get_valuation_multiples"
    description = (
        "Compute deep-value valuation multiples for a ticker: EV/EBIT, EV/EBITDA, "
        "Acquirer's Multiple (Carlisle / EV per operating earnings), FCF yield, "
        "earnings yield (Greenblatt), NCAV / net-net ratio (Graham), P/tangible book, "
        "and dividend yield. All ratios are None when the required inputs are unavailable."
    )
    input_schema = GetValuationMultiplesInput
    output_schema = ValuationData

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, GetValuationMultiplesInput)
        ticker = tool_input.ticker
        try:
            result = yfinance_client().get_valuation_multiples(ticker)
        except Exception as exc:
            return ToolResultError(
                error_code="unknown",
                message=f"get_valuation_multiples failed for {ticker}: {exc}",
                retryable=False,
            )
        if isinstance(result, DataSourceError):
            return error_from_data_source(result)
        return ToolResultOk(data=result)
