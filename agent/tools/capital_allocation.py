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
from data_sources.yfinance_client import CapitalAllocation


class GetCapitalAllocationInput(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z]{1,5}$", description="Stock ticker, e.g. AAPL")


class GetCapitalAllocationTool(Tool):
    name = "get_capital_allocation"
    description = (
        "Grade management capital allocation for a ticker through a Buffett/Munger "
        "owner-friendliness lens: share-count CAGR over available years (negative = net "
        "buybacks, positive = dilution), buyback yield, dividend yield and combined "
        "shareholder yield, dividend growth streak (consecutive years increased), payout "
        "ratio, and net-debt trajectory (delevering vs levering up). Use this to judge "
        "whether management returns capital sensibly — shrinking shares, a growing "
        "dividend, and a delevering balance sheet signal quality; relentless dilution is "
        "a red flag."
    )
    input_schema = GetCapitalAllocationInput
    output_schema = CapitalAllocation

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, GetCapitalAllocationInput)
        ticker = tool_input.ticker
        try:
            result = yfinance_client().get_capital_allocation(ticker)
        except Exception as exc:
            return ToolResultError(
                error_code="unknown",
                message=f"get_capital_allocation failed for {ticker}: {exc}",
                retryable=False,
            )
        if isinstance(result, DataSourceError):
            return error_from_data_source(result)
        return ToolResultOk(data=result)
