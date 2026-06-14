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
from data_sources.yfinance_client import GrowthData


class GetGrowthMetricsInput(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z]{1,5}$", description="Stock ticker, e.g. AAPL")


class GetGrowthMetricsTool(Tool):
    name = "get_growth_metrics"
    description = (
        "Get multi-year growth metrics for a ticker: 3- and 5-year revenue CAGR, "
        "3-year earnings CAGR, and the PEG ratio."
    )
    input_schema = GetGrowthMetricsInput
    output_schema = GrowthData

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, GetGrowthMetricsInput)
        try:
            result = yfinance_client().get_growth_metrics(tool_input.ticker)
        except Exception as exc:
            return ToolResultError(
                error_code="unknown",
                message=f"get_growth_metrics failed for {tool_input.ticker}: {exc}",
                retryable=False,
            )
        if isinstance(result, DataSourceError):
            return error_from_data_source(result)
        return ToolResultOk(data=result)
