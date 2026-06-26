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
from data_sources.yfinance_client import QualityData


class GetQualityMetricsInput(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z]{1,5}$", description="Stock ticker, e.g. AAPL")


class GetQualityMetricsTool(Tool):
    name = "get_quality_metrics"
    description = (
        "Compute Munger-style quality metrics for a ticker: ROIC (current + 3–4 yr series + mean), "
        "ROA, gross-margin level and multi-year stability (stdev — low stdev signals a moat), "
        "cash-conversion ratio (FCF / net income, TTM and multi-year), consecutive years of "
        "positive operating income (consecutive_profit_years), and NCAV trend "
        "(ncav_trend: growing/stable/declining — declining NCAV means the liquidation margin "
        "is eroding). High, stable ROIC and clean cash conversion distinguish quality "
        "compounders from mediocre businesses."
    )
    input_schema = GetQualityMetricsInput
    output_schema = QualityData

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, GetQualityMetricsInput)
        ticker = tool_input.ticker
        try:
            result = yfinance_client().get_quality_metrics(ticker)
        except Exception as exc:
            return ToolResultError(
                error_code="unknown",
                message=f"get_quality_metrics failed for {ticker}: {exc}",
                retryable=False,
            )
        if isinstance(result, DataSourceError):
            return error_from_data_source(result)
        return ToolResultOk(data=result)
