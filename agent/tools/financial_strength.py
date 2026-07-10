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
from data_sources.yfinance_client import FinancialStrengthData


class GetFinancialStrengthInput(BaseModel):
    ticker: str = Field(pattern=TICKER_PATTERN, description="Stock ticker, e.g. AAPL")


class GetFinancialStrengthTool(Tool):
    name = "get_financial_strength"
    description = (
        "Compute balance-sheet quality metrics for a ticker: Piotroski F-score (0–9, with "
        "per-signal breakdown across profitability, leverage/liquidity, and operating efficiency), "
        "Altman Z-score (bankruptcy distance, with distress/grey/safe zone), interest coverage "
        "(EBIT / interest expense), current ratio, and net-debt/EBITDA. Use this tool to screen "
        "out balance-sheet landmines before buying cheap — F-score ≥ 7 signals high quality; "
        "Z-score < 1.81 flags distress risk."
    )
    input_schema = GetFinancialStrengthInput
    output_schema = FinancialStrengthData

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, GetFinancialStrengthInput)
        ticker = tool_input.ticker
        try:
            result = yfinance_client().get_financial_strength(ticker)
        except Exception as exc:
            return ToolResultError(
                error_code="unknown",
                message=f"get_financial_strength failed for {ticker}: {exc}",
                retryable=False,
            )
        if isinstance(result, DataSourceError):
            return error_from_data_source(result)
        return ToolResultOk(data=result)
