from pydantic import BaseModel, Field

from agent.budget import RunContext
from agent.tools._clients import finnhub_client
from agent.tools.base import (
    TICKER_PATTERN,
    Tool,
    ToolResult,
    ToolResultError,
    ToolResultOk,
    error_from_data_source,
)
from data_sources.errors import DataSourceError
from data_sources.finnhub_client import NewsItem


class GetNewsInput(BaseModel):
    ticker: str = Field(pattern=TICKER_PATTERN, description="Stock ticker, e.g. AAPL")
    days: int = Field(default=7, ge=1, le=30, description="Look-back window in days (1-30)")


class NewsResult(BaseModel):
    ticker: str
    days: int
    items: list[NewsItem]


class GetNewsTool(Tool):
    name = "get_news"
    description = (
        "Get recent company news headlines and summaries for a ticker over a "
        "look-back window of up to 30 days (newest first)."
    )
    input_schema = GetNewsInput
    output_schema = NewsResult

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, GetNewsInput)
        fh = finnhub_client()
        if fh is None:
            return ToolResultError(
                error_code="not_found",
                message="Finnhub API key not configured; news is unavailable.",
                retryable=False,
            )
        try:
            result = fh.get_news(tool_input.ticker, tool_input.days)
        except Exception as exc:
            return ToolResultError(
                error_code="unknown",
                message=f"get_news failed for {tool_input.ticker}: {exc}",
                retryable=False,
            )
        if isinstance(result, DataSourceError):
            return error_from_data_source(result)
        return ToolResultOk(
            data=NewsResult(ticker=tool_input.ticker, days=tool_input.days, items=result)
        )
