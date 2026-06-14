from pydantic import BaseModel, Field

from agent.budget import RunContext
from agent.tools._clients import finnhub_client, yfinance_client
from agent.tools.base import (
    Tool,
    ToolResult,
    ToolResultError,
    ToolResultOk,
    error_from_data_source,
)
from data_sources.errors import DataSourceError
from data_sources.finnhub_client import FinnhubFinancials
from data_sources.yfinance_client import FundamentalsData

# yfinance fundamentals are keyed to the last fiscal-year end; once that data is
# older than this threshold we treat it as stale and reach for Finnhub instead.
_STALE_FUNDAMENTALS_H = 48


class GetFundamentalsInput(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z]{1,5}$", description="Stock ticker, e.g. AAPL")


def _finnhub_to_fundamentals(f: FinnhubFinancials) -> FundamentalsData:
    """Project Finnhub's narrower basics onto the FundamentalsData shape."""
    return FundamentalsData(
        ticker=f.ticker,
        as_of=f.as_of,
        pe_ratio=f.pe_ratio,
        pb_ratio=f.pb_ratio,
        roe_pct=f.roe_pct,
        debt_to_equity=None,
        fcf_ttm_usd=None,
        operating_margin_pct=None,
        net_margin_pct=None,
        data_age_hours=0,
        source="finnhub",
    )


class GetFundamentalsTool(Tool):
    name = "get_fundamentals"
    description = (
        "Get valuation and quality fundamentals for a ticker: P/E, P/B, ROE, "
        "debt-to-equity, trailing free cash flow, and operating/net margins. Tries "
        "yfinance first and falls back to Finnhub when yfinance data is stale."
    )
    input_schema = GetFundamentalsInput
    output_schema = FundamentalsData

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, GetFundamentalsInput)
        ticker = tool_input.ticker
        try:
            result = yfinance_client().get_fundamentals(ticker)
            stale_ok = (
                isinstance(result, FundamentalsData)
                and result.data_age_hours > _STALE_FUNDAMENTALS_H
            )
            stale_err = isinstance(result, DataSourceError) and result.error_code == "stale_data"
            if stale_ok or stale_err:
                fh = finnhub_client()
                if fh is not None:
                    fallback = fh.get_basic_financials(ticker)
                    if isinstance(fallback, FinnhubFinancials):
                        result = _finnhub_to_fundamentals(fallback)
        except Exception as exc:
            return ToolResultError(
                error_code="unknown",
                message=f"get_fundamentals failed for {ticker}: {exc}",
                retryable=False,
            )
        if isinstance(result, DataSourceError):
            return error_from_data_source(result)
        return ToolResultOk(data=result)
