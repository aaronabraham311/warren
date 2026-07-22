from pydantic import BaseModel, Field

from agent.budget import RunContext
from agent.tools._clients import finnhub_client, yfinance_client
from agent.tools.base import (
    TICKER_PATTERN,
    Tool,
    ToolResult,
    ToolResultError,
    ToolResultOk,
    error_from_data_source,
)
from data_sources.errors import DataSourceError
from data_sources.finnhub_client import FinnhubFinancials
from data_sources.yfinance_client import FundamentalsData

# yfinance fundamentals are keyed to the last fiscal-year end, so _fiscal_age_hours()
# reports their age in months even when the fetch is current (a company only reports
# annually). A short bar here would treat every company as stale on every run and
# downgrade to Finnhub's narrower basics. Only reach for Finnhub once a full fiscal
# year has lapsed without fresh data — i.e. the company has effectively missed an
# expected annual filing.
_STALE_FUNDAMENTALS_H = 456 * 24  # ~15 months (12mo fiscal cycle + 1 quarter grace)


class GetFundamentalsInput(BaseModel):
    ticker: str = Field(pattern=TICKER_PATTERN, description="Stock ticker, e.g. AAPL")


def _finnhub_to_fundamentals(f: FinnhubFinancials) -> FundamentalsData:
    """Project Finnhub's narrower basics onto the FundamentalsData shape."""
    return FundamentalsData(
        ticker=f.ticker,
        as_of=f.as_of,
        pe_ratio=f.pe_ratio,
        pb_ratio=f.pb_ratio,
        roe_pct=f.roe_pct,
        debt_to_equity=f.debt_to_equity,
        gross_margin_pct=f.gross_margin_pct,
        operating_margin_pct=f.operating_margin_pct,
        net_margin_pct=f.net_margin_pct,
        # Finnhub's basic-financials endpoint supplies neither trailing FCF in a
        # trustworthy unit nor a sector classification (that needs the separate
        # company-profile endpoint), so these stay None on the fallback path.
        fcf_ttm_usd=None,
        sector=None,
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
