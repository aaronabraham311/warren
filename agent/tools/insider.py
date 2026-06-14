from datetime import date
from typing import Literal

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
from data_sources.finnhub_client import FinnhubInsiderTransaction
from data_sources.yfinance_client import OwnershipData


class GetInsiderActivityInput(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z]{1,5}$", description="Stock ticker, e.g. AAPL")
    window_days: int = Field(
        default=90, ge=30, le=365, description="Look-back window in days (30–365)"
    )


class InsiderTransaction(BaseModel):
    name: str
    transaction_type: Literal["buy", "sell", "other"]
    shares: int
    value: float | None
    transaction_date: date


class InsiderActivity(BaseModel):
    ticker: str
    as_of: date
    window_days: int
    transactions: list[InsiderTransaction]
    net_shares_bought: int
    net_value_bought: float | None
    insider_sentiment: Literal["bullish", "bearish", "neutral", "insufficient_data"]
    insider_ownership_pct: float | None
    institutional_ownership_pct: float | None


def _sentiment(
    net_shares: int, has_open_market: bool
) -> Literal["bullish", "bearish", "neutral", "insufficient_data"]:
    if not has_open_market:
        return "insufficient_data"
    if net_shares > 0:
        return "bullish"
    if net_shares < 0:
        return "bearish"
    return "neutral"


def _to_output_txn(t: FinnhubInsiderTransaction) -> InsiderTransaction:
    return InsiderTransaction(
        name=t.name,
        transaction_type=t.transaction_type,
        shares=t.shares,
        value=t.value,
        transaction_date=t.transaction_date,
    )


class GetInsiderActivityTool(Tool):
    name = "get_insider_activity"
    description = (
        "Retrieve recent insider buy/sell transactions for a ticker and summarise the "
        "net insider sentiment (bullish / bearish / neutral / insufficient_data). "
        "Also returns insider ownership % and institutional ownership % where available. "
        "Cluster open-market buying is a classic value signal; heavy insider selling is a "
        "caution flag. Only open-market purchases and sales drive the sentiment score — "
        "awards, option exercises, and gifts are recorded but excluded from the net."
    )
    input_schema = GetInsiderActivityInput
    output_schema = InsiderActivity

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, GetInsiderActivityInput)
        ticker = tool_input.ticker
        window_days = tool_input.window_days

        fh = finnhub_client()
        if fh is None:
            return ToolResultError(
                error_code="not_found",
                message="Finnhub API key not configured; insider activity is unavailable.",
                retryable=False,
            )

        try:
            raw_txns = fh.get_insider_transactions(ticker, window_days)
        except Exception as exc:
            return ToolResultError(
                error_code="unknown",
                message=f"get_insider_activity failed for {ticker}: {exc}",
                retryable=False,
            )
        if isinstance(raw_txns, DataSourceError):
            return error_from_data_source(raw_txns)

        # Aggregate net shares/value from open-market buys and sells only.
        net_shares = 0
        net_value: float | None = None
        has_open_market = False
        for t in raw_txns:
            if t.transaction_type == "buy":
                has_open_market = True
                net_shares += t.shares
                if t.value is not None:
                    net_value = (net_value or 0.0) + t.value
            elif t.transaction_type == "sell":
                has_open_market = True
                net_shares -= t.shares
                if t.value is not None:
                    net_value = (net_value or 0.0) - t.value

        # Ownership % from yfinance — best-effort; failures become None.
        insider_pct: float | None = None
        institutional_pct: float | None = None
        try:
            ownership = yfinance_client().get_ownership(ticker)
            if isinstance(ownership, OwnershipData):
                insider_pct = ownership.insider_pct
                institutional_pct = ownership.institutional_pct
        except Exception:
            pass

        return ToolResultOk(
            data=InsiderActivity(
                ticker=ticker,
                as_of=date.today(),
                window_days=window_days,
                transactions=[_to_output_txn(t) for t in raw_txns],
                net_shares_bought=net_shares,
                net_value_bought=net_value,
                insider_sentiment=_sentiment(net_shares, has_open_market),
                insider_ownership_pct=insider_pct,
                institutional_ownership_pct=institutional_pct,
            )
        )
