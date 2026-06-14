import csv
from pathlib import Path

from pydantic import BaseModel, Field

from agent.budget import RunContext
from agent.tools._clients import yfinance_client
from agent.tools.base import Tool, ToolResult, ToolResultError, ToolResultOk
from data_sources.yfinance_client import PriceData

_PORTFOLIO_FILE = Path("data/portfolio.csv")


class GetHoldingContextInput(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z]{1,5}$", description="Stock ticker, e.g. AAPL")


class HoldingContext(BaseModel):
    ticker: str
    shares: float
    cost_basis: float
    purchase_date: str
    current_price: float | None
    unrealized_pnl_pct: float | None


def _portfolio_row(ticker: str) -> dict[str, str] | None:
    if not _PORTFOLIO_FILE.exists():
        return None
    with _PORTFOLIO_FILE.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("ticker") or "").strip().upper() == ticker:
                return row
    return None


class GetHoldingContextTool(Tool):
    name = "get_holding_context"
    description = (
        "Get the portfolio context for a held ticker: shares, cost basis, purchase "
        "date, current price, and unrealized P&L percent. Returns not_found if the "
        "ticker is not in the portfolio."
    )
    input_schema = GetHoldingContextInput
    output_schema = HoldingContext

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, GetHoldingContextInput)
        ticker = tool_input.ticker
        try:
            row = _portfolio_row(ticker)
            if row is None:
                return ToolResultError(
                    error_code="not_found",
                    message=f"{ticker} is not in the portfolio.",
                    retryable=False,
                )
            shares = float(row["shares"])
            cost_basis = float(row["cost_basis"])
            purchase_date = (row.get("purchase_date") or "").strip()

            price = yfinance_client().get_price(ticker)
            current_price = price.current_price if isinstance(price, PriceData) else None
            pnl_pct = (
                round((current_price - cost_basis) / cost_basis * 100, 2)
                if current_price is not None and cost_basis
                else None
            )
        except Exception as exc:
            return ToolResultError(
                error_code="unknown",
                message=f"get_holding_context failed for {ticker}: {exc}",
                retryable=False,
            )
        return ToolResultOk(
            data=HoldingContext(
                ticker=ticker,
                shares=shares,
                cost_basis=cost_basis,
                purchase_date=purchase_date,
                current_price=current_price,
                unrealized_pnl_pct=pnl_pct,
            )
        )
