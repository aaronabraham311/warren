import json

from agent.budget import RunContext
from agent.tools.base import Tool, ToolResult, ToolResultError, ToolResultOk
from data_sources.yfinance_client import get_quote


class GetQuoteTool(Tool):
    name = "get_quote"
    description = "Get current price, previous close, day change pct, and volume for a ticker."
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g. AAPL, MSFT)",
            }
        },
        "required": ["ticker"],
    }

    def run(self, input: dict[str, object], run_context: RunContext) -> ToolResult:
        ticker = str(input.get("ticker", ""))
        try:
            quote = get_quote(ticker)
            return ToolResultOk(
                content=json.dumps(
                    {
                        "ticker": quote.ticker,
                        "price": quote.price,
                        "previous_close": quote.previous_close,
                        "day_change_pct": quote.day_change_pct,
                        "volume": quote.volume,
                    }
                )
            )
        except Exception as e:
            return ToolResultError(error=f"Failed to fetch quote for {ticker}: {e}")
