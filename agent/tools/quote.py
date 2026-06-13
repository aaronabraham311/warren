import json
from typing import Any

import yfinance as yf

from agent.budget import RunContext
from agent.tools.base import Tool, ToolResult, ToolResultError, ToolResultOk


class GetQuoteTool(Tool):
    name = "get_quote"
    description = "Get current price, previous close, day change pct, and volume for a ticker."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g. AAPL, MSFT)",
            }
        },
        "required": ["ticker"],
    }

    def run(self, input: dict[str, Any], run_context: RunContext) -> ToolResult:
        ticker = input.get("ticker", "")
        try:
            info = yf.Ticker(ticker).fast_info
            price = info.last_price
            prev_close = info.previous_close
            volume = info.three_month_average_volume

            day_change_pct = (
                round((price - prev_close) / prev_close * 100, 2)
                if prev_close and prev_close != 0
                else None
            )

            return ToolResultOk(
                content=json.dumps(
                    {
                        "ticker": ticker.upper(),
                        "price": round(price, 2) if price else None,
                        "previous_close": round(prev_close, 2) if prev_close else None,
                        "day_change_pct": day_change_pct,
                        "volume": int(volume) if volume else None,
                    }
                )
            )
        except Exception as e:
            return ToolResultError(error=f"Failed to fetch quote for {ticker}: {e}")
