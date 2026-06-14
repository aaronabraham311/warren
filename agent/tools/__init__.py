from agent.tools.base import Tool
from agent.tools.filings import ReadFilingTool
from agent.tools.fundamentals import GetFundamentalsTool
from agent.tools.growth import GetGrowthMetricsTool
from agent.tools.holdings import GetHoldingContextTool
from agent.tools.news import GetNewsTool
from agent.tools.quality import GetQualityMetricsTool
from agent.tools.quote import GetQuoteTool
from agent.tools.screen import ScreenUniverseTool
from agent.tools.valuation import GetValuationMultiplesTool

TOOL_REGISTRY: dict[str, Tool] = {
    "get_quote": GetQuoteTool(),
    "get_fundamentals": GetFundamentalsTool(),
    "get_growth_metrics": GetGrowthMetricsTool(),
    "read_filing": ReadFilingTool(),
    "get_news": GetNewsTool(),
    "screen_universe": ScreenUniverseTool(),
    "get_holding_context": GetHoldingContextTool(),
    "get_valuation_multiples": GetValuationMultiplesTool(),
    "get_quality_metrics": GetQualityMetricsTool(),
}

TOOL_DEFINITIONS: list[dict[str, object]] = [t.to_api_dict() for t in TOOL_REGISTRY.values()]
