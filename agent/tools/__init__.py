from agent.tools.adverse_media import GetAdverseMediaTool
from agent.tools.base import Tool
from agent.tools.capital_allocation import GetCapitalAllocationTool
from agent.tools.filings import ReadFilingTool
from agent.tools.financial_strength import GetFinancialStrengthTool
from agent.tools.fundamentals import GetFundamentalsTool
from agent.tools.growth import GetGrowthMetricsTool
from agent.tools.holdings import GetHoldingContextTool
from agent.tools.insider import GetInsiderActivityTool
from agent.tools.intrinsic_value import EstimateIntrinsicValueTool
from agent.tools.news import GetNewsTool
from agent.tools.peers import GetPeerComparisonTool
from agent.tools.persons import GetKeyPersonsTool
from agent.tools.quality import GetQualityMetricsTool
from agent.tools.quote import GetQuoteTool
from agent.tools.screen import ScreenUniverseTool
from agent.tools.valuation import GetValuationMultiplesTool

TOOL_REGISTRY: dict[str, Tool] = {
    "get_adverse_media": GetAdverseMediaTool(),
    "get_quote": GetQuoteTool(),
    "get_fundamentals": GetFundamentalsTool(),
    "get_growth_metrics": GetGrowthMetricsTool(),
    "read_filing": ReadFilingTool(),
    "get_news": GetNewsTool(),
    "screen_universe": ScreenUniverseTool(),
    "get_holding_context": GetHoldingContextTool(),
    "get_valuation_multiples": GetValuationMultiplesTool(),
    "get_quality_metrics": GetQualityMetricsTool(),
    "get_insider_activity": GetInsiderActivityTool(),
    "get_peer_comparison": GetPeerComparisonTool(),
    "get_financial_strength": GetFinancialStrengthTool(),
    "estimate_intrinsic_value": EstimateIntrinsicValueTool(),
    "get_capital_allocation": GetCapitalAllocationTool(),
    "get_key_persons": GetKeyPersonsTool(),
}

TOOL_DEFINITIONS: list[dict[str, object]] = [t.to_api_dict() for t in TOOL_REGISTRY.values()]
