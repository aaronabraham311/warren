from agent.tools.base import Tool
from agent.tools.quote import GetQuoteTool

TOOL_REGISTRY: dict[str, Tool] = {
    "get_quote": GetQuoteTool(),
}

TOOL_DEFINITIONS: list[dict[str, object]] = [t.to_api_dict() for t in TOOL_REGISTRY.values()]
