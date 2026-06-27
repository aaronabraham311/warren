from typing import Literal

from pydantic import BaseModel, Field

from agent.budget import RunContext
from agent.tools._clients import ofac_client
from agent.tools.base import (
    Tool,
    ToolResult,
    ToolResultError,
    ToolResultOk,
    error_from_data_source,
)
from data_sources.errors import DataSourceError
from data_sources.ofac_client import WatchlistResult


class ScreenWatchlistsInput(BaseModel):
    entity_name: str = Field(
        min_length=1,
        max_length=200,
        description="Full name of the entity to screen (person, company, vessel, or aircraft).",
    )
    entity_type: Literal["person", "company", "vessel", "aircraft"] = Field(
        default="person",
        description="Type of entity — affects which OFAC search index is queried.",
    )
    country_hint: str | None = Field(
        default=None,
        pattern=r"^[a-z]{2}$",
        description=(
            "ISO 3166-1 alpha-2 country code hint (e.g. 'ru', 'cn'). "
            "Recorded for context; not used by OFAC API directly."
        ),
    )


class ScreenWatchlistsTool(Tool):
    name = "screen_watchlists"
    description = (
        "Check an entity (person, company, vessel, or aircraft) against the OFAC "
        "Specially Designated Nationals (SDN) and consolidated sanctions lists "
        "(US Treasury, free public API). Returns structured matches with match score, "
        "risk category (sanction), and the OFAC programs that flagged the entity. "
        "Coverage is US sanctions only — an empty match list is not proof of clean "
        "status (OFAC is not exhaustive)."
    )
    input_schema = ScreenWatchlistsInput
    output_schema = WatchlistResult

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, ScreenWatchlistsInput)
        client = ofac_client()
        try:
            result = client.search_entity(
                tool_input.entity_name,
                tool_input.entity_type,
                tool_input.country_hint,
            )
        except Exception as exc:
            return ToolResultError(
                error_code="unknown",
                message=f"screen_watchlists failed for {tool_input.entity_name!r}: {exc}",
                retryable=False,
            )
        if isinstance(result, DataSourceError):
            return error_from_data_source(result)
        return ToolResultOk(data=result)
