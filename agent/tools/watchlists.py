from typing import Literal

from pydantic import BaseModel, Field

from agent.budget import RunContext
from agent.tools._clients import opensanctions_client
from agent.tools.base import (
    Tool,
    ToolResult,
    ToolResultError,
    ToolResultOk,
    error_from_data_source,
)
from data_sources.errors import DataSourceError
from data_sources.opensanctions_client import WatchlistResult


class ScreenWatchlistsInput(BaseModel):
    entity_name: str = Field(
        min_length=1,
        max_length=200,
        description="Full name of the entity to screen (person, company, vessel, or aircraft).",
    )
    entity_type: Literal["person", "company", "vessel", "aircraft"] = Field(
        default="person",
        description="Schema type for the entity. Affects which OpenSanctions properties are matched.",  # noqa: E501
    )
    country_hint: str | None = Field(
        default=None,
        pattern=r"^[a-z]{2}$",
        description=(
            "ISO 3166-1 alpha-2 country code hint (e.g. 'ru', 'cn') to improve match precision."
        ),
    )


class ScreenWatchlistsTool(Tool):
    name = "screen_watchlists"
    description = (
        "Check an entity (person, company, vessel, or aircraft) against the OpenSanctions dataset, "
        "which aggregates hundreds of sanctions lists (OFAC, EU FSF, UN, …), PEP databases, and "
        "criminal-interest indexes. Returns structured matches with match score, risk categories "
        "(sanction, pep, criminal, debarment, other), the datasets that flagged the entity, and "
        "linked entities. An empty match list is not proof of clean status (asymmetry rule)."
    )
    input_schema = ScreenWatchlistsInput
    output_schema = WatchlistResult

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, ScreenWatchlistsInput)
        client = opensanctions_client()
        if client is None:
            return ToolResultError(
                error_code="not_found",
                message=(
                    "OPENSANCTIONS_API_KEY is not configured; watchlist screening is unavailable."
                ),
                retryable=False,
            )
        try:
            result = client.match_entity(
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
