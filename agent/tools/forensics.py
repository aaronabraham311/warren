"""Agent tool exposing versioned, cited evidence from regional filing corpora."""

from datetime import date

from pydantic import BaseModel, Field

from agent.budget import RunContext
from agent.tools._clients import forensic_evidence_client
from agent.tools.base import (
    TICKER_PATTERN,
    Tool,
    ToolResult,
    ToolResultError,
    ToolResultOk,
    error_from_data_source,
)
from data_sources.errors import DataSourceError
from data_sources.forensics import ForensicEvidenceBundle


class GetForensicEvidenceInput(BaseModel):
    ticker: str = Field(pattern=TICKER_PATTERN, description="Regional issuer ticker")
    as_of: date | None = Field(
        default=None,
        description="Optional point-in-time cutoff; documents published later are excluded",
    )
    lookback_years: int = Field(default=10, ge=1, le=20)
    refresh: bool = Field(
        default=False,
        description="Re-extract even when corpus hash and extractor version match a snapshot",
    )


class GetForensicEvidenceTool(Tool):
    name = "get_forensic_evidence"
    description = (
        "Extract a typed, versioned and cited forensic evidence bundle from stored regional "
        "filings for Milan, Madrid or Warsaw. Returns cap-table and stake history, holder "
        "agreements, related-party transactions, auditor history, debt facilities, capital "
        "returns, leadership events and catalysts. Every fact carries original-language source "
        "location/hash/method/confidence. Empty categories and below-threshold ownership are "
        "unknown, not false; partial corpora return usable evidence plus coverage gaps. Call "
        "after cheapness is confirmed and before control or catalyst conclusions."
    )
    input_schema = GetForensicEvidenceInput
    output_schema = ForensicEvidenceBundle

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, GetForensicEvidenceInput)
        try:
            result = forensic_evidence_client().get_evidence(
                tool_input.ticker,
                as_of=tool_input.as_of,
                lookback_years=tool_input.lookback_years,
                refresh=tool_input.refresh,
            )
        except Exception as exc:
            return ToolResultError(
                error_code="unknown",
                message=f"get_forensic_evidence failed for {tool_input.ticker}: {exc}",
                retryable=False,
                stage="forensic_extraction",
                source="regional_filings",
            )
        if isinstance(result, DataSourceError):
            return error_from_data_source(result)
        return ToolResultOk(data=result)
