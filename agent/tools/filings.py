from pydantic import BaseModel, Field

from agent.budget import RunContext
from agent.tools._clients import edgar_client
from agent.tools.base import (
    Tool,
    ToolResult,
    ToolResultError,
    ToolResultOk,
    error_from_data_source,
)
from data_sources.edgar_client import FilingSection, FilingType, SectionName
from data_sources.errors import DataSourceError


class ReadFilingInput(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z]{1,5}$", description="Stock ticker, e.g. AAPL")
    filing_type: FilingType = Field(description="SEC filing form to read")
    section: SectionName = Field(description="Which section of the filing to return")
    fiscal_year: int | None = Field(
        default=None, description="Specific fiscal year; omit for the most recent filing"
    )


class ReadFilingTool(Tool):
    name = "read_filing"
    description = (
        "Read a section of a company's SEC filing (10-K, 10-Q, or 8-K). Sections: "
        "business, risk_factors, mdna, financial_statements, executive_summary. "
        "Returns the extracted text plus the source EDGAR URL."
    )
    input_schema = ReadFilingInput
    output_schema = FilingSection

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, ReadFilingInput)
        try:
            result = edgar_client().get_filing_section(
                tool_input.ticker,
                tool_input.filing_type,
                tool_input.section,
                tool_input.fiscal_year,
            )
        except Exception as exc:
            return ToolResultError(
                error_code="unknown",
                message=f"read_filing failed for {tool_input.ticker}: {exc}",
                retryable=False,
            )
        if isinstance(result, DataSourceError):
            return error_from_data_source(result)
        return ToolResultOk(data=result)
