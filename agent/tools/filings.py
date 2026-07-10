import re

from pydantic import BaseModel, Field

from agent.budget import RunContext
from agent.tools._clients import edgar_client, yfinance_client
from agent.tools.base import (
    TICKER_PATTERN,
    Tool,
    ToolResult,
    ToolResultError,
    ToolResultOk,
    error_from_data_source,
)
from data_sources.edgar_client import FilingSection, FilingType, SectionName
from data_sources.errors import DataSourceError

_GROSS_MARGIN_RE = re.compile(
    r"gross\s+(?:profit\s+)?margin[^.]*?(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)


def _extract_gross_margin(text: str) -> float | None:
    m = _GROSS_MARGIN_RE.search(text)
    return float(m.group(1)) if m else None


class ReadFilingInput(BaseModel):
    ticker: str = Field(pattern=TICKER_PATTERN, description="Stock ticker, e.g. AAPL")
    filing_type: FilingType = Field(description="SEC filing form to read")
    section: SectionName = Field(description="Which section of the filing to return")
    fiscal_year: int | None = Field(
        default=None, description="Specific fiscal year; omit for the most recent filing"
    )
    translate: bool = Field(
        default=False,
        description="Set True when the filing is in a foreign language to flag for translation",
    )
    source_language: str | None = Field(
        default=None,
        description="BCP-47 language tag of the filing's original language, e.g. 'ja' for Japanese",
    )


class ReadFilingTool(Tool):
    name = "read_filing"
    description = (
        "Read a section of a company's SEC filing (10-K, 10-Q, 8-K, or DEF 14A). "
        "Sections: business, risk_factors, mdna, financial_statements, executive_summary, "
        "compensation, related_party. Set translate=True for foreign-language filings. "
        "Returns extracted text, key figures, source URL, and any aggregator-vs-filing "
        "discrepancy note."
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

        updates: dict[str, object] = {
            "translate": tool_input.translate,
            "source_language": tool_input.source_language,
        }
        try:
            yf_fund = yfinance_client().get_fundamentals(tool_input.ticker)
            if not isinstance(yf_fund, DataSourceError) and yf_fund.gross_margin_pct is not None:
                filing_margin = _extract_gross_margin(result.text)
                agg_margin = yf_fund.gross_margin_pct
                diff = None if filing_margin is None else abs(filing_margin - agg_margin)
                if diff is not None and diff > 5.0:
                    updates["aggregator_discrepancy_note"] = (
                        f"Gross margin discrepancy: filing shows {filing_margin:.1f}% "
                        f"vs aggregator {yf_fund.gross_margin_pct:.1f}%"
                    )
        except Exception:
            pass

        return ToolResultOk(data=result.model_copy(update=updates))
