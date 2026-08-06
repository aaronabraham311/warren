import re
from typing import Literal, cast

from pydantic import BaseModel, Field

from agent.budget import RunContext
from agent.tools._clients import edgar_client, stored_filing_client, yfinance_client
from agent.tools.base import (
    TICKER_PATTERN,
    Tool,
    ToolResult,
    ToolResultError,
    ToolResultOk,
    error_from_data_source,
)
from data_sources.edgar_client import FilingType as EdgarFilingType
from data_sources.errors import DataSourceError
from data_sources.filing_models import FilingSection, TranslationStatus

_GROSS_MARGIN_RE = re.compile(
    r"gross\s+(?:profit\s+)?margin[^.]*?(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)

ReadFilingType = Literal[
    "10-K",
    "10-Q",
    "8-K",
    "DEF 14A",
    "annual",
    "half_year",
    "quarterly",
    "inside_information",
    "shareholder_meeting",
    "governance",
    "other_relevant",
]
ReadFilingSection = Literal[
    "business",
    "risk_factors",
    "mdna",
    "financial_statements",
    "executive_summary",
    "compensation",
    "related_party",
    "full_document",
]
_SEC_KIND_MAP: dict[str, str] = {
    "10-K": "annual",
    "10-Q": "quarterly",
    "8-K": "other_relevant",
    "DEF 14A": "governance",
}
_SEC_TYPES = frozenset(_SEC_KIND_MAP)


def _extract_gross_margin(text: str) -> float | None:
    m = _GROSS_MARGIN_RE.search(text)
    return float(m.group(1)) if m else None


class ReadFilingInput(BaseModel):
    ticker: str = Field(pattern=TICKER_PATTERN, description="Stock ticker, e.g. AAPL")
    filing_type: ReadFilingType = Field(
        description=(
            "SEC form or source-neutral regional kind; 10-K maps to annual and "
            "10-Q maps to quarterly for stored regional PDFs"
        )
    )
    section: ReadFilingSection = Field(
        description="Filing section, or full_document for a stored regional PDF"
    )
    fiscal_year: int | None = Field(
        default=None, description="Specific fiscal year; omit for the most recent filing"
    )
    translate: bool = Field(
        default=False,
        description=(
            "Request English translation for a foreign-language stored filing. "
            "The returned status and coverage warnings report whether translation actually ran."
        ),
    )
    source_language: str | None = Field(
        default=None,
        description="Optional BCP-47 language hint; it does not override detector confidence",
    )


class ReadFilingTool(Tool):
    name = "read_filing"
    description = (
        "Read a filing section from SEC/EDGAR or a verified stored regional PDF. "
        "Sections: business, risk_factors, mdna, financial_statements, executive_summary, "
        "compensation, related_party. Regional PDFs may return bounded full-document text when "
        "section boundaries are unavailable. Set translate=True to request, not assume, English "
        "translation. Returns page citations plus explicit OCR, language, translation, and "
        "coverage limits."
    )
    input_schema = ReadFilingInput
    output_schema = FilingSection

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, ReadFilingInput)
        try:
            result = stored_filing_client().get_filing_section(
                tool_input.ticker,
                tool_input.filing_type,
                tool_input.section,
                tool_input.fiscal_year,
                translate=tool_input.translate,
                document_kind=_SEC_KIND_MAP.get(tool_input.filing_type, tool_input.filing_type),
            )
            if (
                isinstance(result, DataSourceError)
                and result.error_code == "not_found"
                and tool_input.filing_type in _SEC_TYPES
                and tool_input.section != "full_document"
            ):
                result = edgar_client().get_filing_section(
                    tool_input.ticker,
                    cast(EdgarFilingType, tool_input.filing_type),
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

        updates: dict[str, object] = {"translate": tool_input.translate}
        if tool_input.source_language is not None and result.source_language is None:
            updates["source_language"] = tool_input.source_language
        effective_language = result.source_language or tool_input.source_language
        if tool_input.translate and result.translation_status is TranslationStatus.NOT_REQUESTED:
            if effective_language and effective_language.lower().split("-", 1)[0] == "en":
                updates["translation_status"] = TranslationStatus.NOT_NEEDED
                updates["output_language"] = "en"
            else:
                updates["translation_status"] = TranslationStatus.FAILED
                updates["output_language"] = effective_language
                updates["coverage_warnings"] = [
                    *result.coverage_warnings,
                    "English translation was requested but no verified translator ran; "
                    "returned source-language text.",
                ]
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
