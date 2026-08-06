"""Evidence-quality rules shared by fixture recording and offline replay."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel

from agent.tools.base import Tool, ToolResult, ToolResultError, ToolResultOk
from data_sources.edgar_client import FilingSection

NEWS_WINDOWS: Final[tuple[int, ...]] = (7, 14, 30)
FILING_CALLS: Final[tuple[tuple[str, str], ...]] = (
    ("10-K", "business"),
    ("10-K", "risk_factors"),
    ("10-K", "mdna"),
    ("10-K", "financial_statements"),
    ("10-K", "executive_summary"),
    ("10-Q", "financial_statements"),
    ("10-Q", "mdna"),
    ("10-Q", "risk_factors"),
    ("10-Q", "executive_summary"),
    ("8-K", "executive_summary"),
    ("DEF 14A", "compensation"),
    ("DEF 14A", "related_party"),
    ("DEF 14A", "executive_summary"),
)

_MIN_FILING_WORDS = 75
_BOILERPLATE_MARKERS = (
    "table of contents",
    "of this report",
    "incorporated herein by reference",
)

# Mandatory qualitative expectations must have usable source evidence.  Each tuple is an
# any-of group.  These labels stay in the eval layer and are never added to runtime prompts.
CURATED_FILING_CONCEPTS: Final[dict[tuple[str, str, str], tuple[tuple[str, ...], ...]]] = {
    ("SBUX", "10-K", "mdna"): (
        ("comparable sales", "comparable store sales", "same-store sales", "traffic"),
    ),
    ("LUMN", "10-K", "mdna"): (("revenue", "legacy", "decline"),),
    ("V", "10-K", "mdna"): (("payment volume", "cross-border", "payments volume"),),
    ("CVX", "10-K", "business"): (("integrated", "upstream", "downstream"),),
    ("COST", "10-K", "mdna"): (("comparable sales", "membership", "renewal"),),
    ("PYPL", "10-K", "business"): (("branded", "unbranded", "checkout"),),
    ("NVDA", "10-K", "business"): (("hyperscaler", "customer concentration", "cloud"),),
}


def _requested_filing_key(ticker: str, tool_input: BaseModel) -> tuple[str, str, str] | None:
    filing_type = getattr(tool_input, "filing_type", None)
    section = getattr(tool_input, "section", None)
    if not isinstance(filing_type, str) or not isinstance(section, str):
        return None
    return ticker.upper(), filing_type, section


def validate_fixture_result(
    ticker: str,
    tool: Tool,
    tool_input: BaseModel,
    result: ToolResult,
) -> None:
    """Reject successful-but-useless evidence and missing mandatory sources."""

    filing_key = _requested_filing_key(ticker, tool_input) if tool.name == "read_filing" else None
    if isinstance(result, ToolResultError):
        if filing_key in CURATED_FILING_CONCEPTS:
            raise ValueError(f"mandatory filing evidence unavailable for {filing_key}")
        return

    assert isinstance(result, ToolResultOk)
    validated = tool.output_schema.model_validate(result.data.model_dump(mode="json"))
    if tool.name != "read_filing":
        return

    filing = FilingSection.model_validate(validated)
    normalized = " ".join(filing.text.casefold().split())
    markers = [marker for marker in _BOILERPLATE_MARKERS if marker in normalized]
    if filing.word_count < _MIN_FILING_WORDS:
        marker_note = f"; boilerplate markers: {', '.join(markers)}" if markers else ""
        raise ValueError(
            f"unusable filing evidence: {filing.word_count} words (minimum "
            f"{_MIN_FILING_WORDS}){marker_note}"
        )
    if markers and filing.word_count < _MIN_FILING_WORDS * 2:
        raise ValueError(f"unusable filing evidence: cross-reference/TOC fragment ({markers[0]})")

    expected_groups = () if filing_key is None else CURATED_FILING_CONCEPTS.get(filing_key, ())
    for alternatives in expected_groups:
        if not any(concept.casefold() in normalized for concept in alternatives):
            raise ValueError(
                f"filing evidence missing curated concept; expected one of {list(alternatives)}"
            )
