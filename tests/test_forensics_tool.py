from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from agent.tools import TOOL_DEFINITIONS, TOOL_REGISTRY
from agent.tools.base import ToolResultError, ToolResultOk
from agent.tools.forensics import GetForensicEvidenceInput, GetForensicEvidenceTool
from data_sources.errors import DataSourceError
from data_sources.forensics import (
    FORENSIC_CATEGORIES,
    CoverageGap,
    CoverageState,
    EvidenceCoverage,
    ForensicEvidenceBundle,
)


def _bundle(ticker: str = "DIR.MI") -> ForensicEvidenceBundle:
    return ForensicEvidenceBundle(
        ticker=ticker,
        venue="XMIL",
        as_of=date(2026, 8, 5),
        lookback_start=date(2016, 8, 5),
        generated_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
        corpus_hash="a" * 64,
        coverage=EvidenceCoverage(
            category_status={category: CoverageState.MISSING for category in FORENSIC_CATEGORIES},
            gaps=[
                CoverageGap(
                    category="cap_table",
                    reason="missing_document",
                    detail="No admission document is stored",
                )
            ],
            documents_considered=0,
            documents_extracted=0,
            documents_failed=0,
        ),
        warnings=["No evidence means unknown, not absence"],
    )


def test_forensic_tool_forwards_point_in_time_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.get_evidence.return_value = _bundle()
    monkeypatch.setattr("agent.tools.forensics.forensic_evidence_client", lambda: client)

    result = GetForensicEvidenceTool().run(
        GetForensicEvidenceInput(
            ticker="DIR.MI", as_of=date(2026, 6, 30), lookback_years=7, refresh=True
        ),
        MagicMock(),
    )

    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, ForensicEvidenceBundle)
    client.get_evidence.assert_called_once_with(
        "DIR.MI", as_of=date(2026, 6, 30), lookback_years=7, refresh=True
    )


def test_forensic_tool_preserves_typed_source_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.get_evidence.return_value = DataSourceError(
        error_code="not_found",
        message="no stored regional corpus",
        stage="discovery",
        source="regional_filings",
    )
    monkeypatch.setattr("agent.tools.forensics.forensic_evidence_client", lambda: client)

    result = GetForensicEvidenceTool().run(GetForensicEvidenceInput(ticker="DIR.MI"), MagicMock())

    assert isinstance(result, ToolResultError)
    assert result.error_code == "not_found"
    assert result.retryable is False
    assert result.stage == "discovery"
    assert result.source == "regional_filings"


@pytest.mark.parametrize("years", [0, 21])
def test_forensic_tool_bounds_lookback(years: int) -> None:
    with pytest.raises(ValidationError):
        GetForensicEvidenceInput(ticker="DIR.MI", lookback_years=years)


def test_forensic_tool_is_registered_with_cited_unknown_semantics() -> None:
    assert "get_forensic_evidence" in TOOL_REGISTRY
    definition = next(item for item in TOOL_DEFINITIONS if item["name"] == "get_forensic_evidence")
    assert "unknown, not false" in str(TOOL_REGISTRY["get_forensic_evidence"].description)
    assert "lookback_years" in str(definition["input_schema"])
