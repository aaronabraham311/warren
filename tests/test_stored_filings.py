from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from data_sources.errors import DataSourceError
from data_sources.filing_models import (
    DocumentPage,
    DocumentText,
    ExtractionMethod,
    FilingSection,
    TranslationStatus,
)
from data_sources.stored_filings import StoredFilingClient
from storage.artifacts import ArtifactStore, StoredArtifact

CHECKSUM = "a" * 64
RETRIEVED = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)


class _Extractor:
    def extract_stored(
        self,
        *,
        filing_id: str,
        source_url: str,
        source_language: str | None,
        artifact: StoredArtifact,
        retrieved_at: datetime,
        source: str | None = None,
    ) -> DocumentText | DataSourceError:
        assert artifact.sha256 == CHECKSUM
        return DocumentText(
            filing_id=filing_id,
            sha256=artifact.sha256,
            source_url=source_url,
            retrieved_at=retrieved_at,
            extraction_method=ExtractionMethod.EMBEDDED_TEXT,
            source_language=source_language,
            language_detector_confidence=0.88,
            pages=[
                DocumentPage(page_number=1, text="Ricavi 2025"),
                DocumentPage(page_number=2, text="Rischi principali"),
            ],
            page_count=2,
            original_char_count=29,
            extracted_char_count=29,
            provenance=[f"source:{source}"],
        )


def _connection(*, with_manifest: bool = True) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE security_identities (
          venue TEXT, isin TEXT, canonical_ticker TEXT, is_active INTEGER, resolved_at TEXT
        );
        CREATE TABLE filing_manifests (
          filing_id TEXT, checksum TEXT, issuer_isin TEXT, venue TEXT, source_system TEXT,
          direct_document_url TEXT, mime_type TEXT, byte_length INTEGER, retrieved_at TEXT,
          source_language TEXT, artifact_key TEXT, created_at TEXT, document_kind TEXT,
          publication_date TEXT, reporting_period_end TEXT, extracted_text_checksum TEXT,
          extracted_text_artifact_key TEXT, translated_text_checksum TEXT,
          translated_text_artifact_key TEXT, parser_version TEXT, extraction_version TEXT,
          translation_version TEXT, status TEXT
        );
        INSERT INTO security_identities VALUES
          ('exgm', 'IT0000000001', 'TEST.MI', 1, '2026-08-05T12:00:00+00:00');
        """
    )
    if with_manifest:
        connection.execute(
            "INSERT INTO filing_manifests "
            "(filing_id, checksum, issuer_isin, venue, source_system, direct_document_url, "
            "mime_type, byte_length, retrieved_at, source_language, artifact_key, created_at, "
            "document_kind, publication_date, reporting_period_end, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "filing_test",
                CHECKSUM,
                "IT0000000001",
                "exgm",
                "borsa_italiana",
                "https://www.borsaitaliana.it/report.pdf",
                "application/pdf",
                123,
                RETRIEVED.isoformat(),
                "it",
                f"aa/{CHECKSUM}.pdf",
                RETRIEVED.isoformat(),
                "annual",
                "2025-04-30",
                "2024-12-31",
                "downloaded",
            ),
        )
        connection.execute(
            "INSERT INTO filing_manifests "
            "(filing_id, checksum, issuer_isin, venue, source_system, direct_document_url, "
            "mime_type, byte_length, retrieved_at, source_language, artifact_key, created_at, "
            "document_kind, publication_date, reporting_period_end, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "filing_governance",
                CHECKSUM,
                "IT0000000001",
                "exgm",
                "borsa_italiana",
                "https://www.borsaitaliana.it/governance.pdf",
                "application/pdf",
                123,
                "2026-08-05T12:00:00+00:00",
                "it",
                f"aa/{CHECKSUM}.pdf",
                "2026-08-05T12:00:00+00:00",
                "governance",
                "2026-08-01",
                None,
                "downloaded",
            ),
        )
    connection.commit()
    return connection


def _client(connection: sqlite3.Connection, tmp_path: Path) -> StoredFilingClient:
    return StoredFilingClient(
        connection,
        ArtifactStore(tmp_path),
        extractor=_Extractor(),
    )


def test_stored_pdf_returns_page_aligned_citations_and_honest_section_warning(
    tmp_path: Path,
) -> None:
    result = _client(_connection(), tmp_path).get_filing_section(
        "TEST.MI", "10-K", "risk_factors", document_kind="annual"
    )

    assert isinstance(result, FilingSection)
    assert result.text == "Ricavi 2025\n\nRischi principali"
    assert [citation.page_number for citation in result.page_citations] == [1, 2]
    assert all(citation.filing_id == "filing_test" for citation in result.page_citations)
    assert result.source_language == "it"
    assert result.output_language == "it"
    assert result.translation_status is TranslationStatus.NOT_REQUESTED
    assert result.filing_id == "filing_test"
    assert result.fiscal_year == 2024
    assert result.filing_date.isoformat() == "2025-04-30"
    assert any("full-document" in warning for warning in result.coverage_warnings)
    assert any("0.88" in warning for warning in result.coverage_warnings)


def test_translation_request_without_provider_returns_original_with_failed_status(
    tmp_path: Path,
) -> None:
    result = _client(_connection(), tmp_path).get_filing_section(
        "TEST.MI", "10-K", "business", translate=True, document_kind="annual"
    )

    assert isinstance(result, FilingSection)
    assert result.translation_status is TranslationStatus.FAILED
    assert result.output_language == "it"
    assert result.text.startswith("Ricavi")
    assert any("no verified translator" in warning for warning in result.coverage_warnings)


def test_missing_manifest_is_typed_discovery_error(tmp_path: Path) -> None:
    result = _client(_connection(with_manifest=False), tmp_path).get_filing_section(
        "TEST.MI", "10-K", "business", document_kind="annual"
    )

    assert isinstance(result, DataSourceError)
    assert result.error_code == "not_found"
    assert result.stage == "discovery"


def test_unknown_regional_identity_is_typed_and_does_not_extract(tmp_path: Path) -> None:
    result = _client(_connection(), tmp_path).get_filing_section(
        "UNKNOWN.MI", "10-K", "business", document_kind="annual"
    )

    assert isinstance(result, DataSourceError)
    assert result.stage == "identity"


def test_selection_uses_exact_kind_and_reporting_year_not_latest_retrieval(
    tmp_path: Path,
) -> None:
    client = _client(_connection(), tmp_path)

    annual = client.get_filing_section(
        "TEST.MI", "annual", "business", fiscal_year=2024, document_kind="annual"
    )
    wrong_year = client.get_filing_section(
        "TEST.MI", "annual", "business", fiscal_year=2026, document_kind="annual"
    )

    assert isinstance(annual, FilingSection)
    assert annual.filing_id == "filing_test"
    assert isinstance(wrong_year, DataSourceError)
    assert wrong_year.error_code == "not_found"
