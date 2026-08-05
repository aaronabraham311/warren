"""Offline contract tests for source-neutral filing discovery and text models."""

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from data_sources.filing_models import (
    DocumentKind,
    DocumentPage,
    DocumentRef,
    DocumentText,
    ExtractionMethod,
    FilingsArchive,
    FilingSection,
    IssuerIdentity,
    PageCitation,
    SourceSystem,
    TranslationStatus,
    stable_filing_id,
)


def _document_ref(*, direct_url: str, upstream_id: str = "issuer-doc-42") -> DocumentRef:
    issuer = IssuerIdentity(
        canonical_ticker="DIR.MI",
        venue="Borsa Italiana",
        isin="IT0000000001",
        legal_name="Example S.p.A.",
    )
    return DocumentRef(
        filing_id=stable_filing_id(
            SourceSystem.BORSA_ITALIANA, issuer.venue, issuer.isin or "", upstream_id
        ),
        issuer=issuer,
        source_system=SourceSystem.BORSA_ITALIANA,
        upstream_id=upstream_id,
        document_kind=DocumentKind.ANNUAL,
        title="2025 annual report",
        publication_date=date(2026, 3, 20),
        reporting_period_end=date(2025, 12, 31),
        original_language="it",
        landing_page_url="https://issuer.example/reports",
        direct_document_url=direct_url,
        mime_type="application/pdf",
        attachment_names=["annual-report.pdf"],
        fetched_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
    )


def test_filing_id_is_stable_when_document_url_changes() -> None:
    first = _document_ref(direct_url="https://cdn-one.example/report.pdf")
    moved = _document_ref(direct_url="https://cdn-two.example/new-path.pdf")

    assert first.filing_id == moved.filing_id
    assert first.direct_document_url != moved.direct_document_url
    assert (
        _document_ref(
            direct_url="https://cdn-two.example/new.pdf", upstream_id="issuer-doc-43"
        ).filing_id
        != first.filing_id
    )


def test_filing_id_is_namespaced_by_source_system() -> None:
    common = ("WSE", "PL0000000001", "12345")

    assert stable_filing_id(SourceSystem.EBI, *common) != stable_filing_id(
        SourceSystem.ESPI_PAP, *common
    )


def test_document_ref_rejects_id_not_grounded_in_source_identity() -> None:
    with pytest.raises(ValidationError, match="filing_id must be derived"):
        _document_ref(direct_url="https://issuer.example/report.pdf").model_copy(
            update={"filing_id": "filing_made_up"}
        ).__class__.model_validate(
            {
                **_document_ref(direct_url="https://issuer.example/report.pdf").model_dump(),
                "filing_id": "filing_made_up",
            }
        )


def test_regional_document_ref_requires_isin() -> None:
    document = _document_ref(direct_url="https://issuer.example/report.pdf")
    payload = document.model_dump()
    payload["issuer"] = {**document.issuer.model_dump(), "isin": None}
    payload["filing_id"] = stable_filing_id(
        document.source_system,
        document.issuer.venue,
        document.issuer.canonical_ticker,
        document.upstream_id,
    )

    with pytest.raises(ValidationError, match="regional filing references require an ISIN"):
        DocumentRef.model_validate(payload)


def test_document_text_preserves_one_to_one_translated_pages() -> None:
    document = DocumentText(
        filing_id="filing_abc",
        sha256="a" * 64,
        source_url="https://issuer.example/report.pdf",
        retrieved_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
        extraction_method=ExtractionMethod.OCR,
        source_language="pl",
        language_detector_confidence=0.98,
        pages=[DocumentPage(page_number=1, text="Oryginał")],
        english_translation_pages=[DocumentPage(page_number=1, text="Original")],
        translation_status=TranslationStatus.TRANSLATED,
        page_count=1,
        original_char_count=100,
        extracted_char_count=8,
        ocr_confidence=0.91,
    )

    assert document.english_translation_pages is not None
    assert document.english_translation_pages[0].page_number == document.pages[0].page_number


def test_document_text_never_implies_unrecorded_translation() -> None:
    with pytest.raises(ValidationError, match="without translated pages"):
        DocumentText(
            filing_id="filing_abc",
            sha256="b" * 64,
            source_url="https://issuer.example/report.pdf",
            retrieved_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
            extraction_method=ExtractionMethod.EMBEDDED_TEXT,
            pages=[DocumentPage(page_number=1, text="Tekst")],
            translation_status=TranslationStatus.TRANSLATED,
            page_count=1,
            original_char_count=5,
            extracted_char_count=5,
        )


def test_document_text_records_partial_translation_page_coverage() -> None:
    document = DocumentText(
        filing_id="filing_abc",
        sha256="c" * 64,
        source_url="https://issuer.example/report.pdf",
        retrieved_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
        extraction_method=ExtractionMethod.OCR,
        pages=[
            DocumentPage(page_number=1, text="Pierwsza"),
            DocumentPage(page_number=2, text="Druga"),
            DocumentPage(page_number=3, text="Trzecia"),
        ],
        english_translation_pages=[
            DocumentPage(page_number=1, text="First"),
            DocumentPage(page_number=3, text="Third"),
        ],
        translation_missing_pages=[2],
        translation_status=TranslationStatus.PARTIAL,
        page_count=3,
        original_char_count=20,
        extracted_char_count=20,
        coverage_notes=["Page 2 translation failed."],
    )

    assert [page.page_number for page in document.english_translation_pages or []] == [1, 3]
    assert document.translation_missing_pages == [2]


def test_document_text_rejects_inexact_partial_translation_coverage() -> None:
    with pytest.raises(ValidationError, match="list every missing source page"):
        DocumentText(
            filing_id="filing_abc",
            sha256="d" * 64,
            source_url="https://issuer.example/report.pdf",
            retrieved_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
            extraction_method=ExtractionMethod.OCR,
            pages=[
                DocumentPage(page_number=1, text="Pierwsza"),
                DocumentPage(page_number=2, text="Druga"),
            ],
            english_translation_pages=[DocumentPage(page_number=1, text="First")],
            translation_missing_pages=[],
            translation_status=TranslationStatus.PARTIAL,
            page_count=2,
            original_char_count=13,
            extracted_char_count=13,
            coverage_notes=["Translation incomplete."],
        )


def test_filings_archive_exposes_ordered_coverage_envelope() -> None:
    filing = _document_ref(direct_url="https://issuer.example/report.pdf")
    archive = FilingsArchive(
        issuer=filing.issuer,
        filings=[filing],
        coverage_start=filing.publication_date,
        coverage_end=filing.publication_date,
        pages_exhausted=True,
        fetched_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
    )

    assert archive.filings == [filing]
    assert archive.pages_exhausted is True


def test_filings_archive_rejects_coverage_that_excludes_results() -> None:
    filing = _document_ref(direct_url="https://issuer.example/report.pdf")

    with pytest.raises(ValidationError, match="coverage bounds cannot exclude"):
        FilingsArchive(
            issuer=filing.issuer,
            filings=[filing],
            coverage_start=date(2026, 3, 21),
            coverage_end=date(2026, 3, 21),
            pages_exhausted=False,
            fetched_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
        )


def test_filing_section_accepts_old_edgar_url_fixture() -> None:
    section = FilingSection.model_validate(
        {
            "ticker": "AAPL",
            "filing_type": "10-K",
            "section": "business",
            "fiscal_year": 2025,
            "filing_date": "2025-11-01",
            "text": "Phones",
            "word_count": 1,
            "truncated": False,
            "edgar_url": "https://www.sec.gov/Archives/report.htm",
            "filing_id": "filing_legacy",
            "page_citations": [2],
        }
    )

    assert section.source_url.endswith("report.htm")
    assert section.edgar_url == section.source_url
    assert "source_url" in section.model_dump()
    assert "edgar_url" not in section.model_dump()
    assert section.page_citations == [
        PageCitation(
            filing_id="filing_legacy",
            page_number=2,
            source_url=section.source_url,
        )
    ]
