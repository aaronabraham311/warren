"""Offline tests for immutable PDF fetch and page-preserving extraction."""

from __future__ import annotations

import io
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pypdf
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from sqlalchemy.orm import Session

from data_sources.errors import DataSourceError
from data_sources.filing_models import (
    DocumentKind,
    DocumentRef,
    ExtractionMethod,
    IssuerIdentity,
    SourceSystem,
    stable_filing_id,
)
from data_sources.pdf_artifacts import (
    FetchedPdf,
    OcrPageText,
    PdfArtifactPipeline,
    PdfTextExtractor,
)
from data_sources.regional_http import HttpBytesDocument, RegionalHttpClient
from storage.artifacts import ArtifactStore
from storage.models import FilingManifest

NOW = datetime(2026, 8, 5, 12, tzinfo=timezone.utc)


def _pdf(*page_texts: str) -> bytes:
    writer = PdfWriter()
    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        if not text:
            continue
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_ref = writer._add_object(font)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def _password_pdf() -> bytes:
    reader = pypdf.PdfReader(io.BytesIO(_pdf("confidential financial report")))
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)
    writer.encrypt("secret")
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def _compressed_content_pdf(*content_streams: bytes) -> bytes:
    writer = PdfWriter()
    for content in content_streams:
        page = writer.add_blank_page(width=612, height=792)
        stream = DecodedStreamObject()
        stream.set_data(content)
        page[NameObject("/Contents")] = writer._add_object(stream.flate_encode())
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def _document() -> DocumentRef:
    identity = IssuerIdentity(canonical_ticker="TEST.MC", venue="BME Growth", isin="ES0000000001")
    return DocumentRef(
        filing_id=stable_filing_id(SourceSystem.BME, identity.venue, identity.isin or "", "42"),
        issuer=identity,
        source_system=SourceSystem.BME,
        upstream_id="42",
        document_kind=DocumentKind.ANNUAL,
        title="Annual report",
        publication_date=date(2025, 12, 31),
        original_language="es",
        landing_page_url="https://www.bmegrowth.es/filing/42",
        direct_document_url="https://www.bmegrowth.es/report.pdf",
        mime_type="application/pdf",
        fetched_at=NOW,
    )


def _http_result(content: bytes, content_type: str = "application/pdf") -> HttpBytesDocument:
    return HttpBytesDocument(
        url="https://www.bmegrowth.es/report.pdf",
        content=content,
        fetched_at=NOW,
        etag='"v1"',
        last_modified="Wed, 05 Aug 2026 12:00:00 GMT",
        content_type=content_type,
        status_code=200,
    )


def test_fetch_verifies_pdf_and_persists_idempotent_manifest(
    tmp_path: Path, db_session: Session
) -> None:
    content = _pdf("The company and the financial report for the year")
    http = MagicMock(spec=RegionalHttpClient)
    http.get_bytes.return_value = _http_result(content)
    pipeline = PdfArtifactPipeline(http, ArtifactStore(tmp_path), session=db_session)

    first = pipeline.fetch(_document())
    second = pipeline.fetch(_document())

    assert isinstance(first, FetchedPdf)
    assert first.artifact.byte_length == len(content)
    assert isinstance(second, FetchedPdf)
    row = db_session.get(FilingManifest, (_document().filing_id, first.artifact.sha256))
    assert row is not None
    assert row.status == "downloaded"
    assert row.artifact_key == first.artifact.relative_key
    assert row.etag == '"v1"'


def test_idempotent_fetch_reconciles_only_null_legacy_discovery_metadata(
    tmp_path: Path, db_session: Session
) -> None:
    content = _pdf("The company and the financial report for the year")
    http = MagicMock(spec=RegionalHttpClient)
    http.get_bytes.return_value = _http_result(content)
    pipeline = PdfArtifactPipeline(http, ArtifactStore(tmp_path), session=db_session)
    first = pipeline.fetch(_document())
    assert isinstance(first, FetchedPdf)
    row = db_session.get(FilingManifest, (_document().filing_id, first.artifact.sha256))
    assert row is not None
    # Simulate a pre-migration row whose newly added nullable fields were not populated.
    for field in ("upstream_id", "document_kind", "title", "publication_date"):
        setattr(row, field, None)
    db_session.flush()

    second = pipeline.fetch(_document())

    assert isinstance(second, FetchedPdf)
    assert row.upstream_id == "42"
    assert row.document_kind == "annual"
    assert row.title == "Annual report"
    assert row.publication_date == date(2025, 12, 31)


def test_idempotent_fetch_rejects_conflicting_immutable_discovery_metadata(
    tmp_path: Path, db_session: Session
) -> None:
    content = _pdf("The company and the financial report for the year")
    http = MagicMock(spec=RegionalHttpClient)
    http.get_bytes.return_value = _http_result(content)
    pipeline = PdfArtifactPipeline(http, ArtifactStore(tmp_path), session=db_session)
    first = pipeline.fetch(_document())
    assert isinstance(first, FetchedPdf)
    row = db_session.get(FilingManifest, (_document().filing_id, first.artifact.sha256))
    assert row is not None
    row.upstream_id = None
    row.document_kind = "governance"
    db_session.flush()

    second = pipeline.fetch(_document())

    assert isinstance(second, DataSourceError)
    assert second.stage == "download"
    assert "document_kind" in second.message
    assert row.upstream_id is None


def test_fetch_rejects_html_disguised_as_pdf_without_storing(tmp_path: Path) -> None:
    http = MagicMock(spec=RegionalHttpClient)
    http.get_bytes.return_value = _http_result(b"<html>blocked</html>", "text/html")
    pipeline = PdfArtifactPipeline(http, ArtifactStore(tmp_path))

    result = pipeline.fetch(_document())

    assert isinstance(result, DataSourceError)
    assert result.stage == "download"
    assert result.source == "bme"
    assert list(tmp_path.rglob("*.pdf")) == []


def test_extraction_preserves_pages_and_selectively_ocrs_sparse_page(tmp_path: Path) -> None:
    content = _pdf("The company and the financial report for the year", "")
    store = ArtifactStore(tmp_path)
    artifact = store.put(content, mime_type="application/pdf")
    calls: list[int] = []

    def ocr(_: bytes, page_number: int) -> OcrPageText:
        calls.append(page_number)
        return OcrPageText("The second page contains financial notes and company details", 0.92)

    extractor = PdfTextExtractor(store, ocr_min_chars=10, ocr_page=ocr)
    result = extractor.extract_stored(
        filing_id=_document().filing_id,
        source_url=_document().direct_document_url,
        source_language="es",
        source="bme",
        artifact=artifact,
        retrieved_at=NOW,
    )

    assert not isinstance(result, DataSourceError)
    assert calls == [2]
    assert [page.page_number for page in result.pages] == [1, 2]
    assert result.extraction_method is ExtractionMethod.OCR
    assert result.source_language == "en"
    assert result.language_detector_confidence is not None
    assert result.ocr_confidence == 0.92
    assert result.truncated is False
    assert result.provenance[1] == f"parser:pypdf/{pypdf.__version__}"
    assert any("differs" in note for note in result.coverage_notes)


def test_selective_ocr_cap_is_explicit_and_page_mapping_remains_complete(
    tmp_path: Path,
) -> None:
    content = _pdf("The company and the financial report for the year", "", "", "")
    store = ArtifactStore(tmp_path)
    artifact = store.put(content, mime_type="application/pdf")
    calls: list[int] = []

    def ocr(_: bytes, page_number: int) -> OcrPageText:
        calls.append(page_number)
        return OcrPageText("Recovered sparse page text")

    result = PdfTextExtractor(
        store, ocr_min_chars=10, ocr_page_limit=1, ocr_page=ocr
    ).extract_stored(
        filing_id="filing_test",
        source_url="https://example.test/report.pdf",
        source_language=None,
        artifact=artifact,
        retrieved_at=NOW,
    )

    assert not isinstance(result, DataSourceError)
    assert calls == [2]
    assert result.page_count == 4
    assert result.truncated is True
    assert any("pages 3, 4" in note for note in result.coverage_notes)


def test_all_sparse_pages_with_failed_ocr_return_typed_error(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    artifact = store.put(_pdf(""), mime_type="application/pdf")

    def failed(_: bytes, page_number: int) -> DataSourceError:
        return DataSourceError(error_code="not_found", message=str(page_number), stage="ocr")

    result = PdfTextExtractor(store, ocr_page=failed).extract_stored(
        filing_id="filing_test",
        source_url="https://example.test/report.pdf",
        source_language=None,
        source="bme",
        artifact=artifact,
        retrieved_at=NOW,
    )

    assert isinstance(result, DataSourceError)
    assert result.stage == "ocr"
    assert result.source == "bme"


def test_page_count_and_uncompressed_content_stream_are_bounded(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    many_pages = store.put(_pdf("one", "two"), mime_type="application/pdf")
    too_many = PdfTextExtractor(store, max_pages=1).extract_stored(
        filing_id="filing_test",
        source_url="https://example.test/report.pdf",
        source_language=None,
        source="bme",
        artifact=many_pages,
        retrieved_at=NOW,
    )
    large_stream = store.put(_pdf("The company financial report"), mime_type="application/pdf")
    too_large = PdfTextExtractor(store, max_page_content_bytes=10).extract_stored(
        filing_id="filing_test",
        source_url="https://example.test/report.pdf",
        source_language=None,
        source="bme",
        artifact=large_stream,
        retrieved_at=NOW,
    )

    assert isinstance(too_many, DataSourceError)
    assert "page limit" in too_many.message
    assert isinstance(too_large, DataSourceError)
    assert "content stream" in too_large.message


def test_compressed_stream_expansion_is_rejected_at_warren_limit(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    compressed_bomb = store.put(
        _compressed_content_pdf(b"q\n" + b" " * 100_000 + b"\nQ"),
        mime_type="application/pdf",
    )

    result = PdfTextExtractor(store, max_page_content_bytes=1_000).extract_stored(
        filing_id="filing_bomb",
        source_url="https://example.test/bomb.pdf",
        source_language=None,
        source="bme",
        artifact=compressed_bomb,
        retrieved_at=NOW,
    )

    assert isinstance(result, DataSourceError)
    assert result.stage == "extract"
    assert "Limit reached" in result.message


def test_cumulative_document_content_cap_rejects_many_individually_small_pages(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    content = b"q\n" + b" " * 700 + b"\nQ"
    artifact = store.put(
        _compressed_content_pdf(content, content),
        mime_type="application/pdf",
    )

    result = PdfTextExtractor(
        store,
        max_page_content_bytes=1_000,
        max_document_content_bytes=1_000,
    ).extract_stored(
        filing_id="filing_cumulative",
        source_url="https://example.test/cumulative.pdf",
        source_language=None,
        source="bme",
        artifact=artifact,
        retrieved_at=NOW,
    )

    assert isinstance(result, DataSourceError)
    assert result.stage == "extract"
    assert "Limit reached" in result.message


def test_corrupt_and_password_protected_pdfs_return_typed_extract_errors(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    corrupt = store.put(b"%PDF-1.7\ncorrupt\n%%EOF", mime_type="application/pdf")
    protected = store.put(_password_pdf(), mime_type="application/pdf")
    extractor = PdfTextExtractor(store)

    corrupt_result = extractor.extract_stored(
        filing_id="filing_corrupt",
        source_url="https://example.test/corrupt.pdf",
        source_language=None,
        source="bme",
        artifact=corrupt,
        retrieved_at=NOW,
    )
    protected_result = extractor.extract_stored(
        filing_id="filing_protected",
        source_url="https://example.test/protected.pdf",
        source_language=None,
        source="bme",
        artifact=protected,
        retrieved_at=NOW,
    )

    assert isinstance(corrupt_result, DataSourceError)
    assert corrupt_result.stage == "extract"
    assert corrupt_result.source == "bme"
    assert isinstance(protected_result, DataSourceError)
    assert "encrypted" in protected_result.message
    assert protected_result.stage == "extract"


def test_multilingual_fixture_detects_language_from_extracted_text(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    artifact = store.put(
        _pdf("La sociedad y el informe financiero de la empresa para el ejercicio"),
        mime_type="application/pdf",
    )

    result = PdfTextExtractor(store, ocr_min_chars=10).extract_stored(
        filing_id="filing_spanish",
        source_url="https://example.test/spanish.pdf",
        source_language=None,
        source="bme",
        artifact=artifact,
        retrieved_at=NOW,
    )

    assert not isinstance(result, DataSourceError)
    assert result.source_language == "es"
    assert result.language_detector_confidence is not None
