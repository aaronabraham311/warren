"""Bounded download, immutable storage, and page-preserving PDF extraction."""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pypdf
import pypdf.filters
from pypdf import PdfReader
from pypdf._page import PageObject
from pypdf.errors import LimitReachedError, PdfReadError
from pypdf.generic import ArrayObject, StreamObject
from sqlalchemy import select
from sqlalchemy.orm import Session

from data_sources.errors import DataSourceError, ErrorStage
from data_sources.filing_models import (
    DocumentPage,
    DocumentRef,
    DocumentText,
    ExtractionMethod,
)
from data_sources.regional_http import HttpBytesDocument, RegionalHttpClient
from storage.artifacts import ArtifactIntegrityError, ArtifactStore, StoredArtifact
from storage.models import FilingManifest

EXTRACTION_VERSION = "warren-pdf/1"
_PYPDF_DECOMPRESSION_LOCK = threading.RLock()


@dataclass(frozen=True)
class FetchedPdf:
    """Verified immutable PDF plus the HTTP provenance needed by its manifest."""

    artifact: StoredArtifact
    source_url: str
    retrieved_at: datetime
    etag: str | None
    last_modified: str | None


@dataclass(frozen=True)
class OcrPageText:
    """Text and optional normalized confidence for one OCR'd source page."""

    text: str
    confidence: float | None = None


OcrPage = Callable[[bytes, int], OcrPageText | DataSourceError]


class PdfArtifactPipeline:
    """Fetch official PDFs and preserve a one-to-one source-page text mapping."""

    def __init__(
        self,
        http: RegionalHttpClient,
        store: ArtifactStore,
        *,
        session: Session | None = None,
        max_bytes: int = 25_000_000,
        max_pages: int = 500,
        max_page_content_bytes: int = 10_000_000,
        max_document_content_bytes: int = 50_000_000,
        max_document_text_characters: int = 10_000_000,
        ocr_min_chars: int = 40,
        ocr_page_limit: int = 20,
        ocr_document_timeout_seconds: float = 120.0,
        ocr_page: OcrPage | None = None,
        _monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_bytes < 1 or max_pages < 1 or ocr_page_limit < 0 or ocr_min_chars < 0:
            raise ValueError("PDF byte/page/OCR bounds must be non-negative and non-zero")
        self._http = http
        self._store = store
        self._session = session
        self._max_bytes = max_bytes
        self._max_pages = max_pages
        self._ocr_min_chars = ocr_min_chars
        self._ocr_page_limit = ocr_page_limit
        self._ocr_page = ocr_page or _poppler_tesseract_ocr
        self._extractor = PdfTextExtractor(
            store,
            session=session,
            max_pages=max_pages,
            max_page_content_bytes=max_page_content_bytes,
            max_document_content_bytes=max_document_content_bytes,
            max_document_text_characters=max_document_text_characters,
            ocr_min_chars=ocr_min_chars,
            ocr_page_limit=ocr_page_limit,
            ocr_document_timeout_seconds=ocr_document_timeout_seconds,
            ocr_page=ocr_page,
            _monotonic=_monotonic,
        )

    def fetch(self, document: DocumentRef) -> FetchedPdf | DataSourceError:
        """Download, verify, and content-address one discovered PDF."""
        response = self._http.get_bytes(
            document.direct_document_url,
            max_bytes=self._max_bytes,
        )
        if isinstance(response, DataSourceError):
            return response
        validation_error = _validate_pdf_response(response, document.source_system.value)
        if validation_error is not None:
            return validation_error
        try:
            artifact = self._store.put(response.content, mime_type="application/pdf")
        except (OSError, ValueError) as exc:
            return _error(
                "parse",
                f"PDF artifact could not be stored: {exc}",
                "download",
                document.source_system.value,
            )
        fetched = FetchedPdf(
            artifact=artifact,
            source_url=response.url,
            retrieved_at=response.fetched_at,
            etag=response.etag,
            last_modified=response.last_modified,
        )
        if self._session is not None:
            persisted = persist_filing_manifest(self._session, document, fetched)
            if isinstance(persisted, DataSourceError):
                return persisted
        return fetched

    def extract(self, document: DocumentRef, fetched: FetchedPdf) -> DocumentText | DataSourceError:
        """Extract every source page, OCR'ing only sparse pages within a hard cap."""
        try:
            content = self._store.read(fetched.artifact)
        except (ArtifactIntegrityError, OSError, ValueError) as exc:
            return _error(
                "parse",
                f"stored PDF failed integrity verification: {exc}",
                "extract",
                document.source_system.value,
            )
        return self._extractor.extract_stored(
            filing_id=document.filing_id,
            source_url=fetched.source_url,
            source_language=document.original_language,
            source=document.source_system.value,
            artifact=fetched.artifact,
            retrieved_at=fetched.retrieved_at,
            content=content,
        )

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
        """Extract an already-manifested artifact without reconstructing discovery data."""
        return self._extractor.extract_stored(
            filing_id=filing_id,
            source_url=source_url,
            source_language=source_language,
            source=source,
            artifact=artifact,
            retrieved_at=retrieved_at,
        )

    def fetch_and_extract(self, document: DocumentRef) -> DocumentText | DataSourceError:
        fetched = self.fetch(document)
        if isinstance(fetched, DataSourceError):
            return fetched
        return self.extract(document, fetched)


class PdfTextExtractor:
    """Page-preserving extractor usable independently of the network fetcher."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        session: Session | None = None,
        max_pages: int = 500,
        max_page_content_bytes: int = 10_000_000,
        max_document_content_bytes: int = 50_000_000,
        max_document_text_characters: int = 10_000_000,
        ocr_min_chars: int = 40,
        ocr_page_limit: int = 20,
        ocr_document_timeout_seconds: float = 120.0,
        ocr_page: OcrPage | None = None,
        _monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            max_pages < 1
            or max_page_content_bytes < 1
            or max_document_content_bytes < 1
            or max_document_text_characters < 1
        ):
            raise ValueError("PDF extraction bounds must be positive")
        if ocr_page_limit < 0 or ocr_min_chars < 0:
            raise ValueError("OCR bounds cannot be negative")
        if ocr_document_timeout_seconds <= 0:
            raise ValueError("OCR document timeout must be positive")
        self._store = store
        self._session = session
        self._max_pages = max_pages
        self._max_page_content_bytes = max_page_content_bytes
        self._max_document_content_bytes = max_document_content_bytes
        self._max_document_text_characters = max_document_text_characters
        self._ocr_min_chars = ocr_min_chars
        self._ocr_page_limit = ocr_page_limit
        self._ocr_document_timeout_seconds = ocr_document_timeout_seconds
        self._ocr_page = ocr_page or _poppler_tesseract_ocr
        self._monotonic = _monotonic

    def extract_stored(
        self,
        *,
        filing_id: str,
        source_url: str,
        source_language: str | None,
        artifact: StoredArtifact,
        retrieved_at: datetime,
        source: str | None = None,
        content: bytes | None = None,
    ) -> DocumentText | DataSourceError:
        try:
            pdf_content = content if content is not None else self._store.read(artifact)
        except (ArtifactIntegrityError, OSError, ValueError) as exc:
            return _error(
                "parse", f"stored PDF failed integrity verification: {exc}", "extract", source
            )
        try:
            reader = PdfReader(io.BytesIO(pdf_content), strict=False)
            if reader.is_encrypted and reader.decrypt("") == 0:
                return _error(
                    "parse", "PDF is encrypted and cannot be extracted", "extract", source
                )
            page_count = len(reader.pages)
        except (PdfReadError, OSError, ValueError) as exc:
            return _error("parse", f"invalid PDF: {exc}", "extract", source)
        if page_count == 0:
            return _error("parse", "PDF contains no pages", "extract", source)
        if page_count > self._max_pages:
            return _error(
                "parse",
                f"PDF has {page_count} pages, exceeding the {self._max_pages}-page limit",
                "extract",
                source,
            )

        embedded: list[str] = []
        document_content_bytes = 0
        document_text_characters = 0
        try:
            # pypdf's process-global default allows a much larger zlib expansion than
            # Warren's filing contract. Serialize the temporary override so concurrent
            # Warren extractions cannot observe or restore each other's limit.
            with _pypdf_decompression_limit(self._max_page_content_bytes + 1):
                for page in reader.pages:
                    remaining_document_bytes = (
                        self._max_document_content_bytes - document_content_bytes
                    )
                    if remaining_document_bytes <= 0:
                        return _error(
                            "parse",
                            "PDF content streams exceed the cumulative document extraction limit",
                            "extract",
                            source,
                        )
                    page_content_bytes = _bounded_page_content_size(
                        page,
                        max_bytes=min(
                            self._max_page_content_bytes,
                            remaining_document_bytes,
                        ),
                    )
                    document_content_bytes += page_content_bytes
                    text = (page.extract_text() or "").strip()
                    document_text_characters += len(text)
                    if document_text_characters > self._max_document_text_characters:
                        return _error(
                            "parse",
                            "PDF extracted text exceeds the cumulative document character limit",
                            "extract",
                            source,
                        )
                    embedded.append(text)
        except (LimitReachedError, PdfReadError, OSError, ValueError) as exc:
            return _error("parse", f"PDF text extraction failed: {exc}", "extract", source)

        original_char_count = sum(len(text) for text in embedded)
        sparse_pages = [
            number
            for number, text in enumerate(embedded, start=1)
            if len(text) < self._ocr_min_chars
        ]
        selected = sparse_pages[: self._ocr_page_limit]
        skipped = sparse_pages[self._ocr_page_limit :]
        output = list(embedded)
        confidences: list[float] = []
        ocr_failures: list[int] = []
        ocr_deadline = self._monotonic() + self._ocr_document_timeout_seconds
        deadline_skipped: list[int] = []
        for index, page_number in enumerate(selected):
            if self._monotonic() >= ocr_deadline:
                deadline_skipped = selected[index:]
                break
            ocr_result = self._ocr_page(pdf_content, page_number)
            if isinstance(ocr_result, DataSourceError):
                ocr_failures.append(page_number)
                continue
            ocr_text = ocr_result.text.strip()
            if ocr_text:
                projected_characters = sum(len(text) for text in output)
                projected_characters += len(ocr_text) - len(output[page_number - 1])
                if projected_characters > self._max_document_text_characters:
                    ocr_failures.append(page_number)
                    continue
                output[page_number - 1] = ocr_text
            else:
                ocr_failures.append(page_number)
            if ocr_result.confidence is not None:
                confidences.append(ocr_result.confidence)

        unresolved = sorted(set(skipped + ocr_failures + deadline_skipped))
        extracted_char_count = sum(len(text) for text in output)
        if extracted_char_count == 0:
            stage: ErrorStage = "ocr" if sparse_pages else "extract"
            return _error("parse", "PDF yielded no extractable text", stage, source)

        used_ocr = any(output[index - 1] != embedded[index - 1] for index in selected)
        notes: list[str] = []
        if skipped:
            notes.append(
                f"Selective OCR cap reached; pages {', '.join(map(str, skipped))} were not OCR'd."
            )
        if ocr_failures:
            notes.append(
                f"OCR unavailable or failed for pages {', '.join(map(str, ocr_failures))}."
            )
        if deadline_skipped:
            notes.append(
                "Document OCR deadline reached; pages "
                f"{', '.join(map(str, deadline_skipped))} were not OCR'd."
            )
        if unresolved:
            notes.append(
                "Sparse pages may have incomplete text; consult the immutable PDF artifact."
            )

        detected_language, language_confidence = _detect_language("\n".join(output))
        if detected_language is None:
            notes.append("Source language could not be detected from the extracted text.")
        elif source_language and detected_language != source_language.lower().split("-", 1)[0]:
            notes.append(
                f"Detected source language {detected_language} differs from discovery metadata "
                f"{source_language}."
            )
        document_text = DocumentText(
            filing_id=filing_id,
            sha256=artifact.sha256,
            source_url=source_url,
            retrieved_at=retrieved_at,
            extraction_method=(
                ExtractionMethod.OCR if used_ocr else ExtractionMethod.EMBEDDED_TEXT
            ),
            source_language=detected_language,
            language_detector_confidence=language_confidence,
            pages=[
                DocumentPage(page_number=number, text=text)
                for number, text in enumerate(output, start=1)
            ],
            page_count=page_count,
            original_char_count=original_char_count,
            extracted_char_count=extracted_char_count,
            ocr_confidence=(sum(confidences) / len(confidences) if confidences else None),
            truncated=bool(unresolved),
            coverage_notes=notes,
            provenance=[
                f"artifact:{artifact.relative_key}",
                f"parser:pypdf/{pypdf.__version__}",
                f"extraction:{EXTRACTION_VERSION}",
            ],
        )
        if self._session is not None:
            manifest = self._session.get(FilingManifest, (filing_id, artifact.sha256))
            if manifest is not None:
                extracted_artifact = self._store.put(
                    document_text.model_dump_json().encode("utf-8"),
                    mime_type="application/json",
                )
                manifest.status = "extracted_partial" if document_text.truncated else "extracted"
                manifest.source_language = document_text.source_language
                manifest.parser_version = f"pypdf/{pypdf.__version__}"
                manifest.extraction_version = EXTRACTION_VERSION
                manifest.extracted_text_checksum = extracted_artifact.sha256
                manifest.extracted_text_artifact_key = extracted_artifact.relative_key
                self._session.flush()
        return document_text


def persist_filing_manifest(
    session: Session, document: DocumentRef, fetched: FetchedPdf
) -> FilingManifest | DataSourceError:
    """Idempotently record one immutable artifact version for a discovered filing."""
    key = (document.filing_id, fetched.artifact.sha256)
    existing = session.get(FilingManifest, key)
    if existing is not None:
        if (
            existing.artifact_key != fetched.artifact.relative_key
            or existing.byte_length != fetched.artifact.byte_length
            or existing.mime_type != fetched.artifact.mime_type
        ):
            return _error(
                "parse",
                "existing filing manifest conflicts with stored artifact",
                "download",
                document.source_system.value,
            )
        immutable_discovery = {
            "issuer_isin": document.issuer.isin,
            "venue": document.issuer.venue,
            "source_system": document.source_system.value,
            "upstream_id": document.upstream_id,
            "document_kind": document.document_kind.value,
            "title": document.title,
            "publication_date": document.publication_date,
            "reporting_period_end": document.reporting_period_end,
        }
        for field, discovered_value in immutable_discovery.items():
            existing_value = getattr(existing, field)
            if (
                discovered_value is not None
                and existing_value is not None
                and existing_value != discovered_value
            ):
                return _error(
                    "parse",
                    f"existing filing manifest conflicts with discovery field {field}",
                    "download",
                    document.source_system.value,
                )
        for field, discovered_value in immutable_discovery.items():
            if discovered_value is not None and getattr(existing, field) is None:
                setattr(existing, field, discovered_value)
        session.flush()
        return existing
    previous = session.scalar(
        select(FilingManifest)
        .where(FilingManifest.filing_id == document.filing_id)
        .order_by(FilingManifest.retrieved_at.desc())
        .limit(1)
    )
    manifest = FilingManifest(
        filing_id=document.filing_id,
        checksum=fetched.artifact.sha256,
        issuer_isin=document.issuer.isin,
        venue=document.issuer.venue,
        source_system=document.source_system.value,
        upstream_id=document.upstream_id,
        document_kind=document.document_kind.value,
        title=document.title,
        publication_date=document.publication_date,
        reporting_period_end=document.reporting_period_end,
        landing_page_url=document.landing_page_url,
        direct_document_url=fetched.source_url,
        mime_type=fetched.artifact.mime_type,
        byte_length=fetched.artifact.byte_length,
        retrieved_at=fetched.retrieved_at,
        etag=fetched.etag,
        last_modified=fetched.last_modified,
        status="downloaded",
        source_language=document.original_language,
        parser_version=None,
        extraction_version=None,
        translation_version=None,
        extracted_text_checksum=None,
        extracted_text_artifact_key=None,
        translated_text_checksum=None,
        translated_text_artifact_key=None,
        artifact_key=fetched.artifact.relative_key,
        supersedes_checksum=previous.checksum if previous is not None else None,
    )
    session.add(manifest)
    session.flush()
    return manifest


def _validate_pdf_response(
    response: HttpBytesDocument, source: str | None
) -> DataSourceError | None:
    content_type = (response.content_type or "").split(";", 1)[0].strip().lower()
    if content_type and content_type not in {"application/pdf", "application/octet-stream"}:
        return _error(
            "parse",
            f"attachment is not a PDF (Content-Type {content_type})",
            "download",
            source,
        )
    if not response.content.startswith(b"%PDF-"):
        return _error("parse", "attachment does not have a PDF signature", "download", source)
    if b"%%EOF" not in response.content[-2048:]:
        return _error("parse", "attachment is missing the PDF end marker", "download", source)
    return None


@contextmanager
def _pypdf_decompression_limit(max_output_bytes: int) -> Iterator[None]:
    """Temporarily lower pypdf's zlib ceiling without cross-thread races."""
    with _PYPDF_DECOMPRESSION_LOCK:
        previous = pypdf.filters.ZLIB_MAX_OUTPUT_LENGTH
        pypdf.filters.ZLIB_MAX_OUTPUT_LENGTH = (
            max_output_bytes if previous == 0 else min(previous, max_output_bytes)
        )
        try:
            yield
        finally:
            pypdf.filters.ZLIB_MAX_OUTPUT_LENGTH = previous


def _bounded_page_content_size(page: PageObject, *, max_bytes: int) -> int:
    """Decode page instruction streams one by one, stopping before concatenation."""
    contents_reference = page.get("/Contents")
    if contents_reference is None:
        return 0
    resolved = contents_reference.get_object()
    streams = resolved if isinstance(resolved, ArrayObject) else [resolved]
    total = 0
    for candidate in streams:
        stream = candidate.get_object()
        if not isinstance(stream, StreamObject):
            raise PdfReadError("page /Contents contains a non-stream object")
        remaining = max_bytes - total
        if remaining <= 0:
            raise LimitReachedError("page content stream limit reached")
        with _pypdf_decompression_limit(remaining + 1):
            decoded_size = len(stream.get_data())
        total += decoded_size
        if total > max_bytes:
            raise LimitReachedError("page content stream limit reached")
    return total


def _poppler_tesseract_ocr(content: bytes, page_number: int) -> OcrPageText | DataSourceError:
    pdftoppm = shutil.which("pdftoppm")
    tesseract = shutil.which("tesseract")
    if pdftoppm is None or tesseract is None:
        return _error("not_found", "selective OCR requires Poppler and Tesseract", "ocr")
    try:
        with tempfile.TemporaryDirectory(prefix="warren-ocr-") as directory:
            root = Path(directory)
            source = root / "source.pdf"
            output = root / "page"
            source.write_bytes(content)
            render = subprocess.run(
                [
                    pdftoppm,
                    "-f",
                    str(page_number),
                    "-l",
                    str(page_number),
                    "-singlefile",
                    "-scale-to",
                    "2400",
                    "-png",
                    str(source),
                    str(output),
                ],
                capture_output=True,
                check=False,
                timeout=30,
            )
            if render.returncode != 0:
                return _error("parse", "Poppler could not render a sparse PDF page", "ocr")
            recognized = subprocess.run(
                [tesseract, str(output.with_suffix(".png")), "stdout", "--psm", "6"],
                capture_output=True,
                check=False,
                timeout=30,
            )
            if recognized.returncode != 0:
                return _error("parse", "Tesseract could not OCR a sparse PDF page", "ocr")
            return OcrPageText(recognized.stdout.decode("utf-8", errors="replace").strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _error("network", f"selective OCR process failed: {exc}", "ocr")


_LANGUAGE_MARKERS: dict[str, frozenset[str]] = {
    "en": frozenset({"and", "the", "of", "to", "in", "for", "company", "financial"}),
    "es": frozenset({"de", "la", "el", "y", "en", "para", "sociedad", "financiero"}),
    "it": frozenset({"di", "la", "il", "e", "in", "per", "società", "bilancio"}),
    "pl": frozenset({"i", "w", "z", "na", "do", "spółka", "roku", "finansowe"}),
}


def _detect_language(text: str) -> tuple[str | None, float | None]:
    """Return a conservative filing-language hint without trusting discovery metadata."""
    words = re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)
    if len(words) < 5:
        return None, None
    counts = {
        language: sum(word in markers for word in words)
        for language, markers in _LANGUAGE_MARKERS.items()
    }
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    language, best = ordered[0]
    second = ordered[1][1]
    if best < 2 or best == second:
        return None, None
    confidence = min(1.0, (best - second + 1) / max(best + 1, 1))
    return language, confidence


def _error(
    code: str, message: str, stage: ErrorStage, source: str | None = None
) -> DataSourceError:
    return DataSourceError(
        error_code=code,
        message=message,
        stage=stage,
        source=source or "filing_pdf",
    )
