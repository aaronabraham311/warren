"""Source-neutral reader for immutable regional filing manifests."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

import pypdf

from data_sources.cache import CacheStore
from data_sources.errors import DataSourceError, ErrorStage
from data_sources.filing_models import (
    DocumentPage,
    DocumentText,
    FilingSection,
    PageCitation,
    SourceSystem,
    TranslationStatus,
)
from data_sources.filing_translation import (
    CachedTranslationStore,
    PageTranslator,
    TranslationLimits,
    translate_document_with_provider,
)
from data_sources.pdf_artifacts import EXTRACTION_VERSION, PdfTextExtractor
from storage.artifacts import ArtifactIntegrityError, ArtifactStore, StoredArtifact

MAX_SECTION_CHARS = 200_000


class StoredPdfExtractor(Protocol):
    def extract_stored(
        self,
        *,
        filing_id: str,
        source_url: str,
        source_language: str | None,
        artifact: StoredArtifact,
        retrieved_at: datetime,
        source: str | None = None,
    ) -> DocumentText | DataSourceError: ...


class StoredFilingClient:
    """Read the latest verified regional PDF already present in the manifest.

    Regional manifests do not yet persist standardized section boundaries. A
    requested section therefore returns bounded full-document text with an explicit
    warning instead of pretending that SEC Item boundaries apply internationally.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        artifact_store: ArtifactStore,
        *,
        extractor: StoredPdfExtractor | None = None,
        translator: PageTranslator | None = None,
        translation_limits: TranslationLimits | None = None,
    ) -> None:
        self._connection = connection
        self._store = artifact_store
        self._extractor = extractor or PdfTextExtractor(artifact_store)
        self._translator = translator
        self._translation_limits = translation_limits
        self._translation_cache = CachedTranslationStore(CacheStore(connection), artifact_store)

    def get_filing_section(
        self,
        ticker: str,
        filing_type: str,
        section: str,
        fiscal_year: int | None = None,
        *,
        translate: bool = False,
        document_kind: str | None = None,
    ) -> FilingSection | DataSourceError:
        try:
            identity = self._connection.execute(
                "SELECT isin, venue FROM security_identities "
                "WHERE canonical_ticker = ? AND is_active = 1 "
                "ORDER BY resolved_at DESC LIMIT 1",
                (ticker.upper(),),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            return _error("not_found", f"regional identity store is unavailable: {exc}", "identity")
        if identity is None:
            return _error("not_found", f"no regional identity for {ticker.upper()}", "identity")
        isin, venue = str(identity[0]), str(identity[1])

        if document_kind is None:
            return _error(
                "parse",
                "regional filing selection requires an explicit source-neutral document kind",
            )
        query = (
            "SELECT filing_id, checksum, source_system, direct_document_url, mime_type, "
            "byte_length, retrieved_at, source_language, artifact_key, publication_date, "
            "reporting_period_end, extracted_text_checksum, extracted_text_artifact_key, "
            "extraction_version FROM filing_manifests WHERE issuer_isin = ? "
            "AND mime_type = 'application/pdf' AND document_kind = ? "
            "AND publication_date IS NOT NULL"
        )
        parameters: list[object] = [isin, document_kind]
        if fiscal_year is not None:
            query += (
                " AND CAST(substr(COALESCE(reporting_period_end, publication_date), 1, 4) "
                "AS INTEGER) = ?"
            )
            parameters.append(fiscal_year)
        query += (
            " ORDER BY reporting_period_end DESC, publication_date DESC, "
            "retrieved_at DESC, created_at DESC LIMIT 1"
        )
        try:
            manifest = self._connection.execute(query, parameters).fetchone()
        except sqlite3.OperationalError as exc:
            return _error("not_found", f"regional filing manifest is unavailable: {exc}")
        if manifest is None:
            year_note = f" for {fiscal_year}" if fiscal_year is not None else ""
            return _error(
                "not_found",
                f"no stored regional PDF filing for {ticker.upper()}{year_note}",
            )

        filing_id = str(manifest[0])
        checksum = str(manifest[1])
        source_value = str(manifest[2])
        source_url = str(manifest[3])
        mime_type = str(manifest[4])
        byte_length = int(manifest[5])
        try:
            retrieved_at = datetime.fromisoformat(str(manifest[6]))
            source_system = SourceSystem(source_value)
            publication_date = datetime.fromisoformat(str(manifest[9])).date()
            reporting_period_end = (
                datetime.fromisoformat(str(manifest[10])).date()
                if manifest[10] is not None
                else None
            )
        except ValueError as exc:
            return _error("parse", f"invalid regional filing manifest: {exc}", source=source_value)
        source_language = str(manifest[7]) if manifest[7] is not None else None
        artifact = StoredArtifact(
            sha256=checksum,
            relative_key=str(manifest[8]),
            byte_length=byte_length,
            mime_type=mime_type,
        )
        cached_document = self._load_extracted_document(
            filing_id=filing_id,
            source_checksum=checksum,
            source_url=source_url,
            extracted_checksum=(str(manifest[11]) if manifest[11] is not None else None),
            extracted_key=(str(manifest[12]) if manifest[12] is not None else None),
            extraction_version=(str(manifest[13]) if manifest[13] is not None else None),
        )
        if isinstance(cached_document, DataSourceError):
            return cached_document
        if cached_document is None:
            document = self._extractor.extract_stored(
                filing_id=filing_id,
                source_url=source_url,
                source_language=source_language,
                source=source_value,
                artifact=artifact,
                retrieved_at=retrieved_at,
            )
            if isinstance(document, DataSourceError):
                return document
            self._persist_extracted_document(document, filing_id=filing_id, checksum=checksum)
        else:
            document = cached_document

        if translate:
            translated = translate_document_with_provider(
                document,
                translator=self._translator,
                cache=self._translation_cache,
                limits=self._translation_limits,
            )
            if isinstance(translated, DataSourceError):
                document = _failed_translation(document, translated.message)
            else:
                document = translated.document
                self._persist_translated_document(
                    document,
                    filing_id=filing_id,
                    checksum=checksum,
                    translator_version=translated.translator_version,
                )

        pages, output_language, language_warning = _output_pages(document, translate=translate)
        text, cited_pages, truncated = _bounded_pages(pages, MAX_SECTION_CHARS)
        warnings = [
            *document.coverage_notes,
            "Regional section boundaries are not standardized; returned bounded "
            f"full-document text for requested section '{section}'.",
        ]
        if language_warning is not None:
            warnings.append(language_warning)
        if document.language_detector_confidence is not None:
            warnings.append(
                "Language detection confidence: "
                f"{document.language_detector_confidence:.2f} "
                f"({document.source_language or 'unknown'})."
            )
        if truncated:
            warnings.append(
                f"Tool output stopped at {MAX_SECTION_CHARS} characters; later pages are omitted."
            )
        citations = [
            PageCitation(filing_id=filing_id, page_number=page.page_number, source_url=source_url)
            for page in cited_pages
        ]
        return FilingSection(
            ticker=ticker.upper(),
            filing_type=filing_type,
            section=section,
            fiscal_year=fiscal_year or (reporting_period_end or publication_date).year,
            filing_date=publication_date,
            text=text,
            word_count=len(text.split()),
            truncated=document.truncated or truncated,
            source_url=source_url,
            filing_id=filing_id,
            venue=str(venue),
            source_system=source_system,
            source_language=document.source_language,
            output_language=output_language,
            page_citations=citations,
            extraction_method=document.extraction_method,
            translation_status=document.translation_status,
            coverage_warnings=list(dict.fromkeys(warnings)),
            translate=translate,
        )

    def _load_extracted_document(
        self,
        *,
        filing_id: str,
        source_checksum: str,
        source_url: str,
        extracted_checksum: str | None,
        extracted_key: str | None,
        extraction_version: str | None,
    ) -> DocumentText | DataSourceError | None:
        if (
            extraction_version != EXTRACTION_VERSION
            or extracted_checksum is None
            or extracted_key is None
        ):
            return None
        artifact = StoredArtifact(
            sha256=extracted_checksum,
            relative_key=extracted_key,
            byte_length=None,
            mime_type="application/json",
        )
        try:
            document = DocumentText.model_validate_json(self._store.read(artifact))
        except (ArtifactIntegrityError, OSError, ValueError) as exc:
            return _error("parse", f"cached extracted filing is invalid: {exc}", "extract")
        if (
            document.filing_id != filing_id
            or document.sha256 != source_checksum
            or document.source_url != source_url
        ):
            return _error(
                "parse", "cached extracted filing provenance does not match its manifest", "extract"
            )
        return document

    def _persist_extracted_document(
        self, document: DocumentText, *, filing_id: str, checksum: str
    ) -> None:
        artifact = self._store.put(
            document.model_dump_json().encode("utf-8"), mime_type="application/json"
        )
        self._connection.execute(
            "UPDATE filing_manifests SET extracted_text_checksum = ?, "
            "extracted_text_artifact_key = ?, source_language = ?, parser_version = ?, "
            "extraction_version = ?, status = ? "
            "WHERE filing_id = ? AND checksum = ?",
            (
                artifact.sha256,
                artifact.relative_key,
                document.source_language,
                f"pypdf/{pypdf.__version__}",
                EXTRACTION_VERSION,
                "extracted_partial" if document.truncated else "extracted",
                filing_id,
                checksum,
            ),
        )
        self._connection.commit()

    def _persist_translated_document(
        self,
        document: DocumentText,
        *,
        filing_id: str,
        checksum: str,
        translator_version: str,
    ) -> None:
        artifact = self._store.put(
            document.model_dump_json().encode("utf-8"), mime_type="application/json"
        )
        self._connection.execute(
            "UPDATE filing_manifests SET translated_text_checksum = ?, "
            "translated_text_artifact_key = ?, translation_version = ?, status = ? "
            "WHERE filing_id = ? AND checksum = ?",
            (
                artifact.sha256,
                artifact.relative_key,
                translator_version,
                f"translation_{document.translation_status.value}",
                filing_id,
                checksum,
            ),
        )
        self._connection.commit()


def _failed_translation(document: DocumentText, message: str) -> DocumentText:
    value = document.model_copy(
        update={
            "english_translation_pages": None,
            "translation_missing_pages": [page.page_number for page in document.pages],
            "translation_status": TranslationStatus.FAILED,
            "coverage_notes": [
                *document.coverage_notes,
                f"Translation failed because no verified translator ran: {message}.",
            ],
        }
    )
    return DocumentText.model_validate(value.model_dump())


def _output_pages(
    document: DocumentText, *, translate: bool
) -> tuple[Sequence[DocumentPage], str | None, str | None]:
    if not translate:
        return document.pages, document.source_language, None
    translated = {page.page_number: page for page in document.english_translation_pages or []}
    if document.translation_status is TranslationStatus.TRANSLATED:
        return list(translated.values()), "en", None
    if document.translation_status is TranslationStatus.NOT_NEEDED:
        return document.pages, document.source_language or "en", None
    if document.translation_status is TranslationStatus.PARTIAL:
        pages = [translated.get(page.page_number, page) for page in document.pages]
        return (
            pages,
            "mixed",
            "Translation is partial; untranslated pages remain in source language.",
        )
    return (
        document.pages,
        document.source_language,
        "English translation was requested but unavailable; returned source-language text.",
    )


def _bounded_pages(
    pages: Sequence[DocumentPage], max_characters: int
) -> tuple[str, list[DocumentPage], bool]:
    chunks: list[str] = []
    cited: list[DocumentPage] = []
    used = 0
    truncated = False
    for page in pages:
        separator = "\n\n" if chunks else ""
        remaining = max_characters - used - len(separator)
        if remaining <= 0:
            truncated = True
            break
        text = page.text
        if len(text) > remaining:
            text = text[:remaining]
            truncated = True
        chunks.append(f"{separator}{text}")
        cited.append(page)
        used += len(separator) + len(text)
        if truncated:
            break
    return "".join(chunks), cited, truncated


def _error(
    code: str,
    message: str,
    stage: ErrorStage = "discovery",
    *,
    source: str = "filing_manifest",
) -> DataSourceError:
    return DataSourceError(error_code=code, message=message, stage=stage, source=source)
