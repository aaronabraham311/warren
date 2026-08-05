"""Source-neutral contracts for discovered and extracted primary filings."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date, datetime
from enum import StrEnum
from typing import Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator

from data_sources.errors import DataSourceError


def _validate_http_url(value: str) -> str:
    """Reject non-web and credential-bearing URLs at the source-neutral boundary."""
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("filing URLs must use HTTP(S) and include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("filing URLs must not contain credentials")
    return value


class SourceSystem(StrEnum):
    BME = "bme"
    BORSA_ITALIANA = "borsa_italiana"
    EURONEXT_LIVE = "euronext_live"
    EBI = "ebi"
    ESPI_PAP = "espi_pap"
    KRS = "krs"
    ISSUER_IR = "issuer_ir"
    EDGAR = "edgar"


class DocumentKind(StrEnum):
    ANNUAL = "annual"
    HALF_YEAR = "half_year"
    QUARTERLY = "quarterly"
    ADMISSION = "admission"
    INSIDE_INFORMATION = "inside_information"
    OTHER_RELEVANT = "other_relevant"
    SHAREHOLDER_MEETING = "shareholder_meeting"
    GOVERNANCE = "governance"
    CORPORATE_ACTION = "corporate_action"
    AUDITOR = "auditor"
    OTHER = "other"


class ExtractionMethod(StrEnum):
    EMBEDDED_TEXT = "embedded_text"
    OCR = "ocr"
    HTML = "html"


class TranslationStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    REQUESTED = "requested"
    NOT_NEEDED = "not_needed"
    TRANSLATED = "translated"
    PARTIAL = "partial"
    FAILED = "failed"


class IssuerIdentity(BaseModel):
    """The stable security identity a filing belongs to."""

    canonical_ticker: str
    venue: str
    isin: str | None = None
    legal_name: str | None = None


def stable_filing_id(
    source_system: SourceSystem | str,
    venue: str,
    isin: str,
    upstream_id: str,
) -> str:
    """Build an opaque ID from source, venue, identity, and upstream document ID.

    EDGAR has no ISIN in its ticker map, so its compatibility adapter supplies the
    canonical ticker as the security identifier. Regional adapters must supply ISIN.
    The source namespace prevents EBI and ESPI/PAP records with the same numeric
    upstream ID from colliding for one issuer and venue.
    """

    source = source_system.value if isinstance(source_system, SourceSystem) else source_system
    parts = (
        source.strip().lower(),
        venue.strip().upper(),
        isin.strip().upper(),
        upstream_id.strip(),
    )
    if not all(parts):
        raise ValueError(
            "source, venue, security identity, and upstream filing ID must be non-empty"
        )
    digest = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]
    return f"filing_{digest}"


class DocumentRef(BaseModel):
    """Discovery metadata for a filing, independent of its archive provider."""

    filing_id: str
    issuer: IssuerIdentity
    source_system: SourceSystem
    upstream_id: str
    document_kind: DocumentKind
    title: str
    publication_date: date
    reporting_period_end: date | None = None
    original_language: str | None = None
    landing_page_url: str
    direct_document_url: str
    mime_type: str | None = None
    attachment_names: list[str] = Field(default_factory=list)
    amended: bool = False
    supersedes_filing_id: str | None = None
    fetched_at: datetime
    etag: str | None = None
    last_modified: str | None = None

    @field_validator("landing_page_url", "direct_document_url")
    @classmethod
    def validate_urls(cls, value: str) -> str:
        return _validate_http_url(value)

    @model_validator(mode="after")
    def filing_id_matches_source_identity(self) -> DocumentRef:
        if self.source_system is not SourceSystem.EDGAR and not self.issuer.isin:
            raise ValueError("regional filing references require an ISIN")
        identity = self.issuer.isin or self.issuer.canonical_ticker
        expected = stable_filing_id(
            self.source_system, self.issuer.venue, identity, self.upstream_id
        )
        if self.filing_id != expected:
            raise ValueError(
                "filing_id must be derived from source, issuer identity, and upstream ID"
            )
        if self.amended and self.supersedes_filing_id is None:
            raise ValueError("an amended filing must identify the filing it supersedes")
        return self


class DocumentPage(BaseModel):
    page_number: int = Field(ge=1)
    text: str


class PageCitation(BaseModel):
    """A public 1-based citation back to one immutable source filing page."""

    filing_id: str | None = None
    page_number: int = Field(ge=1)
    source_url: str

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        return _validate_http_url(value)


class DocumentText(BaseModel):
    """Extracted filing text with page-preserving provenance."""

    filing_id: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_url: str
    retrieved_at: datetime
    extraction_method: ExtractionMethod
    source_language: str | None = None
    language_detector_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    pages: list[DocumentPage]
    english_translation_pages: list[DocumentPage] | None = None
    translation_missing_pages: list[int] = Field(default_factory=list)
    translation_status: TranslationStatus = TranslationStatus.NOT_REQUESTED
    page_count: int = Field(ge=0)
    original_char_count: int = Field(ge=0)
    extracted_char_count: int = Field(ge=0)
    ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    truncated: bool = False
    coverage_notes: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_page_contract(self) -> DocumentText:
        _validate_http_url(self.source_url)
        page_numbers = [page.page_number for page in self.pages]
        if page_numbers != list(range(1, self.page_count + 1)):
            raise ValueError("original pages must be complete, ordered, and 1-based")
        translated = self.english_translation_pages
        if translated is not None:
            translated_numbers = [page.page_number for page in translated]
            if translated_numbers != sorted(set(translated_numbers)):
                raise ValueError("translated pages must be ordered and unique")
            if not set(translated_numbers).issubset(page_numbers):
                raise ValueError("translated pages must correspond to source pages")
            if self.translation_status is TranslationStatus.TRANSLATED:
                if translated_numbers != page_numbers:
                    raise ValueError("complete translation must preserve every source page")
                if self.translation_missing_pages:
                    raise ValueError("complete translation cannot report missing pages")
            elif self.translation_status is TranslationStatus.PARTIAL:
                missing = [number for number in page_numbers if number not in translated_numbers]
                if not translated_numbers or not missing:
                    raise ValueError("partial translation requires translated and missing pages")
                if self.translation_missing_pages != missing:
                    raise ValueError("partial translation must list every missing source page")
                if not self.coverage_notes:
                    raise ValueError("partial translation requires an explicit coverage note")
            else:
                raise ValueError("translated pages require a translated or partial status")
        elif self.translation_status in {
            TranslationStatus.TRANSLATED,
            TranslationStatus.PARTIAL,
        }:
            raise ValueError("translation status cannot imply translation without translated pages")
        elif self.translation_missing_pages:
            raise ValueError("missing translation pages require partial translation status")
        return self


class FilingsArchive(BaseModel):
    """Normalized discovery result and the source's explicit coverage envelope."""

    issuer: IssuerIdentity
    filings: list[DocumentRef]
    coverage_start: date | None = None
    coverage_end: date | None = None
    pages_exhausted: bool
    fetched_at: datetime
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_archive_contract(self) -> FilingsArchive:
        archive_identity = (
            self.issuer.canonical_ticker,
            self.issuer.venue,
            self.issuer.isin,
        )
        if any(
            (
                filing.issuer.canonical_ticker,
                filing.issuer.venue,
                filing.issuer.isin,
            )
            != archive_identity
            for filing in self.filings
        ):
            raise ValueError("every filing must belong to the archive issuer")
        dates = [filing.publication_date for filing in self.filings]
        if dates != sorted(dates, reverse=True):
            raise ValueError("filings must be ordered newest first")
        filing_ids = [filing.filing_id for filing in self.filings]
        if len(filing_ids) != len(set(filing_ids)):
            raise ValueError("filings must not contain duplicate filing IDs")
        if self.coverage_start and self.coverage_end and self.coverage_start > self.coverage_end:
            raise ValueError("coverage_start cannot be after coverage_end")
        if dates:
            if self.coverage_start is None or self.coverage_end is None:
                raise ValueError("non-empty archives require explicit coverage bounds")
            if self.coverage_start > min(dates) or self.coverage_end < max(dates):
                raise ValueError("coverage bounds cannot exclude returned filings")
        return self


class FilingsSource(Protocol):
    """Replaceable regional archive adapter; listing performs no OCR or translation."""

    def list_filings(
        self,
        identity: IssuerIdentity,
        *,
        kinds: Sequence[DocumentKind] | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
        limit: int = 50,
    ) -> FilingsArchive | DataSourceError: ...


class FilingSection(BaseModel):
    """Source-neutral section returned by ``read_filing``.

    ``edgar_url`` remains a read-only compatibility property for existing callers;
    old cached fixtures using that key are accepted during validation.
    """

    ticker: str
    filing_type: str
    section: str
    fiscal_year: int
    filing_date: date
    text: str
    word_count: int
    truncated: bool
    # Default keeps direct construction with the legacy ``edgar_url=`` keyword
    # type-compatible; the before/after validators populate it or reject omission.
    source_url: str = ""
    filing_id: str | None = None
    venue: str | None = None
    source_system: SourceSystem | None = None
    source_language: str | None = None
    output_language: str | None = None
    page_citations: list[PageCitation] = Field(default_factory=list)
    extraction_method: ExtractionMethod | None = None
    translation_status: TranslationStatus = TranslationStatus.NOT_REQUESTED
    coverage_warnings: list[str] = Field(default_factory=list)
    # Legacy request marker; it does not claim that translation occurred.
    translate: bool = False
    key_figures_extracted: list[str] = Field(default_factory=list)
    aggregator_discrepancy_note: str | None = None

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_edgar_url(cls, value: object) -> object:
        if isinstance(value, dict):
            normalized = dict(value)
            if "source_url" not in normalized and "edgar_url" in normalized:
                normalized["source_url"] = normalized["edgar_url"]
            citations = normalized.get("page_citations")
            if isinstance(citations, list) and all(isinstance(item, int) for item in citations):
                normalized["page_citations"] = [
                    {
                        "filing_id": normalized.get("filing_id"),
                        "page_number": item,
                        "source_url": normalized.get("source_url", ""),
                    }
                    for item in citations
                ]
            return normalized
        return value

    @model_validator(mode="after")
    def require_source_url(self) -> FilingSection:
        _validate_http_url(self.source_url)
        for citation in self.page_citations:
            if self.filing_id is not None and citation.filing_id != self.filing_id:
                raise ValueError("citation filing_id must match the section filing_id")
            if citation.source_url != self.source_url:
                raise ValueError("citation source_url must match the section source_url")
        return self

    @property
    def edgar_url(self) -> str:
        """Deprecated EDGAR-specific alias retained for US callers and fixtures."""

        return self.source_url
