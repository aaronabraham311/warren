"""Typed, cited forensic evidence extracted from regional primary filings."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from data_sources.errors import DataSourceError
from data_sources.filing_models import DocumentText
from data_sources.security_master import SecurityMaster
from storage.artifacts import ArtifactIntegrityError, ArtifactStore, StoredArtifact
from storage.models import FilingManifest, ForensicSnapshot

FORENSIC_EXTRACTOR_VERSION = "warren-forensics/1"


class ExtractionMethod(StrEnum):
    XBRL = "xbrl"
    TABLE = "table"
    REGEX = "regex"
    LLM = "llm"
    MANUAL_FIXTURE = "manual_fixture"


class EvidenceStrength(StrEnum):
    CONTRACTUAL = "contractual"
    OBSERVABLE = "observable"
    ASPIRATIONAL = "aspirational"


class CoverageState(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"


class EvidenceRef(BaseModel):
    evidence_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source_url: str = Field(pattern=r"^https://")
    source_channel: str = Field(min_length=1)
    document_type: str = Field(min_length=1)
    published_at: date
    period_end: date | None = None
    page: int = Field(ge=1)
    table_or_section: str | None = None
    original_language: str = Field(min_length=2)
    original_excerpt: str = Field(min_length=1, max_length=2_000)
    translated_excerpt: str | None = Field(default=None, max_length=2_000)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    extraction_method: ExtractionMethod
    confidence: Decimal = Field(ge=0, le=1)
    extractor_version: str = FORENSIC_EXTRACTOR_VERSION


class CitedFact(BaseModel):
    record_id: str = Field(min_length=1)
    evidence_refs: list[EvidenceRef] = Field(min_length=1)
    field_confidence: dict[str, Decimal]

    @model_validator(mode="after")
    def confidence_is_bounded_and_covers_material_fields(self) -> CitedFact:
        if any(value < 0 or value > 1 for value in self.field_confidence.values()):
            raise ValueError("field confidence must be between zero and one")
        populated = {
            name
            for name, value in self.__dict__.items()
            if name not in {"record_id", "evidence_refs", "field_confidence"}
            and value is not None
            and value != []
        }
        missing = populated - self.field_confidence.keys()
        if missing:
            raise ValueError(f"populated forensic fields lack confidence: {sorted(missing)}")
        return self


class HolderPosition(CitedFact):
    holder_raw: str
    holder_canonical: str | None = None
    as_of: date
    shares: Decimal | None = None
    economic_interest_pct: Decimal | None = Field(default=None, ge=0, le=100)
    voting_rights_pct: Decimal | None = Field(default=None, ge=0, le=100)
    direct_pct: Decimal | None = Field(default=None, ge=0, le=100)
    indirect_pct: Decimal | None = Field(default=None, ge=0, le=100)
    scope: Literal["issued_capital", "voting_rights", "free_float", "unknown"]
    holder_type: Literal[
        "founder", "family", "officer", "director", "vehicle", "institutional", "state", "unknown"
    ] = "unknown"
    related_to_issuer: bool | None = None
    ultimate_controller_explicit: bool | None = None


class StakeChange(CitedFact):
    holder_raw: str
    prior_as_of: date | None = None
    current_as_of: date
    prior_shares: Decimal | None = None
    current_shares: Decimal | None = None
    prior_pct: Decimal | None = None
    current_pct: Decimal | None = None
    delta_shares: Decimal | None = None
    delta_pct: Decimal | None = None
    threshold_pct: Decimal | None = None
    direction: Literal["increase", "decrease", "crossed_up", "crossed_down", "unknown"]
    cause: Literal["acquisition", "disposal", "dilution", "unknown"] = "unknown"
    basis: Literal["economic", "voting", "unknown"] = "unknown"
    scope: Literal["issued_capital", "voting_rights", "free_float", "unknown"] = "unknown"
    share_class: str | None = None
    change_is_observed: bool = False

    @model_validator(mode="after")
    def observed_change_is_comparable(self) -> StakeChange:
        shares_comparable = self.prior_shares is not None and self.current_shares is not None
        percentages_comparable = self.prior_pct is not None and self.current_pct is not None
        if self.change_is_observed and (
            self.prior_as_of is None
            or not (shares_comparable or percentages_comparable)
            or self.scope == "unknown"
            or self.basis == "unknown"
        ):
            raise ValueError(
                "observed stake changes require comparable same-unit old/new values, basis, "
                "and scope"
            )
        if self.delta_shares is not None:
            if not shares_comparable:
                raise ValueError("share delta requires comparable prior/current shares")
            if self.delta_shares != self.current_shares - self.prior_shares:  # type: ignore[operator]
                raise ValueError("stake share delta is not exact Decimal arithmetic")
        if self.prior_pct is not None and self.current_pct is not None:
            expected = self.current_pct - self.prior_pct
            if self.delta_pct is not None and self.delta_pct != expected:
                raise ValueError("stake percentage delta is not exact Decimal arithmetic")
        return self


class HolderAgreement(CitedFact):
    agreement_type: Literal[
        "concert_party",
        "voting",
        "transfer_restriction",
        "lock_in",
        "relationship_agreement",
        "other",
    ]
    parties: list[str] = Field(min_length=1)
    aggregate_pct: Decimal | None = None
    affected_shares: Decimal | None = None
    signed_date: date | None = None
    effective_date: date | None = None
    expiry_date: date | None = None
    terms: str | None = None
    termination_trigger: str | None = None
    status: Literal["active", "expired", "terminated", "unknown"] = "unknown"


class RelatedPartyTransaction(CitedFact):
    period_end: date
    counterparty_raw: str
    counterparty_canonical: str | None = None
    relationship: str | None = None
    transaction_type: Literal[
        "purchases",
        "sales",
        "services",
        "loan",
        "guarantee",
        "lease",
        "asset_transfer",
        "compensation",
        "other",
    ]
    amount: Decimal | None = None
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    receivable: Decimal | None = None
    payable: Decimal | None = None
    commitment: Decimal | None = None
    terms: str | None = None
    rate: Decimal | None = None
    maturity: date | None = None
    scope: Literal["consolidated", "separate", "unknown"] = "unknown"
    recurring: bool | None = None
    pct_revenue: Decimal | None = None
    pct_cogs: Decimal | None = None
    pct_assets: Decimal | None = None


class AuditorEvent(CitedFact):
    firm: str | None = None
    signing_partner: str | None = None
    report_period_end: date | None = None
    report_date: date | None = None
    effective_date: date | None = None
    opinion: Literal["unmodified", "qualified", "adverse", "disclaimer", "unknown"] = "unknown"
    going_concern_uncertainty: bool | None = None
    emphasis: list[str] = Field(default_factory=list)
    key_audit_matters: list[str] = Field(default_factory=list)
    event: Literal["appointed", "reappointed", "resigned", "dismissed", "term_expired", "report"]
    stated_reason: str | None = None


class DebtFacility(CitedFact):
    lender: str | None = None
    facility_type: str
    commitment: Decimal | None = None
    outstanding: Decimal | None = None
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    signed_date: date | None = None
    effective_date: date | None = None
    maturity_date: date | None = None
    rate_or_margin: str | None = None
    secured: bool | None = None
    collateral: str | None = None
    covenant_text: str | None = None
    covenant_threshold: Decimal | None = None
    covenant_headroom: Decimal | None = None
    status: Literal[
        "proposed",
        "negotiating",
        "signed",
        "conditions_outstanding",
        "effective",
        "repaid",
        "terminated",
        "unknown",
    ] = "unknown"
    predecessor_record_id: str | None = None

    @model_validator(mode="after")
    def lifecycle_requires_explicit_dates(self) -> DebtFacility:
        if self.status == "signed" and self.signed_date is None:
            raise ValueError("signed debt requires an explicitly cited signed_date")
        if self.status == "effective" and self.effective_date is None:
            raise ValueError("effective debt requires an explicitly cited effective_date")
        return self


class CapitalReturnEvent(CitedFact):
    kind: Literal[
        "dividend",
        "buyback_authorization",
        "buyback_execution",
        "treasury_share_balance",
        "cancellation",
    ]
    state: Literal["proposed", "approved", "paid", "executed", "unknown"]
    announcement_date: date
    record_date: date | None = None
    ex_date: date | None = None
    payment_date: date | None = None
    expiry_date: date | None = None
    per_share: Decimal | None = None
    total_amount: Decimal | None = None
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    authorized_max_shares: Decimal | None = None
    authorized_max_pct: Decimal | None = None
    authorized_max_value: Decimal | None = None
    executed_shares: Decimal | None = None
    executed_value: Decimal | None = None
    remaining_capacity: Decimal | None = None

    @model_validator(mode="after")
    def authorization_and_execution_are_distinct(self) -> CapitalReturnEvent:
        if self.kind == "buyback_authorization":
            if self.state not in {"proposed", "approved"}:
                raise ValueError("buyback authorization cannot claim execution")
            if self.executed_shares is not None or self.executed_value is not None:
                raise ValueError("buyback authorization cannot carry executed amounts")
        if self.kind == "buyback_execution" and self.state != "executed":
            raise ValueError("buyback execution requires executed lifecycle state")
        return self


class LeadershipEvent(CitedFact):
    person: str
    role: str
    event: Literal["appointment", "departure", "role_change"]
    announcement_date: date
    effective_date: date | None = None
    founder_or_family_explicit: bool | None = None
    birth_year: int | None = None
    age_at_source_date: int | None = None
    interim: bool | None = None
    successor: str | None = None
    succession_state: Literal["named", "in_progress", "explicitly_none", "unknown"] = "unknown"
    succession_plan_text: str | None = None
    stated_reason: str | None = None

    @model_validator(mode="after")
    def explicit_no_plan_requires_issuer_language(self) -> LeadershipEvent:
        if self.succession_state == "explicitly_none" and not self.succession_plan_text:
            raise ValueError("explicitly-none succession requires cited issuer plan text")
        return self


class CatalystEvidence(CitedFact):
    catalyst_type: Literal[
        "acquisition",
        "disposal",
        "takeover",
        "delisting",
        "spin_off",
        "dividend",
        "buyback",
        "refinancing",
        "regulatory",
        "succession",
        "accumulation",
        "operational",
        "other",
    ]
    stage: Literal[
        "rumor",
        "intention",
        "strategic_review",
        "board_authorized",
        "signed",
        "conditions_outstanding",
        "completed",
        "terminated",
    ]
    strength: EvidenceStrength
    description: str
    counterparty: str | None = None
    amount: Decimal | None = None
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    expected_by: date | None = None
    long_stop_date: date | None = None
    conditions: list[str] = Field(default_factory=list)
    linked_record_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def stage_and_strength_are_consistent(self) -> CatalystEvidence:
        expected = {
            "rumor": EvidenceStrength.ASPIRATIONAL,
            "intention": EvidenceStrength.ASPIRATIONAL,
            "strategic_review": EvidenceStrength.ASPIRATIONAL,
            "board_authorized": EvidenceStrength.OBSERVABLE,
            "signed": EvidenceStrength.CONTRACTUAL,
            "conditions_outstanding": EvidenceStrength.CONTRACTUAL,
            "completed": EvidenceStrength.OBSERVABLE,
        }
        required = expected.get(self.stage)
        if required is not None and self.strength is not required:
            raise ValueError(f"catalyst stage {self.stage} requires {required.value} strength")
        return self


class CoverageGap(BaseModel):
    category: str
    reason: Literal[
        "missing_document",
        "missing_year",
        "extraction_failed",
        "ocr_partial",
        "translation_partial",
        "not_disclosed",
        "context_limit",
    ]
    document_type: str | None = None
    year: int | None = None
    detail: str


class EvidenceCoverage(BaseModel):
    category_status: dict[str, CoverageState]
    gaps: list[CoverageGap] = Field(default_factory=list)
    documents_considered: int = Field(ge=0)
    documents_extracted: int = Field(ge=0)
    documents_failed: int = Field(ge=0)
    truncated: bool = False


class ForensicEvidenceBundle(BaseModel):
    ticker: str
    venue: str
    as_of: date
    lookback_start: date
    generated_at: datetime
    extractor_version: str = FORENSIC_EXTRACTOR_VERSION
    corpus_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    coverage: EvidenceCoverage
    cap_table: list[HolderPosition] = Field(default_factory=list)
    stake_events: list[StakeChange] = Field(default_factory=list)
    agreements: list[HolderAgreement] = Field(default_factory=list)
    related_party_transactions: list[RelatedPartyTransaction] = Field(default_factory=list)
    auditor_history: list[AuditorEvent] = Field(default_factory=list)
    debt_facilities: list[DebtFacility] = Field(default_factory=list)
    capital_returns: list[CapitalReturnEvent] = Field(default_factory=list)
    leadership_events: list[LeadershipEvent] = Field(default_factory=list)
    catalysts: list[CatalystEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sources: list[EvidenceRef] = Field(default_factory=list)


FORENSIC_CATEGORIES = (
    "cap_table",
    "stake_events",
    "agreements",
    "related_party_transactions",
    "auditor_history",
    "debt_facilities",
    "capital_returns",
    "leadership_events",
    "catalysts",
)


class ForensicEvidenceClient:
    """Build and cache point-in-time evidence from already-extracted G16 artifacts.

    This boundary is deliberately offline: ``refresh`` recomputes the local corpus but
    never falls through to regional network adapters when a filing artifact is absent.
    """

    def __init__(self, session: Session, artifact_store: ArtifactStore) -> None:
        self._session = session
        self._artifacts = artifact_store
        self._security_master = SecurityMaster(session)

    def get_evidence(
        self,
        ticker: str,
        *,
        as_of: date | None = None,
        lookback_years: int = 10,
        refresh: bool = False,
    ) -> ForensicEvidenceBundle | DataSourceError:
        if not 1 <= lookback_years <= 20:
            return DataSourceError(
                error_code="parse",
                message="forensic lookback_years must be between 1 and 20",
                stage="extract",
                source="forensics",
            )
        effective_as_of = as_of or date.today()
        lookback_start = _subtract_years(effective_as_of, lookback_years)
        resolved = self._security_master.resolve(ticker=ticker)
        if isinstance(resolved, DataSourceError):
            return resolved
        available_by = datetime.combine(effective_as_of, time.max, tzinfo=timezone.utc)
        manifests = list(
            self._session.scalars(
                select(FilingManifest)
                .where(
                    FilingManifest.issuer_isin == resolved.isin,
                    FilingManifest.publication_date.is_not(None),
                    FilingManifest.publication_date >= lookback_start,
                    FilingManifest.publication_date <= effective_as_of,
                    # Conservative no-lookahead rule: a version Warren had not retrieved by
                    # the decision date cannot replace the version then available.
                    FilingManifest.retrieved_at <= available_by,
                )
                .order_by(
                    FilingManifest.publication_date.asc(),
                    FilingManifest.retrieved_at.asc(),
                )
            )
        )
        effective = _effective_manifests(manifests)
        # Hash every version in the point-in-time corpus, including corrected versions.
        # Extraction uses only the effective lineage member, but the snapshot retains the
        # existence of the superseded source in its immutable cache key.
        corpus_hash = _corpus_hash(manifests)
        snapshot_key = (
            resolved.canonical_ticker,
            effective_as_of,
            lookback_start,
            FORENSIC_EXTRACTOR_VERSION,
            corpus_hash,
        )
        if not refresh:
            snapshot = self._session.get(ForensicSnapshot, snapshot_key)
            if snapshot is not None:
                try:
                    return ForensicEvidenceBundle.model_validate(snapshot.evidence_json)
                except ValueError as exc:
                    return DataSourceError(
                        error_code="parse",
                        message=f"stored forensic snapshot is invalid: {exc}",
                        stage="extract",
                        source="forensics",
                    )

        documents: list[tuple[FilingManifest, DocumentText]] = []
        gaps: list[CoverageGap] = []
        failed = 0
        for manifest in effective:
            if not manifest.extracted_text_checksum or not manifest.extracted_text_artifact_key:
                failed += 1
                gaps.append(
                    CoverageGap(
                        category="all",
                        reason="missing_document",
                        document_type=manifest.document_kind,
                        year=(
                            manifest.publication_date.year if manifest.publication_date else None
                        ),
                        detail=f"{manifest.filing_id} has no extracted text artifact",
                    )
                )
                continue
            # Prefer the G16 translated DocumentText artifact when present: it embeds both
            # original pages and their page-linked translations. Raw facts still must match
            # original text; translation is supporting context only.
            derived_checksum = manifest.translated_text_checksum or manifest.extracted_text_checksum
            derived_key = (
                manifest.translated_text_artifact_key or manifest.extracted_text_artifact_key
            )
            assert derived_checksum is not None and derived_key is not None
            artifact = StoredArtifact(
                sha256=derived_checksum,
                relative_key=derived_key,
                byte_length=0,
                mime_type="application/json",
            )
            try:
                document = DocumentText.model_validate_json(self._artifacts.read(artifact))
            except (ArtifactIntegrityError, OSError, ValueError) as exc:
                failed += 1
                gaps.append(
                    CoverageGap(
                        category="all",
                        reason="extraction_failed",
                        document_type=manifest.document_kind,
                        year=(
                            manifest.publication_date.year if manifest.publication_date else None
                        ),
                        detail=f"{manifest.filing_id}: {type(exc).__name__}",
                    )
                )
                continue
            mismatch = _document_manifest_mismatch(manifest, document)
            if mismatch is not None:
                failed += 1
                gaps.append(
                    CoverageGap(
                        category="all",
                        reason="extraction_failed",
                        document_type=manifest.document_kind,
                        year=(
                            manifest.publication_date.year if manifest.publication_date else None
                        ),
                        detail=f"{manifest.filing_id}: {mismatch}",
                    )
                )
                continue
            if document.translation_status.value == "partial":
                gaps.append(
                    CoverageGap(
                        category="all",
                        reason="translation_partial",
                        document_type=manifest.document_kind,
                        year=(
                            manifest.publication_date.year if manifest.publication_date else None
                        ),
                        detail=(
                            f"{manifest.filing_id} translation is missing pages "
                            f"{document.translation_missing_pages}"
                        ),
                    )
                )
            documents.append((manifest, document))

        from data_sources.forensic_extraction import extract_forensic_documents

        extracted = extract_forensic_documents(documents)
        gaps.extend(extracted.gaps)
        category_status = {
            category: (CoverageState.PARTIAL if documents else CoverageState.MISSING)
            for category in FORENSIC_CATEGORIES
        }
        for category in FORENSIC_CATEGORIES:
            records = getattr(extracted, category)
            if records:
                # A parsed record proves only that record. Completeness needs a venue/type/
                # year coverage matrix, which is intentionally not inferred from one PDF.
                category_status[category] = CoverageState.PARTIAL
            elif documents:
                gaps.append(
                    CoverageGap(
                        category=category,
                        reason="missing_document",
                        detail=(
                            "No relevant complete source set was available; absence is unknown."
                        ),
                    )
                )
        sources_by_id = {
            ref.evidence_id: ref
            for category in FORENSIC_CATEGORIES
            for fact in getattr(extracted, category)
            for ref in fact.evidence_refs
        }
        generated_at = datetime.now().astimezone()
        warnings = [
            "Forensic extraction is bounded to locally stored primary filings; "
            "missing disclosure remains unknown."
        ]
        if refresh:
            warnings.append(
                "refresh recomputed the local corpus and did not perform network ingestion"
            )
        bundle = ForensicEvidenceBundle(
            ticker=resolved.canonical_ticker,
            venue=resolved.venue,
            as_of=effective_as_of,
            lookback_start=lookback_start,
            generated_at=generated_at,
            corpus_hash=corpus_hash,
            coverage=EvidenceCoverage(
                category_status=category_status,
                gaps=gaps,
                documents_considered=len(effective),
                documents_extracted=len(documents),
                documents_failed=failed,
            ),
            warnings=warnings,
            sources=list(sources_by_id.values()),
            **{category: getattr(extracted, category) for category in FORENSIC_CATEGORIES},
        )
        snapshot = ForensicSnapshot(
            ticker=bundle.ticker,
            as_of=bundle.as_of,
            lookback_start=bundle.lookback_start,
            extractor_version=bundle.extractor_version,
            corpus_hash=bundle.corpus_hash,
            venue=bundle.venue,
            generated_at=bundle.generated_at,
            evidence_json=json.loads(bundle.model_dump_json()),
            coverage_json=json.loads(bundle.coverage.model_dump_json()),
            warnings_json=bundle.warnings,
        )
        try:
            self._session.merge(snapshot)
            self._session.commit()
        except SQLAlchemyError as exc:
            self._session.rollback()
            return DataSourceError(
                error_code="parse",
                message=f"forensic snapshot could not be persisted: {exc}",
                stage="extract",
                source="forensics",
            )
        return bundle


def _subtract_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def _effective_manifests(manifests: list[FilingManifest]) -> list[FilingManifest]:
    # Checksums can legitimately be shared by mirrored filing IDs. Supersession is scoped
    # to one stable document lineage so a correction cannot erase another official source.
    superseded = {
        (item.filing_id, item.supersedes_checksum) for item in manifests if item.supersedes_checksum
    }
    latest: dict[str, FilingManifest] = {}
    for manifest in manifests:
        if (manifest.filing_id, manifest.checksum) in superseded:
            continue
        prior = latest.get(manifest.filing_id)
        if prior is None or manifest.retrieved_at >= prior.retrieved_at:
            latest[manifest.filing_id] = manifest
    return sorted(
        latest.values(), key=lambda item: (item.publication_date or date.min, item.filing_id)
    )


def _corpus_hash(manifests: list[FilingManifest]) -> str:
    rows = [
        {
            "filing_id": item.filing_id,
            "checksum": item.checksum,
            "extracted": item.extracted_text_checksum,
            "translated": item.translated_text_checksum,
            "parser": item.parser_version,
            "extraction": item.extraction_version,
            "translation": item.translation_version,
            "supersedes": item.supersedes_checksum,
        }
        for item in manifests
    ]
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _document_manifest_mismatch(manifest: FilingManifest, document: DocumentText) -> str | None:
    if document.filing_id != manifest.filing_id:
        return "derived text filing_id does not match its manifest"
    if document.sha256 != manifest.checksum:
        return "derived text source checksum does not match the raw filing manifest"
    if document.source_url.rstrip("/") != manifest.direct_document_url.rstrip("/"):
        return "derived text source URL does not match its manifest"
    return None
