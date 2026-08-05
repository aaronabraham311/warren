"""Deterministic, fail-closed forensic extraction from bounded filing pages."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation

from data_sources.filing_models import DocumentText
from data_sources.forensics import (
    FORENSIC_EXTRACTOR_VERSION,
    AuditorEvent,
    CapitalReturnEvent,
    CatalystEvidence,
    CoverageGap,
    DebtFacility,
    EvidenceRef,
    EvidenceStrength,
    ExtractionMethod,
    HolderAgreement,
    HolderPosition,
    LeadershipEvent,
    RelatedPartyTransaction,
    StakeChange,
)
from storage.models import FilingManifest


@dataclass
class ForensicExtractionBatch:
    cap_table: list[HolderPosition] = field(default_factory=list)
    stake_events: list[StakeChange] = field(default_factory=list)
    agreements: list[HolderAgreement] = field(default_factory=list)
    related_party_transactions: list[RelatedPartyTransaction] = field(default_factory=list)
    auditor_history: list[AuditorEvent] = field(default_factory=list)
    debt_facilities: list[DebtFacility] = field(default_factory=list)
    capital_returns: list[CapitalReturnEvent] = field(default_factory=list)
    leadership_events: list[LeadershipEvent] = field(default_factory=list)
    catalysts: list[CatalystEvidence] = field(default_factory=list)
    gaps: list[CoverageGap] = field(default_factory=list)


_AUDIT_OPINIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "disclaimer",
        (
            "disclaimer of opinion",
            "denegación de opinión",
            "impossibilità di esprimere un giudizio",
            "odmowa wyrażenia opinii",
        ),
    ),
    (
        "adverse",
        ("adverse opinion", "opinión desfavorable", "giudizio negativo", "opinia negatywna"),
    ),
    (
        "qualified",
        (
            "qualified opinion",
            "opinión con salvedades",
            "giudizio con rilievi",
            "opinia z zastrzeżeniem",
        ),
    ),
    (
        "unmodified",
        (
            "unmodified opinion",
            "opinión favorable",
            "giudizio senza rilievi",
            "opinia bez zastrzeżeń",
        ),
    ),
)
_HOLDER = re.compile(
    r"(?:holder|accionista|azionista|akcjonariusz)\s*[:\-]\s*(?P<name>[^\n|;]{2,100}).{0,80}?"
    r"(?:voting rights|derechos de voto|diritti di voto|praw głosu)\s*[:\-]?\s*"
    r"(?P<pct>\d{1,3}(?:[.,]\d+)?)\s*%",
    re.IGNORECASE,
)
_RPT = re.compile(
    r"related[- ]party\s+(?P<kind>purchase|sale|service|loan|guarantee|lease)s?\s+with\s+"
    r"(?P<name>[^\n:;]{2,100})\s*[:\-]\s*(?P<amount>\d[\d ,.]*\d|\d)\s*(?P<currency>EUR|PLN|USD)",
    re.IGNORECASE,
)
_DEBT = re.compile(
    r"(?P<status>signed|effective)(?:\s+on)?\s+(?P<date>\d{4}-\d{2}-\d{2})\s+"
    r"(?P<facility>[^\n.;]{2,80}facility).{0,80}?"
    r"(?P<amount>\d[\d ,.]*\d|\d)\s*(?P<currency>EUR|PLN|USD)",
    re.IGNORECASE,
)
_BUYBACK = re.compile(
    r"buyback\s+(?P<state>authorization|authorised|authorized|execution|executed).{0,100}?"
    r"(?P<shares>\d[\d ,.]*\d|\d)\s+shares",
    re.IGNORECASE,
)
_LOCK_IN = re.compile(r"lock[- ]in agreement.{0,300}", re.IGNORECASE)
_CATALYST = re.compile(
    r"(?P<stage>signed(?:\s+binding)?|completed|board authori[sz]ed|"
    r"intends?\s+to|strategic review of)\s+"
    r"(?P<kind>acquisition|disposal|takeover|delisting|spin[- ]off|dividend|buyback|"
    r"refinancing|regulatory decision|succession|accumulation|operational restructuring)\b"
    r"(?P<description>[^\n.]{0,200})",
    re.IGNORECASE,
)


def extract_forensic_documents(
    documents: list[tuple[FilingManifest, DocumentText]],
) -> ForensicExtractionBatch:
    output = ForensicExtractionBatch()
    for manifest, document in documents:
        translated = {
            page.page_number: page.text for page in document.english_translation_pages or []
        }
        for page in document.pages:
            text = page.text
            if not text.strip():
                output.gaps.append(
                    CoverageGap(
                        category="all",
                        reason="ocr_partial",
                        document_type=manifest.document_kind,
                        year=(
                            manifest.publication_date.year if manifest.publication_date else None
                        ),
                        detail=(
                            f"{manifest.filing_id} page {page.page_number} has no extracted text"
                        ),
                    )
                )
                continue
            try:
                _extract_page(
                    output,
                    manifest,
                    document,
                    page.page_number,
                    text,
                    translated.get(page.page_number),
                )
            except (InvalidOperation, ValueError) as exc:
                # One malformed/ambiguous span must not erase safely cited facts from the
                # remaining corpus. Keep partial evidence and expose the failed page.
                output.gaps.append(
                    CoverageGap(
                        category="all",
                        reason="extraction_failed",
                        document_type=manifest.document_kind,
                        year=(
                            manifest.publication_date.year if manifest.publication_date else None
                        ),
                        detail=(
                            f"{manifest.filing_id} page {page.page_number}: {type(exc).__name__}"
                        ),
                    )
                )
    return output


def _extract_page(
    output: ForensicExtractionBatch,
    manifest: FilingManifest,
    document: DocumentText,
    page_number: int,
    text: str,
    translated_text: str | None,
) -> None:
    # Deterministic facts may only be extracted from the original source page. Translation
    # is retained as separately linked context, but it cannot supply a value whose original
    # span cannot be cited.
    searchable = text
    language = document.source_language or manifest.source_language or "und"
    for match in _HOLDER.finditer(searchable):
        pct = _decimal(match.group("pct"), language=language, percentage=True)
        ref = _evidence(
            manifest, document, page_number, text, translated_text, match.group(0), "holder"
        )
        output.cap_table.append(
            HolderPosition(
                record_id=f"holder:{ref.evidence_id}",
                evidence_refs=[ref],
                field_confidence={
                    "holder_raw": Decimal("0.98"),
                    "as_of": Decimal("0.95"),
                    "voting_rights_pct": Decimal("0.98"),
                    "scope": Decimal("0.98"),
                    "holder_type": Decimal("0.50"),
                },
                holder_raw=match.group("name").strip(),
                as_of=manifest.reporting_period_end or manifest.publication_date or date.min,
                voting_rights_pct=pct,
                scope="voting_rights",
            )
        )
    lower = searchable.lower()
    for opinion, phrases in _AUDIT_OPINIONS:
        phrase = next((value for value in phrases if value in lower), None)
        if phrase is None:
            continue
        ref = _evidence(manifest, document, page_number, text, translated_text, phrase, "audit")
        output.auditor_history.append(
            AuditorEvent(
                record_id=f"audit:{ref.evidence_id}",
                evidence_refs=[ref],
                field_confidence={
                    "report_period_end": Decimal("0.95"),
                    "opinion": Decimal("0.99"),
                    "event": Decimal("0.99"),
                },
                report_period_end=manifest.reporting_period_end,
                opinion=opinion,
                event="report",
            )
        )
        break
    for match in _RPT.finditer(searchable):
        kind = match.group("kind").lower()
        transaction_type = {
            "purchase": "purchases",
            "sale": "sales",
            "service": "services",
        }.get(kind, kind)
        ref = _evidence(
            manifest, document, page_number, text, translated_text, match.group(0), "rpt"
        )
        output.related_party_transactions.append(
            RelatedPartyTransaction(
                record_id=f"rpt:{ref.evidence_id}",
                evidence_refs=[ref],
                field_confidence={
                    "period_end": Decimal("0.95"),
                    "counterparty_raw": Decimal("0.95"),
                    "transaction_type": Decimal("0.95"),
                    "amount": Decimal("0.95"),
                    "currency": Decimal("0.99"),
                    "scope": Decimal("0.50"),
                },
                period_end=manifest.reporting_period_end or manifest.publication_date or date.min,
                counterparty_raw=match.group("name").strip(),
                transaction_type=transaction_type,
                amount=_decimal(match.group("amount"), language=language),
                currency=match.group("currency").upper(),
            )
        )
    for match in _DEBT.finditer(searchable):
        ref = _evidence(
            manifest, document, page_number, text, translated_text, match.group(0), "debt"
        )
        lifecycle_date = date.fromisoformat(match.group("date"))
        status = match.group("status").lower()
        output.debt_facilities.append(
            DebtFacility(
                record_id=f"debt:{ref.evidence_id}",
                evidence_refs=[ref],
                field_confidence={
                    "facility_type": Decimal("0.90"),
                    "commitment": Decimal("0.95"),
                    "currency": Decimal("0.99"),
                    "status": Decimal("0.95"),
                    ("signed_date" if status == "signed" else "effective_date"): Decimal("0.95"),
                },
                facility_type=match.group("facility").strip(),
                commitment=_decimal(match.group("amount"), language=language),
                currency=match.group("currency").upper(),
                status=status,
                signed_date=lifecycle_date if status == "signed" else None,
                effective_date=lifecycle_date if status == "effective" else None,
            )
        )
    for match in _BUYBACK.finditer(searchable):
        raw_state = match.group("state").lower()
        execution = raw_state in {"execution", "executed"}
        ref = _evidence(
            manifest, document, page_number, text, translated_text, match.group(0), "return"
        )
        output.capital_returns.append(
            CapitalReturnEvent(
                record_id=f"return:{ref.evidence_id}",
                evidence_refs=[ref],
                field_confidence={
                    "kind": Decimal("0.99"),
                    "state": Decimal("0.99"),
                    "announcement_date": Decimal("0.95"),
                    ("executed_shares" if execution else "authorized_max_shares"): Decimal("0.95"),
                },
                kind="buyback_execution" if execution else "buyback_authorization",
                state="executed" if execution else "approved",
                announcement_date=manifest.publication_date or date.min,
                executed_shares=(
                    _decimal(match.group("shares"), language=language) if execution else None
                ),
                authorized_max_shares=(
                    None if execution else _decimal(match.group("shares"), language=language)
                ),
            )
        )
    lock = _LOCK_IN.search(searchable)
    if lock is not None:
        output.gaps.append(
            CoverageGap(
                category="agreements",
                reason="extraction_failed",
                document_type=manifest.document_kind,
                year=(manifest.publication_date.year if manifest.publication_date else None),
                detail=(
                    f"{manifest.filing_id} page {page_number}: explicit lock-in language "
                    "found but named parties were not safely parsed"
                ),
            )
        )
    for match in _CATALYST.finditer(searchable):
        raw_stage = match.group("stage").lower()
        if raw_stage.startswith("intend"):
            stage, strength = "intention", EvidenceStrength.ASPIRATIONAL
        elif raw_stage.startswith("strategic review"):
            stage, strength = "strategic_review", EvidenceStrength.ASPIRATIONAL
        elif raw_stage.startswith("board authori"):
            stage, strength = "board_authorized", EvidenceStrength.OBSERVABLE
        elif raw_stage == "completed":
            stage, strength = "completed", EvidenceStrength.OBSERVABLE
        else:
            stage, strength = "signed", EvidenceStrength.CONTRACTUAL
        raw_kind = match.group("kind").lower().replace("-", " ")
        catalyst_type = {
            "spin off": "spin_off",
            "regulatory decision": "regulatory",
            "operational restructuring": "operational",
        }.get(raw_kind, raw_kind)
        description = f"{match.group('kind')}{match.group('description')}".strip()
        ref = _evidence(
            manifest, document, page_number, text, translated_text, match.group(0), "catalyst"
        )
        output.catalysts.append(
            CatalystEvidence(
                record_id=f"catalyst:{ref.evidence_id}",
                evidence_refs=[ref],
                field_confidence={
                    "catalyst_type": Decimal("0.50"),
                    "stage": Decimal("0.95"),
                    "strength": Decimal("0.95"),
                    "description": Decimal("0.90"),
                },
                catalyst_type=catalyst_type,
                stage=stage,
                strength=strength,
                description=description,
            )
        )


def _evidence(
    manifest: FilingManifest,
    document: DocumentText,
    page_number: int,
    original_page: str,
    translated_page: str | None,
    matched: str,
    category: str,
) -> EvidenceRef:
    excerpt = _bounded_excerpt(original_page, matched)
    translated_excerpt = _matching_excerpt(translated_page, matched)
    seed = f"{manifest.filing_id}\x1f{document.sha256}\x1f{page_number}\x1f{category}\x1f{matched}"
    return EvidenceRef(
        evidence_id=hashlib.sha256(seed.encode()).hexdigest()[:24],
        document_id=manifest.filing_id,
        source_url=manifest.direct_document_url,
        source_channel=manifest.source_system,
        document_type=manifest.document_kind or "unknown",
        published_at=manifest.publication_date or date.min,
        period_end=manifest.reporting_period_end,
        page=page_number,
        original_language=document.source_language or manifest.source_language or "und",
        original_excerpt=excerpt,
        translated_excerpt=translated_excerpt,
        content_sha256=document.sha256,
        extraction_method=ExtractionMethod.REGEX,
        confidence=Decimal("0.90"),
        extractor_version=FORENSIC_EXTRACTOR_VERSION,
    )


def _bounded_excerpt(page: str, matched: str, limit: int = 600) -> str:
    offset = page.lower().find(matched.lower())
    if offset < 0:
        offset = 0
    start = max(0, offset - 120)
    return page[start : start + limit].strip() or matched[:limit]


def _matching_excerpt(page: str | None, matched: str) -> str | None:
    if page is None or matched.lower() not in page.lower():
        return None
    return _bounded_excerpt(page, matched)


def _decimal(value: str, *, language: str, percentage: bool = False) -> Decimal:
    """Parse a locale-grounded number, rejecting separators with unknown meaning."""

    normalized = value.strip().replace("\u00a0", " ").replace(" ", "")
    base_language = language.lower().split("-", 1)[0]
    if not re.fullmatch(r"\d+(?:[.,]\d+)*", normalized):
        raise ValueError("numeric span contains unsupported characters")

    if base_language == "en":
        if "," in normalized:
            groups = normalized.split(",")
            if not all(len(group) == 3 for group in groups[1:]):
                raise ValueError("ambiguous English thousands separator")
            normalized = "".join(groups)
    elif base_language in {"es", "it", "pl"}:
        if "." in normalized:
            groups = normalized.split(".")
            if not all(len(group) == 3 for group in groups[1:]):
                raise ValueError("ambiguous regional thousands separator")
            normalized = "".join(groups)
        if "," in normalized:
            if normalized.count(",") != 1:
                raise ValueError("ambiguous regional decimal separator")
            normalized = normalized.replace(",", ".")
    elif "," in normalized or "." in normalized:
        if not percentage:
            raise ValueError("numeric separator is ambiguous without source language")
        if normalized.count(",") == 1 and "." not in normalized:
            normalized = normalized.replace(",", ".")
        elif normalized.count(".") != 1 or "," in normalized:
            raise ValueError("percentage separator is ambiguous")
    return Decimal(normalized)
