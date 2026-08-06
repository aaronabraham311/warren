from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from data_sources.filing_models import (
    DocumentPage,
    DocumentText,
    ExtractionMethod,
    TranslationStatus,
)
from data_sources.forensic_extraction import extract_forensic_documents
from data_sources.forensics import (
    CapitalReturnEvent,
    CatalystEvidence,
    DebtFacility,
    EvidenceRef,
    HolderPosition,
    LeadershipEvent,
    StakeChange,
)
from storage.models import FilingManifest


def _manifest() -> FilingManifest:
    return FilingManifest(
        filing_id="filing_official_1",
        checksum="a" * 64,
        issuer_isin="ES0105509006",
        venue="bme_growth",
        source_system="bme",
        upstream_id="42",
        document_kind="annual",
        title="Annual report",
        publication_date=date(2026, 4, 30),
        reporting_period_end=date(2025, 12, 31),
        landing_page_url="https://example.com/filing",
        direct_document_url="https://example.com/filing.pdf",
        mime_type="application/pdf",
        byte_length=1,
        retrieved_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        status="extracted",
        artifact_key=f"aa/{'a' * 64}.pdf",
    )


def _document(*pages: str) -> DocumentText:
    return DocumentText(
        filing_id="filing_official_1",
        sha256="a" * 64,
        source_url="https://example.com/filing.pdf",
        retrieved_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        extraction_method=ExtractionMethod.EMBEDDED_TEXT,
        source_language="en",
        pages=[DocumentPage(page_number=index, text=text) for index, text in enumerate(pages, 1)],
        page_count=len(pages),
        original_char_count=sum(map(len, pages)),
        extracted_char_count=sum(map(len, pages)),
    )


def _evidence() -> EvidenceRef:
    return EvidenceRef(
        evidence_id="e1",
        document_id="d1",
        source_url="https://example.com/filing.pdf",
        source_channel="bme",
        document_type="annual",
        published_at=date(2026, 4, 30),
        page=1,
        original_language="es",
        original_excerpt="Accionista: Persona; derechos de voto: 74,99%",
        content_sha256="a" * 64,
        extraction_method="regex",
        confidence=Decimal("0.9"),
    )


def test_naked_or_unscored_forensic_fact_is_rejected() -> None:
    with pytest.raises(ValidationError):
        HolderPosition(
            record_id="holder-1",
            evidence_refs=[],
            field_confidence={},
            holder_raw="Persona",
            as_of=date(2025, 12, 31),
            scope="voting_rights",
        )
    with pytest.raises(ValidationError, match="lack confidence"):
        HolderPosition(
            record_id="holder-1",
            evidence_refs=[_evidence()],
            field_confidence={"holder_raw": Decimal("0.9")},
            holder_raw="Persona",
            as_of=date(2025, 12, 31),
            scope="voting_rights",
        )


def test_stake_delta_requires_comparable_scope_and_exact_decimal_math() -> None:
    common = {
        "record_id": "stake-1",
        "evidence_refs": [_evidence()],
        "field_confidence": {
            "holder_raw": Decimal("1"),
            "prior_as_of": Decimal("1"),
            "current_as_of": Decimal("1"),
            "prior_pct": Decimal("1"),
            "current_pct": Decimal("1"),
            "delta_pct": Decimal("1"),
            "direction": Decimal("1"),
            "cause": Decimal("1"),
            "basis": Decimal("1"),
            "scope": Decimal("1"),
            "change_is_observed": Decimal("1"),
        },
        "holder_raw": "Persona",
        "prior_as_of": date(2024, 12, 31),
        "current_as_of": date(2025, 12, 31),
        "prior_pct": Decimal("5.01"),
        "current_pct": Decimal("7.25"),
        "direction": "increase",
        "basis": "voting",
        "scope": "voting_rights",
        "change_is_observed": True,
    }
    result = StakeChange(**common, delta_pct=Decimal("2.24"))
    assert result.delta_pct == Decimal("2.24")
    with pytest.raises(ValidationError, match="exact Decimal"):
        StakeChange(**common, delta_pct=Decimal("2.23"))
    with pytest.raises(ValidationError, match="comparable"):
        StakeChange(**{**common, "scope": "unknown"}, delta_pct=Decimal("2.24"))


@pytest.mark.parametrize(
    ("phrase", "opinion"),
    [
        ("qualified opinion", "qualified"),
        ("opinión desfavorable", "adverse"),
        ("giudizio senza rilievi", "unmodified"),
        ("odmowa wyrażenia opinii", "disclaimer"),
    ],
)
def test_multilingual_audit_opinions_are_cited(phrase: str, opinion: str) -> None:
    result = extract_forensic_documents([(_manifest(), _document(f"Auditor: {phrase}."))])
    assert result.auditor_history[0].opinion == opinion
    assert result.auditor_history[0].evidence_refs[0].page == 1
    assert phrase in result.auditor_history[0].evidence_refs[0].original_excerpt


def test_authorization_is_not_execution_and_catalyst_strength_is_evidence_based() -> None:
    result = extract_forensic_documents(
        [
            (
                _manifest(),
                _document(
                    "Buyback authorization for 100,000 shares. "
                    "The company signed binding disposal agreement."
                ),
            )
        ]
    )
    assert result.capital_returns[0].kind == "buyback_authorization"
    assert result.capital_returns[0].executed_shares is None
    assert result.catalysts[0].strength == "contractual"


def test_age_alone_never_creates_succession_or_catalyst_evidence() -> None:
    result = extract_forensic_documents(
        [(_manifest(), _document("The founder is 74 years old. No other statement is made."))]
    )
    assert result.leadership_events == []
    assert result.catalysts == []


def test_translated_only_match_cannot_masquerade_as_original_evidence() -> None:
    original = "Operaciones vinculadas descritas en la nota, sin importe en esta página."
    translated = "Related-party service with Example Parent: 10 EUR."
    document = DocumentText(
        filing_id="filing_official_1",
        sha256="a" * 64,
        source_url="https://example.com/filing.pdf",
        retrieved_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        extraction_method=ExtractionMethod.EMBEDDED_TEXT,
        source_language="es",
        pages=[DocumentPage(page_number=1, text=original)],
        english_translation_pages=[DocumentPage(page_number=1, text=translated)],
        translation_status=TranslationStatus.TRANSLATED,
        page_count=1,
        original_char_count=len(original),
        extracted_char_count=len(original),
    )

    result = extract_forensic_documents([(_manifest(), document)])

    assert result.related_party_transactions == []


def test_lock_in_without_parsed_names_never_invents_parties() -> None:
    result = extract_forensic_documents(
        [(_manifest(), _document("A lock-in agreement applies under the admission document."))]
    )

    assert result.agreements == []
    assert any(gap.category == "agreements" for gap in result.gaps)


def test_signed_audit_report_is_not_a_contractual_catalyst() -> None:
    result = extract_forensic_documents(
        [(_manifest(), _document("The directors signed audit report representations."))]
    )

    assert result.catalysts == []


def test_related_party_service_maps_to_typed_services_without_aborting_page() -> None:
    result = extract_forensic_documents(
        [(_manifest(), _document("Related-party service with Parent SA: 10 EUR."))]
    )

    assert result.related_party_transactions[0].transaction_type == "services"
    assert result.related_party_transactions[0].amount == Decimal("10")


@pytest.mark.parametrize(
    ("language", "amount", "expected"),
    [("en", "1,234.56", Decimal("1234.56")), ("es", "1.234,56", Decimal("1234.56"))],
)
def test_grouped_related_party_amounts_follow_source_locale(
    language: str, amount: str, expected: Decimal
) -> None:
    document = _document(f"Related-party service with Parent SA: {amount} EUR.").model_copy(
        update={"source_language": language}
    )

    result = extract_forensic_documents([(_manifest(), document)])

    assert result.related_party_transactions[0].amount == expected


def test_repeated_matches_receive_distinct_evidence_ids() -> None:
    statement = "Related-party service with Parent SA: 10 EUR."

    result = extract_forensic_documents([(_manifest(), _document(f"{statement} {statement}"))])

    first, second = result.related_party_transactions
    assert first.evidence_refs[0].evidence_id != second.evidence_refs[0].evidence_id


def test_ambiguous_number_without_language_fails_closed_as_coverage_gap() -> None:
    document = _document("Buyback authorization for 100,000 shares.").model_copy(
        update={"source_language": None}
    )

    result = extract_forensic_documents([(_manifest(), document)])

    assert result.capital_returns == []
    assert any(gap.reason == "extraction_failed" for gap in result.gaps)


def test_lifecycle_models_reject_contradictory_or_uncited_states() -> None:
    ref = _evidence()
    with pytest.raises(ValidationError, match="signed_date"):
        DebtFacility(
            record_id="debt-1",
            evidence_refs=[ref],
            field_confidence={"facility_type": Decimal("1"), "status": Decimal("1")},
            facility_type="term facility",
            status="signed",
        )
    with pytest.raises(ValidationError, match="cannot carry executed"):
        CapitalReturnEvent(
            record_id="return-1",
            evidence_refs=[ref],
            field_confidence={
                "kind": Decimal("1"),
                "state": Decimal("1"),
                "announcement_date": Decimal("1"),
                "executed_shares": Decimal("1"),
            },
            kind="buyback_authorization",
            state="approved",
            announcement_date=date(2026, 4, 30),
            executed_shares=Decimal("10"),
        )
    with pytest.raises(ValidationError, match="requires aspirational"):
        CatalystEvidence(
            record_id="catalyst-1",
            evidence_refs=[ref],
            field_confidence={
                "catalyst_type": Decimal("1"),
                "stage": Decimal("1"),
                "strength": Decimal("1"),
                "description": Decimal("1"),
            },
            catalyst_type="takeover",
            stage="rumor",
            strength="contractual",
            description="market rumor",
        )
    with pytest.raises(ValidationError, match="issuer plan text"):
        LeadershipEvent(
            record_id="leader-1",
            evidence_refs=[ref],
            field_confidence={
                "person": Decimal("1"),
                "role": Decimal("1"),
                "event": Decimal("1"),
                "announcement_date": Decimal("1"),
                "succession_state": Decimal("1"),
            },
            person="Founder",
            role="Chair",
            event="role_change",
            announcement_date=date(2026, 4, 30),
            succession_state="explicitly_none",
        )
