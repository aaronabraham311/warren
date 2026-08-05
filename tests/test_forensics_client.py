from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from data_sources.filing_models import DocumentPage, DocumentText, ExtractionMethod
from data_sources.forensics import ForensicEvidenceBundle, ForensicEvidenceClient
from storage.artifacts import ArtifactStore
from storage.models import FilingManifest, ForensicSnapshot, SecurityIdentityRecord


def _seed(
    session: Session,
    store: ArtifactStore,
    *,
    filing_id: str,
    publication_date: date,
    text: str,
) -> None:
    raw_checksum = ("b" if filing_id.endswith("1") else "c") * 64
    document = DocumentText(
        filing_id=filing_id,
        sha256=raw_checksum,
        source_url=f"https://example.com/{filing_id}.pdf",
        retrieved_at=datetime.combine(publication_date, datetime.min.time(), tzinfo=timezone.utc),
        extraction_method=ExtractionMethod.EMBEDDED_TEXT,
        source_language="es",
        pages=[DocumentPage(page_number=1, text=text)],
        page_count=1,
        original_char_count=len(text),
        extracted_char_count=len(text),
    )
    extracted = store.put(document.model_dump_json().encode(), mime_type="application/json")
    session.add(
        FilingManifest(
            filing_id=filing_id,
            checksum=raw_checksum,
            issuer_isin="ES0105509006",
            venue="bme_growth",
            source_system="bme",
            upstream_id=filing_id,
            document_kind="annual",
            title="Official annual report",
            publication_date=publication_date,
            reporting_period_end=date(publication_date.year - 1, 12, 31),
            landing_page_url="https://example.com/issuer",
            direct_document_url=document.source_url,
            mime_type="application/pdf",
            byte_length=100,
            retrieved_at=datetime.combine(
                publication_date, datetime.min.time(), tzinfo=timezone.utc
            ),
            status="extracted",
            source_language="es",
            parser_version="pypdf/6.14.2",
            extraction_version="warren-pdf/1",
            extracted_text_checksum=extracted.sha256,
            extracted_text_artifact_key=extracted.relative_key,
            artifact_key=f"bb/{'b' * 64}.pdf",
        )
    )


def test_client_is_point_in_time_offline_and_snapshot_invalidates_with_corpus(
    db_session: Session, tmp_path: Path
) -> None:
    db_session.add(
        SecurityIdentityRecord(
            venue="bme_growth",
            isin="ES0105509006",
            canonical_ticker="480S.MC",
            mic="XGRO",
            exchange_symbol="480S",
            legal_name="SOLUCIONES CUATROOCHENTA, S.A.",
            identity_source_url="https://example.com/security-master",
            resolved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            is_active=True,
        )
    )
    store = ArtifactStore(tmp_path)
    _seed(
        db_session,
        store,
        filing_id="filing_1",
        publication_date=date(2025, 4, 30),
        text="Accionista: Familia Uno; derechos de voto: 74,99%.",
    )
    db_session.flush()
    client = ForensicEvidenceClient(db_session, store)

    first = client.get_evidence("480S.MC", as_of=date(2025, 12, 31))

    assert isinstance(first, ForensicEvidenceBundle)
    assert first.cap_table[0].voting_rights_pct == Decimal("74.99")
    assert first.cap_table[0].evidence_refs[0].document_id == "filing_1"
    assert first.coverage.documents_extracted == 1
    assert db_session.scalar(select(ForensicSnapshot)) is not None

    cached = client.get_evidence("480S.MC", as_of=date(2025, 12, 31))
    assert isinstance(cached, ForensicEvidenceBundle)
    assert cached.corpus_hash == first.corpus_hash

    _seed(
        db_session,
        store,
        filing_id="filing_2",
        publication_date=date(2026, 4, 30),
        text="Buyback execution of 10,000 shares.",
    )
    db_session.flush()
    still_point_in_time = client.get_evidence("480S.MC", as_of=date(2025, 12, 31))
    assert isinstance(still_point_in_time, ForensicEvidenceBundle)
    assert still_point_in_time.corpus_hash == first.corpus_hash

    updated = client.get_evidence("480S.MC", as_of=date(2026, 12, 31))
    assert isinstance(updated, ForensicEvidenceBundle)
    assert updated.corpus_hash != first.corpus_hash
    assert updated.capital_returns[0].kind == "buyback_execution"
