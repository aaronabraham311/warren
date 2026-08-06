from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from storage.artifacts import ArtifactIntegrityError, ArtifactStore, StoredArtifact
from storage.models import FilingManifest


def test_artifact_store_is_content_addressed_and_deduplicates(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    first = store.put(b"%PDF-1.7\nprimary evidence", mime_type="application/pdf")
    second = store.put(b"%PDF-1.7\nprimary evidence", mime_type="application/pdf")

    assert first == second
    assert first.relative_key == f"{first.sha256[:2]}/{first.sha256}.pdf"
    assert len(list(tmp_path.rglob("*.pdf"))) == 1
    assert store.read(first) == b"%PDF-1.7\nprimary evidence"


def test_changed_upstream_bytes_create_a_new_immutable_key(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    original = store.put(b"original", mime_type="text/plain")
    amended = store.put(b"amended", mime_type="text/plain")

    assert original.sha256 != amended.sha256
    assert original.relative_key != amended.relative_key
    assert store.read(original) == b"original"
    assert store.read(amended) == b"amended"


def test_artifact_store_rejects_unsupported_mime_and_unsafe_key(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ValueError, match="Unsupported"):
        store.put(b"binary", mime_type="application/octet-stream")

    valid = store.put(b"document", mime_type="text/plain")
    unsafe = StoredArtifact(
        sha256=valid.sha256,
        relative_key="../document.txt",
        byte_length=valid.byte_length,
        mime_type=valid.mime_type,
    )
    with pytest.raises(ArtifactIntegrityError, match="key"):
        store.read(unsafe)


def test_artifact_store_detects_on_disk_corruption(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    artifact = store.put(b"evidence", mime_type="text/plain")
    (tmp_path / artifact.relative_key).write_bytes(b"tampered")

    with pytest.raises(ArtifactIntegrityError, match="checksum mismatch"):
        store.read(artifact)


def test_artifact_store_detects_recorded_size_mismatch(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    artifact = store.put(b"evidence", mime_type="text/plain")
    assert artifact.byte_length is not None
    wrong_size = StoredArtifact(
        sha256=artifact.sha256,
        relative_key=artifact.relative_key,
        byte_length=artifact.byte_length + 1,
        mime_type=artifact.mime_type,
    )

    with pytest.raises(ArtifactIntegrityError, match="byte length mismatch"):
        store.read(wrong_size)


def test_manifest_preserves_versions_and_deduplicated_artifact(
    db_session: Session, tmp_path: Path
) -> None:
    store = ArtifactStore(tmp_path)
    artifact = store.put(b"same attachment", mime_type="application/pdf")
    amended_artifact = store.put(b"corrected attachment", mime_type="application/pdf")
    retrieved = datetime(2026, 8, 5, tzinfo=timezone.utc)
    first = FilingManifest(
        filing_id="newconnect:PLTEST000001:ebi-101",
        checksum=artifact.sha256,
        issuer_isin="PLTEST000001",
        venue="newconnect",
        source_system="ebi",
        landing_page_url="https://example.test/ebi/101",
        direct_document_url="https://example.test/files/report.pdf",
        mime_type=artifact.mime_type,
        byte_length=artifact.byte_length,
        retrieved_at=retrieved,
        etag='"source-v1"',
        last_modified="Wed, 05 Aug 2026 12:00:00 GMT",
        status="downloaded",
        source_language="pl",
        parser_version=None,
        extraction_version=None,
        translation_version=None,
        artifact_key=artifact.relative_key,
        supersedes_checksum=None,
    )
    mirror = FilingManifest(
        filing_id="newconnect:PLTEST000001:espi-202",
        checksum=artifact.sha256,
        issuer_isin="PLTEST000001",
        venue="newconnect",
        source_system="espi_pap",
        landing_page_url="https://example.test/espi/202",
        direct_document_url="https://example.test/files/report.pdf",
        mime_type=artifact.mime_type,
        byte_length=artifact.byte_length,
        retrieved_at=retrieved,
        etag=None,
        last_modified=None,
        status="downloaded",
        source_language="pl",
        parser_version=None,
        extraction_version=None,
        translation_version=None,
        artifact_key=artifact.relative_key,
        supersedes_checksum=None,
    )
    amended = FilingManifest(
        filing_id=first.filing_id,
        checksum=amended_artifact.sha256,
        issuer_isin="PLTEST000001",
        venue="newconnect",
        source_system="ebi",
        landing_page_url="https://example.test/ebi/101-correction",
        direct_document_url="https://example.test/files/report-corrected.pdf",
        mime_type=amended_artifact.mime_type,
        byte_length=amended_artifact.byte_length,
        retrieved_at=retrieved,
        etag='"source-v2"',
        last_modified="Wed, 05 Aug 2026 13:00:00 GMT",
        status="downloaded",
        source_language="pl",
        parser_version="pdf-inspector/0.2.6",
        extraction_version=None,
        translation_version=None,
        artifact_key=amended_artifact.relative_key,
        supersedes_checksum=artifact.sha256,
    )
    db_session.add_all([first, mirror, amended])
    db_session.commit()

    rows = db_session.scalars(select(FilingManifest).order_by(FilingManifest.filing_id)).all()
    assert len(rows) == 3
    assert sum(row.checksum == artifact.sha256 for row in rows) == 2
    assert sum(row.artifact_key == artifact.relative_key for row in rows) == 2
    assert amended.supersedes_checksum == artifact.sha256
    assert amended.issuer_isin == "PLTEST000001"
    assert amended.parser_version == "pdf-inspector/0.2.6"
