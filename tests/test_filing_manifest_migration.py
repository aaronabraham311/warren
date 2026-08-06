from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

import storage.engine as storage_engine

_PREVIOUS_REVISION = "4d6f8a1b2c3d"


def test_filing_manifest_migration_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    database = tmp_path / "migration.db"
    monkeypatch.setenv("WARREN_DB", str(database))
    monkeypatch.setattr(storage_engine, "engine", None)
    config = Config(str(Path(__file__).parents[1] / "alembic.ini"))

    command.upgrade(config, "head")
    engine = storage_engine.get_engine()
    upgraded = inspect(engine)
    assert "filing_manifests" in upgraded.get_table_names()
    assert {column["name"] for column in upgraded.get_columns("filing_manifests")} >= {
        "filing_id",
        "checksum",
        "issuer_isin",
        "artifact_key",
        "etag",
        "last_modified",
        "parser_version",
        "supersedes_checksum",
        "upstream_id",
        "document_kind",
        "title",
        "publication_date",
        "reporting_period_end",
        "extracted_text_checksum",
        "extracted_text_artifact_key",
        "translated_text_checksum",
        "translated_text_artifact_key",
    }
    with engine.connect() as connection:
        index_rows = connection.exec_driver_sql(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'index' AND name IN "
            "('idx_filing_manifests_issuer_date', "
            "'idx_filing_manifests_document_versions')"
        ).all()
        index_sql: dict[str, str] = {str(row[0]): str(row[1]) for row in index_rows}
    check_names = {item["name"] for item in upgraded.get_check_constraints("filing_manifests")}
    foreign_keys = upgraded.get_foreign_keys("filing_manifests")
    assert "issuer_isin, retrieved_at DESC" in index_sql["idx_filing_manifests_issuer_date"]
    assert "filing_id, retrieved_at DESC" in index_sql["idx_filing_manifests_document_versions"]
    assert check_names == {
        "ck_filing_manifests_byte_length",
        "ck_filing_manifests_checksum_length",
    }
    assert any(
        foreign_key["constrained_columns"] == ["filing_id", "supersedes_checksum"]
        for foreign_key in foreign_keys
    )

    command.downgrade(config, "a5e0b1e349aa")
    base_manifest_columns = {
        column["name"] for column in inspect(engine).get_columns("filing_manifests")
    }
    assert "document_kind" not in base_manifest_columns
    assert "extracted_text_artifact_key" not in base_manifest_columns
    command.upgrade(config, "head")
    assert "document_kind" in {
        column["name"] for column in inspect(engine).get_columns("filing_manifests")
    }

    command.downgrade(config, _PREVIOUS_REVISION)
    assert "filing_manifests" not in inspect(engine).get_table_names()

    command.upgrade(config, "head")
    assert "filing_manifests" in inspect(engine).get_table_names()
    engine.dispose()
