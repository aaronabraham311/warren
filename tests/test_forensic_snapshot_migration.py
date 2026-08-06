from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

import storage.engine as storage_engine


def test_forensic_snapshot_migration_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    database = tmp_path / "forensic-migration.db"
    monkeypatch.setenv("WARREN_DB", str(database))
    monkeypatch.setattr(storage_engine, "engine", None)
    config = Config(str(Path(__file__).parents[1] / "alembic.ini"))

    command.upgrade(config, "head")
    engine = storage_engine.get_engine()
    inspector = inspect(engine)
    assert "forensic_snapshots" in inspector.get_table_names()
    columns = {column["name"] for column in inspector.get_columns("forensic_snapshots")}
    assert columns == {
        "ticker",
        "issuer_isin",
        "as_of",
        "lookback_start",
        "extractor_version",
        "corpus_hash",
        "venue",
        "generated_at",
        "evidence_json",
        "coverage_json",
        "warnings_json",
    }
    pk = inspector.get_pk_constraint("forensic_snapshots")["constrained_columns"]
    assert pk == [
        "ticker",
        "issuer_isin",
        "venue",
        "as_of",
        "lookback_start",
        "extractor_version",
        "corpus_hash",
    ]

    command.downgrade(config, "c4aac1e13582")
    assert "forensic_snapshots" not in inspect(engine).get_table_names()

    command.upgrade(config, "head")
    assert "forensic_snapshots" in inspect(engine).get_table_names()
    engine.dispose()
