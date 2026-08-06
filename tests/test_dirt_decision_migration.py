from __future__ import annotations

import json
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

import storage.engine as storage_engine

_PREVIOUS_REVISION = "d7a4c8e2f913"


def _assert_analysis_constraints(engine: Engine) -> None:
    inspector = inspect(engine)
    foreign_keys = inspector.get_foreign_keys("analyses")
    assert any(
        fk["constrained_columns"] == ["run_id"]
        and fk["referred_table"] == "runs"
        and fk["referred_columns"] == ["id"]
        for fk in foreign_keys
    )
    unique_constraints = inspector.get_unique_constraints("analyses")
    assert any(
        constraint["column_names"] == ["run_id", "ticker"] for constraint in unique_constraints
    )


def test_dirt_decision_migration_round_trip_preserves_populated_analysis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    database = tmp_path / "dirt-decision-migration.db"
    monkeypatch.setenv("WARREN_DB", str(database))
    monkeypatch.setattr(storage_engine, "engine", None)
    config = Config(str(Path(__file__).parents[1] / "alembic.ini"))

    command.upgrade(config, _PREVIOUS_REVISION)
    engine = storage_engine.get_engine()
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "INSERT INTO runs (id, status) VALUES (?, ?)", ("run-g18", "success")
        )
        connection.exec_driver_sql(
            "INSERT INTO analyses "
            "(run_id, ticker, analysis_type, recommendation, confidence, thesis, "
            "lynch_signals, buffett_signals, key_risks, data_quality_notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "run-g18",
                "KPL.WA",
                "discovery",
                "buy",
                0.7,
                "A populated pre-G18 analysis must survive both table recreations.",
                json.dumps({"pros": [], "cons": []}),
                json.dumps({"pros": [], "cons": []}),
                json.dumps(["illiquidity"]),
                json.dumps(["regional coverage"]),
            ),
        )

    command.upgrade(config, "head")
    upgraded_columns = {column["name"] for column in inspect(engine).get_columns("analyses")}
    assert {
        "dirt_signals",
        "dirt_decision",
        "decision_outcome",
        "probability_weighted_irr",
    } <= upgraded_columns
    _assert_analysis_constraints(engine)

    dirt_signals = {"ev_ebit": 4.2, "closability_status": "supported"}
    dirt_decision = {
        "outcome": "buy",
        "probability_weighted_irr": 0.224,
        "hurdle_irr": 0.20,
        "required_entry_price": 12.5,
    }
    with engine.begin() as connection:
        row = connection.exec_driver_sql(
            "SELECT ticker, dirt_signals, dirt_decision, decision_outcome, "
            "probability_weighted_irr FROM analyses WHERE run_id = ?",
            ("run-g18",),
        ).one()
        assert row[0] == "KPL.WA"
        assert row[1:] == (None, None, None, None)
        connection.exec_driver_sql(
            "UPDATE analyses SET dirt_signals = ?, dirt_decision = ?, "
            "decision_outcome = ?, probability_weighted_irr = ? WHERE run_id = ?",
            (
                json.dumps(dirt_signals),
                json.dumps(dirt_decision),
                "buy",
                0.224,
                "run-g18",
            ),
        )
        stored = connection.exec_driver_sql(
            "SELECT dirt_signals, dirt_decision, decision_outcome, "
            "probability_weighted_irr FROM analyses WHERE run_id = ?",
            ("run-g18",),
        ).one()
        assert json.loads(stored[0]) == dirt_signals
        assert json.loads(stored[1]) == dirt_decision
        assert stored[2] == "buy"
        assert stored[3] == pytest.approx(0.224)

    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.exec_driver_sql(
            "INSERT INTO analyses (run_id, ticker) VALUES (?, ?)",
            ("run-g18", "KPL.WA"),
        )
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.exec_driver_sql(
            "INSERT INTO analyses (run_id, ticker) VALUES (?, ?)",
            ("missing-run", "ORPHAN"),
        )

    # Simulate the debris an interrupted downgrade leaves behind. The migration's
    # explicit guard must remove it before Alembic starts its own batch recreation.
    with engine.begin() as connection:
        connection.exec_driver_sql('CREATE TABLE "_alembic_tmp_analyses" (id INTEGER)')
    command.downgrade(config, _PREVIOUS_REVISION)

    downgraded_columns = {column["name"] for column in inspect(engine).get_columns("analyses")}
    assert "dirt_decision" not in downgraded_columns
    assert "dirt_signals" not in downgraded_columns
    assert "decision_outcome" not in downgraded_columns
    assert "probability_weighted_irr" not in downgraded_columns
    _assert_analysis_constraints(engine)
    with engine.connect() as connection:
        preserved = connection.exec_driver_sql(
            "SELECT ticker, recommendation, thesis FROM analyses WHERE run_id = ?",
            ("run-g18",),
        ).one()
        assert preserved == (
            "KPL.WA",
            "buy",
            "A populated pre-G18 analysis must survive both table recreations.",
        )
        stale = connection.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE name = '_alembic_tmp_analyses'"
        ).all()
        assert stale == []

    command.upgrade(config, "head")
    assert "dirt_decision" in {column["name"] for column in inspect(engine).get_columns("analyses")}
    engine.dispose()
