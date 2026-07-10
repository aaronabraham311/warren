"""Tests for the Streamlit Metrics page — data layer + headless AppTest integration.

Data-layer tests exercise the new cache_read_tokens_for_run/cache_hit_rate/
recommendation_distribution/monthly_cost queries in isolation. The AppTest cases run
`dashboard/pages/metrics.py` headlessly against a temp file DB + logs dir to cover the
acceptance criteria that depend on actual rendering (red cost bars, the cache hit rate
metric, and the monthly warning banner).
"""

import json
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

import storage.engine as eng
from dashboard.data import (
    cache_hit_rate,
    cache_read_tokens_for_run,
    get_recent_runs_with_tokens,
    monthly_cost,
    recommendation_distribution,
)
from storage.models import Analysis, Base, Run, ToolCall

_METRICS_PAGE = str(Path(__file__).resolve().parents[1] / "dashboard" / "pages" / "metrics.py")
_BASE = datetime(2026, 6, 29, 7, 0, 0, tzinfo=timezone.utc)


def _make_run(
    run_id: str,
    *,
    started: datetime = _BASE,
    total_cost_usd: float | None = 0.5,
    total_input_tokens: int | None = 1000,
    total_output_tokens: int | None = 200,
    status: str = "success",
) -> Run:
    return Run(
        id=run_id,
        started_at=started,
        status=status,
        total_cost_usd=total_cost_usd,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
    )


def _make_analysis(run_id: str, ticker: str, recommendation: str) -> Analysis:
    return Analysis(
        run_id=run_id, ticker=ticker, analysis_type="holding", recommendation=recommendation
    )


# --------------------------------------------------------------------------- #
# Data layer
# --------------------------------------------------------------------------- #


def test_cache_read_tokens_for_run_sums_llm_calls(tmp_path: Path) -> None:
    log = tmp_path / "run-1.jsonl"
    lines = [
        {"event": "llm_call", "cache_read_tokens": 100},
        {"event": "tool_call", "cached": True},
        {"event": "llm_call", "cache_read_tokens": 250},
    ]
    log.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    assert cache_read_tokens_for_run("run-1", base_dir=tmp_path) == 350


def test_cache_read_tokens_for_run_missing_file(tmp_path: Path) -> None:
    assert cache_read_tokens_for_run("nope", base_dir=tmp_path) == 0


def test_get_recent_runs_with_tokens_orders_newest_first_and_limits(
    db_session: Session, tmp_path: Path
) -> None:
    older = _make_run("old", started=_BASE - timedelta(days=1))
    newer = _make_run("new", started=_BASE)
    db_session.add_all([older, newer])
    db_session.commit()

    rows = get_recent_runs_with_tokens(db_session, limit=1, base_dir=tmp_path)
    assert [row.run_id for row in rows] == ["new"]


def test_get_recent_runs_with_tokens_joins_cache_read_tokens(
    db_session: Session, tmp_path: Path
) -> None:
    db_session.add(_make_run("run-1"))
    db_session.commit()
    log = tmp_path / "run-1.jsonl"
    log.write_text(
        json.dumps({"event": "llm_call", "cache_read_tokens": 42}) + "\n", encoding="utf-8"
    )

    rows = get_recent_runs_with_tokens(db_session, base_dir=tmp_path)
    assert rows[0].cache_read_tokens == 42


def test_cache_hit_rate_none_when_no_tool_calls(db_session: Session) -> None:
    assert cache_hit_rate(db_session) is None


def test_cache_hit_rate_computes_fraction(db_session: Session) -> None:
    db_session.add(_make_run("run-1"))
    db_session.flush()
    db_session.add_all(
        [
            ToolCall(run_id="run-1", tool_name="get_quote", cached=True),
            ToolCall(run_id="run-1", tool_name="get_quote", cached=True),
            ToolCall(run_id="run-1", tool_name="get_news", cached=False),
        ]
    )
    db_session.commit()

    assert cache_hit_rate(db_session) == pytest.approx(2 / 3)


def test_recommendation_distribution_counts_all_buckets(db_session: Session) -> None:
    db_session.add(_make_run("run-1"))
    db_session.flush()
    db_session.add_all(
        [
            _make_analysis("run-1", "AAPL", "buy"),
            _make_analysis("run-1", "MSFT", "buy"),
            _make_analysis("run-1", "GOOG", "sell"),
            _make_analysis("run-1", "KO", "hold"),
        ]
    )
    db_session.commit()

    counts = {row.recommendation: row.count for row in recommendation_distribution(db_session)}
    assert counts == {"buy": 2, "sell": 1, "hold": 1}


def test_monthly_cost_groups_and_orders_newest_first(db_session: Session) -> None:
    db_session.add_all(
        [
            _make_run(
                "jan-1", started=datetime(2026, 1, 5, tzinfo=timezone.utc), total_cost_usd=1.0
            ),
            _make_run(
                "jan-2", started=datetime(2026, 1, 20, tzinfo=timezone.utc), total_cost_usd=2.0
            ),
            _make_run(
                "feb-1", started=datetime(2026, 2, 1, tzinfo=timezone.utc), total_cost_usd=5.0
            ),
        ]
    )
    db_session.commit()

    months = monthly_cost(db_session)
    assert [row.month for row in months] == ["2026-02", "2026-01"]
    assert months[1].total_cost_usd == pytest.approx(3.0)


def test_monthly_cost_respects_months_cap(db_session: Session) -> None:
    db_session.add_all(
        [
            _make_run(
                f"m{i}", started=datetime(2026, i, 1, tzinfo=timezone.utc), total_cost_usd=1.0
            )
            for i in range(1, 8)
        ]
    )
    db_session.commit()

    assert len(monthly_cost(db_session, months=6)) == 6


# --------------------------------------------------------------------------- #
# AppTest integration
# --------------------------------------------------------------------------- #


@pytest.fixture()
def metrics_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[SimpleNamespace, None, None]:
    """Temp file-backed warren.db + logs dir wired into storage.engine for AppTest."""
    db_path = tmp_path / "warren.db"
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    monkeypatch.setenv("WARREN_DB", str(db_path))
    monkeypatch.setenv("WARREN_LOGS_DIR", str(logs_dir))
    monkeypatch.setattr(eng, "engine", None)
    engine = eng.get_engine()
    Base.metadata.create_all(engine)
    yield SimpleNamespace(engine=engine, logs_dir=logs_dir)
    eng.engine = None


def test_apptest_empty_db_shows_info(metrics_env: SimpleNamespace) -> None:
    at = AppTest.from_file(_METRICS_PAGE).run()
    assert not at.exception
    assert any("No runs yet" in info.value for info in at.info)


def test_apptest_renders_charts_without_error(metrics_env: SimpleNamespace) -> None:
    with Session(metrics_env.engine) as session:
        session.add(_make_run("run-1", total_cost_usd=0.4))
        session.flush()
        session.add_all(
            [
                ToolCall(run_id="run-1", tool_name="get_quote", cached=True),
                ToolCall(run_id="run-1", tool_name="get_news", cached=False),
                _make_analysis("run-1", "AAPL", "buy"),
                _make_analysis("run-1", "MSFT", "hold"),
            ]
        )
        session.commit()

    at = AppTest.from_file(_METRICS_PAGE).run()
    assert not at.exception
    metric_values = {m.label: m.value for m in at.metric}
    assert metric_values["Cache hit rate"] == "50%"


def test_apptest_cache_hit_rate_dash_with_no_tool_calls(metrics_env: SimpleNamespace) -> None:
    with Session(metrics_env.engine) as session:
        session.add(_make_run("run-1"))
        session.commit()

    at = AppTest.from_file(_METRICS_PAGE).run()
    assert not at.exception
    metric_values = {m.label: m.value for m in at.metric}
    assert metric_values["Cache hit rate"] == "—"


def test_apptest_monthly_warning_banner_above_threshold(metrics_env: SimpleNamespace) -> None:
    with Session(metrics_env.engine) as session:
        session.add(_make_run("run-1", total_cost_usd=25.0, started=datetime.now(timezone.utc)))
        session.commit()

    at = AppTest.from_file(_METRICS_PAGE).run()
    assert not at.exception
    assert any("approaching the $20 ceiling" in w.value for w in at.warning)


def test_apptest_no_monthly_warning_below_threshold(metrics_env: SimpleNamespace) -> None:
    with Session(metrics_env.engine) as session:
        session.add(_make_run("run-1", total_cost_usd=5.0, started=datetime.now(timezone.utc)))
        session.commit()

    at = AppTest.from_file(_METRICS_PAGE).run()
    assert not at.exception
    assert len(at.warning) == 0
