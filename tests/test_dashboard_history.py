"""Tests for the Streamlit History page — data layer + headless AppTest integration.

Data-layer tests exercise `search_analyses`'s filters (ticker LIKE, recommendation IN,
date range, confidence range), the newest-first ordering, and the prompt-version join.
The AppTest cases run `dashboard/pages/history.py` headlessly against a temp file DB to
cover the acceptance criteria that depend on rendering (version tag in the label, the
empty-result info, expandable cards).
"""

from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

import storage.engine as eng
from dashboard.data import search_analyses
from storage.models import Analysis, Base, PromptVersion, Run

_HISTORY_PAGE = str(Path(__file__).resolve().parents[1] / "dashboard" / "pages" / "history.py")
_BASE = datetime(2026, 6, 29, 7, 0, 0, tzinfo=timezone.utc)


def _make_run(
    run_id: str, *, prompt_version_id: int | None = None, started: datetime = _BASE
) -> Run:
    return Run(id=run_id, prompt_version_id=prompt_version_id, started_at=started, status="success")


def _make_analysis(
    run_id: str,
    ticker: str,
    *,
    recommendation: str = "hold",
    confidence: float = 0.5,
    created_at: datetime = _BASE,
    analysis_type: str = "holding",
) -> Analysis:
    return Analysis(
        run_id=run_id,
        ticker=ticker,
        analysis_type=analysis_type,
        recommendation=recommendation,
        confidence=confidence,
        thesis=f"{ticker} thesis.",
        lynch_signals=["fast grower"],
        buffett_signals=["wide moat"],
        key_risks=["valuation"],
        data_quality_notes=[],
        created_at=created_at,
    )


# --------------------------------------------------------------------------- #
# Data layer
# --------------------------------------------------------------------------- #


def test_search_filters_by_ticker_newest_first(db_session: Session) -> None:
    # Two AAPL rows live in different runs (the (run_id, ticker) unique constraint).
    db_session.add_all([_make_run("r1"), _make_run("r2")])
    db_session.add_all(
        [
            _make_analysis("r1", "AAPL", created_at=_BASE - timedelta(days=2)),
            _make_analysis("r2", "AAPL", created_at=_BASE),  # newer
            _make_analysis("r1", "MSFT"),
            _make_analysis("r1", "GOOG"),
        ]
    )
    db_session.commit()

    results = search_analyses(db_session, ticker="AAPL")
    assert [r.analysis.ticker for r in results] == ["AAPL", "AAPL"]
    # Newest first: the _BASE row precedes the two-days-earlier one (SQLite drops tzinfo,
    # so compare the calendar date, not the aware datetime).
    created = [r.analysis.created_at for r in results]
    assert all(c is not None for c in created)
    assert [c.date() for c in created if c is not None] == [
        _BASE.date(),
        (_BASE - timedelta(days=2)).date(),
    ]


def test_search_ticker_is_substring_and_case_insensitive(db_session: Session) -> None:
    db_session.add(_make_run("r1"))
    db_session.add_all([_make_analysis("r1", "AAPL"), _make_analysis("r1", "APPN")])
    db_session.commit()

    # "app" lowercases→uppercases to "APP", a substring of both AAPL and APPN? No —
    # APP matches APPN only via LIKE %APP%, and AAPL does not contain "APP".
    results = search_analyses(db_session, ticker="app")
    assert {r.analysis.ticker for r in results} == {"APPN"}


def test_search_filters_by_recommendation(db_session: Session) -> None:
    db_session.add(_make_run("r1"))
    db_session.add_all(
        [
            _make_analysis("r1", "AAPL", recommendation="sell"),
            _make_analysis("r1", "MSFT", recommendation="buy"),
            _make_analysis("r1", "GOOG", recommendation="sell"),
            _make_analysis("r1", "KO", recommendation="hold"),
        ]
    )
    db_session.commit()

    results = search_analyses(db_session, recommendations=["sell"])
    assert {r.analysis.ticker for r in results} == {"AAPL", "GOOG"}
    assert all(r.analysis.recommendation == "sell" for r in results)


def test_search_filters_by_date_range(db_session: Session) -> None:
    db_session.add(_make_run("r1"))
    db_session.add_all(
        [
            _make_analysis("r1", "OLD", created_at=datetime(2026, 6, 1, tzinfo=timezone.utc)),
            _make_analysis("r1", "MID", created_at=datetime(2026, 6, 15, tzinfo=timezone.utc)),
            _make_analysis("r1", "NEW", created_at=datetime(2026, 6, 28, tzinfo=timezone.utc)),
        ]
    )
    db_session.commit()

    results = search_analyses(
        db_session,
        date_from=datetime(2026, 6, 10).date(),
        date_to=datetime(2026, 6, 20).date(),
    )
    assert {r.analysis.ticker for r in results} == {"MID"}


def test_search_filters_by_confidence_range(db_session: Session) -> None:
    db_session.add(_make_run("r1"))
    db_session.add_all(
        [
            _make_analysis("r1", "LO", confidence=0.10),
            _make_analysis("r1", "MID", confidence=0.55),
            _make_analysis("r1", "HI", confidence=0.95),
        ]
    )
    db_session.commit()

    results = search_analyses(db_session, conf_min=0.5, conf_max=0.8)
    assert {r.analysis.ticker for r in results} == {"MID"}


def test_search_returns_prompt_version_tag(db_session: Session) -> None:
    db_session.add(PromptVersion(id=1, version_tag="v1-baseline"))
    db_session.add(_make_run("versioned", prompt_version_id=1))
    db_session.add(_make_run("unversioned", prompt_version_id=None))
    db_session.add_all([_make_analysis("versioned", "AAPL"), _make_analysis("unversioned", "MSFT")])
    db_session.commit()

    by_ticker = {r.analysis.ticker: r.prompt_version for r in search_analyses(db_session)}
    assert by_ticker["AAPL"] == "v1-baseline"
    assert by_ticker["MSFT"] is None


def test_search_respects_limit(db_session: Session) -> None:
    db_session.add(_make_run("r1"))
    db_session.add_all([_make_analysis("r1", f"T{i}") for i in range(10)])
    db_session.commit()

    assert len(search_analyses(db_session, limit=3)) == 3


# --------------------------------------------------------------------------- #
# AppTest integration
# --------------------------------------------------------------------------- #


@pytest.fixture()
def history_env(
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


def _seed(engine: Engine) -> None:
    with Session(engine) as session:
        session.add(PromptVersion(id=1, version_tag="v1-baseline"))
        session.add(_make_run("versioned", prompt_version_id=1))
        session.add(_make_run("plain", prompt_version_id=None))
        session.flush()
        session.add_all(
            [
                _make_analysis("versioned", "AAPL", recommendation="buy", confidence=0.9),
                _make_analysis("plain", "MSFT", recommendation="sell", confidence=0.8),
            ]
        )
        session.commit()


def test_apptest_empty_db_shows_info(history_env: SimpleNamespace) -> None:
    at = AppTest.from_file(_HISTORY_PAGE).run()
    assert not at.exception
    assert any("No results" in info.value for info in at.info)


def test_apptest_renders_rows_with_version_tag(history_env: SimpleNamespace) -> None:
    _seed(history_env.engine)
    at = AppTest.from_file(_HISTORY_PAGE).run()
    assert not at.exception
    labels = [e.label for e in at.expander]
    aapl = next(label for label in labels if "AAPL" in label)
    msft = next(label for label in labels if "MSFT" in label)
    assert "v1-baseline" in aapl  # versioned run shows its tag
    assert "unknown version" in msft  # unlinked run falls back


def test_apptest_ticker_filter_narrows_rows(history_env: SimpleNamespace) -> None:
    _seed(history_env.engine)
    at = AppTest.from_file(_HISTORY_PAGE).run()
    at.sidebar.text_input[0].set_value("AAPL").run()
    assert not at.exception
    labels = [e.label for e in at.expander]
    assert any("AAPL" in label for label in labels)
    assert not any("MSFT" in label for label in labels)


def test_apptest_card_expands_with_thesis(history_env: SimpleNamespace) -> None:
    _seed(history_env.engine)
    at = AppTest.from_file(_HISTORY_PAGE).run()
    assert not at.exception
    markdown_text = " ".join(m.value for m in at.markdown)
    assert "AAPL thesis." in markdown_text
    assert "Lynch signals" in markdown_text
    assert "Key risks" in markdown_text
