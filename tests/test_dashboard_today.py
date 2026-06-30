"""Tests for the Streamlit Today page — data layer + headless AppTest integration.

Data-layer tests exercise the §9.Q3 sort and the JSONL trace filter in isolation.
The AppTest cases run `dashboard/pages/today.py` headlessly against a temp file DB
to cover the acceptance criteria that depend on actual rendering (auto-expand, the
⚠️ data-quality badge, the empty-DB stop).
"""

import json
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

import storage.engine as eng
from dashboard.data import (
    get_analyses_for_run,
    get_latest_run,
    read_reasoning_trace,
    run_duration_seconds,
)
from dashboard.seed_demo import seed_demo
from storage.models import Analysis, Base, Run

_TODAY_PAGE = str(Path(__file__).resolve().parents[1] / "dashboard" / "pages" / "today.py")
_RUN_START = datetime(2026, 6, 29, 7, 0, 0, tzinfo=timezone.utc)


def _make_run(run_id: str = "run-1") -> Run:
    return Run(
        id=run_id,
        started_at=_RUN_START,
        completed_at=_RUN_START + timedelta(minutes=3, seconds=20),
        status="success",
        total_cost_usd=0.1234,
        num_tool_calls=7,
    )


def _make_analysis(
    run_id: str,
    ticker: str,
    *,
    analysis_type: str = "holding",
    recommendation: str = "hold",
    confidence: float = 0.5,
    data_quality_notes: list[str] | None = None,
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
        data_quality_notes=data_quality_notes or [],
    )


# --------------------------------------------------------------------------- #
# Data layer
# --------------------------------------------------------------------------- #


def test_get_latest_run_returns_most_recent(db_session: Session) -> None:
    old = _make_run("old")
    old.started_at = _RUN_START - timedelta(days=1)
    new = _make_run("new")
    db_session.add_all([old, new])
    db_session.commit()

    latest = get_latest_run(db_session)
    assert latest is not None
    assert latest.id == "new"


def test_get_latest_run_none_when_empty(db_session: Session) -> None:
    assert get_latest_run(db_session) is None


def test_sort_order_non_hold_before_hold_then_confidence_desc(db_session: Session) -> None:
    db_session.add(_make_run())
    db_session.add_all(
        [
            _make_analysis("run-1", "HOLDHI", recommendation="hold", confidence=0.95),
            _make_analysis("run-1", "BUYLO", recommendation="buy", confidence=0.55),
            _make_analysis("run-1", "BUYHI", recommendation="buy", confidence=0.80),
            _make_analysis("run-1", "SELLMID", recommendation="sell", confidence=0.70),
            _make_analysis("run-1", "HOLDLO", recommendation="hold", confidence=0.40),
        ]
    )
    db_session.commit()

    ordered = [a.ticker for a in get_analyses_for_run(db_session, "run-1")]
    # Non-hold first (by confidence desc), then holds (by confidence desc).
    assert ordered == ["BUYHI", "SELLMID", "BUYLO", "HOLDHI", "HOLDLO"]


def test_run_duration_seconds() -> None:
    assert run_duration_seconds(_make_run()) == pytest.approx(200.0)
    incomplete = _make_run()
    incomplete.completed_at = None
    assert run_duration_seconds(incomplete) is None


def test_read_reasoning_trace_filters_by_ticker_and_event(tmp_path: Path) -> None:
    log = tmp_path / "run-1.jsonl"
    lines = [
        {"event": "run_started", "ticker": None, "run_id": "run-1"},
        {"event": "tool_call", "ticker": "AAPL", "tool": "get_quote", "latency_ms": 12},
        {"event": "llm_call", "ticker": "AAPL", "input_tokens": 100, "output_tokens": 20},
        {"event": "tool_call", "ticker": "MSFT", "tool": "get_news", "latency_ms": 30},
        {"event": "ticker_completed", "ticker": "AAPL"},
    ]
    log.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    events = read_reasoning_trace("run-1", "AAPL", base_dir=tmp_path)
    assert [e["event"] for e in events] == ["tool_call", "llm_call"]
    assert all(e["ticker"] == "AAPL" for e in events)


def test_read_reasoning_trace_missing_file(tmp_path: Path) -> None:
    assert read_reasoning_trace("nope", "AAPL", base_dir=tmp_path) == []


# --------------------------------------------------------------------------- #
# AppTest integration
# --------------------------------------------------------------------------- #


@pytest.fixture()
def today_env(
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


def _seed(engine: Engine, analyses: list[Analysis], run: Run | None = None) -> None:
    with Session(engine) as session:
        session.add(run or _make_run())
        session.flush()  # insert the run before its analyses (FK enforced on file DB)
        session.add_all(analyses)
        session.commit()


def test_apptest_empty_db_shows_info(today_env: SimpleNamespace) -> None:
    at = AppTest.from_file(_TODAY_PAGE).run()
    assert not at.exception
    assert len(at.info) >= 1
    assert "No runs yet" in at.info[0].value


def test_apptest_renders_cards_without_error(today_env: SimpleNamespace) -> None:
    _seed(
        today_env.engine,
        [
            _make_analysis("run-1", "AAPL", recommendation="buy", confidence=0.9),
            _make_analysis(
                "run-1", "MSFT", analysis_type="discovery", recommendation="sell", confidence=0.8
            ),
        ],
    )
    at = AppTest.from_file(_TODAY_PAGE).run()
    assert not at.exception
    labels = [e.label for e in at.expander]
    assert any("AAPL" in label for label in labels)
    assert any("MSFT" in label for label in labels)
    headers = [h.value for h in at.header]
    assert any("Portfolio Holdings (1)" in h for h in headers)
    assert any("Discovery Candidates (1)" in h for h in headers)


def test_apptest_auto_expands_high_confidence_action(today_env: SimpleNamespace) -> None:
    _seed(
        today_env.engine,
        [
            _make_analysis("run-1", "BUYHI", recommendation="buy", confidence=0.9),
            _make_analysis("run-1", "HOLDHI", recommendation="hold", confidence=0.95),
            _make_analysis("run-1", "BUYLO", recommendation="buy", confidence=0.5),
        ],
    )
    at = AppTest.from_file(_TODAY_PAGE).run()
    assert not at.exception
    expanded = {e.label: e.proto.expanded for e in at.expander}
    buy_hi = next(label for label in expanded if "BUYHI" in label)
    hold_hi = next(label for label in expanded if "HOLDHI" in label)
    buy_lo = next(label for label in expanded if "BUYLO" in label)
    assert expanded[buy_hi] is True  # non-hold, confidence > 0.6
    assert expanded[hold_hi] is False  # hold never auto-expands
    assert expanded[buy_lo] is False  # confidence not > 0.6


def test_apptest_reasoning_trace_shows_args_and_output(today_env: SimpleNamespace) -> None:
    _seed(today_env.engine, [_make_analysis("run-1", "AAPL", recommendation="buy", confidence=0.9)])
    log = today_env.logs_dir / "run-1.jsonl"
    events = [
        {
            "event": "llm_call",
            "ticker": "AAPL",
            "model": "claude-opus-4-8",
            "input_tokens": 9000,
            "output_tokens": 500,
            "cost_usd": 0.04,
        },
        {
            "event": "tool_call",
            "ticker": "AAPL",
            "tool": "get_quote",
            "input": {"ticker": "AAPL"},
            "output": json.dumps({"price": 207.4}),
            "status": "ok",
            "cached": False,
            "latency_ms": 120,
        },
        {
            "event": "tool_call",
            "ticker": "OTHER",
            "tool": "get_news",
            "input": {"ticker": "OTHER"},
            "output": "{}",
            "latency_ms": 50,
        },
    ]
    log.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    at = AppTest.from_file(_TODAY_PAGE).run()
    assert not at.exception
    # The full trace renders the tool name, args and output (st.json) — even though the
    # inner expander is collapsed, AppTest still builds its element subtree.
    markdown_text = " ".join(m.value for m in at.markdown)
    assert "get_quote" in markdown_text
    assert "LLM turn" in markdown_text
    assert len(at.json) >= 1  # args + output rendered as JSON viewers
    # The other ticker's events must not leak into AAPL's trace.
    assert "get_news" not in markdown_text


def test_seed_demo_writes_run_analyses_and_full_trace(tmp_path: Path) -> None:
    db_path = tmp_path / "demo.db"
    logs_dir = tmp_path / "logs"
    seed_demo(str(db_path), str(logs_dir))

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        run = get_latest_run(session)
        assert run is not None
        analyses = get_analyses_for_run(session, run.id)
    assert len(analyses) == 6
    # Non-hold recommendations sort ahead of holds (the §9.Q3 ordering the page relies on).
    assert analyses[0].recommendation != "hold"

    trace = read_reasoning_trace(run.id, "AAPL", base_dir=logs_dir)
    tool_events = [e for e in trace if e["event"] == "tool_call"]
    assert tool_events, "expected tool_call events in the demo trace"
    # Every demo tool call carries args and an output payload — the whole point of the change.
    assert all("input" in e and "output" in e for e in tool_events)
    assert any(e["event"] == "llm_call" for e in trace)


def test_apptest_data_quality_badge_in_label(today_env: SimpleNamespace) -> None:
    _seed(
        today_env.engine,
        [
            _make_analysis(
                "run-1",
                "AAPL",
                recommendation="buy",
                confidence=0.9,
                data_quality_notes=["stale fundamentals"],
            ),
        ],
    )
    at = AppTest.from_file(_TODAY_PAGE).run()
    assert not at.exception
    aapl_label = next(e.label for e in at.expander if "AAPL" in e.label)
    assert "⚠️" in aapl_label
