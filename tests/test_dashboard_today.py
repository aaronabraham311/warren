"""Tests for the Streamlit Today page — data layer + headless AppTest integration.

Data-layer tests exercise the §9.Q3 sort and the JSONL trace filter in isolation.
The AppTest cases run `dashboard/pages/today.py` headlessly against a temp file DB
to cover the acceptance criteria that depend on actual rendering (auto-expand, the
⚠️ data-quality badge, the empty-DB stop).
"""

import json
import subprocess
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
    cooldown_suppressed_count,
    get_analyses_for_run,
    get_latest_run,
    previous_recommendation,
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
    dirt_decision: dict[str, object] | None = None,
) -> Analysis:
    outcome = None if dirt_decision is None else dirt_decision.get("outcome")
    weighted_irr = None if dirt_decision is None else dirt_decision.get("probability_weighted_irr")
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
        dirt_decision=dirt_decision,
        decision_outcome=outcome if isinstance(outcome, str) else None,
        probability_weighted_irr=(
            float(weighted_irr)
            if isinstance(weighted_irr, (int, float)) and not isinstance(weighted_irr, bool)
            else None
        ),
    )


def _decision(outcome: str = "buy", weighted_irr: float = 0.27) -> dict[str, object]:
    return {
        "valuation_date": "2026-06-29",
        "currency": "USD",
        "current_price": 50.0,
        "horizon_years": 2,
        "hurdle_irr": 0.2,
        "probability_weighted_irr": weighted_irr,
        "hurdle_cleared": weighted_irr >= 0.2,
        "required_entry_price": 56.25,
        "outcome": outcome,
        "outcome_reason": "Cited return and catalyst support the outcome",
        "scenarios": [
            {
                "case": case,
                "probability": probability,
                "terminal_price": terminal,
                "terminal_date": "2028-06-29",
                "total_dividends": 4.0,
                "total_return": total_return,
                "irr": irr,
            }
            for case, probability, terminal, total_return, irr in [
                ("bear", 0.25, 42.0, -0.08, -0.04),
                ("base", 0.5, 75.0, 0.58, 0.26),
                ("bull", 0.25, 105.0, 1.18, 0.48),
            ]
        ],
        "downside_floor": {
            "basis": "tangible_book",
            "gross": 48.0,
            "adjusted": 40.8,
            "coverage": 0.816,
            "confidence": "medium",
            "adjustments": ["Exclude goodwill"],
        },
        "catalysts": [
            {
                "description": "Board-authorized tender",
                "category": "capital_return",
                "evidence_strength": "contractual",
                "expected_by": "2027-03-31",
                "source_ref": "filing:tender",
            }
        ],
        "failure_thesis": "The tender is withdrawn and asset coverage falls.",
        "entry_conditions": [{"description": "Price at or below hurdle", "threshold": 56.25}],
        "blocking_unknowns": ["Final tender size"],
        "monitoring_metrics": [
            {"metric": "tangible_book_per_share", "failure_threshold": 40.0},
            {"metric": "tender_completion_pct", "warning_threshold": 25.0},
        ],
        "calculation_version": "dce_irr_v1",
    }


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


def test_today_sort_prioritizes_decision_outcome_then_weighted_irr(db_session: Session) -> None:
    db_session.add(_make_run())
    db_session.add_all(
        [
            _make_analysis("run-1", "LEGACY", recommendation="buy", confidence=0.99),
            _make_analysis("run-1", "PASS", dirt_decision=_decision("pass", 0.40)),
            _make_analysis("run-1", "WATCH", dirt_decision=_decision("watchlist", 0.45)),
            _make_analysis("run-1", "BUYLO", dirt_decision=_decision("buy", 0.22)),
            _make_analysis("run-1", "BUYHI", dirt_decision=_decision("buy", 0.35)),
        ]
    )
    db_session.commit()

    ordered = [analysis.ticker for analysis in get_analyses_for_run(db_session, "run-1")]
    assert ordered == ["BUYHI", "BUYLO", "WATCH", "PASS", "LEGACY"]


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


def test_cooldown_suppressed_count_reads_event(tmp_path: Path) -> None:
    log = tmp_path / "run-1.jsonl"
    lines = [
        {"event": "run_started", "run_id": "run-1"},
        {"event": "discovery_cooldown_applied", "suppressed_count": 4, "suppressed_tickers": []},
    ]
    log.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    assert cooldown_suppressed_count("run-1", base_dir=tmp_path) == 4


def test_cooldown_suppressed_count_zero_when_event_absent(tmp_path: Path) -> None:
    log = tmp_path / "run-1.jsonl"
    log.write_text(json.dumps({"event": "run_started", "run_id": "run-1"}) + "\n", encoding="utf-8")
    assert cooldown_suppressed_count("run-1", base_dir=tmp_path) == 0


def test_cooldown_suppressed_count_zero_when_file_missing(tmp_path: Path) -> None:
    assert cooldown_suppressed_count("nope", base_dir=tmp_path) == 0


def test_previous_recommendation_returns_most_recent_before(db_session: Session) -> None:
    db_session.add_all([_make_run("run-1"), _make_run("run-2")])
    earlier = _make_analysis("run-1", "AAPL", recommendation="sell", confidence=0.5)
    earlier.created_at = _RUN_START - timedelta(days=5)
    later = _make_analysis("run-2", "AAPL", recommendation="hold", confidence=0.5)
    later.created_at = _RUN_START - timedelta(days=1)
    db_session.add_all([earlier, later])
    db_session.commit()

    prior = previous_recommendation(db_session, "AAPL", _RUN_START)
    assert prior == "hold"


def test_previous_recommendation_none_when_no_earlier_analysis(db_session: Session) -> None:
    db_session.add(_make_run())
    db_session.add(_make_analysis("run-1", "NVDA", recommendation="buy", confidence=0.9))
    db_session.commit()

    assert previous_recommendation(db_session, "NVDA", _RUN_START - timedelta(days=100)) is None


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


def test_apptest_renders_dirt_decision_badge_and_contract(today_env: SimpleNamespace) -> None:
    _seed(
        today_env.engine,
        [
            _make_analysis(
                "run-1",
                "DIRT",
                analysis_type="discovery",
                recommendation="buy",
                confidence=0.9,
                dirt_decision=_decision(),
            )
        ],
    )

    at = AppTest.from_file(_TODAY_PAGE).run()

    assert not at.exception
    card = next(expander for expander in at.expander if "DIRT" in expander.label)
    assert "BUY" in card.label
    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics["Weighted IRR"] == "27.0%"
    assert metrics["Hurdle"] == "20.0%"
    markdown = " ".join(item.value for item in at.markdown)
    for expected in [
        "DIRT decision contract",
        "Scenarios",
        "Downside floor",
        "Catalysts",
        "Failure thesis",
        "Entry conditions",
        "Blocking unknowns",
        "Monitoring",
    ]:
        assert expected in markdown
    assert len(at.table) == 1


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
        run_tool_calls = run.num_tool_calls
    assert len(analyses) == 9
    # Non-hold recommendations sort ahead of holds (the §9.Q3 ordering the page relies on).
    assert analyses[0].recommendation != "hold"
    decisions = {row.decision_outcome: row for row in analyses if row.dirt_decision is not None}
    assert set(decisions) == {"buy", "watchlist", "pass"}
    assert decisions["buy"].probability_weighted_irr is not None
    assert decisions["buy"].probability_weighted_irr >= 0.20
    assert decisions["watchlist"].probability_weighted_irr == pytest.approx(0.17, abs=0.02)
    pass_decision = decisions["pass"].dirt_decision
    assert pass_decision is not None
    assert "structural" in str(pass_decision["outcome_reason"]).lower()
    assert run_tool_calls == sum(row.tool_calls_made or 0 for row in analyses)

    trace = read_reasoning_trace(run.id, "AAPL", base_dir=logs_dir)
    tool_events = [e for e in trace if e["event"] == "tool_call"]
    assert tool_events, "expected tool_call events in the demo trace"
    # Every demo tool call carries args and an output payload — the whole point of the change.
    assert all("input" in e and "output" in e for e in tool_events)
    assert any(e["event"] == "llm_call" for e in trace)
    dirt_trace = read_reasoning_trace(run.id, "DIR.MI", base_dir=logs_dir)
    decision_event = next(
        event for event in dirt_trace if event.get("tool") == "model_dirt_scenarios"
    )
    assert json.loads(str(decision_event["output"])) == decisions["buy"].dirt_decision


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


def test_apptest_suppressed_count_metric_column(today_env: SimpleNamespace) -> None:
    _seed(today_env.engine, [_make_analysis("run-1", "AAPL")])
    log = today_env.logs_dir / "run-1.jsonl"
    log.write_text(
        json.dumps({"event": "discovery_cooldown_applied", "suppressed_count": 4}) + "\n",
        encoding="utf-8",
    )
    at = AppTest.from_file(_TODAY_PAGE).run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Suppressed (cooldown)"] == "4"


def test_apptest_run_status_banner_on_failed_run(today_env: SimpleNamespace) -> None:
    failed_run = _make_run()
    failed_run.status = "failed"
    failed_run.error_msg = "cost ceiling reached after AAPL"
    _seed(today_env.engine, [_make_analysis("run-1", "AAPL")], run=failed_run)
    at = AppTest.from_file(_TODAY_PAGE).run()
    assert not at.exception
    error_text = " ".join(e.value for e in at.error)
    assert "failed" in error_text
    assert "cost ceiling reached after AAPL" in error_text


def test_apptest_no_status_banner_on_success(today_env: SimpleNamespace) -> None:
    _seed(today_env.engine, [_make_analysis("run-1", "AAPL")])
    at = AppTest.from_file(_TODAY_PAGE).run()
    assert not at.exception
    assert len(at.error) == 0


def test_apptest_budget_banner_over_monthly_ceiling(today_env: SimpleNamespace) -> None:
    over_budget_run = _make_run()
    over_budget_run.total_cost_usd = 19.0
    _seed(today_env.engine, [_make_analysis("run-1", "AAPL")], run=over_budget_run)
    at = AppTest.from_file(_TODAY_PAGE).run()
    assert not at.exception
    warning_text = " ".join(w.value for w in at.warning)
    assert "approaching the $20 ceiling" in warning_text


def test_apptest_no_budget_banner_under_ceiling(today_env: SimpleNamespace) -> None:
    _seed(today_env.engine, [_make_analysis("run-1", "AAPL")])  # default cost 0.1234
    at = AppTest.from_file(_TODAY_PAGE).run()
    assert not at.exception
    warning_text = " ".join(w.value for w in at.warning)
    assert "approaching the $20 ceiling" not in warning_text


def test_apptest_prior_recommendation_delta_shown(today_env: SimpleNamespace) -> None:
    with Session(today_env.engine) as session:
        session.add(Run(id="run-0", started_at=_RUN_START - timedelta(days=1)))
        session.flush()  # insert the run before its analysis (FK enforced on file DB)
        older = _make_analysis("run-0", "AAPL", recommendation="hold", confidence=0.5)
        older.created_at = _RUN_START - timedelta(days=1)
        session.add(older)
        session.commit()
    _seed(
        today_env.engine,
        [_make_analysis("run-1", "AAPL", recommendation="buy", confidence=0.9)],
    )
    at = AppTest.from_file(_TODAY_PAGE).run()
    assert not at.exception
    aapl_label = next(e.label for e in at.expander if "AAPL" in e.label)
    assert "was HOLD" in aapl_label


def test_apptest_no_delta_when_recommendation_unchanged(today_env: SimpleNamespace) -> None:
    with Session(today_env.engine) as session:
        session.add(Run(id="run-0", started_at=_RUN_START - timedelta(days=1)))
        session.flush()
        older = _make_analysis("run-0", "AAPL", recommendation="buy", confidence=0.5)
        older.created_at = _RUN_START - timedelta(days=1)
        session.add(older)
        session.commit()
    _seed(
        today_env.engine,
        [_make_analysis("run-1", "AAPL", recommendation="buy", confidence=0.9)],
    )
    at = AppTest.from_file(_TODAY_PAGE).run()
    assert not at.exception
    aapl_label = next(e.label for e in at.expander if "AAPL" in e.label)
    assert "was" not in aapl_label


def test_apptest_run_now_button_success(
    today_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(today_env.engine, [_make_analysis("run-1", "AAPL")])

    def _fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    at = AppTest.from_file(_TODAY_PAGE).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    success_text = " ".join(s.value for s in at.success)
    assert "Run completed" in success_text


def test_apptest_run_now_button_failure(
    today_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(today_env.engine, [_make_analysis("run-1", "AAPL")])

    def _fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    at = AppTest.from_file(_TODAY_PAGE).run()
    at.button[0].click().run()
    assert not at.exception
    error_text = " ".join(e.value for e in at.error)
    assert "Run failed" in error_text
    assert "boom" in error_text
