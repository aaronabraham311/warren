import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from agent.terminal.queries import (
    MAX_HISTORY_LIMIT,
    get_run,
    get_trace,
    list_history,
    list_portfolio,
    list_watchlist,
)
from storage.models import Analysis, Holding, Run, Watchlist

_NOW = datetime(2026, 8, 6, 12, 0)


def _run(run_id: str, started_at: datetime | None, *, status: str = "success") -> Run:
    return Run(
        id=run_id,
        started_at=started_at,
        completed_at=started_at + timedelta(minutes=2) if started_at is not None else None,
        status=status,
        total_input_tokens=120,
        total_output_tokens=40,
        total_cost_usd=0.25,
        num_tool_calls=3,
    )


def _analysis(
    run_id: str,
    ticker: str,
    *,
    recommendation: str = "hold",
    thesis: str = "Durable business.",
) -> Analysis:
    return Analysis(
        run_id=run_id,
        ticker=ticker,
        analysis_type="holding",
        recommendation=recommendation,
        confidence=0.7,
        thesis=thesis,
        lynch_signals={"pros": ["growth"], "cons": []},
        buffett_signals={"pros": ["moat"], "cons": []},
        key_risks=["valuation"],
        data_quality_notes=["one stale field"],
        tool_calls_made=3,
        tokens_used=160,
    )


def test_list_history_is_deterministic_and_includes_analysis_order(db_session: Session) -> None:
    db_session.add_all(
        [
            _run("run-a", _NOW),
            _run("run-b", _NOW),
            _run("run-undated", None),
        ]
    )
    db_session.flush()
    db_session.add_all(
        [
            _analysis("run-b", "MSFT"),
            _analysis("run-b", "AAPL"),
            _analysis("run-a", "KO"),
        ]
    )
    db_session.commit()

    rows = list_history()

    assert [row.run_id for row in rows] == ["run-b", "run-a", "run-undated"]
    assert rows[0].tickers == ("MSFT", "AAPL")
    assert rows[0].total_cost_usd == pytest.approx(0.25)


def test_list_history_filters_exact_ticker_case_insensitively(db_session: Session) -> None:
    db_session.add_all([_run("exact", _NOW), _run("other", _NOW - timedelta(days=1))])
    db_session.flush()
    db_session.add_all([_analysis("exact", "BRK.B"), _analysis("other", "BRK")])
    db_session.commit()

    assert [row.run_id for row in list_history("brk.b")] == ["exact"]


def test_list_history_bounds_limit_and_nonpositive_is_empty(db_session: Session) -> None:
    db_session.add_all(
        [_run(f"run-{index:03d}", _NOW + timedelta(minutes=index)) for index in range(110)]
    )
    db_session.commit()

    assert len(list_history(limit=1_000)) == MAX_HISTORY_LIMIT
    assert list_history(limit=0) == ()


def test_get_run_returns_full_projected_details_in_input_order(db_session: Session) -> None:
    run = _run("run-detail", _NOW, status="failed")
    run.error_msg = "budget exhausted"
    db_session.add(run)
    db_session.flush()
    db_session.add_all(
        [
            _analysis("run-detail", "MSFT", recommendation="buy", thesis="Second ticker."),
            _analysis("run-detail", "AAPL", recommendation="hold", thesis="First ticker."),
        ]
    )
    db_session.commit()

    result = get_run("run-detail")

    assert result is not None
    assert result.status == "failed"
    assert result.error_msg == "budget exhausted"
    assert result.total_input_tokens == 120
    assert [analysis.ticker for analysis in result.analyses] == ["MSFT", "AAPL"]
    assert result.analyses[0].lynch_signals == {"pros": ["growth"], "cons": []}
    assert result.analyses[1].key_risks == ("valuation",)


def test_get_run_returns_none_for_unknown_id(db_engine: object) -> None:
    assert get_run("missing") is None


def test_get_run_reconstructs_partial_ticker_order_from_wal(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_session.add(_run("run-partial", _NOW))
    db_session.flush()
    db_session.add(_analysis("run-partial", "MSFT"))
    db_session.commit()
    records = [
        {"event": "run_started", "run_id": "run-partial", "tickers": ["AAPL", "MSFT"]},
        {
            "event": "ticker_failed",
            "run_id": "run-partial",
            "ticker": "AAPL",
            "error": "upstream unavailable",
        },
    ]
    (tmp_path / "run-partial.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WARREN_LOGS_DIR", str(tmp_path))

    detail = get_run("run-partial")

    assert detail is not None
    assert [result.ticker for result in detail.ticker_results] == ["AAPL", "MSFT"]
    assert detail.ticker_results[0].error == "upstream unavailable"
    assert detail.ticker_results[1].analysis is not None


def test_get_trace_preserves_wal_order_and_builds_safe_summaries(tmp_path: Path) -> None:
    records = [
        {"ts": "2026-08-06T12:00:00Z", "run_id": "r1", "event": "run_started"},
        {
            "ts": "2026-08-06T12:00:01Z",
            "run_id": "r1",
            "event": "tool_call",
            "ticker": "AAPL",
            "tool": "get_quote",
            "status": "success",
            "cached": True,
            "input": {"ticker": "AAPL"},
            "output": "large payload stays in fields",
        },
    ]
    (tmp_path / "r1.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    trace = get_trace("r1", log_dir=tmp_path)

    assert trace is not None
    assert [event.event for event in trace.events] == ["run_started", "tool_call"]
    assert [event.sequence for event in trace.events] == [1, 2]
    assert trace.events[1].ticker == "AAPL"
    assert trace.events[1].summary == "tool=get_quote, status=success, cached=True"
    assert trace.warnings == ()


def test_trace_summary_shows_sidecar_path_without_inlining_output(tmp_path: Path) -> None:
    marker = json.dumps({"truncated": True, "path": "r1/tool_outputs/0001.json"})
    (tmp_path / "r1.jsonl").write_text(
        json.dumps(
            {
                "event": "tool_call",
                "run_id": "r1",
                "tool": "read_filing",
                "output": marker,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    trace = get_trace("r1", log_dir=tmp_path)

    assert trace is not None
    assert "output_file=r1/tool_outputs/0001.json" in trace.events[0].summary


def test_get_trace_tolerates_torn_final_line_with_warning(tmp_path: Path) -> None:
    (tmp_path / "r-torn.jsonl").write_text(
        json.dumps({"event": "run_started", "run_id": "r-torn"}) + "\n{" + '"event":',
        encoding="utf-8",
    )

    trace = get_trace("r-torn", log_dir=tmp_path)

    assert trace is not None
    assert [event.event for event in trace.events] == ["run_started"]
    assert len(trace.warnings) == 1
    assert trace.warnings[0].line_number == 2
    assert "torn final" in trace.warnings[0].message


def test_get_trace_tolerates_torn_final_utf8_character(tmp_path: Path) -> None:
    (tmp_path / "r-utf8.jsonl").write_bytes(
        json.dumps({"event": "run_started", "run_id": "r-utf8"}).encode()
        + b'\n{"event":"tool_call","output":"\xe2'
    )

    trace = get_trace("r-utf8", log_dir=tmp_path)

    assert trace is not None
    assert [event.event for event in trace.events] == ["run_started"]
    assert len(trace.warnings) == 1
    assert "torn final" in trace.warnings[0].message


def test_get_trace_surfaces_missing_authoritative_file(tmp_path: Path) -> None:
    trace = get_trace("missing", log_dir=tmp_path)

    assert trace is not None
    assert trace.events == ()
    assert trace.warnings[0].line_number is None
    assert "unavailable" in trace.warnings[0].message


def test_get_trace_without_id_uses_latest_projected_run(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_session.add_all([_run("old", _NOW), _run("new", _NOW + timedelta(minutes=1))])
    db_session.commit()
    (tmp_path / "new.jsonl").write_text(
        json.dumps({"event": "run_completed", "run_id": "new"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WARREN_LOGS_DIR", str(tmp_path))

    trace = get_trace()

    assert trace is not None
    assert trace.run_id == "new"
    assert [event.event for event in trace.events] == ["run_completed"]


def test_queries_do_not_require_an_api_key(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    db_session.add(_run("offline", _NOW))
    db_session.commit()
    (tmp_path / "offline.jsonl").write_text(
        json.dumps({"event": "run_started", "run_id": "offline"}) + "\n",
        encoding="utf-8",
    )

    assert list_history()[0].run_id == "offline"
    assert get_run("offline") is not None
    assert get_trace("offline", log_dir=tmp_path) is not None


def test_list_portfolio_returns_stored_snapshot_in_ticker_order(db_session: Session) -> None:
    db_session.add_all(
        [
            Holding(
                ticker="MSFT",
                shares=2.0,
                cost_basis=300.0,
                current_price=410.0,
                purchase_date=date(2024, 1, 2),
                updated_at=_NOW,
            ),
            Holding(ticker="AAPL", shares=5.0, cost_basis=150.0, current_price=205.0),
        ]
    )
    db_session.commit()

    entries = list_portfolio()

    assert [entry.ticker for entry in entries] == ["AAPL", "MSFT"]
    assert entries[1].shares == pytest.approx(2.0)
    assert entries[1].purchase_date == date(2024, 1, 2)


def test_list_watchlist_returns_stored_notes_in_ticker_order(db_session: Session) -> None:
    db_session.add_all(
        [
            Watchlist(ticker="MSFT", notes="wait for valuation"),
            Watchlist(ticker="AAPL", notes="watch margins"),
        ]
    )
    db_session.commit()

    entries = list_watchlist()

    assert [(entry.ticker, entry.notes) for entry in entries] == [
        ("AAPL", "watch margins"),
        ("MSFT", "wait for valuation"),
    ]
