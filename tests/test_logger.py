"""Tests for the JSONL WAL (RunLogger) and its reconciliation into SQLite."""

import json
from pathlib import Path
from unittest.mock import patch

import anthropic
import pytest
from sqlalchemy.orm import Session

from agent.models import SONNET_4_6
from storage.cost import compute_cost
from storage.engine import write_run_start
from storage.logger import RunLogger
from storage.models import Run, ToolCall
from storage.recovery import reconcile_orphans


def _llm_response(
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    cache_creation: int,
) -> anthropic.types.Message:
    usage = anthropic.types.Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
    )
    return anthropic.types.Message(
        id="msg_test",
        type="message",
        role="assistant",
        content=[anthropic.types.TextBlock(type="text", text="ok")],
        model=SONNET_4_6,
        stop_reason="end_turn",
        stop_sequence=None,
        usage=usage,
    )


def _read_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _emit_full_run(logger: RunLogger) -> None:
    logger.log("run_started", tickers=["AAPL"])
    logger.log("ticker_started", ticker="AAPL", phase="deep", model=SONNET_4_6)
    logger.log(
        "llm_call_started",
        ticker="AAPL",
        phase="deep",
        model=SONNET_4_6,
        purpose="synthesis",
        iteration=0,
    )
    logger.log_llm_call(
        _llm_response(4523, 1840, cache_read=4200, cache_creation=323),
        ticker="AAPL",
        phase="deep",
        model=SONNET_4_6,
        latency_ms=3200,
    )
    logger.log_tool_started(ticker="AAPL", tool_name="get_quote")
    logger.log_tool_call(
        tool_name="get_quote",
        tool_input={"ticker": "AAPL"},
        output='{"price": 182.5}',
        cached=False,
        latency_ms=420,
        status="ok",
        ticker="AAPL",
        error_msg=None,
    )
    logger.log(
        "ticker_completed",
        ticker="AAPL",
        recommendation="hold",
        confidence=0.72,
        iterations=2,
        tokens=6363,
        cost_usd=0.03,
        termination="success",
    )
    logger.log("run_completed", status="success", total_cost_usd=0.03, duration_seconds=12.0)


def test_run_produces_jsonl_file(tmp_path: Path) -> None:
    logger = RunLogger("r_file", tmp_path)
    logger.log("run_started", tickers=["AAPL"])
    logger.close()
    assert (tmp_path / "r_file.jsonl").exists()


def test_six_event_types_in_sequence(tmp_path: Path) -> None:
    logger = RunLogger("r_seq", tmp_path)
    _emit_full_run(logger)
    logger.close()

    events = [e["event"] for e in _read_lines(tmp_path / "r_seq.jsonl")]
    assert events == [
        "run_started",
        "ticker_started",
        "llm_call_started",
        "llm_call",
        "tool_call_started",
        "tool_call",
        "ticker_completed",
        "run_completed",
    ]


def test_cache_tokens_separate_and_cost_correct(tmp_path: Path) -> None:
    logger = RunLogger("r_cost", tmp_path)
    logger.log_llm_call(
        _llm_response(4523, 1840, cache_read=4200, cache_creation=323),
        ticker="AAPL",
        phase="deep",
        model=SONNET_4_6,
        latency_ms=100,
    )
    logger.close()

    (call,) = [e for e in _read_lines(tmp_path / "r_cost.jsonl") if e["event"] == "llm_call"]
    # Cache fields are captured separately, not folded into input_tokens.
    assert call["input_tokens"] == 4523
    assert call["cache_read_tokens"] == 4200
    assert call["cache_creation_tokens"] == 323
    assert call["output_tokens"] == 1840
    expected = compute_cost(SONNET_4_6, 4523, 4200, 323, 1840)
    assert call["cost_usd"] == expected


def test_every_line_is_valid_json(tmp_path: Path) -> None:
    """Each event is a complete, atomically-written line — no partial/corrupt JSON."""
    logger = RunLogger("r_atomic", tmp_path)
    _emit_full_run(logger)
    logger.close()
    for line in (tmp_path / "r_atomic.jsonl").read_text().splitlines():
        if line.strip():
            json.loads(line)  # raises if any line is torn


def test_flush_to_db_populates_runs_and_tool_calls(
    db_engine: object, db_session: Session, tmp_path: Path
) -> None:
    logger = RunLogger("r_flush", tmp_path)
    _emit_full_run(logger)
    logger.flush_to_db(db_session)

    run = db_session.get(Run, "r_flush")
    assert run is not None
    assert run.status == "success"
    assert run.total_input_tokens == 4523
    assert run.total_output_tokens == 1840
    assert run.num_tool_calls == 1
    assert run.total_cost_usd == compute_cost(SONNET_4_6, 4523, 4200, 323, 1840)

    rows = db_session.query(ToolCall).filter_by(run_id="r_flush").all()
    assert len(rows) == 1
    assert rows[0].tool_name == "get_quote"
    assert rows[0].input_json == '{"ticker": "AAPL"}'
    assert rows[0].latency_ms == 420


def test_flush_to_db_is_idempotent(db_engine: object, db_session: Session, tmp_path: Path) -> None:
    logger = RunLogger("r_idem", tmp_path)
    _emit_full_run(logger)
    logger.flush_to_db(db_session)
    logger.flush_to_db(db_session)  # replay must not duplicate rows

    rows = db_session.query(ToolCall).filter_by(run_id="r_idem").all()
    assert len(rows) == 1


def test_cancelled_trace_projects_cancelled_status(
    db_engine: object, db_session: Session, tmp_path: Path
) -> None:
    logger = RunLogger("r-cancelled", tmp_path)
    logger.log("run_started", tickers=["AAPL"])
    logger.log("run_completed", status="cancelled")
    logger.flush_to_db(db_session)

    run = db_session.get(Run, "r-cancelled")
    assert run is not None
    assert run.status == "cancelled"


def test_large_tool_sidecar_uses_configured_log_directory(tmp_path: Path) -> None:
    logger = RunLogger("configured-sidecar", tmp_path)
    output = json.dumps({"payload": "x" * 9_000})
    logger.log_tool_call(
        tool_name="get_quote",
        tool_input={"ticker": "AAPL"},
        output=output,
        cached=False,
        latency_ms=12,
        status="ok",
        ticker="AAPL",
        error_msg=None,
    )
    logger.close()

    event = json.loads(logger.path.read_text(encoding="utf-8"))
    marker = json.loads(event["output"])
    sidecar = Path(marker["path"])
    assert sidecar == tmp_path / "configured-sidecar" / "tool_outputs" / "0.json"
    assert sidecar.read_text(encoding="utf-8") == output


def test_sidecar_fsync_failure_never_appends_a_durable_marker(tmp_path: Path) -> None:
    logger = RunLogger("sidecar-failure", tmp_path)
    output = json.dumps({"payload": "x" * 9_000})
    with (
        patch("storage.engine.os.fsync", side_effect=OSError("disk failure")),
        pytest.raises(OSError, match="disk failure"),
    ):
        logger.log_tool_call(
            tool_name="get_quote",
            tool_input={"ticker": "AAPL"},
            output=output,
            cached=False,
            latency_ms=12,
            status="ok",
            ticker="AAPL",
            error_msg=None,
        )
    logger.close()
    assert logger.path.read_text(encoding="utf-8") == ""
    assert not list((tmp_path / "sidecar-failure" / "tool_outputs").glob("*.json"))


def test_incomplete_trace_marked_failed(
    db_engine: object, db_session: Session, tmp_path: Path
) -> None:
    """A trace with no run_completed event (a crash) reconciles to status=failed."""
    logger = RunLogger("r_crash", tmp_path)
    logger.log("run_started", tickers=["AAPL"])
    logger.log_tool_call(
        tool_name="get_quote",
        tool_input={"ticker": "AAPL"},
        output='{"price": 1}',
        cached=False,
        latency_ms=10,
        status="ok",
        ticker="AAPL",
        error_msg=None,
    )
    # no run_completed
    logger.flush_to_db(db_session)

    run = db_session.get(Run, "r_crash")
    assert run is not None
    assert run.status == "failed"
    assert run.error_msg is not None
    assert db_session.query(ToolCall).filter_by(run_id="r_crash").count() == 1


def test_reconcile_orphans_sweeps_running_runs(
    db_engine: object, db_session: Session, tmp_path: Path
) -> None:
    from datetime import datetime, timezone

    # A run marked "running" with a trace on disk but never flushed (crashed mid-run).
    write_run_start("r_orphan", datetime.now(timezone.utc))
    logger = RunLogger("r_orphan", tmp_path)
    logger.log("run_started", tickers=["AAPL"])
    logger.log_tool_call(
        tool_name="get_quote",
        tool_input={"ticker": "AAPL"},
        output='{"price": 1}',
        cached=False,
        latency_ms=10,
        status="ok",
        ticker="AAPL",
        error_msg=None,
    )
    logger.close()

    count = reconcile_orphans(tmp_path)
    assert count == 1

    db_session.expire_all()
    run = db_session.get(Run, "r_orphan")
    assert run is not None
    assert run.status == "failed"  # no run_completed → crashed
    assert db_session.query(ToolCall).filter_by(run_id="r_orphan").count() == 1


def test_reconcile_orphans_tolerates_torn_final_utf8_record(
    db_engine: object, db_session: Session, tmp_path: Path
) -> None:
    from datetime import datetime, timezone

    write_run_start("r_utf8_orphan", datetime.now(timezone.utc))
    trace = tmp_path / "r_utf8_orphan.jsonl"
    trace.write_bytes(
        json.dumps({"event": "run_started", "run_id": "r_utf8_orphan"}).encode()
        + b'\n{"event":"tool_call","output":"\xe2'
    )

    assert reconcile_orphans(tmp_path) == 1
    db_session.expire_all()
    run = db_session.get(Run, "r_utf8_orphan")
    assert run is not None
    assert run.status == "failed"
