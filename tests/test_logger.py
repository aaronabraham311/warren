"""Tests for the JSONL WAL (RunLogger) and its reconciliation into SQLite."""

import json
from pathlib import Path

import anthropic
from sqlalchemy.orm import Session

from agent.models import GEMINI_3_6_FLASH, SONNET_4_6, TERRA_5_6
from agent.providers.base import ProviderResponse, TextBlock, Usage
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
    logger.log_llm_call(
        _llm_response(4523, 1840, cache_read=4200, cache_creation=323),
        ticker="AAPL",
        phase="deep",
        model=SONNET_4_6,
        latency_ms=3200,
    )
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
        "llm_call",
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
    assert call["reasoning_tokens"] is None
    assert call["tool_use_tokens"] == 0
    assert call["provider"] == "anthropic"
    assert call["service_tier"] == "default"
    assert isinstance(call["raw_usage"], dict)
    expected = compute_cost(SONNET_4_6, 4523, 4200, 323, 1840)
    assert call["cost_usd"] == expected


def test_normalized_openai_usage_is_logged_and_priced_once(
    tmp_path: Path, db_session: Session
) -> None:
    response = ProviderResponse(
        blocks=(TextBlock("ok"),),
        stop_reason="end_turn",
        model_id=TERRA_5_6,
        usage=Usage(
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=200,
            cache_write_tokens=100,
            reasoning_tokens=250,
            total_tokens=1800,
            raw={
                "input_tokens": 1200,
                "output_tokens": 500,
                "output_tokens_details": {"reasoning_tokens": 250},
            },
        ),
    )
    logger = RunLogger("r_openai", tmp_path)
    logger.log_llm_call(
        response,
        ticker="AAPL",
        phase="deep",
        provider="openai",
        service_tier="flex",
        reasoning_effort="medium",
        latency_ms=20,
    )
    logger.close()
    logger.flush_to_db(db_session)

    (call,) = _read_lines(tmp_path / "r_openai.jsonl")
    assert call["provider"] == "openai"
    assert call["model"] == TERRA_5_6
    assert call["service_tier"] == "flex"
    assert call["reasoning_effort"] == "medium"
    assert call["input_tokens"] == 1000
    assert call["cache_read_tokens"] == 200
    assert call["cache_creation_tokens"] == 100
    assert call["output_tokens"] == 500
    assert call["reasoning_tokens"] == 250
    assert call["raw_usage"] == response.usage.raw
    assert call["cost_usd"] == compute_cost(
        TERRA_5_6, 1000, 200, 100, 500, provider="openai", service_tier="flex"
    )
    run = db_session.get(Run, "r_openai")
    assert run is not None
    assert run.total_input_tokens == 1000
    assert run.total_output_tokens == 500
    assert run.total_cost_usd == call["cost_usd"]


def test_normalized_gemini_tool_use_is_an_input_subdivision(tmp_path: Path) -> None:
    response = ProviderResponse(
        blocks=(TextBlock("ok"),),
        stop_reason="end_turn",
        model_id=GEMINI_3_6_FLASH,
        usage=Usage(
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=200,
            reasoning_tokens=300,
            tool_use_tokens=150,
            total_tokens=1700,
            raw={"tool_use_prompt_token_count": 150, "thoughts_token_count": 300},
        ),
    )
    logger = RunLogger("r_gemini", tmp_path)
    logger.log_llm_call(
        response,
        ticker="AAPL",
        phase="deep",
        provider="gemini",
        service_tier="flex",
        reasoning_effort="high",
        latency_ms=20,
    )
    logger.close()

    (call,) = _read_lines(tmp_path / "r_gemini.jsonl")
    assert call["output_tokens"] == 500
    assert call["reasoning_tokens"] == 300
    assert call["tool_use_tokens"] == 150
    assert call["cost_usd"] == compute_cost(
        GEMINI_3_6_FLASH, 1000, 200, 0, 500, provider="gemini", service_tier="flex"
    )


def test_normalized_response_uses_requested_model_for_billing(tmp_path: Path) -> None:
    response = ProviderResponse(
        blocks=(TextBlock("ok"),),
        stop_reason="completed",
        model_id="gpt-5.6-terra-2026-08-01",
        usage=Usage(input_tokens=1000, output_tokens=500),
    )
    logger = RunLogger("r_canonical", tmp_path)
    logger.log_llm_call(
        response,
        ticker="AAPL",
        phase="deep",
        model=TERRA_5_6,
        provider="openai",
        service_tier="flex",
        latency_ms=20,
    )
    logger.close()

    (call,) = _read_lines(tmp_path / "r_canonical.jsonl")
    assert call["model"] == TERRA_5_6
    assert call["response_model"] == "gpt-5.6-terra-2026-08-01"
    assert call["cost_usd"] == compute_cost(
        TERRA_5_6, 1000, 0, 0, 500, provider="openai", service_tier="flex"
    )


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
