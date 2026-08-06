from __future__ import annotations

import json
from pathlib import Path

from agent.lifecycle import validate_trace
from storage.logger import TRACE_SCHEMA_VERSION, RunLogger


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_logger_envelope_sequences_and_pairs_operations(tmp_path: Path) -> None:
    clock_values = iter([1.0, 1.1, 1.2, 1.5, 1.6, 1.8, 1.9, 2.0])
    logger = RunLogger("run-1", tmp_path, monotonic=lambda: next(clock_values))
    logger.log("run_started", tickers=["AMD"])
    logger.log("ticker_started", ticker="AMD")
    logger.log(
        "llm_call_started",
        ticker="AMD",
        model="sonnet",
        purpose="synthesis",
        iteration=1,
    )
    logger.log(
        "llm_call",
        ticker="AMD",
        model="sonnet",
        purpose="synthesis",
        iteration=1,
        latency_ms=300,
    )
    logger.log_tool_started(ticker="AMD", tool_name="get_quote")
    logger.log(
        "tool_call",
        ticker="AMD",
        tool="get_quote",
        status="ok",
        latency_ms=200,
    )
    logger.log("ticker_completed", ticker="AMD", status="success")
    logger.log("run_completed", status="success", duration_seconds=1.0)
    logger.close()

    records = _records(tmp_path / "run-1.jsonl")
    assert [record["sequence"] for record in records] == list(range(1, 9))
    assert {record["schema_version"] for record in records} == {TRACE_SCHEMA_VERSION}
    assert [record["monotonic_ms"] for record in records] == [
        1000,
        1100,
        1200,
        1500,
        1600,
        1800,
        1900,
        2000,
    ]
    assert records[2]["operation_id"] == records[3]["operation_id"]
    assert records[4]["operation_id"] == records[5]["operation_id"]
    assert records[1]["parent_operation_id"] == records[0]["operation_id"]

    summary = validate_trace(records)
    assert summary.verdict == "healthy"
    assert summary.current_or_final_phase == "success"
    assert summary.unmatched_starts == 0
    assert [(item.kind, item.duration_ms) for item in summary.slowest] == [
        ("model", 300),
        ("run", 1000),
        ("tool", 200),
    ]


def test_validator_reports_sequence_orphan_parent_duration_and_missing_outcome() -> None:
    summary = validate_trace(
        [
            {
                "schema_version": 1,
                "sequence": 1,
                "event": "run_started",
                "operation_id": "run-op",
            },
            {
                "schema_version": 1,
                "sequence": 3,
                "event": "tool_call_started",
                "operation_id": "tool-op",
                "parent_operation_id": "missing",
                "tool": "get_quote",
            },
            {
                "schema_version": 1,
                "sequence": 4,
                "event": "tool_call",
                "operation_id": "orphan",
                "tool": "get_quote",
                "latency_ms": -1,
                "status": "error",
                "retry_count": 2,
            },
        ]
    )

    codes = {issue.code for issue in summary.issues}
    assert codes == {
        "invalid_parent",
        "missing_run_outcome",
        "negative_duration",
        "orphan_outcome",
        "sequence_gap",
        "unmatched_start",
    }
    assert summary.verdict == "degraded"
    assert summary.unmatched_starts == 2
    assert summary.retries == 2
    assert summary.failures == 1


def test_validator_ignores_unknown_future_event_payload() -> None:
    summary = validate_trace(
        [
            {
                "schema_version": TRACE_SCHEMA_VERSION + 1,
                "sequence": 1,
                "event": "future_secret_event",
                "unknown": {"nested": "payload"},
            }
        ]
    )
    assert summary.verdict == "healthy"
