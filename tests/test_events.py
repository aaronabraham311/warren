import json
from pathlib import Path
from unittest.mock import patch

from agent.events import (
    EventSink,
    LlmCallCompleted,
    RunCancelled,
    RunEvent,
    RunStarted,
    ToolCallCompleted,
    event_from_wal_record,
)
from storage.logger import RunLogger


class _CollectingSink(EventSink):
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def emit(self, event: RunEvent) -> None:
        self.events.append(event)


def test_wal_projection_is_typed_and_redacts_payloads() -> None:
    event = event_from_wal_record(
        {
            "event": "tool_call",
            "run_id": "r1",
            "ticker": "AAPL",
            "tool": "get_quote",
            "input": {"api_key": "secret"},
            "output": "raw payload",
            "status": "ok",
            "cached": True,
            "latency_ms": 12,
            "retry_count": 1,
        }
    )
    assert isinstance(event, ToolCallCompleted)
    assert event.run_id == "r1"
    assert event.ticker == "AAPL"
    assert event.tool_name == "get_quote"
    assert event.status == "ok"
    assert event.cached is True
    assert event.latency_ms == 12
    assert event.retry_count == 1
    assert "secret" not in repr(event)
    assert "raw payload" not in repr(event)


def test_logger_emits_only_after_fsync(tmp_path: Path) -> None:
    order: list[str] = []

    class _OrderingSink:
        def emit(self, event: RunEvent) -> None:
            assert isinstance(event, RunStarted)
            order.append("emit")

    logger = RunLogger("r-order", tmp_path, event_sink=_OrderingSink())
    with patch("storage.logger.os.fsync", side_effect=lambda _fd: order.append("fsync")):
        logger.log("run_started", tickers=["AAPL"])
    logger.close()
    assert order == ["fsync", "emit"]


def test_display_sink_failure_does_not_lose_durable_wal(tmp_path: Path) -> None:
    class _BrokenSink:
        def emit(self, event: RunEvent) -> None:
            del event
            raise RuntimeError("renderer broke")

    logger = RunLogger("r-broken", tmp_path, event_sink=_BrokenSink())
    logger.log("run_started", tickers=["AAPL"])
    logger.close()
    record = json.loads((tmp_path / "r-broken.jsonl").read_text())
    assert record["event"] == "run_started"


def test_completion_projection_distinguishes_cancelled_and_llm_metrics() -> None:
    cancelled = event_from_wal_record(
        {"event": "run_completed", "run_id": "r1", "status": "cancelled"}
    )
    assert isinstance(cancelled, RunCancelled)
    assert cancelled.run_id == "r1"
    llm = event_from_wal_record(
        {
            "event": "llm_call",
            "run_id": "r1",
            "ticker": "AAPL",
            "model": "model",
            "latency_ms": 9,
            "input_tokens": 10,
            "output_tokens": 2,
            "cache_read_tokens": 7,
            "cache_creation_tokens": 3,
            "cost_usd": 0.01,
            "prompt": "must not escape",
        }
    )
    assert isinstance(llm, LlmCallCompleted)
    assert llm.input_tokens == 10
    assert llm.output_tokens == 2
    assert llm.cache_read_tokens == 7
    assert llm.cache_creation_tokens == 3
    assert llm.cost_usd == 0.01
