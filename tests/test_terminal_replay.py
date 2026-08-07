from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from agent.terminal.replay import ReplayMode, export_failure_bundle, replay_trace
from storage.logger import RunLogger

_CANARY = "warren-canary-must-never-escape"


def _write_sensitive_trace(tmp_path: Path) -> Path:
    logger = RunLogger("run-replay", tmp_path)
    logger.log("run_started", tickers=["AMD"])
    logger.log_tool_started(ticker="AMD", tool_name="get_quote")
    logger.log_tool_call(
        tool_name="get_quote",
        tool_input={
            "ticker": "AMD",
            "headers": {"Authorization": f"Bearer {_CANARY}"},
            "prompt": _CANARY,
        },
        output=json.dumps({"raw": _CANARY, "portfolio_value": 1_000_000}),
        cached=False,
        latency_ms=120,
        status="error",
        ticker="AMD",
        error_msg=f"authorization={_CANARY} at /Users/private-user/secrets.txt",
    )
    logger.log("run_completed", status="success", total_cost_usd=0.0, duration_seconds=0.2)
    logger.close()
    return tmp_path / "run-replay.jsonl"


@pytest.mark.parametrize("mode", ["tty", "pipe", "no_color", "dumb"])
def test_trace_replay_is_offline_semantic_and_mode_deterministic(
    tmp_path: Path,
    mode: ReplayMode,
) -> None:
    trace = _write_sensitive_trace(tmp_path)
    result = replay_trace(trace, width=60, height=10, mode=mode)

    visible = "\n".join(result.snapshot.cells)
    assert "Market quote" in visible
    assert result.snapshot.cursor_visible is True
    assert result.integrity.verdict == "healthy"
    assert _CANARY not in repr(result.events)


def test_sequence_checkpoint_keeps_intermediate_live_cursor_hidden(tmp_path: Path) -> None:
    trace = _write_sensitive_trace(tmp_path)

    result = replay_trace(trace, width=80, height=10, through_sequence=2)

    assert result.snapshot.name == "sequence-2"
    assert result.snapshot.cursor_visible is False
    assert "Using: Market quote" in "\n".join(result.snapshot.cells)


def test_private_bundle_drops_payloads_secrets_values_and_local_paths(tmp_path: Path) -> None:
    trace = _write_sensitive_trace(tmp_path)
    bundle = export_failure_bundle(trace, tmp_path / "bundle" / "failure.json", width=60)

    text = bundle.read_text()
    payload = json.loads(text)
    assert stat.S_IMODE(bundle.stat().st_mode) == 0o600
    assert _CANARY not in text
    assert "1000000" not in text
    assert "private-user" not in text
    assert "<home>/secrets.txt" in text
    assert all("input" not in event and "output" not in event for event in payload["events"])
    assert any(event["dropped_field_count"] >= 2 for event in payload["events"])


def test_replay_rejects_invalid_dimensions_speed_and_malformed_trace(tmp_path: Path) -> None:
    trace = _write_sensitive_trace(tmp_path)
    with pytest.raises(ValueError, match="dimensions"):
        replay_trace(trace, width=0)
    with pytest.raises(ValueError, match="speed"):
        replay_trace(trace, playback_speed=0)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"event": "run_started"}\nnot-json\n')
    with pytest.raises(ValueError, match="line 2"):
        replay_trace(malformed)
