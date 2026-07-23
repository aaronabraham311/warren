import json
from pathlib import Path

import pytest

from eval.analysis.trace_tools import load_tool_calls, main


def _write_trace(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def test_load_tool_calls_groups_by_ticker(tmp_path: Path) -> None:
    trace = tmp_path / "run-1.jsonl"
    _write_trace(
        trace,
        [
            {"event": "run_started", "run_id": "run-1"},
            {"event": "tool_call", "ticker": "AAPL", "tool": "get_quote"},
            {"event": "tool_call", "ticker": "AAPL", "tool": "get_fundamentals"},
            {"event": "tool_call", "ticker": "MSFT", "tool": "get_quote"},
        ],
    )

    calls = load_tool_calls("run-1", tmp_path)

    assert calls == {
        "AAPL": ["get_quote", "get_fundamentals"],
        "MSFT": ["get_quote"],
    }


def test_main_flags_uncalled_core_coverage_tools(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trace = tmp_path / "run-2.jsonl"
    _write_trace(
        trace,
        [
            {"event": "tool_call", "ticker": "AAPL", "tool": "get_quote"},
            {"event": "tool_call", "ticker": "AAPL", "tool": "read_filing"},
        ],
    )

    exit_code = main(["run-2", "--logs-dir", str(tmp_path)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "AAPL: get_quote, read_filing" in out
    assert "! never called:" in out
    assert "get_capital_allocation" in out
    assert "read_filing" not in out.split("never called:")[1]
