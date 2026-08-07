"""Process-like restart coverage for the terminal's read-only persistence commands."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import storage.engine as storage_engine
from agent.cancellation import CancellationToken
from agent.service import RunRequest, RunResult
from agent.terminal.app import create_app
from agent.terminal.renderer import TerminalRenderer
from agent.terminal.settings import TerminalSettings
from storage.models import Analysis, Base, Run, Watchlist


class _ForbiddenExecutor:
    """Fail loudly if an offline persistence command crosses into the run service."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        request: RunRequest,
        *,
        event_sink: TerminalRenderer,
        cancellation: CancellationToken,
    ) -> RunResult:
        del request, event_sink, cancellation
        self.calls += 1
        raise AssertionError("read-only terminal commands must not invoke the run service")


def _analysis(
    run_id: str,
    ticker: str,
    recommendation: str,
    thesis: str,
    *,
    termination_reason: str = "success",
) -> Analysis:
    return Analysis(
        run_id=run_id,
        ticker=ticker,
        analysis_type="holding",
        recommendation=recommendation,
        confidence=0.75,
        thesis=thesis,
        lynch_signals={"pros": ["growth"], "cons": []},
        buffett_signals={"pros": ["moat"], "cons": ["valuation"]},
        key_risks=["execution"],
        data_quality_notes=["one stale field"],
        tool_calls_made=2,
        tokens_used=150,
        termination_reason=termination_reason,
    )


def test_fresh_terminal_reads_sqlite_and_authoritative_wal_without_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restarted shell retains DB/WAL order and tolerates a torn final trace line."""
    database_path = tmp_path / "warren.db"
    log_dir = tmp_path / "logs" / "runs"
    log_dir.mkdir(parents=True)
    started = datetime(2026, 8, 6, 8, 0)

    # Seed persistence through a separate engine, then dispose it. The terminal below
    # must discover the file from its environment just as a new process would.
    seed_engine = create_engine(f"sqlite:///{database_path}")
    Base.metadata.create_all(seed_engine)
    with Session(seed_engine) as session:
        session.add_all(
            [
                Run(
                    id="run-old",
                    started_at=started,
                    completed_at=started + timedelta(minutes=1),
                    status="success",
                    total_input_tokens=50,
                    total_output_tokens=10,
                    total_cost_usd=0.01,
                    num_tool_calls=1,
                ),
                Run(
                    id="run-new",
                    started_at=started + timedelta(hours=1),
                    completed_at=started + timedelta(hours=1, minutes=2),
                    status="success",
                    total_input_tokens=120,
                    total_output_tokens=30,
                    total_cost_usd=0.04,
                    num_tool_calls=2,
                ),
            ]
        )
        session.flush()
        session.add_all(
            [
                _analysis("run-old", "KO", "hold", "Older stored thesis."),
                _analysis(
                    "run-new",
                    "MSFT",
                    "buy",
                    "Persisted partial analysis.",
                    termination_reason="iteration_capped",
                ),
                Watchlist(ticker="COST", notes="wait for a wider margin of safety"),
            ]
        )
        session.commit()
    seed_engine.dispose()

    durable_records = [
        {
            "ts": "2026-08-06T09:00:00Z",
            "run_id": "run-new",
            "event": "run_started",
            "tickers": ["AAPL", "MSFT"],
        },
        {
            "ts": "2026-08-06T09:00:00.500000Z",
            "run_id": "run-new",
            "event": "ticker_failed",
            "ticker": "AAPL",
            "error": "upstream unavailable",
        },
        {
            "ts": "2026-08-06T09:00:01Z",
            "run_id": "run-new",
            "event": "tool_call",
            "ticker": "MSFT",
            "tool": "get_quote",
            "status": "success",
            "cached": True,
        },
    ]
    trace_path = log_dir / "run-new.jsonl"
    trace_path.write_bytes(
        b"\n".join(json.dumps(record).encode() for record in durable_records)
        + b'\n{"event":"tool_call","output":"\xe2'
    )

    monkeypatch.setenv("WARREN_DB", str(database_path))
    monkeypatch.setenv("WARREN_LOGS_DIR", str(log_dir))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Reset the module singleton so this app cannot reuse an engine from either the
    # test seed or another test, mirroring import state in a newly started process.
    previous_engine = storage_engine.engine
    if previous_engine is not None:
        previous_engine.dispose()
    monkeypatch.setattr(storage_engine, "engine", None)

    stdout = StringIO()
    stderr = StringIO()
    executor = _ForbiddenExecutor()
    app = create_app(
        stdin=StringIO("/history\n/show run-new\n/trace run-new\n/watchlist\n/quit\n"),
        stdout=stdout,
        stderr=stderr,
        executor=executor,
        settings=TerminalSettings(color="never"),
        api_key_available=lambda: False,
        width=200,
    )

    assert app.run() == 0
    assert executor.calls == 0

    output = stdout.getvalue()
    assert output.index("run-new | success") < output.index("run-old | success")
    assert output.index("AAPL | ERROR | upstream unavailable") < output.index(
        "MSFT | BUY | 75% | Persisted partial analysis."
    )
    assert "Termination: iteration_capped" in output
    assert output.index("2026-08-06T09:00:00Z | run_started") < output.index(
        "2026-08-06T09:00:01Z | tool_call | MSFT"
    )
    assert "tool=get_quote, status=success, cached=True" in output
    assert "COST | wait for a wider margin of safety" in output
    assert "Trace warning: Ignored a torn final JSONL record." in stderr.getvalue()
    assert "ANTHROPIC_API_KEY" not in stdout.getvalue() + stderr.getvalue()

    restarted_engine = storage_engine.engine
    if restarted_engine is not None:
        restarted_engine.dispose()
