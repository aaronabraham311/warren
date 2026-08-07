from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from agent.budget import Budget
from agent.cancellation import CancellationToken, NeverCancelToken, RunCancelledError
from agent.events import NullEventSink
from agent.locking import RunLock
from agent.loop import CostAbortedError
from agent.models import AnalysisOutput, LynchBuffettSignals
from agent.persona import DefaultPersona
from agent.routing import PhaseBasedRouting
from agent.service import (
    RunMode,
    RunRequest,
    TickerRunResult,
    _run_targets,
    execute_run,
)
from storage.logger import RunLogger
from storage.models import Analysis, Run


def _analysis(ticker: str = "AAPL") -> AnalysisOutput:
    return AnalysisOutput(
        ticker=ticker,
        analysis_type="holding",
        recommendation="hold",
        confidence=0.72,
        thesis="A sufficiently detailed structured thesis for the service test.",
        lynch_signals=LynchBuffettSignals(pros=["Growth"], cons=["Price"]),
        buffett_signals=LynchBuffettSignals(pros=["Moat"], cons=["Valuation"]),
        key_risks=["Execution"],
    )


def test_run_request_normalizes_and_validates_tickers() -> None:
    request = RunRequest(mode=RunMode.TICKERS, tickers=[" aapl ", "brk.b"])
    assert request.tickers == ["AAPL", "BRK.B"]

    with pytest.raises(ValidationError, match="duplicate ticker"):
        RunRequest(mode=RunMode.TICKERS, tickers=["AAPL", "aapl"])
    with pytest.raises(ValidationError, match="requires at least one"):
        RunRequest(mode=RunMode.TICKERS)
    with pytest.raises(ValidationError, match="does not accept"):
        RunRequest(mode=RunMode.PORTFOLIO, tickers=["AAPL"])


def test_run_targets_keeps_order_and_continues_after_ticker_failure(tmp_path: Path) -> None:
    logger = RunLogger("partial-run", tmp_path)
    second = _analysis("MSFT")
    with patch(
        "agent.service._analyze_and_persist",
        side_effect=[ValueError("bad first ticker"), second],
    ) as analyze:
        status, error_msg, results = _run_targets(
            [("AAPL", "holding"), ("MSFT", "discovery")],
            run_id="partial-run",
            budget=Budget(),
            logger=logger,
            persona=DefaultPersona(),
            routing_policy=PhaseBasedRouting(),
            client=MagicMock(),
            portfolio_context="",
            cancellation=NeverCancelToken(),
            event_sink=NullEventSink(),
        )
    logger.close()

    assert analyze.call_count == 2
    assert status == "success"
    assert error_msg is None
    assert results == [
        TickerRunResult("AAPL", "holding", error="bad first ticker"),
        TickerRunResult("MSFT", "discovery", analysis=second),
    ]


def test_run_targets_stops_after_cost_abort(tmp_path: Path) -> None:
    logger = RunLogger("cost-run", tmp_path)
    with patch(
        "agent.service._analyze_and_persist",
        side_effect=CostAbortedError("ceiling reached"),
    ) as analyze:
        status, error_msg, results = _run_targets(
            [("AAPL", "holding"), ("MSFT", "holding")],
            run_id="cost-run",
            budget=Budget(),
            logger=logger,
            persona=DefaultPersona(),
            routing_policy=PhaseBasedRouting(),
            client=MagicMock(),
            portfolio_context="",
            cancellation=NeverCancelToken(),
            event_sink=NullEventSink(),
        )
    logger.close()

    assert analyze.call_count == 1
    assert status == "cost_aborted"
    assert error_msg == "ceiling reached"
    assert results == [TickerRunResult("AAPL", "holding", error="ceiling reached")]


def test_run_targets_checks_cancellation_before_first_ticker(tmp_path: Path) -> None:
    logger = RunLogger("cancel-run", tmp_path)
    cancellation = CancellationToken()
    cancellation.cancel()
    with (
        patch("agent.service._analyze_and_persist") as analyze,
        pytest.raises(RunCancelledError),
    ):
        _run_targets(
            [("AAPL", "holding")],
            run_id="cancel-run",
            budget=Budget(),
            logger=logger,
            persona=DefaultPersona(),
            routing_policy=PhaseBasedRouting(),
            client=MagicMock(),
            portfolio_context="",
            cancellation=cancellation,
            event_sink=NullEventSink(),
        )
    logger.close()
    analyze.assert_not_called()


def test_execute_run_persists_analysis_and_reconciles_trace(
    db_engine: object,
    db_session: Session,
    tmp_path: Path,
) -> None:
    del db_engine
    output = _analysis()
    logs = tmp_path / "logs"
    state = tmp_path / "state"
    with (
        patch("agent.service.migrate"),
        patch("agent.service.reconcile_orphans", return_value=0),
        patch("agent.service.sync_input_data"),
        patch("agent.service.load_portfolio", return_value=[]),
        patch("agent.service.load_watchlist", return_value=[]),
        patch("agent.service.analyze_ticker", return_value=output),
    ):
        result = execute_run(
            RunRequest(mode=RunMode.TICKERS, tickers=["AAPL"]),
            event_sink=NullEventSink(),
            cancellation=NeverCancelToken(),
            log_dir=logs,
            runtime_state_dir=state,
            client=MagicMock(),
        )

    assert result.status == "success"
    assert result.analyses == (output,)
    assert result.ticker_results[0].ticker == "AAPL"
    stored = db_session.scalar(select(Analysis).where(Analysis.run_id == result.run_id))
    assert stored is not None
    assert stored.ticker == "AAPL"
    events = [
        json.loads(line)
        for line in (logs / f"{result.run_id}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "run_started",
        "ticker_started",
        "ticker_completed",
        "run_completed",
    ]


def test_keyboard_interrupt_is_persisted_as_cancelled_and_releases_lock(
    db_engine: object,
    db_session: Session,
    tmp_path: Path,
) -> None:
    del db_engine
    logs = tmp_path / "logs"
    state = tmp_path / "state"
    with (
        patch("agent.service.migrate"),
        patch("agent.service.reconcile_orphans", return_value=0),
        patch("agent.service.sync_input_data"),
        patch("agent.service.load_portfolio", return_value=[]),
        patch("agent.service.load_watchlist", return_value=[]),
        patch("agent.service.analyze_ticker", side_effect=KeyboardInterrupt),
    ):
        result = execute_run(
            RunRequest(mode=RunMode.TICKERS, tickers=["AAPL"]),
            log_dir=logs,
            runtime_state_dir=state,
            client=MagicMock(),
        )

    assert result.status == "cancelled"
    db_session.expire_all()
    stored = db_session.get(Run, result.run_id)
    assert stored is not None
    assert stored.status == "cancelled"
    with RunLock(state).acquire(run_id="next-run", mode="portfolio"):
        pass


@pytest.mark.parametrize("failure_point", ["build_portfolio_context", "RunLogger"])
def test_pre_lifecycle_setup_failure_does_not_leave_running_row(
    failure_point: str,
    db_engine: object,
    db_session: Session,
    tmp_path: Path,
) -> None:
    del db_engine
    patches = [
        patch("agent.service.migrate"),
        patch("agent.service.reconcile_orphans", return_value=0),
        patch("agent.service.sync_input_data"),
        patch("agent.service.load_portfolio", return_value=[]),
        patch("agent.service.load_watchlist", return_value=[]),
        patch(f"agent.service.{failure_point}", side_effect=OSError("setup failed")),
    ]
    for active_patch in patches:
        active_patch.start()
    try:
        with pytest.raises(OSError, match="setup failed"):
            execute_run(
                RunRequest(mode=RunMode.TICKERS, tickers=["AAPL"]),
                log_dir=tmp_path / "logs",
                runtime_state_dir=tmp_path / "state",
                client=MagicMock(),
            )
    finally:
        for active_patch in reversed(patches):
            active_patch.stop()

    assert list(db_session.scalars(select(Run))) == []
