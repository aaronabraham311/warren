"""Integration tests for the agent loop.

All Anthropic API calls are monkeypatched; yfinance is patched where relevant.
SQLite runs in-memory via the db_engine fixture.
"""

import json
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import anthropic
import pytest
from sqlalchemy.orm import Session

from agent.budget import Budget, RunContext
from agent.loop import (
    CostAbortedError,
    SchemaRepairError,
    _iteration_limit,
    analyze_ticker,
)
from agent.models import AnalysisOutput, DirtSignals, LynchBuffettSignals
from agent.persona import DefaultPersona, DirtPersona
from agent.routing import HardcodedSonnetRouting
from agent.tools.base import ErrorCode, ToolResult, ToolResultError, ToolResultOk
from agent.tools.quote import GetQuoteInput, GetQuoteTool
from agent.tools.valuation_history import GetValuationHistoryInput
from data_sources.yfinance_client import PriceData
from storage.engine import upsert_analysis, write_run_end, write_run_start
from storage.logger import RunLogger
from storage.models import Analysis, AnalysisData, Run, ToolCall
from tests.conftest import VALID_ANALYSIS_JSON, make_end_turn, make_tool_use


def _last_user_text(content: list[dict[str, object]] | str) -> str:
    """Extract text from user message content (str or cached block list)."""
    if isinstance(content, str):
        return content
    return str(content[0].get("text", ""))


def _ctx(
    run_id: str = "run-test",
    logger: RunLogger | None = None,
    budget: Budget | None = None,
) -> RunContext:
    # Every RunContext needs a logger; default to a throwaway trace in a temp dir for
    # tests that don't assert on the JSONL/DB projection.
    if logger is None:
        logger = RunLogger(run_id, Path(tempfile.mkdtemp()))
    return RunContext(run_id=run_id, budget=budget or Budget(), logger=logger)


def _persona() -> DefaultPersona:
    return DefaultPersona()


def _routing() -> HardcodedSonnetRouting:
    return HardcodedSonnetRouting()


# ── Happy path ────────────────────────────────────────────────────────────────


def test_happy_path(db_engine: object, mock_claude: MagicMock, db_session: Session) -> None:
    mock_claude(
        [
            make_tool_use("get_quote", {"ticker": "AAPL"}),
            make_end_turn(VALID_ANALYSIS_JSON),
        ]
    )
    ctx = _ctx("run-happy")
    write_run_start("run-happy", datetime.now(timezone.utc))

    result = analyze_ticker("AAPL", _persona(), _routing(), ctx)

    assert isinstance(result, AnalysisOutput)
    assert result.ticker == "AAPL"
    assert result.recommendation in ("buy", "sell", "hold")

    upsert_analysis(
        "run-happy",
        "AAPL",
        AnalysisData(
            analysis_type=result.analysis_type,
            recommendation=result.recommendation,
            confidence=result.confidence,
            thesis=result.thesis,
            lynch_signals=result.lynch_signals.model_dump(),
            buffett_signals=result.buffett_signals.model_dump(),
            key_risks=result.key_risks,
            data_quality_notes=result.data_quality_notes,
            tool_calls_made=ctx.budget.total_tool_calls,
            tokens_used=ctx.budget.total_input_tokens + ctx.budget.total_output_tokens,
            termination_reason=result.termination_reason,
        ),
    )
    write_run_end(
        "run-happy",
        "success",
        ctx.budget.total_input_tokens,
        ctx.budget.total_output_tokens,
        ctx.budget.total_cost_usd,
        ctx.budget.total_tool_calls,
        datetime.now(timezone.utc),
    )

    run = db_session.get(Run, "run-happy")
    assert run is not None
    assert run.status == "success"

    analysis = db_session.query(Analysis).filter_by(run_id="run-happy", ticker="AAPL").first()
    assert analysis is not None
    assert analysis.recommendation == "hold"
    assert analysis.confidence == pytest.approx(0.72)


# ── Tool-call persistence ─────────────────────────────────────────────────────


def test_tool_call_persisted(
    db_engine: object, mock_claude: MagicMock, db_session: Session, tmp_path: Path
) -> None:
    mock_claude(
        [
            make_tool_use("get_quote", {"ticker": "TSLA"}),
            make_end_turn(VALID_ANALYSIS_JSON),
        ]
    )
    logger = RunLogger("run-persist", tmp_path)
    ctx = _ctx("run-persist", logger=logger)
    mock_fast_info = MagicMock()
    mock_fast_info.last_price = 210.0
    mock_fast_info.previous_close = 205.0
    mock_fast_info.three_month_average_volume = 40_000_000
    with patch("data_sources.yfinance_client.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.fast_info = mock_fast_info
        analyze_ticker("TSLA", _persona(), _routing(), ctx)
    # tool_calls are a projection of the JSONL WAL — materialised at reconcile time.
    logger.flush_to_db(db_session)

    rows = db_session.query(ToolCall).filter_by(run_id="run-persist").all()
    assert len(rows) == 1
    row = rows[0]
    assert row.tool_name == "get_quote"
    assert row.input_json == '{"ticker": "TSLA"}'
    assert row.error_msg is None
    assert row.cached is False
    assert row.latency_ms is not None
    assert row.latency_ms >= 0


def test_tool_call_error_persisted(
    db_engine: object, mock_claude: MagicMock, db_session: Session, tmp_path: Path
) -> None:
    mock_claude(
        [
            make_tool_use("get_quote", {"ticker": "AAPL"}),
            make_end_turn(VALID_ANALYSIS_JSON),
        ]
    )
    logger = RunLogger("run-err", tmp_path)
    with patch("data_sources.yfinance_client.yf.Ticker", side_effect=RuntimeError("network error")):
        analyze_ticker("AAPL", _persona(), _routing(), _ctx("run-err", logger=logger))
    logger.flush_to_db(db_session)

    row = db_session.query(ToolCall).filter_by(run_id="run-err").first()
    assert row is not None
    assert row.error_msg is not None
    assert "network error" in row.error_msg


# ── Schema repair ─────────────────────────────────────────────────────────────


def test_schema_repair_success(mock_claude: MagicMock) -> None:
    mock_claude(
        [
            make_end_turn("this is not json at all"),
            make_end_turn(VALID_ANALYSIS_JSON),
        ]
    )
    result = analyze_ticker("AAPL", _persona(), _routing(), _ctx())
    assert result.recommendation == "hold"


def test_schema_repair_failure(mock_claude: MagicMock) -> None:
    mock_claude(
        [
            make_end_turn("bad json #1"),
            make_end_turn("bad json #2"),
        ]
    )
    with pytest.raises(SchemaRepairError):
        analyze_ticker("AAPL", _persona(), _routing(), _ctx())


# ── Iteration cap ─────────────────────────────────────────────────────────────


_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK"]


def test_iteration_cap(db_engine: object, mock_claude: MagicMock) -> None:
    # Use a different ticker each time so tool-loop detection (same input ≥3x) doesn't fire first
    tool_responses = [
        make_tool_use("get_quote", {"ticker": _TICKERS[i]}, tool_id=f"toolu_{i:02d}")
        for i in range(8)
    ]
    mock_client = mock_claude(tool_responses + [make_end_turn(VALID_ANALYSIS_JSON)])
    result = analyze_ticker("AAPL", _persona(), _routing(), _ctx())
    assert isinstance(result, AnalysisOutput)
    last_call_kwargs = mock_client.messages.create.call_args_list[-1][1]
    last_user_msg = last_call_kwargs["messages"][-1]["content"]
    assert "iteration_capped" in _last_user_text(last_user_msg)


# ── Token cap ─────────────────────────────────────────────────────────────────


def test_token_cap(mock_claude: MagicMock) -> None:
    budget = Budget(max_input_tokens=100)
    budget.total_input_tokens = 100  # already at limit
    ctx = _ctx("run-tokencap", budget=budget)
    mock_client = mock_claude([make_end_turn(VALID_ANALYSIS_JSON)])
    result = analyze_ticker("AAPL", _persona(), _routing(), ctx)
    assert isinstance(result, AnalysisOutput)
    last_call_kwargs = mock_client.messages.create.call_args_list[-1][1]
    last_user_msg = last_call_kwargs["messages"][-1]["content"]
    assert "token_capped" in _last_user_text(last_user_msg)


# ── Cost aborted ──────────────────────────────────────────────────────────────


def test_cost_aborted(mock_claude: MagicMock) -> None:
    budget = Budget(max_cost_usd=0.001)
    budget.total_cost_usd = 0.001  # already at ceiling
    ctx = _ctx("run-costabort", budget=budget)
    mock_claude([])
    with pytest.raises(CostAbortedError):
        analyze_ticker("AAPL", _persona(), _routing(), ctx)


# ── Tool-loop broken ──────────────────────────────────────────────────────────


def test_tool_loop_broken(db_engine: object, mock_claude: MagicMock) -> None:
    same_tool = make_tool_use("get_quote", {"ticker": "AAPL"})
    mock_client = mock_claude([same_tool, same_tool, same_tool, make_end_turn(VALID_ANALYSIS_JSON)])
    result = analyze_ticker("AAPL", _persona(), _routing(), _ctx())
    assert isinstance(result, AnalysisOutput)
    last_call_kwargs = mock_client.messages.create.call_args_list[-1][1]
    last_user_msg = last_call_kwargs["messages"][-1]["content"]
    assert "tool_loop_broken" in _last_user_text(last_user_msg)


# ── Forced-final invalid JSON → SchemaRepairError ────────────────────────────


def test_iteration_cap_invalid_json(db_engine: object, mock_claude: MagicMock) -> None:
    tool_responses = [
        make_tool_use("get_quote", {"ticker": _TICKERS[i]}, tool_id=f"toolu_{i:02d}")
        for i in range(8)
    ]
    mock_claude(tool_responses + [make_end_turn("not valid json at all")])
    with pytest.raises(SchemaRepairError):
        analyze_ticker("AAPL", _persona(), _routing(), _ctx())


def test_tool_loop_broken_invalid_json(db_engine: object, mock_claude: MagicMock) -> None:
    same_tool = make_tool_use("get_quote", {"ticker": "AAPL"})
    mock_claude([same_tool, same_tool, same_tool, make_end_turn("not valid json at all")])
    with pytest.raises(SchemaRepairError):
        analyze_ticker("AAPL", _persona(), _routing(), _ctx())


# ── Tool error recovered ──────────────────────────────────────────────────────


def test_tool_error_recovered(db_engine: object, mock_claude: MagicMock) -> None:
    mock_claude(
        [
            make_tool_use("get_quote", {"ticker": "AAPL"}),
            make_end_turn(VALID_ANALYSIS_JSON),
        ]
    )
    with patch("data_sources.yfinance_client.yf.Ticker", side_effect=RuntimeError("network error")):
        result = analyze_ticker("AAPL", _persona(), _routing(), _ctx())
    assert isinstance(result, AnalysisOutput)


# ── GetQuoteTool unit tests ───────────────────────────────────────────────────


def test_get_quote_tool() -> None:
    mock_fast_info = MagicMock()
    mock_fast_info.last_price = 182.50
    mock_fast_info.previous_close = 180.00
    mock_fast_info.three_month_average_volume = 55_000_000

    with patch("data_sources.yfinance_client.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.fast_info = mock_fast_info
        tool = GetQuoteTool()
        result = tool.run(GetQuoteInput(ticker="AAPL"), _ctx())

    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, PriceData)
    assert result.data.ticker == "AAPL"
    assert result.data.current_price == 182.50
    assert result.data.day_change_pct == pytest.approx(1.39)
    assert result.data.volume == 55_000_000


def test_get_quote_tool_error() -> None:
    # yfinance raising surfaces as a DataSourceError(network) from the client, which the
    # tool maps to a structured ToolResultError — never a raised exception.
    with patch("data_sources.yfinance_client.yf.Ticker", side_effect=Exception("timeout")):
        tool = GetQuoteTool()
        result = tool.run(GetQuoteInput(ticker="AAPL"), _ctx())

    assert isinstance(result, ToolResultError)
    assert result.error_code == "network"


# ── Budget cost calculation ───────────────────────────────────────────────────


def test_budget_cost_calculation() -> None:
    budget = Budget()
    budget.record_usage(input_tokens=1000, output_tokens=500, cache_read_tokens=200)
    # Sonnet 4.6: $3/$15 per MTok in/out, $0.30/MTok cache_read
    expected = (1000 * 3.0 + 500 * 15.0 + 200 * 0.30) / 1_000_000
    assert budget.total_cost_usd == pytest.approx(expected, rel=1e-6)
    assert budget.total_input_tokens == 1000
    assert budget.total_output_tokens == 500


def test_budget_token_exceeded() -> None:
    budget = Budget(max_input_tokens=1000)
    budget.total_input_tokens = 1001
    assert budget.token_exceeded() is True

    budget2 = Budget(max_input_tokens=1000)
    budget2.total_input_tokens = 999
    assert budget2.token_exceeded() is False


def test_budget_cost_exceeded() -> None:
    budget = Budget(max_cost_usd=1.0)
    budget.total_cost_usd = 1.0
    assert budget.cost_exceeded() is True

    budget2 = Budget(max_cost_usd=1.0)
    budget2.total_cost_usd = 0.99
    assert budget2.cost_exceeded() is False


# ── JSON list column round-trip ───────────────────────────────────────────────


def test_analysis_list_columns_roundtrip(db_engine: object, db_session: Session) -> None:
    """lynch_signals/buffett_signals/key_risks/data_quality_notes survive a write/read."""
    from datetime import datetime, timezone

    write_run_start("run-jsontest", datetime.now(timezone.utc))
    upsert_analysis(
        "run-jsontest",
        "AAPL",
        AnalysisData(
            analysis_type="holding",
            recommendation="buy",
            confidence=0.8,
            thesis="Strong moat.",
            lynch_signals={"pros": ["dominant brand", "consistent earnings"], "cons": []},
            buffett_signals={"pros": ["high ROE"], "cons": []},
            key_risks=["valuation stretched"],
            data_quality_notes=[],
            tool_calls_made=1,
            tokens_used=500,
        ),
    )

    row = db_session.query(Analysis).filter_by(run_id="run-jsontest", ticker="AAPL").one()
    assert row.lynch_signals == {"pros": ["dominant brand", "consistent earnings"], "cons": []}
    assert row.buffett_signals == {"pros": ["high ROE"], "cons": []}
    assert row.key_risks == ["valuation stretched"]
    assert row.data_quality_notes == []


# ── Retry / backoff ───────────────────────────────────────────────────────────


def _ok_result() -> ToolResultOk:
    return ToolResultOk(
        data=PriceData(
            ticker="AAPL",
            current_price=180.0,
            previous_close=178.0,
            day_change_pct=1.12,
            volume=50_000_000,
            as_of=datetime.now(timezone.utc),
            data_age_hours=0,
        )
    )


def _err(code: ErrorCode, retryable: bool = True) -> ToolResultError:
    return ToolResultError(error_code=code, message=f"{code} error", retryable=retryable)


def _stub_registry(results: list[ToolResult]) -> dict[str, MagicMock]:
    """Return a patched TOOL_REGISTRY dict with a stub get_quote tool."""
    mock_tool = MagicMock()
    mock_tool.input_schema = GetQuoteInput
    mock_tool.run.side_effect = results
    return {"get_quote": mock_tool}


def _run_with_mock_sleep(
    mock_claude: Callable[[list[anthropic.types.Message]], MagicMock],
    registry: dict[str, MagicMock],
    *,
    responses: list[anthropic.types.Message] | None = None,
) -> tuple[AnalysisOutput, MagicMock]:
    """Wire up mock_claude + stub registry, run analyze_ticker with a mock sleep.

    Returns (result, sleep_mock) so callers can assert on sleep_mock.call_count /
    sleep_mock.call_args_list.
    """
    if responses is None:
        responses = [
            make_tool_use("get_quote", {"ticker": "AAPL"}),
            make_end_turn(VALID_ANALYSIS_JSON),
        ]
    mock_claude(responses)
    sleep_mock = MagicMock()
    with patch("agent.loop.TOOL_REGISTRY", registry):
        result = analyze_ticker(
            "AAPL", DefaultPersona(), HardcodedSonnetRouting(), _ctx(), _sleep=sleep_mock
        )
    return result, sleep_mock


def test_retry_rate_limit_success(mock_claude: MagicMock) -> None:
    """rate_limit x2 then ok — two sleeps, tool called 3 times."""
    registry = _stub_registry([_err("rate_limit"), _err("rate_limit"), _ok_result()])
    result, sleep_mock = _run_with_mock_sleep(mock_claude, registry)
    assert isinstance(result, AnalysisOutput)
    assert registry["get_quote"].run.call_count == 3
    assert sleep_mock.call_count == 2
    assert sleep_mock.call_args_list[0][0][0] == pytest.approx(1.0)
    assert sleep_mock.call_args_list[1][0][0] == pytest.approx(2.0)


def test_retry_rate_limit_exhausted(mock_claude: MagicMock) -> None:
    """rate_limit returned 4 times (> max 3 retries) — final error reaches agent."""
    registry = _stub_registry(
        [_err("rate_limit"), _err("rate_limit"), _err("rate_limit"), _err("rate_limit")]
    )
    result, sleep_mock = _run_with_mock_sleep(mock_claude, registry)
    assert isinstance(result, AnalysisOutput)
    assert registry["get_quote"].run.call_count == 4  # 1 initial + 3 retries
    assert sleep_mock.call_count == 3


def test_retry_network_success(mock_claude: MagicMock) -> None:
    """network x1 then ok — one sleep, tool called 2 times."""
    registry = _stub_registry([_err("network"), _ok_result()])
    result, sleep_mock = _run_with_mock_sleep(mock_claude, registry)
    assert isinstance(result, AnalysisOutput)
    assert registry["get_quote"].run.call_count == 2
    assert sleep_mock.call_count == 1
    assert sleep_mock.call_args_list[0][0][0] == pytest.approx(1.0)


def test_retry_network_exhausted(mock_claude: MagicMock) -> None:
    """network returned 3 times (> max 2 retries) — final error reaches agent."""
    registry = _stub_registry([_err("network"), _err("network"), _err("network")])
    result, sleep_mock = _run_with_mock_sleep(mock_claude, registry)
    assert isinstance(result, AnalysisOutput)
    assert registry["get_quote"].run.call_count == 3  # 1 initial + 2 retries
    assert sleep_mock.call_count == 2


def test_no_retry_not_found(mock_claude: MagicMock) -> None:
    """not_found is never retried regardless of retryable flag."""
    registry = _stub_registry([_err("not_found", retryable=False)])
    result, sleep_mock = _run_with_mock_sleep(mock_claude, registry)
    assert isinstance(result, AnalysisOutput)
    assert registry["get_quote"].run.call_count == 1
    assert sleep_mock.call_count == 0


def test_no_retry_stale_data(mock_claude: MagicMock) -> None:
    """stale_data is never retried."""
    registry = _stub_registry([_err("stale_data", retryable=False)])
    result, sleep_mock = _run_with_mock_sleep(mock_claude, registry)
    assert isinstance(result, AnalysisOutput)
    assert registry["get_quote"].run.call_count == 1
    assert sleep_mock.call_count == 0


def test_no_retry_when_retryable_false(mock_claude: MagicMock) -> None:
    """retryable=False on a normally-retryable code skips retry."""
    registry = _stub_registry([_err("rate_limit", retryable=False)])
    result, sleep_mock = _run_with_mock_sleep(mock_claude, registry)
    assert isinstance(result, AnalysisOutput)
    assert registry["get_quote"].run.call_count == 1
    assert sleep_mock.call_count == 0


def test_unknown_error_logged_to_stderr(
    mock_claude: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    """unknown errors are logged to stderr once and not retried."""
    registry = _stub_registry([_err("unknown", retryable=False)])
    result, sleep_mock = _run_with_mock_sleep(mock_claude, registry)
    assert isinstance(result, AnalysisOutput)
    assert registry["get_quote"].run.call_count == 1
    assert sleep_mock.call_count == 0
    captured = capsys.readouterr()
    assert "[warren] unknown tool error" in captured.err


# ── WAL retry fields ───────────────────────────────────────────────────────────


def _wal_tool_events(logger: RunLogger) -> list[dict[str, object]]:
    import json

    return [
        json.loads(line)
        for line in logger.path.read_text().splitlines()
        if json.loads(line).get("event") == "tool_call"
    ]


def test_dirt_valuation_history_call_is_recorded_in_wal(
    mock_claude: MagicMock, tmp_path: Path
) -> None:
    """The G15 persona path can execute the required own-history tool and audit it."""
    mock_tool = MagicMock()
    mock_tool.input_schema = GetValuationHistoryInput
    mock_tool.run.return_value = _ok_result()
    logger = RunLogger("run-dirt-history", tmp_path)
    mock_claude(
        [
            make_tool_use("get_valuation_history", {"ticker": "KPL.WA"}),
            make_end_turn(VALID_ANALYSIS_JSON),
            make_end_turn(VALID_ANALYSIS_JSON),
        ]
    )

    with patch("agent.loop.TOOL_REGISTRY", {"get_valuation_history": mock_tool}):
        with pytest.raises(SchemaRepairError, match="Schema repair failed"):
            analyze_ticker(
                "KPL.WA",
                DirtPersona(),
                HardcodedSonnetRouting(),
                _ctx("run-dirt-history", logger=logger),
            )

    events = _wal_tool_events(logger)
    assert [event["tool"] for event in events] == ["get_valuation_history"]
    assert events[0]["ticker"] == "KPL.WA"


def test_wal_retry_count_recorded(mock_claude: MagicMock, tmp_path: Path) -> None:
    """rate_limit x2 then ok — WAL records retry_count=2, last_retry_error='rate_limit'."""
    registry = _stub_registry([_err("rate_limit"), _err("rate_limit"), _ok_result()])
    logger = RunLogger("run-wal-retry", tmp_path)
    mock_claude(
        [
            make_tool_use("get_quote", {"ticker": "AAPL"}),
            make_end_turn(VALID_ANALYSIS_JSON),
        ]
    )
    sleep_mock = MagicMock()
    with patch("agent.loop.TOOL_REGISTRY", registry):
        analyze_ticker(
            "AAPL",
            DefaultPersona(),
            HardcodedSonnetRouting(),
            _ctx("run-wal-retry", logger=logger),
            _sleep=sleep_mock,
        )

    events = _wal_tool_events(logger)
    assert len(events) == 1
    assert events[0]["retry_count"] == 2
    assert events[0]["last_retry_error"] == "rate_limit"


def test_wal_no_retry_fields_zero(mock_claude: MagicMock, tmp_path: Path) -> None:
    """Clean success — WAL records retry_count=0, last_retry_error=None."""
    registry = _stub_registry([_ok_result()])
    logger = RunLogger("run-wal-noretry", tmp_path)
    mock_claude(
        [
            make_tool_use("get_quote", {"ticker": "AAPL"}),
            make_end_turn(VALID_ANALYSIS_JSON),
        ]
    )
    sleep_mock = MagicMock()
    with patch("agent.loop.TOOL_REGISTRY", registry):
        analyze_ticker(
            "AAPL",
            DefaultPersona(),
            HardcodedSonnetRouting(),
            _ctx("run-wal-noretry", logger=logger),
            _sleep=sleep_mock,
        )

    events = _wal_tool_events(logger)
    assert len(events) == 1
    assert events[0]["retry_count"] == 0
    assert events[0]["last_retry_error"] is None


def test_wal_discovery_cooldown_applied_records_suppressed_count(tmp_path: Path) -> None:
    """agent/run.py's nightly-mode logging call — the event the Today page's
    suppressed-count metric (dashboard.data.cooldown_suppressed_count) reads back."""
    from agent.cooldown import CooldownResult

    logger = RunLogger("run-cooldown", tmp_path)
    cooldown_result = CooldownResult(active=["AAPL"], suppressed=["INTC", "TSLA"])
    logger.log(
        "discovery_cooldown_applied",
        suppressed_count=len(cooldown_result.suppressed),
        suppressed_tickers=cooldown_result.suppressed,
    )
    logger.close()

    lines = (tmp_path / "run-cooldown.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    event = next(e for e in events if e["event"] == "discovery_cooldown_applied")
    assert event["suppressed_count"] == 2
    assert event["suppressed_tickers"] == ["INTC", "TSLA"]


# ── DirtSignals ───────────────────────────────────────────────────────────────


def test_dirt_signals_defaults_to_none() -> None:
    result = AnalysisOutput.model_validate_json(VALID_ANALYSIS_JSON)
    assert result.dirt_signals is None


def test_dirt_signals_round_trips() -> None:
    signals = DirtSignals(
        ev_ebit=4.2,
        price_to_ncav=0.85,
        ncav_discount_pct=15.3,
        net_cash_positive=True,
        consecutive_profit_years=7,
        buyback_active=True,
        insider_sentiment="positive",
        analyst_coverage_count=2,
        aggregator_discrepancies_found=False,
        controller_identified=True,
        controller_name="Founding Family S.p.A.",
        controller_economic_interest_pct=74.99,
        controller_voting_rights_pct=74.99,
        catalyst_strength="observable",
        catalyst_stage="board_authorized",
        catalyst_description="Board-authorized dividend",
        forensic_evidence_ids=["ev-cap-1", "ev-dividend-2"],
        daily_turnover_usd=3_000.0,
        free_float_pct=11.13,
        position_size_cap_usd=6_000.0,
        founder_age_years=74,
        own_history_pb_percentile=5.0,
        closability_status="constrained",
        closability_score=0.2,
        closability_confidence=0.9,
        closability_reasons=["74.99% controller and no observable capital return"],
    )
    output = AnalysisOutput(
        ticker="BARC",
        analysis_type="discovery",
        recommendation="buy",
        confidence=0.8,
        thesis="Deep-value play with NCAV discount and net-cash balance sheet.",
        lynch_signals=LynchBuffettSignals(pros=[], cons=[]),
        buffett_signals=LynchBuffettSignals(pros=[], cons=[]),
        key_risks=["cyclical earnings", "macro headwinds"],
        dirt_signals=signals,
    )
    reparsed = AnalysisOutput.model_validate_json(output.model_dump_json())
    assert reparsed.dirt_signals is not None
    assert reparsed.dirt_signals.ev_ebit == 4.2
    assert reparsed.dirt_signals.price_to_ncav == 0.85
    assert reparsed.dirt_signals.ncav_discount_pct == 15.3
    assert reparsed.dirt_signals.net_cash_positive is True
    assert reparsed.dirt_signals.consecutive_profit_years == 7
    assert reparsed.dirt_signals.buyback_active is True
    assert reparsed.dirt_signals.insider_sentiment == "positive"
    assert reparsed.dirt_signals.analyst_coverage_count == 2
    assert reparsed.dirt_signals.aggregator_discrepancies_found is False
    assert reparsed.dirt_signals.controller_identified is True
    assert reparsed.dirt_signals.controller_name == "Founding Family S.p.A."
    assert reparsed.dirt_signals.controller_economic_interest_pct == 74.99
    assert reparsed.dirt_signals.catalyst_strength == "observable"
    assert reparsed.dirt_signals.catalyst_stage == "board_authorized"
    assert reparsed.dirt_signals.forensic_evidence_ids == ["ev-cap-1", "ev-dividend-2"]
    assert reparsed.dirt_signals.daily_turnover_usd == 3_000.0
    assert reparsed.dirt_signals.free_float_pct == 11.13
    assert reparsed.dirt_signals.position_size_cap_usd == 6_000.0
    assert reparsed.dirt_signals.founder_age_years == 74
    assert reparsed.dirt_signals.own_history_pb_percentile == 5.0
    assert reparsed.dirt_signals.closability_status == "constrained"
    assert reparsed.dirt_signals.closability_score == 0.2
    assert reparsed.dirt_signals.closability_confidence == 0.9


def test_dirt_signals_aggregator_discrepancies_defaults_false() -> None:
    partial = DirtSignals(ev_ebit=5.0)
    assert partial.aggregator_discrepancies_found is False


# ── W4: Idempotent upsert ─────────────────────────────────────────────────────


def test_upsert_analysis_idempotent(db_engine: object, db_session: Session) -> None:
    """Calling upsert_analysis twice with the same (run_id, ticker) yields one row."""
    from datetime import datetime, timezone

    write_run_start("run-idem", datetime.now(timezone.utc))
    data = AnalysisData(
        analysis_type="holding",
        recommendation="hold",
        confidence=0.6,
        thesis="Test idempotency of upsert.",
        lynch_signals={"pros": ["simple business"], "cons": []},
        buffett_signals={"pros": [], "cons": ["no moat"]},
        key_risks=["execution risk"],
        data_quality_notes=[],
        tool_calls_made=2,
        tokens_used=300,
        termination_reason="success",
    )
    upsert_analysis("run-idem", "AAPL", data)
    upsert_analysis("run-idem", "AAPL", data)

    rows = db_session.query(Analysis).filter_by(run_id="run-idem", ticker="AAPL").all()
    assert len(rows) == 1


# ── W4: termination_reason set by loop ───────────────────────────────────────


def test_schema_repair_sets_termination_reason(mock_claude: MagicMock) -> None:
    """Schema repair success sets termination_reason='schema_repair_success'."""
    mock_claude(
        [
            make_end_turn("this is not json at all"),
            make_end_turn(VALID_ANALYSIS_JSON),
        ]
    )
    result = analyze_ticker("AAPL", _persona(), _routing(), _ctx())
    assert result.termination_reason == "schema_repair_success"


def test_iteration_cap_sets_termination_reason(db_engine: object, mock_claude: MagicMock) -> None:
    """Hitting the iteration cap sets termination_reason='iteration_capped'."""
    _TICKERS_LOCAL = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK"]
    tool_responses = [
        make_tool_use("get_quote", {"ticker": _TICKERS_LOCAL[i]}, tool_id=f"toolu_{i:02d}")
        for i in range(8)
    ]
    mock_claude(tool_responses + [make_end_turn(VALID_ANALYSIS_JSON)])
    result = analyze_ticker("AAPL", _persona(), _routing(), _ctx())
    assert result.termination_reason == "iteration_capped"


def test_dirt_persona_has_room_for_required_decision_call() -> None:
    assert _iteration_limit(DefaultPersona()) == 8
    assert _iteration_limit(DirtPersona()) == 12


# ── W4: analyses rows validate against AnalysisOutput ────────────────────────


def test_analyses_validate_against_output_schema(db_engine: object, db_session: Session) -> None:
    """All analyses rows round-trip through AnalysisOutput.model_validate."""
    from datetime import datetime, timezone

    write_run_start("run-validate", datetime.now(timezone.utc))
    upsert_analysis(
        "run-validate",
        "MSFT",
        AnalysisData(
            analysis_type="holding",
            recommendation="buy",
            confidence=0.8,
            thesis="Strong cloud growth and durable enterprise moat justify a buy.",
            lynch_signals={"pros": ["fast grower"], "cons": []},
            buffett_signals={"pros": ["high ROIC"], "cons": []},
            key_risks=["valuation stretched", "macro slowdown"],
            data_quality_notes=[],
            tool_calls_made=3,
            tokens_used=450,
            termination_reason="success",
        ),
    )

    row = db_session.query(Analysis).filter_by(run_id="run-validate", ticker="MSFT").one()
    validated = AnalysisOutput.model_validate(
        {
            "ticker": row.ticker,
            "analysis_type": row.analysis_type,
            "recommendation": row.recommendation,
            "confidence": row.confidence,
            "thesis": row.thesis,
            "lynch_signals": row.lynch_signals,
            "buffett_signals": row.buffett_signals,
            "key_risks": row.key_risks,
            "data_quality_notes": row.data_quality_notes or [],
            "tool_calls_made": row.tool_calls_made or 0,
            "tokens_used": row.tokens_used or 0,
            "termination_reason": row.termination_reason or "success",
        }
    )
    assert validated.ticker == "MSFT"
    assert validated.recommendation == "buy"
    assert validated.termination_reason == "success"


# ── W4: multi-ticker full-run writes multiple rows ────────────────────────────


def test_full_run_writes_multiple_rows(
    db_engine: object, mock_claude: MagicMock, db_session: Session, tmp_path: Path
) -> None:
    """Analysing two tickers in sequence writes two rows to the analyses table."""
    from datetime import datetime, timezone

    mock_claude(
        [
            make_tool_use("get_quote", {"ticker": "AAPL"}),
            make_end_turn(VALID_ANALYSIS_JSON),
            make_tool_use("get_quote", {"ticker": "MSFT"}),
            make_end_turn(VALID_ANALYSIS_JSON.replace('"AAPL"', '"MSFT"')),
        ]
    )
    run_id = "run-multirow"
    write_run_start(run_id, datetime.now(timezone.utc))
    budget = Budget()
    logger = RunLogger(run_id, tmp_path)

    for ticker, analysis_type in [("AAPL", "holding"), ("MSFT", "discovery")]:
        ctx = RunContext(run_id=run_id, budget=budget, logger=logger)
        result = analyze_ticker(ticker, _persona(), _routing(), ctx)
        result.analysis_type = analysis_type  # type: ignore[assignment]
        upsert_analysis(
            run_id,
            ticker,
            AnalysisData(
                analysis_type=result.analysis_type,
                recommendation=result.recommendation,
                confidence=result.confidence,
                thesis=result.thesis,
                lynch_signals=result.lynch_signals.model_dump(),
                buffett_signals=result.buffett_signals.model_dump(),
                key_risks=result.key_risks,
                data_quality_notes=result.data_quality_notes,
                tool_calls_made=budget.total_tool_calls,
                tokens_used=budget.total_input_tokens + budget.total_output_tokens,
                termination_reason=result.termination_reason,
            ),
        )

    rows = db_session.query(Analysis).filter_by(run_id=run_id).all()
    assert len(rows) == 2
    tickers_written = {r.ticker for r in rows}
    assert tickers_written == {"AAPL", "MSFT"}
    assert all(r.termination_reason == "success" for r in rows)
