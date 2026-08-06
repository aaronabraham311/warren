import json
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.budget import Budget, RunContext
from agent.models import DirtDecisionContract, DirtDownsideFloorAssumption, DirtScenarioAssumption
from agent.tools import TOOL_REGISTRY
from agent.tools.base import Tool, ToolResultError, ToolResultOk
from agent.tools.dirt_scenarios import ModelDirtScenariosInput
from dashboard.seed_demo import _BUY_DECISION
from data_sources.yfinance_client import PriceData
from eval.tool_fixtures import (
    FixtureMiss,
    FixtureToolRunner,
    has_tool_fixtures,
    record_tool_result,
    tool_fixture_path,
    tool_input_hash,
)
from storage.logger import RunLogger


@pytest.fixture()
def ctx(tmp_path: Path) -> RunContext:
    return RunContext(run_id="eval-test", budget=Budget(), logger=RunLogger("eval-test", tmp_path))


def _quote_tool() -> Tool:
    return TOOL_REGISTRY["get_quote"]


def _price() -> PriceData:
    return PriceData(
        ticker="AAPL",
        current_price=190.5,
        previous_close=188.0,
        day_change_pct=1.33,
        volume=50_000_000,
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        data_age_hours=1,
    )


def test_input_hash_is_stable_and_order_independent() -> None:
    assert tool_input_hash({"ticker": "AAPL", "a": 1}) == tool_input_hash(
        {"a": 1, "ticker": "AAPL"}
    )
    assert tool_input_hash({"ticker": "AAPL"}) != tool_input_hash({"ticker": "MSFT"})


def test_fixture_path_layout(tmp_path: Path) -> None:
    path = tool_fixture_path("AAPL", "get_quote", {"ticker": "AAPL"}, tmp_path)
    assert path.parent == tmp_path / "AAPL" / "tools" / "get_quote"
    assert path.name == f"{tool_input_hash({'ticker': 'AAPL'})}.json"


def test_has_tool_fixtures_false_when_absent(tmp_path: Path) -> None:
    assert not has_tool_fixtures("NKE", tmp_path)
    (tmp_path / "NKE" / "tools" / "get_quote").mkdir(parents=True)
    assert not has_tool_fixtures("NKE", tmp_path), "an empty tools/ dir is not coverage"


def test_record_then_replay_round_trips_ok_result(tmp_path: Path, ctx: RunContext) -> None:
    tool = _quote_tool()
    tool_input = tool.input_schema.model_validate({"ticker": "AAPL"})
    record_tool_result(
        "AAPL", "get_quote", {"ticker": "AAPL"}, ToolResultOk(data=_price()), tmp_path
    )

    assert has_tool_fixtures("AAPL", tmp_path)
    result = FixtureToolRunner("AAPL", tmp_path).run(tool, tool_input, ctx)

    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, PriceData), "payload rehydrated into the concrete model"
    assert result.data.current_price == 190.5
    assert result.cached is True


def test_record_then_replay_round_trips_error_result(tmp_path: Path, ctx: RunContext) -> None:
    tool = _quote_tool()
    tool_input = tool.input_schema.model_validate({"ticker": "AAPL"})
    record_tool_result(
        "AAPL",
        "get_quote",
        {"ticker": "AAPL"},
        ToolResultError(error_code="stale_data", message="too old", retryable=False),
        tmp_path,
    )

    result = FixtureToolRunner("AAPL", tmp_path).run(tool, tool_input, ctx)
    assert isinstance(result, ToolResultError)
    assert result.error_code == "stale_data"
    assert result.message == "too old"


def test_missing_fixture_returns_non_retryable_not_found(tmp_path: Path, ctx: RunContext) -> None:
    """retryable=False is load-bearing: a retryable miss sends the loop into backoff."""
    tool = _quote_tool()
    tool_input = tool.input_schema.model_validate({"ticker": "MSFT"})
    runner = FixtureToolRunner("MSFT", tmp_path)

    result = runner.run(tool, tool_input, ctx)

    assert isinstance(result, ToolResultError)
    assert result.error_code == "not_found"
    assert result.retryable is False
    assert runner.misses == [FixtureMiss("get_quote", tool_input_hash({"ticker": "MSFT"}))]


def test_fixture_runner_never_calls_the_real_tool(tmp_path: Path, ctx: RunContext) -> None:
    """Replay must not construct a data-source client — the offline guarantee."""

    class _ExplodingTool(Tool):
        name = "get_quote"
        description = "boom"
        input_schema = _quote_tool().input_schema
        output_schema = PriceData

        def run(self, tool_input: object, ctx: object) -> ToolResultOk:
            raise AssertionError("FixtureToolRunner must never dispatch to the real tool")

    tool = _ExplodingTool()
    tool_input = tool.input_schema.model_validate({"ticker": "AAPL"})
    record_tool_result(
        "AAPL", "get_quote", {"ticker": "AAPL"}, ToolResultOk(data=_price()), tmp_path
    )

    result = FixtureToolRunner("AAPL", tmp_path).run(tool, tool_input, ctx)
    assert isinstance(result, ToolResultOk)


def test_fixture_runner_recomputes_scenarios_without_dispatching_tool_run(
    tmp_path: Path, ctx: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision = DirtDecisionContract.model_validate(_BUY_DECISION)
    tool_input = ModelDirtScenariosInput(
        valuation_date=decision.valuation_date,
        currency=decision.currency,
        current_price=decision.current_price,
        horizon_years=decision.horizon_years,
        scenarios=[
            DirtScenarioAssumption.model_validate(
                scenario.model_dump(
                    exclude={"total_dividends", "total_value", "total_return", "irr"}
                )
            )
            for scenario in decision.scenarios
        ],
        downside_floor=DirtDownsideFloorAssumption.model_validate(
            decision.downside_floor.model_dump(exclude={"adjusted", "coverage"})
        ),
        catalysts=decision.catalysts,
        failure_thesis=decision.failure_thesis,
        outcome=decision.outcome,
        outcome_reason=decision.outcome_reason,
        entry_conditions=decision.entry_conditions,
        blocking_unknowns=decision.blocking_unknowns,
        monitoring_metrics=decision.monitoring_metrics,
    )
    tool = TOOL_REGISTRY["model_dirt_scenarios"]
    monkeypatch.setattr(
        tool,
        "run",
        MagicMock(side_effect=AssertionError("FixtureToolRunner must not dispatch Tool.run")),
    )

    runner = FixtureToolRunner("AAPL", tmp_path)
    result = runner.run(tool, tool_input, ctx)

    assert isinstance(result, ToolResultOk)
    assert result.cached is True
    assert result.data == decision
    assert runner.served["model_dirt_scenarios"] is result
    assert runner.misses == []


def test_recorded_file_is_deterministic_json(tmp_path: Path) -> None:
    path = record_tool_result(
        "AAPL", "get_quote", {"ticker": "AAPL"}, ToolResultOk(data=_price()), tmp_path
    )
    payload = json.loads(path.read_text())
    assert payload["status"] == "ok"
    assert payload["data"]["ticker"] == "AAPL"
    # sort_keys=True → byte-stable across re-records, so a fixture diff is a real change.
    assert path.read_text() == json.dumps(payload, indent=2, sort_keys=True)


# ── staleness ─────────────────────────────────────────────────────────────────


def test_recorded_file_carries_a_recorded_at_stamp(tmp_path: Path) -> None:
    path = record_tool_result(
        "AAPL", "get_quote", {"ticker": "AAPL"}, ToolResultOk(data=_price()), tmp_path
    )
    assert "recorded_at" in json.loads(path.read_text())


def test_stale_fixture_warns_on_replay(tmp_path: Path, ctx: RunContext) -> None:
    old = datetime.now(timezone.utc) - timedelta(days=95)
    record_tool_result(
        "AAPL", "get_quote", {"ticker": "AAPL"}, ToolResultOk(data=_price()), tmp_path, old
    )
    tool = _quote_tool()

    with pytest.warns(UserWarning, match="95 days old"):
        FixtureToolRunner("AAPL", tmp_path).run(tool, tool.input_schema(ticker="AAPL"), ctx)


def test_fresh_fixture_does_not_warn(tmp_path: Path, ctx: RunContext) -> None:
    record_tool_result(
        "AAPL", "get_quote", {"ticker": "AAPL"}, ToolResultOk(data=_price()), tmp_path
    )
    tool = _quote_tool()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FixtureToolRunner("AAPL", tmp_path).run(tool, tool.input_schema(ticker="AAPL"), ctx)


def test_fixture_without_recorded_at_is_silent(tmp_path: Path, ctx: RunContext) -> None:
    """An undated fixture has unknown age; a warning we can't substantiate is noise."""
    path = tool_fixture_path("AAPL", "get_quote", {"ticker": "AAPL"}, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"status": "ok", "data": _price().model_dump(mode="json")}))
    tool = _quote_tool()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FixtureToolRunner("AAPL", tmp_path).run(tool, tool.input_schema(ticker="AAPL"), ctx)
