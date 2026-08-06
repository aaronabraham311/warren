from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from agent.budget import Budget, RunContext
from agent.loop import _schema_repair_prompt, _validate_persona_output, analyze_ticker
from agent.models import (
    AnalysisOutput,
    DirtCashFlow,
    DirtCatalyst,
    DirtDownsideFloorAssumption,
    DirtEntryCondition,
    DirtMonitoringMetric,
    DirtScenarioAssumption,
    DirtSignals,
    LynchBuffettSignals,
)
from agent.persona import DefaultPersona, DirtPersona
from agent.routing import HardcodedSonnetRouting, PhaseBasedRouting
from agent.run import _analyze_and_persist
from agent.tools import TOOL_REGISTRY
from agent.tools.base import ToolResultError, ToolResultOk
from agent.tools.dirt_scenarios import (
    ModelDirtScenariosInput,
    ModelDirtScenariosTool,
    _weighted_irr,
    _xnpv,
    model_dirt_scenarios,
)
from storage.engine import upsert_analysis, write_run_start
from storage.logger import RunLogger
from storage.models import Analysis, AnalysisData
from tests.conftest import make_end_turn, make_tool_use

VALUATION_DATE = date(2026, 1, 1)
TERMINAL_DATE = date(2028, 1, 1)


def _scenario(
    case: Literal["bear", "base", "bull"],
    probability: float,
    terminal_price: float,
    *,
    assumption: str = "Operations remain viable",
    dividends: tuple[tuple[date, float], ...] = ((date(2027, 1, 1), 5.0),),
) -> DirtScenarioAssumption:
    flows = [
        DirtCashFlow(date=flow_date, amount=amount, kind="dividend", source_ref="filing:p12")
        for flow_date, amount in dividends
    ]
    flows.append(
        DirtCashFlow(
            date=TERMINAL_DATE,
            amount=terminal_price,
            kind="terminal_sale",
            source_ref="valuation:case",
        )
    )
    return DirtScenarioAssumption(
        case=case,
        probability=probability,
        assumption=assumption,
        rationale="Cited operating and valuation assumptions",
        terminal_price=terminal_price,
        terminal_date=TERMINAL_DATE,
        cash_flows=flows,
    )


def _floor(**updates: object) -> DirtDownsideFloorAssumption:
    values: dict[str, object] = {
        "basis": "tangible_book",
        "gross": 90.0,
        "haircut": 0.2,
        "source_ref": "filing:balance-sheet",
        "as_of": date(2025, 12, 31),
        "adjustments": ["Exclude goodwill"],
        "confidence": "medium",
    }
    values.update(updates)
    return DirtDownsideFloorAssumption.model_validate(values)


def _input(**updates: object) -> ModelDirtScenariosInput:
    values: dict[str, object] = {
        "valuation_date": VALUATION_DATE,
        "currency": "USD",
        "current_price": 100.0,
        "horizon_years": 2,
        "scenarios": [
            _scenario("bear", 0.25, 80.0),
            _scenario("base", 0.50, 160.0),
            _scenario("bull", 0.25, 240.0),
        ],
        "downside_floor": _floor(),
        "catalysts": [
            DirtCatalyst(
                description="Board-authorized tender",
                category="capital_return",
                evidence_strength="contractual",
                expected_by=date(2027, 6, 30),
                source_ref="filing:tender",
                failure_condition="Tender authorization is withdrawn",
            )
        ],
        "failure_thesis": "Asset coverage fails or the tender is withdrawn",
        "outcome": "buy",
        "outcome_reason": "The cited catalyst clears the return hurdle",
        "entry_conditions": [],
        "blocking_unknowns": [],
        "monitoring_metrics": [
            DirtMonitoringMetric(
                metric="tangible_book_per_share",
                current_value=90.0,
                failure_threshold=70.0,
                cadence="quarterly",
                source_ref="filing:balance-sheet",
                rationale="Protects the downside floor",
            ),
            DirtMonitoringMetric(
                metric="tender_completion_pct",
                current_value=0.0,
                warning_threshold=25.0,
                cadence="event",
                source_ref="filing:tender",
                rationale="Tracks the discount-closing mechanism",
            ),
        ],
    }
    values.update(updates)
    return ModelDirtScenariosInput.model_validate(values)


def test_models_exact_dated_cash_flows_and_required_entry_price() -> None:
    inp = _input()
    contract = model_dirt_scenarios(inp)

    assert [scenario.case for scenario in contract.scenarios] == ["bear", "base", "bull"]
    assert contract.scenarios[1].total_dividends == 5.0
    assert contract.scenarios[1].total_value == 165.0
    assert contract.scenarios[1].total_return == pytest.approx(0.65)
    assert contract.probability_weighted_irr == pytest.approx(
        sum(item.probability * item.irr for item in contract.scenarios), abs=2e-8
    )
    assert contract.hurdle_cleared is True
    assert contract.downside_floor.adjusted == 72.0
    assert contract.downside_floor.coverage == 0.72
    assert contract.calculation_version == "dce_irr_v1"
    required_irr = _weighted_irr(inp.scenarios, VALUATION_DATE, contract.required_entry_price)
    assert required_irr == pytest.approx(0.20, abs=1e-7)


def test_xirr_uses_actual_dates_over_365() -> None:
    inp = _input(
        scenarios=[
            _scenario("bear", 0.25, 121.0, dividends=()),
            _scenario("base", 0.50, 121.0, dividends=()),
            _scenario("bull", 0.25, 121.0, dividends=()),
        ],
        outcome="pass",
        monitoring_metrics=[],
    )
    contract = model_dirt_scenarios(inp)

    assert contract.scenarios[0].irr == pytest.approx(0.10, abs=1e-8)
    scenario = contract.scenarios[0]
    dated = [(VALUATION_DATE, -100.0), *((flow.date, flow.amount) for flow in scenario.cash_flows)]
    assert _xnpv(scenario.irr, VALUATION_DATE, dated) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"current_price": 160.0}, "IRR of at least 20%"),
        (
            {
                "downside_floor": _floor(
                    basis="none",
                    gross=None,
                    haircut=None,
                    source_ref=None,
                    as_of=None,
                    adjustments=[],
                    confidence="unavailable",
                )
            },
            "stated downside floor",
        ),
        ({"blocking_unknowns": ["Controller identity unresolved"]}, "blocking unknowns"),
        ({"monitoring_metrics": []}, "at least two monitoring metrics"),
        ({"downside_floor": _floor(gross=0.0)}, "positive adjusted downside floor"),
        ({"downside_floor": _floor(haircut=1.0)}, "positive adjusted downside floor"),
    ],
)
def test_buy_invariants_are_enforced(update: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        model_dirt_scenarios(_input(**update))


def test_buy_requires_timed_non_aspirational_catalyst() -> None:
    catalyst = DirtCatalyst(
        description="Management may consider a sale",
        category="strategic_action",
        evidence_strength="aspirational",
        expected_by=None,
        source_ref="interview:1",
        failure_condition="No formal process begins",
    )
    with pytest.raises(ValueError, match="timed observable or contractual"):
        model_dirt_scenarios(_input(catalysts=[catalyst]))

    stale = catalyst.model_copy(
        update={"evidence_strength": "contractual", "expected_by": date(2025, 12, 31)}
    )
    with pytest.raises(ValueError, match="timed observable or contractual"):
        model_dirt_scenarios(_input(catalysts=[stale]))


def test_watchlist_requires_an_entry_condition() -> None:
    with pytest.raises(ValueError, match="watchlist requires"):
        model_dirt_scenarios(_input(outcome="watchlist", monitoring_metrics=[]))

    contract = model_dirt_scenarios(
        _input(
            outcome="watchlist",
            monitoring_metrics=[],
            entry_conditions=[
                DirtEntryCondition(
                    description="Buy only at the hurdle price",
                    metric="share_price",
                    operator="lte",
                    threshold=80.0,
                    currency="USD",
                )
            ],
        )
    )
    assert contract.outcome == "watchlist"


def test_pass_above_hurdle_requires_a_typed_blocking_unknown() -> None:
    entry = DirtEntryCondition(
        description="Buy at the hurdle price",
        metric="share_price",
        operator="lte",
        threshold=80.0,
        currency="USD",
    )
    with pytest.raises(ValueError, match="blocking unknown"):
        model_dirt_scenarios(
            _input(
                outcome="pass",
                outcome_reason="The opportunity is unattractive",
                entry_conditions=[entry],
                monitoring_metrics=[],
            )
        )
    decision = model_dirt_scenarios(
        _input(
            outcome="pass",
            outcome_reason="A structural control trap prevents minority value realization",
            entry_conditions=[entry],
            blocking_unknowns=["Controller blocks minority value realization"],
            monitoring_metrics=[],
        )
    )
    assert decision.outcome == "pass"


def test_bear_case_below_floor_requires_explicit_impairment_or_unreachable_assumption() -> None:
    scenarios = [
        _scenario("bear", 0.25, 40.0),
        _scenario("base", 0.50, 160.0),
        _scenario("bull", 0.25, 240.0),
    ]
    with pytest.raises(ValueError, match="impairment/unreachable"):
        model_dirt_scenarios(_input(scenarios=scenarios))

    scenarios[0] = _scenario("bear", 0.25, 40.0, assumption="Severe asset impairment")
    assert model_dirt_scenarios(_input(scenarios=scenarios)).scenarios[0].total_value == 45.0

    dividend_masked = [
        _scenario("bear", 0.25, 70.0, dividends=((date(2027, 1, 1), 5.0),)),
        _scenario("base", 0.50, 160.0),
        _scenario("bull", 0.25, 240.0),
    ]
    with pytest.raises(ValueError, match="impairment/unreachable"):
        model_dirt_scenarios(_input(scenarios=dividend_masked))


def test_scenario_set_and_terminal_sale_are_validated() -> None:
    duplicate = [
        _scenario("bear", 0.2, 80),
        _scenario("bear", 0.3, 100),
        _scenario("bull", 0.5, 200),
    ]
    with pytest.raises(ValueError, match="one bear, base, and bull"):
        model_dirt_scenarios(_input(outcome="pass", scenarios=duplicate, monitoring_metrics=[]))

    bad = _scenario("bear", 0.25, 80)
    bad.cash_flows[-1] = bad.cash_flows[-1].model_copy(update={"amount": 81.0})
    with pytest.raises(ValueError, match="terminal sale must match"):
        model_dirt_scenarios(
            _input(
                outcome="pass",
                scenarios=[bad, _scenario("base", 0.5, 160), _scenario("bull", 0.25, 240)],
                monitoring_metrics=[],
            )
        )


def test_none_floor_rejects_false_precision() -> None:
    with pytest.raises(ValidationError, match="requires gross, haircut, and source_ref to be null"):
        DirtDownsideFloorAssumption(
            basis="none",
            gross=10.0,
            haircut=None,
            source_ref=None,
            confidence="unavailable",
        )


def test_required_evidence_text_rejects_whitespace() -> None:
    with pytest.raises(ValidationError):
        DirtCatalyst(
            description=" ",
            category="capital_return",
            evidence_strength="observable",
            expected_by=date(2027, 1, 1),
            source_ref=" ",
            failure_condition=" ",
        )


def test_decision_models_forbid_unknown_fields_and_non_finite_numbers() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        DirtCashFlow(  # type: ignore[call-arg]
            date=date(2027, 1, 1), amount=1.0, kind="dividend", source_ref="filing:1", typo=True
        )
    with pytest.raises(ValidationError, match="finite_number"):
        DirtCashFlow(
            date=date(2027, 1, 1),
            amount=float("nan"),
            kind="dividend",
            source_ref="filing:1",
        )


def test_dirt_schema_repair_requires_the_missing_tool_call() -> None:
    prompt = _schema_repair_prompt(DirtPersona(), None)
    assert "call model_dirt_scenarios" in prompt
    assert "dirt_decision copied exactly" in prompt


def test_tool_is_pure_registered_and_returns_validation_errors_as_data() -> None:
    assert TOOL_REGISTRY["model_dirt_scenarios"].output_schema.__name__ == "DirtDecisionContract"
    result = ModelDirtScenariosTool().run(_input(), MagicMock())
    assert isinstance(result, ToolResultOk)

    invalid = _input(outcome="watchlist", monitoring_metrics=[])
    error = ModelDirtScenariosTool().run(invalid, MagicMock())
    assert isinstance(error, ToolResultError)
    assert error.error_code == "parse"
    assert error.retryable is False


def test_persona_requires_exact_served_decision_and_maps_outcome() -> None:
    served = model_dirt_scenarios(_input())
    output = AnalysisOutput(
        ticker="AAPL",
        analysis_type="discovery",
        recommendation="buy",
        confidence=0.8,
        thesis="A sufficiently detailed deterministic DIRT decision thesis.",
        lynch_signals=LynchBuffettSignals(pros=[], cons=[]),
        buffett_signals=LynchBuffettSignals(pros=[], cons=[]),
        key_risks=["Catalyst execution"],
        dirt_signals=DirtSignals(ev_ebit=5.0),
        dirt_decision=served,
    )
    assert _validate_persona_output(output, DirtPersona(), served) is output

    tampered = output.model_copy(
        update={"dirt_decision": served.model_copy(update={"required_entry_price": 1.0})}
    )
    with pytest.raises(ValueError, match="exactly match"):
        _validate_persona_output(tampered, DirtPersona(), served)

    wrong_mapping = output.model_copy(update={"recommendation": "hold"})
    with pytest.raises(ValueError, match="recommendation must map"):
        _validate_persona_output(wrong_mapping, DirtPersona(), served)

    with pytest.raises(ValueError, match="must be null"):
        _validate_persona_output(output, DefaultPersona(), None)


def test_loop_captures_and_accepts_served_decision(mock_claude: MagicMock, tmp_path: Path) -> None:
    inp = _input()
    served = model_dirt_scenarios(inp)
    output = AnalysisOutput(
        ticker="AAPL",
        analysis_type="discovery",
        recommendation="buy",
        confidence=0.8,
        thesis="A sufficiently detailed deterministic DIRT decision thesis.",
        lynch_signals=LynchBuffettSignals(pros=[], cons=[]),
        buffett_signals=LynchBuffettSignals(pros=[], cons=[]),
        key_risks=["Catalyst execution"],
        dirt_signals=DirtSignals(ev_ebit=5.0),
        dirt_decision=served,
    )
    mock_claude(
        [
            make_tool_use("model_dirt_scenarios", inp.model_dump(mode="json")),
            make_end_turn(output.model_dump_json()),
        ]
    )
    logger = RunLogger("loop-dirt-decision", tmp_path)
    result = analyze_ticker(
        "AAPL",
        DirtPersona(),
        HardcodedSonnetRouting(),
        RunContext(run_id="loop-dirt-decision", budget=Budget(), logger=logger),
    )
    assert result.dirt_decision == served


def test_run_persists_and_logs_synchronized_decision_projection(
    db_engine: object, db_session: Session, tmp_path: Path
) -> None:
    from datetime import datetime, timezone

    output = AnalysisOutput(
        ticker="AAPL",
        analysis_type="discovery",
        recommendation="buy",
        confidence=0.8,
        thesis="A sufficiently detailed deterministic DIRT decision thesis.",
        lynch_signals=LynchBuffettSignals(pros=[], cons=[]),
        buffett_signals=LynchBuffettSignals(pros=[], cons=[]),
        key_risks=["Catalyst execution"],
        dirt_signals=DirtSignals(ev_ebit=5.0),
        dirt_decision=model_dirt_scenarios(_input()),
    )
    write_run_start("persist-dirt-decision", datetime.now(timezone.utc))
    logger = RunLogger("persist-dirt-decision", tmp_path)
    with patch("agent.run.analyze_ticker", return_value=output):
        _analyze_and_persist(
            "AAPL",
            "discovery",
            "persist-dirt-decision",
            Budget(),
            logger,
            DirtPersona(),
            PhaseBasedRouting(),
            MagicMock(),
            "",
        )
    row = db_session.query(Analysis).filter_by(run_id="persist-dirt-decision").one()
    assert row.dirt_decision is not None
    assert row.dirt_decision["outcome"] == row.decision_outcome == "buy"
    assert row.dirt_decision["probability_weighted_irr"] == pytest.approx(
        row.probability_weighted_irr
    )
    events = [json.loads(line) for line in logger.path.read_text().splitlines()]
    completed = next(event for event in events if event["event"] == "ticker_completed")
    assert completed["decision_outcome"] == row.decision_outcome
    assert completed["probability_weighted_irr"] == pytest.approx(row.probability_weighted_irr)


def test_storage_derives_decision_projections_from_contract(
    db_engine: object, db_session: Session
) -> None:
    from datetime import datetime, timezone

    decision = model_dirt_scenarios(_input()).model_dump(mode="json")
    write_run_start("derived-decision-projection", datetime.now(timezone.utc))
    upsert_analysis(
        "derived-decision-projection",
        "AAPL",
        AnalysisData(
            analysis_type="discovery",
            recommendation="buy",
            confidence=0.8,
            thesis="A sufficiently detailed deterministic DIRT decision thesis.",
            lynch_signals={"pros": [], "cons": []},
            buffett_signals={"pros": [], "cons": []},
            key_risks=["Catalyst execution"],
            data_quality_notes=[],
            tool_calls_made=1,
            tokens_used=100,
            dirt_decision=decision,
            decision_outcome="pass",
            probability_weighted_irr=-1.0,
        ),
    )
    db_session.expire_all()

    row = db_session.query(Analysis).filter_by(run_id="derived-decision-projection").one()
    assert row.decision_outcome == decision["outcome"] == "buy"
    assert row.probability_weighted_irr == pytest.approx(decision["probability_weighted_irr"])
