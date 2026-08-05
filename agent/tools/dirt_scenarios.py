"""Pure DIRT scenario math and the validated G18 decision contract.

The tool deliberately performs no I/O.  It turns cited, dated assumptions into a
decision record whose returns cannot drift from the cash flows shown to the user.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from agent.budget import RunContext
from agent.models import (
    DirtCashFlow,
    DirtCatalyst,
    DirtDecisionContract,
    DirtDownsideFloor,
    DirtDownsideFloorAssumption,
    DirtEntryCondition,
    DirtMonitoringMetric,
    DirtScenario,
    DirtScenarioAssumption,
)
from agent.tools.base import Tool, ToolResult, ToolResultError, ToolResultOk


class ModelDirtScenariosInput(BaseModel):
    valuation_date: date
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    current_price: float = Field(gt=0.0)
    horizon_years: Literal[2, 3]
    scenarios: list[DirtScenarioAssumption] = Field(min_length=3, max_length=3)
    downside_floor: DirtDownsideFloorAssumption
    catalysts: list[DirtCatalyst] = Field(min_length=1)
    failure_thesis: str = Field(min_length=1)
    outcome: Literal["buy", "watchlist", "pass"]
    outcome_reason: str = Field(min_length=1)
    entry_conditions: list[DirtEntryCondition] = Field(default_factory=list)
    blocking_unknowns: list[str] = Field(default_factory=list)
    monitoring_metrics: list[DirtMonitoringMetric] = Field(default_factory=list)


def _xnpv(rate: float, initial_date: date, cash_flows: list[tuple[date, float]]) -> float:
    if rate <= -1.0:
        raise ValueError("XIRR rate must be greater than -100%")
    return sum(
        amount / math.pow(1.0 + rate, (cash_date - initial_date).days / 365.0)
        for cash_date, amount in cash_flows
    )


def _xirr(initial_date: date, entry_price: float, cash_flows: list[DirtCashFlow]) -> float:
    """Return the unique XIRR for one initial outflow and later positive receipts."""
    if entry_price <= 0:
        raise ValueError("entry price must be positive")
    if not cash_flows or any(flow.date <= initial_date for flow in cash_flows):
        raise ValueError("all scenario cash flows must occur after the valuation date")
    dated = [(initial_date, -entry_price), *((flow.date, flow.amount) for flow in cash_flows)]
    lo = -0.999999999
    hi = 1.0
    lo_value = _xnpv(lo, initial_date, dated)
    hi_value = _xnpv(hi, initial_date, dated)
    while hi_value > 0.0 and hi < 1_000_000.0:
        hi *= 2.0
        hi_value = _xnpv(hi, initial_date, dated)
    if lo_value <= 0.0 or hi_value >= 0.0:
        raise ValueError("cash flows do not have a bracketed conventional XIRR")
    for _ in range(160):
        mid = (lo + hi) / 2.0
        if _xnpv(mid, initial_date, dated) > 0.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _validate_scenario(
    scenario: DirtScenarioAssumption, valuation_date: date, horizon_years: int
) -> None:
    horizon_days = (scenario.terminal_date - valuation_date).days
    if horizon_days <= 0:
        raise ValueError(f"{scenario.case} terminal date must follow the valuation date")
    # Leap years make an exact anniversary differ by a day; allow that calendar effect only.
    if abs(horizon_days - horizon_years * 365) > 2:
        raise ValueError(f"{scenario.case} terminal date must match the stated horizon")
    sales = [flow for flow in scenario.cash_flows if flow.kind == "terminal_sale"]
    if len(sales) != 1:
        raise ValueError(f"{scenario.case} requires exactly one terminal_sale cash flow")
    sale = sales[0]
    if sale.date != scenario.terminal_date or abs(sale.amount - scenario.terminal_price) > 1e-9:
        raise ValueError(f"{scenario.case} terminal sale must match terminal date and price")
    if any(flow.date > scenario.terminal_date for flow in scenario.cash_flows):
        raise ValueError(f"{scenario.case} cash flows cannot follow its terminal date")


def _weighted_irr(
    scenarios: list[DirtScenarioAssumption], valuation_date: date, entry_price: float
) -> float:
    return sum(
        scenario.probability * _xirr(valuation_date, entry_price, scenario.cash_flows)
        for scenario in scenarios
    )


def _required_entry_price(
    scenarios: list[DirtScenarioAssumption], valuation_date: date, hurdle: float
) -> float:
    """Price at which the probability-weighted, unrounded scenario XIRR equals hurdle."""
    lo = 1e-9
    hi = max(sum(flow.amount for flow in item.cash_flows) for item in scenarios)
    while _weighted_irr(scenarios, valuation_date, hi) > hurdle:
        hi *= 2.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        if _weighted_irr(scenarios, valuation_date, mid) > hurdle:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def model_dirt_scenarios(inp: ModelDirtScenariosInput) -> DirtDecisionContract:
    """Compute and validate a decision contract without network, storage, or clock access."""
    cases = [scenario.case for scenario in inp.scenarios]
    if sorted(cases) != ["base", "bear", "bull"]:
        raise ValueError("scenarios must contain exactly one bear, base, and bull case")
    if abs(sum(item.probability for item in inp.scenarios) - 1.0) > 1e-9:
        raise ValueError("scenario probabilities must sum to 1")
    for scenario in inp.scenarios:
        _validate_scenario(scenario, inp.valuation_date, inp.horizon_years)

    scenarios: list[DirtScenario] = []
    for item in inp.scenarios:
        dividends = sum(flow.amount for flow in item.cash_flows if flow.kind == "dividend")
        total_value = item.terminal_price + dividends
        irr = _xirr(inp.valuation_date, inp.current_price, item.cash_flows)
        scenarios.append(
            DirtScenario(
                **item.model_dump(),
                total_dividends=dividends,
                total_value=total_value,
                total_return=total_value / inp.current_price - 1.0,
                irr=irr,
            )
        )

    weighted_irr = _weighted_irr(inp.scenarios, inp.valuation_date, inp.current_price)
    floor = inp.downside_floor
    adjusted = None if floor.gross is None else floor.gross * (1.0 - (floor.haircut or 0.0))
    downside_floor = DirtDownsideFloor(
        **floor.model_dump(),
        adjusted=adjusted,
        coverage=adjusted / inp.current_price if adjusted is not None else None,
    )

    bear = next(item for item in scenarios if item.case == "bear")
    if adjusted is not None and bear.terminal_price < adjusted:
        breach_text = f"{bear.assumption} {bear.rationale}".lower()
        if "impairment" not in breach_text and "unreachable" not in breach_text:
            raise ValueError(
                "a bear case below the downside floor must identify impairment/unreachable"
            )

    if inp.outcome == "buy":
        timed_support = any(
            catalyst.evidence_strength in {"contractual", "observable"}
            and catalyst.expected_by is not None
            and inp.valuation_date
            < catalyst.expected_by
            <= max(scenario.terminal_date for scenario in inp.scenarios)
            for catalyst in inp.catalysts
        )
        if weighted_irr < 0.20:
            raise ValueError("buy requires probability-weighted IRR of at least 20%")
        if floor.basis == "none":
            raise ValueError("buy requires a stated downside floor")
        if not timed_support:
            raise ValueError("buy requires a timed observable or contractual catalyst")
        if inp.blocking_unknowns:
            raise ValueError("buy cannot contain blocking unknowns")
        if len(inp.monitoring_metrics) < 2:
            raise ValueError("buy requires at least two monitoring metrics")
    elif inp.outcome == "watchlist" and not inp.entry_conditions:
        raise ValueError("watchlist requires at least one entry condition")
    elif inp.outcome == "pass" and weighted_irr >= 0.20:
        reason = inp.outcome_reason.lower()
        credible_catalyst = any(
            catalyst.evidence_strength in {"contractual", "observable"}
            and catalyst.expected_by is not None
            and inp.valuation_date < catalyst.expected_by
            for catalyst in inp.catalysts
        )
        if (
            credible_catalyst
            and not inp.blocking_unknowns
            and "structural" not in reason
            and "value trap" not in reason
        ):
            raise ValueError(
                "pass above the hurdle with a credible catalyst requires a structural trap "
                "or blocking unknown"
            )

    required_price = _required_entry_price(inp.scenarios, inp.valuation_date, 0.20)
    return DirtDecisionContract(
        valuation_date=inp.valuation_date,
        currency=inp.currency,
        current_price=inp.current_price,
        horizon_years=inp.horizon_years,
        scenarios=scenarios,
        probability_weighted_irr=weighted_irr,
        hurdle_cleared=weighted_irr >= 0.20,
        downside_floor=downside_floor,
        catalysts=inp.catalysts,
        failure_thesis=inp.failure_thesis,
        outcome=inp.outcome,
        outcome_reason=inp.outcome_reason,
        required_entry_price=required_price,
        entry_conditions=inp.entry_conditions,
        blocking_unknowns=inp.blocking_unknowns,
        monitoring_metrics=inp.monitoring_metrics,
    )


class ModelDirtScenariosTool(Tool):
    name = "model_dirt_scenarios"
    description = (
        "Pure offline DIRT decision calculator. Computes dated bear/base/bull XIRRs, "
        "probability-weighted return, downside-floor coverage, and the entry price required "
        "for a 20% hurdle, then rejects recommendations that violate the decision contract."
    )
    input_schema = ModelDirtScenariosInput
    output_schema = DirtDecisionContract

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, ModelDirtScenariosInput)
        try:
            return ToolResultOk(data=model_dirt_scenarios(tool_input))
        except ValueError as exc:
            return ToolResultError(
                error_code="parse",
                message=f"invalid DIRT decision contract: {exc}",
                retryable=False,
            )
