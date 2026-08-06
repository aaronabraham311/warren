"""Grade one ``AnalysisOutput`` against one golden-set ``EvalExample``.

Grading asserts *membership in an envelope*, never equality with a single expected answer
(see ``eval.golden_set``). Each assertion becomes a ``CheckResult`` carrying its own
severity:

    must    — any failure sets ``EvalGrade.passed = False``
    should  — failure is recorded for inspection but does not fail the example

Five check families run, in this order:

    recommendation_in_allowed               must
    thesis_mentions_{...}                   must    one per ThesisMention (any_of semantics)
    thesis_not_mention_{...}                must    one per forbidden term
    {buffett,lynch}_{pros,cons}_min_count   should  only when the YAML sets min_count
    key_risks_include_one_of                must    only when must_include_one_of is set
    numerical_grounding                     must    count of specific numbers in the thesis

A ``should`` severity for the signal counts is deliberate: the number of pros a model
surfaces is a stylistic choice that drifts between prompt versions without indicating a
regression, whereas a forbidden term or an out-of-envelope recommendation does.
"""

import json
import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from agent.models import (
    AnalysisOutput,
    DirtDecisionContract,
    DirtDownsideFloorAssumption,
    DirtScenarioAssumption,
    LynchBuffettSignals,
)
from agent.tools.dirt_scenarios import ModelDirtScenariosInput, model_dirt_scenarios
from data_sources.forensics import ForensicEvidenceBundle
from eval.golden_set import (
    ClosabilityExpectation,
    DeepValueExpectation,
    EvalExample,
    SignalsExpectation,
)
from eval.judge import JudgeVerdict, SemanticRequest, ThesisJudge

Severity = Literal["must", "should"]

# A "specific number" is a bare integer/decimal, optionally a percentage: 12, 3.4, 18.7%.
_NUMBER_RE = re.compile(r"\d+\.?\d*%?")

_THESIS_EXCERPT_CHARS = 100

# A stable substring of the G9 DIRT universe-limitation note emitted into
# data_quality_notes (see DIRT_SYSTEM_PROMPT in agent/persona.py). Substring, not the full
# string, so wording tweaks elsewhere in the note don't spuriously fail the check.
_UNIVERSE_NOTE_SUBSTRING = "DIRT universe: US small-caps (Russell 2000)"

_VALUE_TRAP_FRAMINGS = [
    "debt burden, refinancing, or ability to service debt",
    "negative free cash flow, recurring cash burn, or funding runway",
    "NCAV erosion, asset deterioration, or loss of the claimed asset-value floor",
    "trading illiquidity or inability to realize value",
    "missing catalyst, minority control, or a controlling-holder discount",
]
_VALUE_TRAP_RUBRIC = (
    "Assess whether the apparent cheapness is a value trap. Pass only when the analysis uses "
    "specific evidence about at least one supplied dimension and reaches a supported adverse "
    "or protective conclusion. Net cash, covered debt service, durable NCAV, or positive cash "
    "generation may support a protective conclusion. The bare words 'value trap', 'risk', "
    "'balance sheet', or 'fragile' without analysis do not pass."
)


class CheckResult(BaseModel):
    check_name: str
    passed: bool
    expected: str
    actual: str
    severity: Severity


class EvalGrade(BaseModel):
    ticker: str
    passed: bool
    checks: list[CheckResult]
    overall_notes: str


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _finalize(ticker: str, checks: list[CheckResult]) -> EvalGrade:
    passed = all(c.passed for c in checks if c.severity == "must")
    n_passed = sum(1 for c in checks if c.passed)
    return EvalGrade(
        ticker=ticker,
        passed=passed,
        checks=checks,
        overall_notes=f"{n_passed}/{len(checks)} checks passed",
    )


def failed_grade(
    ticker: str,
    check_name: str,
    expected: str,
    actual: str,
    notes: str,
) -> EvalGrade:
    """A grade carrying one ``must`` failure — for tickers that never reached grading.

    Used for ``fixture_missing`` (no recorded tool outputs) and ``run_completed`` (the
    agent raised). Both must fail the example, and both must be visible in ``eval_runs``.
    """
    return EvalGrade(
        ticker=ticker,
        passed=False,
        checks=[
            CheckResult(
                check_name=check_name,
                passed=False,
                expected=expected,
                actual=actual,
                severity="must",
            )
        ],
        overall_notes=notes,
    )


def _signal_count_checks(
    prefix: str,
    expectation: SignalsExpectation,
    actual: LynchBuffettSignals,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    for field_name, expected_count in (("pros", expectation.pros), ("cons", expectation.cons)):
        if expected_count is None:
            continue
        n = len(actual.pros if field_name == "pros" else actual.cons)
        checks.append(
            CheckResult(
                check_name=f"{prefix}_{field_name}_min_count",
                passed=n >= expected_count.min_count,
                expected=f">= {expected_count.min_count}",
                actual=str(n),
                severity="should",
            )
        )
    return checks


def _deep_value_checks(
    result: AnalysisOutput,
    expectation: DeepValueExpectation,
    thesis_lower: str,
    forensic_evidence: ForensicEvidenceBundle | None = None,
) -> list[CheckResult]:
    """Deep-value (DIRT) check family — one ``must`` check per configured toggle.

    Each check is satisfied from the structured ``dirt_signals`` block and/or the thesis
    text, so a model that reasons about the concept in prose still passes even when it did
    not populate the corresponding numeric field.
    """
    checks: list[CheckResult] = []
    dirt = result.dirt_signals

    if expectation.require_ev_ebit:
        found = (dirt is not None and dirt.ev_ebit is not None) or "ev/ebit" in thesis_lower
        checks.append(
            CheckResult(
                check_name="ev_ebit_present",
                passed=found,
                expected="dirt_signals.ev_ebit populated or EV/EBIT cited in thesis",
                actual=f"ev_ebit={None if dirt is None else dirt.ev_ebit}",
                severity="must",
            )
        )

    if expectation.require_ncav:
        found = (
            dirt is not None
            and (dirt.price_to_ncav is not None or dirt.ncav_discount_pct is not None)
        ) or "ncav" in thesis_lower
        checks.append(
            CheckResult(
                check_name="ncav_cited",
                passed=found,
                expected="dirt_signals price_to_ncav/ncav_discount_pct populated or NCAV in thesis",
                actual=(
                    "no NCAV signal or mention"
                    if dirt is None
                    else f"price_to_ncav={dirt.price_to_ncav}, discount={dirt.ncav_discount_pct}"
                ),
                severity="must",
            )
        )

    if expectation.require_universe_note:
        notes_blob = " ".join(result.data_quality_notes)
        checks.append(
            CheckResult(
                check_name="universe_note_present",
                passed=_UNIVERSE_NOTE_SUBSTRING in notes_blob,
                expected=f"data_quality_notes contains {_UNIVERSE_NOTE_SUBSTRING!r}",
                actual="present" if _UNIVERSE_NOTE_SUBSTRING in notes_blob else "absent",
                severity="must",
            )
        )

    if expectation.require_forensic_citations:
        claim_types: list[str] = []
        if dirt is not None and (
            dirt.controller_identified is not None
            or dirt.controller_name is not None
            or dirt.controller_economic_interest_pct is not None
            or dirt.controller_voting_rights_pct is not None
        ):
            claim_types.append("ownership")
        if dirt is not None and dirt.buyback_active is not None:
            claim_types.append("buyback")
        if dirt is not None and (
            dirt.catalyst_strength is not None
            or dirt.catalyst_stage is not None
            or dirt.catalyst_description is not None
        ):
            claim_types.append("catalyst")
        if "related-party" in thesis_lower or "related party" in thesis_lower:
            claim_types.append("related_party")
        evidence_ids = set(() if dirt is None else dirt.forensic_evidence_ids)
        category_ids: dict[str, set[str]] = {
            "ownership": set(),
            "buyback": set(),
            "catalyst": set(),
            "related_party": set(),
        }
        if forensic_evidence is not None:
            category_ids["ownership"] = {
                ref.evidence_id
                for fact in (
                    *forensic_evidence.cap_table,
                    *forensic_evidence.stake_events,
                    *forensic_evidence.agreements,
                )
                for ref in fact.evidence_refs
            }
            category_ids["buyback"] = {
                ref.evidence_id
                for fact in forensic_evidence.capital_returns
                for ref in fact.evidence_refs
            }
            category_ids["catalyst"] = {
                ref.evidence_id
                for fact in forensic_evidence.catalysts
                for ref in fact.evidence_refs
            }
            category_ids["related_party"] = {
                ref.evidence_id
                for fact in forensic_evidence.related_party_transactions
                for ref in fact.evidence_refs
            }
        untraced = [
            claim for claim in claim_types if not evidence_ids.intersection(category_ids[claim])
        ]
        unknown_ids = evidence_ids - set().union(*category_ids.values())
        traced = not claim_types or (
            forensic_evidence is not None and not untraced and not unknown_ids
        )
        checks.append(
            CheckResult(
                check_name="forensic_claims_cited",
                passed=traced,
                expected=(
                    "ownership, related-party, buyback and catalyst claims carry "
                    "dirt_signals.forensic_evidence_ids"
                ),
                actual=(
                    f"claims={claim_types}, untraced={untraced}, unknown_ids={sorted(unknown_ids)}"
                    if claim_types
                    else "no forensic claims made"
                ),
                severity="must",
            )
        )

    decision = result.dirt_decision
    if expectation.require_decision_contract:
        checks.append(
            CheckResult(
                check_name="dirt_decision_present",
                passed=decision is not None,
                expected="a complete dirt_decision contract",
                actual="present" if decision is not None else "absent",
                severity="must",
            )
        )

    if expectation.allowed_decision_outcomes:
        outcome = None if decision is None else decision.outcome
        checks.append(
            CheckResult(
                check_name="dirt_decision_outcome_allowed",
                passed=outcome in expectation.allowed_decision_outcomes,
                expected=f"one of {expectation.allowed_decision_outcomes}",
                actual=str(outcome),
                severity="must",
            )
        )

    if expectation.require_decision_recomputation:
        recomputed: DirtDecisionContract | None = None
        error: str | None = None
        if decision is not None:
            try:
                inputs = ModelDirtScenariosInput(
                    valuation_date=decision.valuation_date,
                    currency=decision.currency,
                    current_price=decision.current_price,
                    horizon_years=decision.horizon_years,
                    scenarios=[
                        DirtScenarioAssumption.model_validate(
                            item.model_dump(
                                exclude={"total_dividends", "total_value", "total_return", "irr"}
                            )
                        )
                        for item in decision.scenarios
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
                recomputed = model_dirt_scenarios(inputs)
            except (ValueError, TypeError) as exc:
                error = str(exc)
        matches = (
            decision is not None
            and recomputed is not None
            and decision.model_dump(mode="json") == recomputed.model_dump(mode="json")
        )
        checks.append(
            CheckResult(
                check_name="dirt_decision_recomputes",
                passed=matches,
                expected="all scenario returns and decision fields recompute deterministically",
                actual=error or ("match" if matches else "computed fields differ"),
                severity="must",
            )
        )

    return checks


def _closability_checks(
    result: AnalysisOutput,
    expectation: ClosabilityExpectation,
    forensic_evidence: ForensicEvidenceBundle | None,
) -> list[CheckResult]:
    """Grade G14's compact decision fields without turning missing coverage into failure."""

    dirt = result.dirt_signals
    status = None if dirt is None else getattr(dirt, "closability_status", None)
    score = None if dirt is None else getattr(dirt, "closability_score", None)
    confidence = None if dirt is None else getattr(dirt, "closability_confidence", None)
    reasons = [] if dirt is None else getattr(dirt, "closability_reasons", [])
    checks = [
        CheckResult(
            check_name="closability_status_allowed",
            passed=status in expectation.allowed_status,
            expected=f"one of {expectation.allowed_status}",
            actual=str(status),
            severity="must",
        ),
        CheckResult(
            check_name="closability_reasons_present",
            passed=len(reasons) >= expectation.min_reasons,
            expected=f">= {expectation.min_reasons} cited reason(s)",
            actual=str(len(reasons)),
            severity="must",
        ),
    ]
    for label, value, lower, upper in (
        ("score", score, expectation.min_score, expectation.max_score),
        ("confidence", confidence, expectation.min_confidence, expectation.max_confidence),
    ):
        if lower is None and upper is None:
            continue
        passed = value is not None
        if value is not None and lower is not None:
            passed = value >= lower
        if value is not None and upper is not None:
            passed = passed and value <= upper
        checks.append(
            CheckResult(
                check_name=f"closability_{label}_in_envelope",
                passed=passed,
                expected=f"{lower if lower is not None else 0.0} <= {label} <= "
                f"{upper if upper is not None else 1.0}",
                actual=str(value),
                severity="must",
            )
        )

    if expectation.require_unknown_semantics:
        controller = None if dirt is None else dirt.controller_identified
        reason_blob = " ".join(reasons).lower()
        explains_gap = any(
            term in reason_blob
            for term in ("unknown", "coverage", "missing", "partial", "conflict")
        )
        checks.append(
            CheckResult(
                check_name="closability_unknown_stays_unknown",
                passed=status == "unknown" and controller is None and explains_gap,
                expected=(
                    "unknown status, controller_identified=None, and an explicit "
                    "coverage/missing/conflict reason"
                ),
                actual=f"status={status}, controller={controller}, reasons={reasons}",
                severity="must",
            )
        )

    if expectation.require_observable_or_contractual_catalyst:
        strength = None if dirt is None else dirt.catalyst_strength
        evidence_ids = set(() if dirt is None else dirt.forensic_evidence_ids)
        catalyst_ids = (
            set()
            if forensic_evidence is None
            else {
                ref.evidence_id
                for catalyst in forensic_evidence.catalysts
                for ref in catalyst.evidence_refs
            }
        )
        checks.append(
            CheckResult(
                check_name="closability_catalyst_is_cited_and_observable",
                passed=(
                    strength in {"observable", "contractual"}
                    and bool(evidence_ids.intersection(catalyst_ids))
                ),
                expected="observable/contractual catalyst with forensic evidence ID",
                actual=f"strength={strength}, evidence_ids={evidence_ids}",
                severity="must",
            )
        )
    return checks


def _analysis_evidence(result: AnalysisOutput) -> str:
    """Blinded candidate text for recommendation-consistency judging."""
    return json.dumps(
        {
            "recommendation": result.recommendation,
            "thesis": result.thesis,
            "buffett_signals": result.buffett_signals.model_dump(),
            "lynch_signals": result.lynch_signals.model_dump(),
            "key_risks": result.key_risks,
        },
        sort_keys=True,
    )


def _judge_many(
    judge: ThesisJudge,
    requests: Sequence[SemanticRequest],
) -> dict[str, JudgeVerdict]:
    """Use the batch seam so a live panel makes one request per judge and ticker."""
    return judge.judge_many(requests)


def grade_analysis(
    result: AnalysisOutput,
    example: EvalExample,
    judge: ThesisJudge | None = None,
    forensic_evidence: ForensicEvidenceBundle | None = None,
) -> EvalGrade:
    """Grade *result* against *example*, batching all semantic checks when possible."""
    expectations = example.expectations
    thesis_lower = result.thesis.lower()
    excerpt = result.thesis[:_THESIS_EXCERPT_CHARS]
    checks: list[CheckResult] = []
    semantic: list[tuple[SemanticRequest, str, Severity, str]] = []

    allowed = expectations.recommendation.allowed
    checks.append(
        CheckResult(
            check_name="recommendation_in_allowed",
            passed=result.recommendation in allowed,
            expected=f"one of {sorted(allowed)}",
            actual=result.recommendation,
            severity="must",
        )
    )

    consistency_id = "recommendation_evidence_consistent"
    semantic.append(
        (
            SemanticRequest(
                check_id=consistency_id,
                text=_analysis_evidence(result),
                concept=[
                    "the recommendation follows from the thesis, signals, and risks",
                    "the recommendation acknowledges material contrary evidence",
                ],
                ticker=example.ticker,
                rubric=(
                    "Pass when the stated buy/sell/hold label is a defensible conclusion from "
                    "the supplied evidence and material counter-evidence is not ignored. Do not "
                    "compare against, infer, or enforce a golden recommendation label."
                ),
            ),
            consistency_id,
            "must",
            "recommendation is consistent with its own evidence",
        )
    )

    for mention in expectations.thesis_must_mention:
        check_name = f"thesis_mentions_{_slug('_or_'.join(mention.any_of[:2]))}"
        if judge is not None:
            semantic.append(
                (
                    SemanticRequest(
                        check_id=check_name,
                        text=result.thesis,
                        concept=mention.any_of,
                        ticker=example.ticker,
                    ),
                    check_name,
                    "must",
                    f"engages topic {mention.any_of}",
                )
            )
            continue
        found = [kw for kw in mention.any_of if kw.lower() in thesis_lower]
        checks.append(
            CheckResult(
                check_name=check_name,
                passed=bool(found),
                expected=f"any of {mention.any_of}",
                actual=f"found {found}" if found else f"none present; thesis={excerpt!r}",
                severity="must",
            )
        )

    for forbidden in expectations.thesis_must_not_mention:
        haystack = thesis_lower
        if forbidden.lower() == "risk-free":
            # "risk-free rate" is standard DCF vocabulary (the discount-rate anchor), not the
            # overconfidence hype this guard targets ("guaranteed", "can't lose", "no downside").
            # Strip the collocation so a legitimate discount-rate mention doesn't trip the check.
            haystack = haystack.replace("risk-free rate", "").replace("risk free rate", "")
        present = forbidden.lower() in haystack
        checks.append(
            CheckResult(
                check_name=f"thesis_not_mention_{_slug(forbidden)}",
                passed=not present,
                expected=f"not mention {forbidden!r}",
                actual="found" if present else "not found",
                severity="must",
            )
        )

    checks += _signal_count_checks("buffett", expectations.buffett_signals, result.buffett_signals)
    checks += _signal_count_checks("lynch", expectations.lynch_signals, result.lynch_signals)

    required_risks = expectations.key_risks.must_include_one_of
    if required_risks:
        framings = [f"{risk.concept}: {'; '.join(risk.any_of)}" for risk in required_risks]
        semantic.append(
            (
                SemanticRequest(
                    check_id="key_risks_include_one_of",
                    text="\n".join(result.key_risks),
                    concept=framings,
                    ticker=example.ticker,
                    rubric=(
                        "Pass when the key-risks section substantively explains at least one "
                        "configured risk concept. A bare label or keyword with no company-specific "
                        "meaning does not pass."
                    ),
                ),
                "key_risks_include_one_of",
                "must",
                f"substantively covers any of {[risk.concept for risk in required_risks]}",
            )
        )

    min_numbers = expectations.numerical_grounding.min_specific_numbers
    n_numbers = len(_NUMBER_RE.findall(result.thesis))
    checks.append(
        CheckResult(
            check_name="numerical_grounding",
            passed=n_numbers >= min_numbers,
            expected=f">= {min_numbers} numbers",
            actual=str(n_numbers),
            severity="must",
        )
    )

    if expectations.deep_value is not None:
        checks += _deep_value_checks(
            result,
            expectations.deep_value,
            thesis_lower,
            forensic_evidence,
        )
        if expectations.deep_value.require_value_trap_assessment:
            semantic.append(
                (
                    SemanticRequest(
                        check_id="value_trap_assessed",
                        text=result.thesis + "\nKey risks:\n" + "\n".join(result.key_risks),
                        concept=_VALUE_TRAP_FRAMINGS,
                        ticker=example.ticker,
                        rubric=_VALUE_TRAP_RUBRIC,
                    ),
                    "value_trap_assessed",
                    "must",
                    "evidence-backed adverse or protective value-trap assessment",
                )
            )
    if expectations.closability is not None:
        checks += _closability_checks(result, expectations.closability, forensic_evidence)

    if judge is None:
        for _request, check_name, severity, expected in semantic:
            checks.append(
                CheckResult(
                    check_name=check_name,
                    passed=False,
                    expected=expected,
                    actual="not evaluated: semantic judge not configured",
                    severity="should" if check_name == consistency_id else severity,
                )
            )
        return _finalize(example.ticker, checks)

    requests = [item[0] for item in semantic]
    try:
        verdicts = _judge_many(judge, requests)
    except Exception as exc:  # preserve deterministic grades and expose judge failure
        checks.append(
            CheckResult(
                check_name="judge_available",
                passed=False,
                expected="semantic judge returns every requested verdict",
                actual=f"{type(exc).__name__}: {exc}",
                severity="must",
            )
        )
        for _request, check_name, severity, expected in semantic:
            checks.append(
                CheckResult(
                    check_name=check_name,
                    passed=False,
                    expected=expected,
                    actual="not evaluated because the semantic judge failed",
                    severity=severity,
                )
            )
        return _finalize(example.ticker, checks)

    for request, check_name, severity, expected in semantic:
        verdict = verdicts.get(request.check_id)
        if verdict is None:
            checks.append(
                CheckResult(
                    check_name=check_name,
                    passed=False,
                    expected=expected,
                    actual="judge omitted this verdict",
                    severity=severity,
                )
            )
            continue
        checks.append(
            CheckResult(
                check_name=check_name,
                passed=verdict.passes,
                expected=expected,
                actual=verdict.reasoning,
                severity=severity,
            )
        )
        if len(verdict.votes) > 1:
            votes = ", ".join(f"{vote.judge_id}={vote.passes}" for vote in verdict.votes)
            checks.append(
                CheckResult(
                    check_name=f"judge_agreement_{check_name}",
                    passed=verdict.agreement,
                    expected="all blinded judges agree",
                    actual=votes,
                    severity="must",
                )
            )

    return _finalize(example.ticker, checks)
