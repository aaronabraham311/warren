from datetime import date

import pytest
from pydantic import ValidationError

from agent.models import AnalysisOutput, DirtDecisionContract, DirtSignals, LynchBuffettSignals
from dashboard.seed_demo import _BUY_DECISION
from data_sources.forensics import (
    CatalystEvidence,
    EvidenceRef,
    ForensicEvidenceBundle,
    HolderPosition,
)
from eval.golden_set import (
    ClosabilityExpectation,
    DeepValueExpectation,
    EvalExample,
    EvalExpectations,
    KeyRisksExpectation,
    NumericalGrounding,
    RecommendationExpectation,
    SignalCount,
    SignalsExpectation,
    ThesisMention,
    load_eval_example,
)
from eval.grader import _UNIVERSE_NOTE_SUBSTRING, failed_grade, grade_analysis
from eval.judge import JudgeVerdict


class _FakeJudge:
    """Deterministic, offline ThesisJudge: returns a scripted verdict and records calls."""

    def __init__(self, passes: bool, reasoning: str = "scripted") -> None:
        self._verdict = JudgeVerdict(passes=passes, reasoning=reasoning)
        self.calls: list[tuple[str, list[str], str]] = []

    def judge(self, *, thesis: str, concept: list[str], ticker: str) -> JudgeVerdict:
        self.calls.append((thesis, concept, ticker))
        return self._verdict


def _analysis(
    ticker: str = "AAPL",
    recommendation: str = "hold",
    thesis: str = "Trades at 28.4x earnings with 22% ROE and 6.1% FCF yield; moat intact.",
    buffett_pros: list[str] | None = None,
    lynch_pros: list[str] | None = None,
    key_risks: list[str] | None = None,
) -> AnalysisOutput:
    return AnalysisOutput(
        ticker=ticker,
        analysis_type="holding",
        recommendation=recommendation,
        confidence=0.7,
        thesis=thesis,
        lynch_signals=LynchBuffettSignals(pros=lynch_pros or ["steady grower"], cons=[]),
        buffett_signals=LynchBuffettSignals(
            pros=buffett_pros if buffett_pros is not None else ["moat", "high ROE"], cons=[]
        ),
        key_risks=key_risks or ["valuation stretched"],
    )


def _example(
    ticker: str = "AAPL",
    allowed: list[str] | None = None,
    must_mention: list[ThesisMention] | None = None,
    must_not_mention: list[str] | None = None,
    buffett: SignalsExpectation | None = None,
    key_risks: KeyRisksExpectation | None = None,
    min_numbers: int = 3,
) -> EvalExample:
    return EvalExample(
        ticker=ticker,
        notes="test fixture",
        last_curated=date(2026, 1, 1),
        expectations=EvalExpectations(
            recommendation=RecommendationExpectation(allowed=allowed or ["hold", "buy"]),
            thesis_must_mention=must_mention or [],
            thesis_must_not_mention=must_not_mention or [],
            buffett_signals=buffett or SignalsExpectation(),
            key_risks=key_risks or KeyRisksExpectation(),
            numerical_grounding=NumericalGrounding(min_specific_numbers=min_numbers),
        ),
    )


def test_recommendation_in_allowed_passes_and_fails() -> None:
    ok = grade_analysis(_analysis(recommendation="hold"), _example(allowed=["hold"]))
    assert ok.passed

    bad = grade_analysis(_analysis(recommendation="sell"), _example(allowed=["hold", "buy"]))
    assert not bad.passed
    check = next(c for c in bad.checks if c.check_name == "recommendation_in_allowed")
    assert check.severity == "must"
    assert check.actual == "sell"


def test_thesis_must_mention_uses_any_of_semantics() -> None:
    example = _example(must_mention=[ThesisMention(any_of=["services", "moat"])])
    # "moat" is present but "services" is not — any_of is satisfied.
    assert grade_analysis(_analysis(), example).passed

    missing = _example(must_mention=[ThesisMention(any_of=["dividend", "buyback"])])
    grade = grade_analysis(_analysis(), missing)
    assert not grade.passed
    assert any(c.check_name.startswith("thesis_mentions_") and not c.passed for c in grade.checks)


def test_judge_passes_thesis_that_misses_the_keyword_but_engages_the_topic() -> None:
    """The whole point: a topic-on, vocabulary-off thesis fails substring but passes the judge."""
    # "cost curve" is nowhere in the thesis, so the substring path would fail this.
    example = _example(must_mention=[ThesisMention(any_of=["cost curve", "breakeven"])])
    thesis = "CVX at 12.1x, 18% ROE, 4% yield: low on the industry's marginal production expense."
    judge = _FakeJudge(passes=True, reasoning="Thesis reasons about relative production cost.")

    # Substring path fails it...
    assert not grade_analysis(_analysis(thesis=thesis), example).passed
    # ...judge path passes it.
    grade = grade_analysis(_analysis(thesis=thesis), example, judge)
    assert grade.passed
    check = next(c for c in grade.checks if c.check_name.startswith("thesis_mentions_"))
    assert check.severity == "must"
    assert check.actual == "Thesis reasons about relative production cost."
    # The judge saw the full thesis, the concept keywords, and the ticker.
    assert judge.calls == [(thesis, ["cost curve", "breakeven"], "AAPL")]


def test_judge_negative_fails_the_example_at_must_severity() -> None:
    example = _example(must_mention=[ThesisMention(any_of=["moat"])])
    # "moat" IS present, so substring would pass — but the judge overrides to a fail.
    judge = _FakeJudge(passes=False, reasoning="Mentions moat but never analyzes it.")
    grade = grade_analysis(_analysis(), example, judge)
    assert not grade.passed
    check = next(c for c in grade.checks if c.check_name.startswith("thesis_mentions_"))
    assert not check.passed
    assert check.severity == "must"


def test_judge_is_not_called_when_absent() -> None:
    """judge=None keeps the substring path — the default and back-compat guarantee."""
    example = _example(must_mention=[ThesisMention(any_of=["moat"])])
    judge = _FakeJudge(passes=False)
    grade_analysis(_analysis(), example)  # no judge arg
    assert judge.calls == []


def test_thesis_must_not_mention_fails_when_forbidden_term_present() -> None:
    thesis = "A 10.5% grower at 12.1x, but this is not investment advice; 3 catalysts."
    grade = grade_analysis(
        _analysis(thesis=thesis), _example(must_not_mention=["not investment advice"])
    )
    assert not grade.passed
    check = next(c for c in grade.checks if c.check_name.startswith("thesis_not_mention_"))
    assert check.actual == "found"


def test_should_failure_does_not_fail_the_example() -> None:
    """A `should` check can fail while grade.passed stays True — the core severity rule."""
    example = _example(buffett=SignalsExpectation(pros=SignalCount(min_count=5)))
    grade = grade_analysis(_analysis(buffett_pros=["moat"]), example)

    count_check = next(c for c in grade.checks if c.check_name == "buffett_pros_min_count")
    assert not count_check.passed
    assert count_check.severity == "should"
    assert grade.passed


def test_must_failure_fails_the_example() -> None:
    grade = grade_analysis(_analysis(recommendation="sell"), _example(allowed=["buy"]))
    assert not grade.passed
    assert all(c.severity == "must" for c in grade.checks if not c.passed)


def test_signal_count_check_omitted_when_yaml_does_not_set_it() -> None:
    grade = grade_analysis(_analysis(), _example(buffett=SignalsExpectation()))
    assert not any(c.check_name.endswith("_min_count") for c in grade.checks)


def test_key_risks_include_one_of() -> None:
    example = _example(key_risks=KeyRisksExpectation(must_include_one_of=["china", "regulatory"]))
    ok = grade_analysis(_analysis(key_risks=["China exposure is material"]), example)
    assert ok.passed

    bad = grade_analysis(_analysis(key_risks=["valuation"]), example)
    assert not bad.passed
    assert (
        next(c for c in bad.checks if c.check_name == "key_risks_include_one_of").severity == "must"
    )


def test_numerical_grounding_counts_specific_numbers() -> None:
    grade = grade_analysis(_analysis(thesis="A great company at a fair price."), _example())
    check = next(c for c in grade.checks if c.check_name == "numerical_grounding")
    assert not check.passed
    assert check.actual == "0"
    assert not grade.passed

    grounded = grade_analysis(_analysis(), _example(min_numbers=3))
    assert next(c for c in grounded.checks if c.check_name == "numerical_grounding").passed


def test_overall_notes_reports_check_tally() -> None:
    grade = grade_analysis(_analysis(), _example())
    n_passed = sum(1 for c in grade.checks if c.passed)
    assert grade.overall_notes == f"{n_passed}/{len(grade.checks)} checks passed"


def test_failed_grade_is_a_single_must_failure() -> None:
    grade = failed_grade("NKE", "fixture_missing", "fixtures", "none found", "skipped")
    assert not grade.passed
    assert [c.check_name for c in grade.checks] == ["fixture_missing"]
    assert grade.checks[0].severity == "must"
    assert grade.overall_notes == "skipped"


_UNIVERSE_NOTE = (
    "DIRT universe: US small-caps (Russell 2000) plus Euronext Growth Milan (.MI), "
    "Bolsa de Madrid (.MC), and GPW Warsaw (.WA); market-cap gates are USD-normalized; "
    "aggregator reliability still degrades for micro-caps (sub-$300M USD), and non-US names "
    "often lack SEC/EDGAR filings."
)


def _dirt_analysis(
    dirt_signals: DirtSignals | None = None,
    key_risks: list[str] | None = None,
    data_quality_notes: list[str] | None = None,
    thesis: str = "Cheap at 6.1x EV/EBIT, trades at 0.8x NCAV with net cash of 55M.",
) -> AnalysisOutput:
    return AnalysisOutput(
        ticker="DIR.MI",
        analysis_type="discovery",
        recommendation="buy",
        confidence=0.6,
        thesis=thesis,
        lynch_signals=LynchBuffettSignals(pros=[], cons=[]),
        buffett_signals=LynchBuffettSignals(pros=[], cons=[]),
        key_risks=key_risks or ["value trap risk if earnings roll over"],
        data_quality_notes=data_quality_notes or [_UNIVERSE_NOTE],
        dirt_signals=dirt_signals
        if dirt_signals is not None
        else DirtSignals(ev_ebit=6.1, price_to_ncav=0.8, ncav_discount_pct=20.0),
    )


def _dirt_example() -> EvalExample:
    return EvalExample(
        ticker="DIR.MI",
        notes="deep-value gem",
        last_curated=date(2026, 1, 1),
        persona="dirt",
        expectations=EvalExpectations(
            recommendation=RecommendationExpectation(allowed=["buy", "hold"]),
            numerical_grounding=NumericalGrounding(min_specific_numbers=0),
            deep_value=DeepValueExpectation(
                require_ev_ebit=True,
                require_ncav=True,
                require_value_trap_risk=True,
                require_universe_note=True,
            ),
        ),
    )


def _closability_example(expectation: ClosabilityExpectation) -> EvalExample:
    example = _dirt_example()
    example.expectations.closability = expectation
    return example


def _forensic_with_catalyst(evidence_id: str) -> ForensicEvidenceBundle:
    ref = EvidenceRef.model_construct(  # type: ignore[call-arg]
        evidence_id=evidence_id
    )
    return ForensicEvidenceBundle.model_construct(  # type: ignore[call-arg]
        cap_table=[],
        stake_events=[],
        agreements=[],
        related_party_transactions=[],
        auditor_history=[],
        debt_facilities=[],
        capital_returns=[],
        leadership_events=[],
        catalysts=[
            CatalystEvidence.model_construct(  # type: ignore[call-arg]
                evidence_refs=[ref]
            )
        ],
    )


_DIRT_CHECKS = {
    "ev_ebit_present",
    "ncav_cited",
    "value_trap_risk_surfaced",
    "universe_note_present",
}


def test_deep_value_checks_pass_when_all_present() -> None:
    grade = grade_analysis(_dirt_analysis(), _dirt_example())
    dirt = [c for c in grade.checks if c.check_name in _DIRT_CHECKS]
    assert {c.check_name for c in dirt} == _DIRT_CHECKS
    assert all(c.passed and c.severity == "must" for c in dirt)
    assert grade.passed, grade.overall_notes


def test_ev_ebit_check_fails_when_absent_from_signals_and_thesis() -> None:
    analysis = _dirt_analysis(
        dirt_signals=DirtSignals(ev_ebit=None, price_to_ncav=0.8),
        thesis="A cheap, profitable small-cap trading below its net current asset value.",
    )
    grade = grade_analysis(analysis, _dirt_example())
    check = next(c for c in grade.checks if c.check_name == "ev_ebit_present")
    assert not check.passed
    assert not grade.passed


def test_ncav_check_fails_when_absent_from_signals_and_thesis() -> None:
    analysis = _dirt_analysis(
        dirt_signals=DirtSignals(ev_ebit=6.1, price_to_ncav=None, ncav_discount_pct=None),
        thesis="Cheap at 6.1x EV/EBIT with a clean balance sheet and net cash.",
    )
    grade = grade_analysis(analysis, _dirt_example())
    check = next(c for c in grade.checks if c.check_name == "ncav_cited")
    assert not check.passed
    assert not grade.passed


def test_value_trap_check_fails_when_no_such_risk_surfaced() -> None:
    analysis = _dirt_analysis(key_risks=["FX translation drag on reported revenue"])
    analysis.thesis = "Cheap at 6.1x EV/EBIT, trades at 0.8x NCAV; earnings are steady."
    grade = grade_analysis(analysis, _dirt_example())
    check = next(c for c in grade.checks if c.check_name == "value_trap_risk_surfaced")
    assert not check.passed
    assert not grade.passed


def test_universe_note_check_fails_when_missing() -> None:
    analysis = _dirt_analysis(data_quality_notes=["some other note"])
    grade = grade_analysis(analysis, _dirt_example())
    check = next(c for c in grade.checks if c.check_name == "universe_note_present")
    assert not check.passed
    assert not grade.passed


def test_forensic_claims_require_evidence_ids_when_opted_in() -> None:
    with pytest.raises(ValidationError, match="require cited evidence IDs"):
        DirtSignals(
            ev_ebit=6.1,
            price_to_ncav=0.8,
            controller_identified=True,
            controller_name="Founding Family",
        )


def test_forensic_claims_pass_with_compact_evidence_ids() -> None:
    example = _dirt_example()
    assert example.expectations.deep_value is not None
    example.expectations.deep_value.require_forensic_citations = True
    analysis = _dirt_analysis(
        dirt_signals=DirtSignals(
            ev_ebit=6.1,
            price_to_ncav=0.8,
            controller_identified=True,
            controller_name="Founding Family",
            catalyst_strength="observable",
            catalyst_stage="board_authorized",
            catalyst_description="Board-authorized asset sale",
            forensic_evidence_ids=["evidence-cap-table-1", "evidence-catalyst-2"],
        ),
        thesis="Cheap at 6.1x EV/EBIT and 0.8x NCAV with an observable catalyst.",
    )
    ownership_ref = EvidenceRef.model_construct(  # type: ignore[call-arg]
        evidence_id="evidence-cap-table-1"
    )
    catalyst_ref = EvidenceRef.model_construct(  # type: ignore[call-arg]
        evidence_id="evidence-catalyst-2"
    )
    forensic = ForensicEvidenceBundle.model_construct(  # type: ignore[call-arg]
        cap_table=[
            HolderPosition.model_construct(  # type: ignore[call-arg]
                evidence_refs=[ownership_ref]
            )
        ],
        stake_events=[],
        agreements=[],
        related_party_transactions=[],
        auditor_history=[],
        debt_facilities=[],
        capital_returns=[],
        leadership_events=[],
        catalysts=[
            CatalystEvidence.model_construct(  # type: ignore[call-arg]
                evidence_refs=[catalyst_ref]
            )
        ],
    )
    grade = grade_analysis(analysis, example, forensic_evidence=forensic)
    check = next(c for c in grade.checks if c.check_name == "forensic_claims_cited")
    assert check.passed
    assert check.severity == "must"


def test_closability_checks_are_opt_in() -> None:
    grade = grade_analysis(_dirt_analysis(), _dirt_example())
    assert not any(check.check_name.startswith("closability_") for check in grade.checks)


def test_sparse_forensic_coverage_stays_unknown() -> None:
    analysis = _dirt_analysis(
        dirt_signals=DirtSignals(
            ev_ebit=6.1,
            price_to_ncav=0.8,
            controller_identified=None,
            closability_status="unknown",
            closability_score=0.5,
            closability_confidence=0.2,
            closability_reasons=["Partial forensic coverage; controller status is unknown."],
        )
    )
    example = _closability_example(
        ClosabilityExpectation(
            allowed_status=["unknown"],
            min_score=0.4,
            max_score=0.6,
            max_confidence=0.4,
            require_unknown_semantics=True,
        )
    )

    grade = grade_analysis(analysis, example)

    closability_checks = [
        check for check in grade.checks if check.check_name.startswith("closability_")
    ]
    assert closability_checks
    assert all(check.passed for check in closability_checks)


def test_unknown_closability_rejects_false_controller_inference() -> None:
    analysis = _dirt_analysis(
        dirt_signals=DirtSignals(
            ev_ebit=6.1,
            price_to_ncav=0.8,
            closability_status="unknown",
            closability_score=0.5,
            closability_confidence=0.2,
            closability_reasons=["Coverage is partial."],
        )
    )
    assert analysis.dirt_signals is not None
    analysis.dirt_signals.controller_identified = False
    example = _closability_example(
        ClosabilityExpectation(
            allowed_status=["unknown"],
            require_unknown_semantics=True,
        )
    )

    grade = grade_analysis(analysis, example)

    check = next(
        check for check in grade.checks if check.check_name == "closability_unknown_stays_unknown"
    )
    assert not check.passed
    assert not grade.passed


def test_supported_closability_requires_cited_observable_catalyst() -> None:
    example = _closability_example(
        ClosabilityExpectation(
            allowed_status=["supported"],
            require_observable_or_contractual_catalyst=True,
        )
    )
    uncited = _dirt_analysis(
        dirt_signals=DirtSignals(
            ev_ebit=6.1,
            price_to_ncav=0.8,
            closability_status="supported",
            closability_score=0.8,
            closability_confidence=0.8,
            closability_reasons=["Board-authorized tender offer."],
            catalyst_strength="observable",
            catalyst_stage="board_authorized",
            catalyst_description="Board-authorized tender offer",
            forensic_evidence_ids=["temporarily-cited"],
        )
    )
    assert uncited.dirt_signals is not None
    uncited.dirt_signals.forensic_evidence_ids = []
    cited = uncited.model_copy(deep=True)
    assert cited.dirt_signals is not None
    cited.dirt_signals.forensic_evidence_ids = ["catalyst-board-1"]
    hallucinated = cited.model_copy(deep=True)
    assert hallucinated.dirt_signals is not None
    hallucinated.dirt_signals.forensic_evidence_ids = ["not-in-the-served-bundle"]

    failed = grade_analysis(uncited, example)
    passed = grade_analysis(
        cited,
        example,
        forensic_evidence=_forensic_with_catalyst("catalyst-board-1"),
    )
    hallucinated_grade = grade_analysis(
        hallucinated,
        example,
        forensic_evidence=_forensic_with_catalyst("catalyst-board-1"),
    )

    assert not next(
        check
        for check in failed.checks
        if check.check_name == "closability_catalyst_is_cited_and_observable"
    ).passed
    assert next(
        check
        for check in passed.checks
        if check.check_name == "closability_catalyst_is_cited_and_observable"
    ).passed
    assert not next(
        check
        for check in hallucinated_grade.checks
        if check.check_name == "closability_catalyst_is_cited_and_observable"
    ).passed


def test_dirt_decision_contract_recomputes_and_matches_served_tool() -> None:
    decision = DirtDecisionContract.model_validate(_BUY_DECISION)
    analysis = _dirt_analysis()
    analysis.dirt_decision = decision
    example = _dirt_example()
    assert example.expectations.deep_value is not None
    example.expectations.deep_value.require_decision_contract = True
    example.expectations.deep_value.require_decision_recomputation = True
    example.expectations.deep_value.require_served_decision_match = True
    example.expectations.deep_value.allowed_decision_outcomes = ["buy"]

    grade = grade_analysis(analysis, example, served_dirt_decision=decision)

    decision_checks = [
        check for check in grade.checks if check.check_name.startswith("dirt_decision_")
    ]
    assert {check.check_name for check in decision_checks} == {
        "dirt_decision_present",
        "dirt_decision_outcome_allowed",
        "dirt_decision_recomputes",
        "dirt_decision_matches_served_tool",
    }
    assert all(check.passed for check in decision_checks)


def test_dirt_decision_grader_rejects_tampered_math_and_unserved_contract() -> None:
    decision = DirtDecisionContract.model_validate(_BUY_DECISION)
    tampered = decision.model_copy(
        update={"probability_weighted_irr": decision.probability_weighted_irr + 0.05}
    )
    analysis = _dirt_analysis()
    analysis.dirt_decision = tampered
    example = _dirt_example()
    assert example.expectations.deep_value is not None
    example.expectations.deep_value.require_decision_recomputation = True
    example.expectations.deep_value.require_served_decision_match = True

    grade = grade_analysis(analysis, example, served_dirt_decision=decision)

    assert not next(
        check for check in grade.checks if check.check_name == "dirt_decision_recomputes"
    ).passed
    assert not next(
        check for check in grade.checks if check.check_name == "dirt_decision_matches_served_tool"
    ).passed
def test_deep_value_checks_omitted_for_default_examples() -> None:
    """No deep_value block → none of the DIRT checks are emitted (back-compat)."""
    grade = grade_analysis(_analysis(), _example())
    assert not any(
        c.check_name
        in {"ev_ebit_present", "ncav_cited", "value_trap_risk_surfaced", "universe_note_present"}
        for c in grade.checks
    )


def test_universe_note_substring_matches_the_g9_verbatim_note() -> None:
    assert _UNIVERSE_NOTE_SUBSTRING in _UNIVERSE_NOTE


def test_gem_yamls_load_and_validate_as_dirt() -> None:
    from eval.golden_set import EXAMPLES_DIR

    for stem in ("dir_mi", "cirsa_mc", "kpl_wa"):
        example = load_eval_example(EXAMPLES_DIR / f"{stem}.yaml")
        assert example.persona == "dirt"
        assert example.expectations.deep_value is not None
        assert example.expectations.deep_value.require_universe_note


def test_grades_the_real_golden_set_examples() -> None:
    """The grader must accept every committed expectation file without raising."""
    from eval.golden_set import load_all_examples

    for example in load_all_examples():
        grade = grade_analysis(_analysis(ticker=example.ticker), example)
        assert grade.ticker == example.ticker
        assert grade.checks
