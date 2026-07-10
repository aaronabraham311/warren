from datetime import date

from agent.models import AnalysisOutput, LynchBuffettSignals
from eval.golden_set import (
    EvalExample,
    EvalExpectations,
    KeyRisksExpectation,
    NumericalGrounding,
    RecommendationExpectation,
    SignalCount,
    SignalsExpectation,
    ThesisMention,
)
from eval.grader import failed_grade, grade_analysis


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


def test_grades_the_real_golden_set_examples() -> None:
    """The grader must accept every committed expectation file without raising."""
    from eval.golden_set import load_all_examples

    for example in load_all_examples():
        grade = grade_analysis(_analysis(ticker=example.ticker), example)
        assert grade.ticker == example.ticker
        assert grade.checks
