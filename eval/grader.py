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

import re
from typing import Literal

from pydantic import BaseModel

from agent.models import AnalysisOutput, LynchBuffettSignals
from eval.golden_set import EvalExample, SignalsExpectation
from eval.judge import ThesisJudge

Severity = Literal["must", "should"]

# A "specific number" is a bare integer/decimal, optionally a percentage: 12, 3.4, 18.7%.
_NUMBER_RE = re.compile(r"\d+\.?\d*%?")

_THESIS_EXCERPT_CHARS = 100


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


def grade_analysis(
    result: AnalysisOutput,
    example: EvalExample,
    judge: ThesisJudge | None = None,
) -> EvalGrade:
    """Grade *result* against *example*.

    When *judge* is supplied, each ``thesis_must_mention`` expectation is graded
    semantically (does the thesis reason about the topic, in any phrasing) rather than by
    substring match. Every other check family stays deterministic, and ``judge=None``
    preserves the exact substring behavior — the grader's default and the path all unit
    tests exercise.
    """
    expectations = example.expectations
    thesis_lower = result.thesis.lower()
    excerpt = result.thesis[:_THESIS_EXCERPT_CHARS]
    checks: list[CheckResult] = []

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

    for mention in expectations.thesis_must_mention:
        check_name = f"thesis_mentions_{_slug('_or_'.join(mention.any_of[:2]))}"
        if judge is not None:
            # Semantic grading: does the thesis reason about the topic in any phrasing?
            verdict = judge.judge(
                thesis=result.thesis, concept=mention.any_of, ticker=example.ticker
            )
            checks.append(
                CheckResult(
                    check_name=check_name,
                    passed=verdict.passes,
                    expected=f"engages topic {mention.any_of}",
                    actual=verdict.reasoning,
                    severity="must",
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
        present = forbidden.lower() in thesis_lower
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
        risks_blob = " ".join(result.key_risks).lower()
        hits = [r for r in required_risks if r.lower() in risks_blob]
        checks.append(
            CheckResult(
                check_name="key_risks_include_one_of",
                passed=bool(hits),
                expected=f"any of {required_risks}",
                actual=f"found {hits}" if hits else f"none present in {result.key_risks}",
                severity="must",
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

    return _finalize(example.ticker, checks)
