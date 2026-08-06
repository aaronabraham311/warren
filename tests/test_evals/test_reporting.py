from typing import cast

from eval.grader import CheckResult, EvalGrade, Severity
from eval.reporting import build_eval_report


def _check(name: str, passed: bool, severity: Severity = "must") -> CheckResult:
    return CheckResult(
        check_name=name,
        passed=passed,
        expected="expected",
        actual="actual",
        severity=severity,
    )


def test_report_distinguishes_failure_depth_and_families() -> None:
    grades = [
        EvalGrade(
            ticker="AAPL",
            passed=False,
            checks=[
                _check("fixture_completeness", True),
                _check("thesis_mentions_capital_return", False),
                _check("buffett_pros_min_count", False, "should"),
            ],
            overall_notes="1/3 checks passed",
        ),
        EvalGrade(
            ticker="LUMN",
            passed=True,
            checks=[_check("thesis_mentions_secular_decline", True)],
            overall_notes="1/1 checks passed",
        ),
    ]

    report = build_eval_report(grades)

    assert report["strict_ticker_pass"] == {"passed": 1, "total": 2}
    assert report["mandatory_checks"] == {"passed": 2, "total": 3}
    assert report["secondary_checks"] == {"passed": 0, "total": 1}
    assert report["failed_mandatory_by_family"] == {"synthesis_coverage": 1}
    assert report["fixture_evidence_parity"] == "complete"
    assert report["synthesis_coverage"] == {
        "status": "measured",
        "passed": 1,
        "total": 2,
        "rate": 0.5,
    }
    ticker = cast(list[dict[str, object]], report["tickers"])[0]
    assert ticker["critical_failures"] == 1
    assert ticker["secondary_failures"] == 1


def test_report_marks_unobservable_metrics_as_not_measured() -> None:
    grade = EvalGrade(
        ticker="AAPL",
        passed=True,
        checks=[_check("recommendation_in_allowed", True)],
        overall_notes="1/1 checks passed",
    )

    report = build_eval_report([grade])

    assert report["tool_planning_coverage"] == {
        "status": "not_measured",
        "passed": 0,
        "total": 0,
        "rate": None,
    }
    assert report["judge_disagreement_rate"] is None
