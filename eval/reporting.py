"""Aggregate eval reporting that preserves strict pass/fail while showing failure depth."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from eval.grader import CheckResult, EvalGrade


def check_family(check: CheckResult) -> str:
    name = check.check_name
    if name.startswith("thesis_mentions_"):
        return "synthesis_coverage"
    if name.startswith("thesis_not_mention_"):
        return "synthesis_guardrail"
    if name.startswith(("buffett_", "lynch_")):
        return "signal_coverage"
    if name.startswith(("key_risks_", "value_trap_")):
        return "risk_analysis"
    if name.startswith("recommendation_"):
        return "recommendation"
    if name in {"fixture_missing", "fixture_completeness", "fixture_evidence_parity"}:
        return "fixture_evidence"
    if name == "structured_output_valid":
        return "structured_output"
    if name in {"run_completed", "provider_terminal"}:
        return "run_execution"
    if name.startswith("judge_"):
        return "judge"
    if name.startswith("planning_"):
        return "tool_planning"
    if name.startswith("synthesis_"):
        return "synthesis_coverage"
    if name in {"ev_ebit_present", "ncav_cited", "universe_note_present"}:
        return "deep_value"
    return name


def _coverage(checks: list[CheckResult], family: str) -> dict[str, object]:
    relevant = [check for check in checks if check_family(check) == family]
    if not relevant:
        return {"status": "not_measured", "passed": 0, "total": 0, "rate": None}
    passed = sum(check.passed for check in relevant)
    return {
        "status": "measured",
        "passed": passed,
        "total": len(relevant),
        "rate": passed / len(relevant),
    }


def build_eval_report(grades: list[EvalGrade]) -> dict[str, object]:
    checks = [check for grade in grades for check in grade.checks]
    mandatory = [check for check in checks if check.severity == "must"]
    secondary = [check for check in checks if check.severity == "should"]
    failed_families = Counter(check_family(check) for check in mandatory if not check.passed)
    fixture_checks = [check for check in checks if check_family(check) == "fixture_evidence"]
    schema_checks = [check for check in checks if check_family(check) == "structured_output"]
    judge_checks = [check for check in checks if check.check_name.startswith("judge_agreement")]

    tickers: list[dict[str, object]] = []
    for grade in grades:
        failures = [check for check in grade.checks if not check.passed]
        tickers.append(
            {
                "ticker": grade.ticker,
                "strict_passed": grade.passed,
                "checks_passed": sum(check.passed for check in grade.checks),
                "checks_total": len(grade.checks),
                "failure_count": len(failures),
                "critical_failures": sum(check.severity == "must" for check in failures),
                "secondary_failures": sum(check.severity == "should" for check in failures),
                "failure_families": sorted({check_family(check) for check in failures}),
            }
        )

    return {
        "schema_version": 1,
        "strict_ticker_pass": {
            "passed": sum(grade.passed for grade in grades),
            "total": len(grades),
        },
        "mandatory_checks": {
            "passed": sum(check.passed for check in mandatory),
            "total": len(mandatory),
        },
        "secondary_checks": {
            "passed": sum(check.passed for check in secondary),
            "total": len(secondary),
        },
        "failed_mandatory_by_family": dict(sorted(failed_families.items())),
        "fixture_evidence_parity": (
            "not_measured"
            if not fixture_checks
            else "complete"
            if all(check.passed for check in fixture_checks)
            else "incomplete"
        ),
        "structured_output_failure_rate": (
            None
            if not schema_checks
            else sum(not check.passed for check in schema_checks) / len(schema_checks)
        ),
        "tool_planning_coverage": _coverage(checks, "tool_planning"),
        "synthesis_coverage": _coverage(checks, "synthesis_coverage"),
        "judge_disagreement_rate": (
            None
            if not judge_checks
            else sum(not check.passed for check in judge_checks) / len(judge_checks)
        ),
        "tickers": tickers,
    }


def write_eval_report(output_path: Path, grades: list[EvalGrade]) -> Path:
    report_path = output_path.with_suffix(output_path.suffix + ".report")
    report_path.write_text(
        json.dumps(build_eval_report(grades), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report_path
