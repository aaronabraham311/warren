import json
from pathlib import Path

import pytest

from eval.analysis.diff_runs import load_grades_from_json, main


def _write_eval_json(path: Path, grades: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(grades))


def test_load_grades_from_json_matches_dashboard_shape(tmp_path: Path) -> None:
    path = tmp_path / "eval-a.json"
    _write_eval_json(
        path,
        [
            {
                "ticker": "AAPL",
                "passed": True,
                "checks": [
                    {
                        "check_name": "recommendation_in_allowed",
                        "passed": True,
                        "expected": "hold or buy",
                        "actual": "hold",
                        "severity": "must",
                    }
                ],
                "overall_notes": "1/1 checks passed",
            }
        ],
    )

    grades = load_grades_from_json(path)

    assert set(grades) == {"AAPL"}
    check = grades["AAPL"]["recommendation_in_allowed"]
    assert check.passed is True
    assert check.severity == "must"


def test_diff_runs_reports_fixes_and_regressions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = tmp_path / "eval-before.json"
    current = tmp_path / "eval-after.json"
    _write_eval_json(
        baseline,
        [
            {
                "ticker": "AAPL",
                "passed": False,
                "checks": [
                    {
                        "check_name": "numerical_grounding",
                        "passed": False,
                        "expected": ">=3 numbers",
                        "actual": "1 number",
                        "severity": "must",
                    },
                    {
                        "check_name": "recommendation_in_allowed",
                        "passed": True,
                        "expected": "hold or buy",
                        "actual": "hold",
                        "severity": "must",
                    },
                ],
                "overall_notes": "1/2 checks passed",
            }
        ],
    )
    _write_eval_json(
        current,
        [
            {
                "ticker": "AAPL",
                "passed": False,
                "checks": [
                    {
                        "check_name": "numerical_grounding",
                        "passed": True,
                        "expected": ">=3 numbers",
                        "actual": "4 numbers",
                        "severity": "must",
                    },
                    {
                        "check_name": "recommendation_in_allowed",
                        "passed": False,
                        "expected": "hold or buy",
                        "actual": "sell",
                        "severity": "must",
                    },
                ],
                "overall_notes": "1/2 checks passed",
            }
        ],
    )

    exit_code = main([str(baseline), str(current)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "net: 1 fixed, 1 regressed" in out
    assert "+ numerical_grounding" in out
    assert "- recommendation_in_allowed" in out
