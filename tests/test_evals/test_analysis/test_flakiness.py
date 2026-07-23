import json
from pathlib import Path

import pytest

from eval.analysis.flakiness import aggregate, main


def _grade(ticker: str, check_name: str, passed: bool) -> dict[str, object]:
    return {
        "ticker": ticker,
        "passed": passed,
        "checks": [
            {
                "check_name": check_name,
                "passed": passed,
                "expected": "x",
                "actual": "y",
                "severity": "must",
            }
        ],
        "overall_notes": "",
    }


def test_aggregate_computes_per_check_pass_rate() -> None:
    all_grades = [
        [_grade("AAPL", "numerical_grounding", True)],
        [_grade("AAPL", "numerical_grounding", False)],
        [_grade("AAPL", "numerical_grounding", True)],
    ]

    rates = aggregate(all_grades)

    assert rates[("AAPL", "numerical_grounding")] == pytest.approx(2 / 3)


def test_offline_mode_flags_a_flaky_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stable = [_grade("AAPL", "recommendation_in_allowed", True)]
    flaky_pass = [
        _grade("AAPL", "recommendation_in_allowed", True),
        _grade("MSFT", "numerical_grounding", True),
    ]
    flaky_fail = [
        _grade("AAPL", "recommendation_in_allowed", True),
        _grade("MSFT", "numerical_grounding", False),
    ]

    p1, p2, p3 = tmp_path / "r1.json", tmp_path / "r2.json", tmp_path / "r3.json"
    p1.write_text(json.dumps(stable))
    p2.write_text(json.dumps(flaky_pass))
    p3.write_text(json.dumps(flaky_fail))

    exit_code = main(["--from", str(p1), str(p2), str(p3)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "MSFT" in out
    assert "numerical_grounding" in out
    assert "AAPL" not in out  # recommendation_in_allowed passed every run — not flaky
