import json
from pathlib import Path

import pytest

from eval.analysis.failures import check_family, main


@pytest.mark.parametrize(
    ("check_name", "family"),
    [
        ("thesis_mentions_moat", "thesis_mentions_"),
        ("thesis_not_mention_guaranteed", "thesis_not_mention_"),
        ("numerical_grounding", "numerical_grounding"),
        ("some_unknown_check", "some_unknown_check"),
    ],
)
def test_check_family_groups_dynamic_check_names(check_name: str, family: str) -> None:
    assert check_family(check_name) == family


def test_main_groups_failures_by_family_and_ticker(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    eval_json = tmp_path / "eval-run.json"
    eval_json.write_text(
        json.dumps(
            [
                {
                    "ticker": "AAPL",
                    "passed": False,
                    "checks": [
                        {
                            "check_name": "thesis_mentions_moat",
                            "passed": False,
                            "expected": "moat",
                            "actual": "no match",
                            "severity": "must",
                        },
                        {
                            "check_name": "numerical_grounding",
                            "passed": False,
                            "expected": ">=3 numbers",
                            "actual": "1 number",
                            "severity": "must",
                        },
                        {
                            "check_name": "buffett_pros_min_count",
                            "passed": False,
                            "expected": ">=2",
                            "actual": "1",
                            "severity": "should",
                        },
                    ],
                    "overall_notes": "1/3 checks passed",
                },
                {
                    "ticker": "MSFT",
                    "passed": False,
                    "checks": [
                        {
                            "check_name": "thesis_mentions_cloud",
                            "passed": False,
                            "expected": "cloud",
                            "actual": "no match",
                            "severity": "must",
                        }
                    ],
                    "overall_notes": "0/1 checks passed",
                },
            ]
        )
    )

    exit_code = main([str(eval_json)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "thesis_mentions_ (2 failures)" in out
    assert "numerical_grounding (1 failures)" in out
    assert "AAPL: expected 'moat'" in out
    assert "MSFT: expected 'cloud'" in out
    # 'should' severity failures are excluded from the aggregation
    assert "buffett_pros_min_count" not in out
