"""Structured per-ticker, per-check diff between two eval JSON outputs.

    python -m eval.analysis.diff_runs runs/eval-before.json runs/eval-after.json

Pure and offline: reads the two JSON files (the ``list[EvalGrade.model_dump()]``
``--output`` of ``python -m agent.eval --golden-set``) and reuses
``dashboard.data.diff_eval_runs`` for the actual diffing — the same logic the
dashboard's Eval page uses to diff two DB-backed runs, just fed grades built from
files instead of ``EvalRun`` rows.
"""

import argparse
import json
import sys
from pathlib import Path

from dashboard.data import EvalCheckResult, EvalRunDiff, diff_eval_runs


def load_grades_from_json(path: Path) -> dict[str, dict[str, EvalCheckResult]]:
    """Parse an eval JSON output into the {ticker: {check_name: EvalCheckResult}} shape
    that dashboard.data.load_eval_grades builds from the DB, so diff_eval_runs is agnostic
    to whether grades came from SQLite or a file."""
    raw = json.loads(path.read_text())
    grades: dict[str, dict[str, EvalCheckResult]] = {}
    for grade in raw:
        grades[grade["ticker"]] = {
            check["check_name"]: EvalCheckResult(
                check_name=check["check_name"],
                passed=check["passed"],
                expected=check["expected"],
                actual=check["actual"],
                severity=check["severity"],
            )
            for check in grade["checks"]
        }
    return grades


def _print_diff(diff: EvalRunDiff) -> None:
    print(f"{diff.baseline_run_id} -> {diff.current_run_id}")
    print(f"net: {diff.fixes} fixed, {diff.regressions} regressed\n")
    for ticker_diff in diff.ticker_diffs:
        print(f"{ticker_diff.ticker}:")
        for change in ticker_diff.changes:
            marker = {"fix": "+", "regression": "-", "other": "~"}[change.kind]
            print(
                f"  {marker} {change.check_name}: {change.old} -> {change.new} "
                f"(expected {change.expected!r}, actual {change.actual!r})"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", type=Path, help="Baseline eval JSON output")
    parser.add_argument("current", type=Path, help="Current eval JSON output")
    args = parser.parse_args(argv)

    grades_a = load_grades_from_json(args.baseline)
    grades_b = load_grades_from_json(args.current)
    diff = diff_eval_runs(args.baseline.stem, args.current.stem, grades_a, grades_b)
    _print_diff(diff)
    return 0


if __name__ == "__main__":
    sys.exit(main())
