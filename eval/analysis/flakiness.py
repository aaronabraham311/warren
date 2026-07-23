"""Distinguish real eval failures from temperature=0 / LLM-judge run-to-run variance.

    python -m eval.analysis.flakiness --runs 5              (live: N fresh replays)
    python -m eval.analysis.flakiness --from runs/*.json     (offline: aggregate existing outputs)

Live mode calls ``eval.runner.run_eval`` N times (costing N times the golden-set API
spend) rather than reimplementing the replay loop. Offline mode just aggregates
existing ``--output`` JSON files, so a flakiness read doesn't require re-spending API
budget if the runs already exist on disk.
"""

import argparse
import json
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from eval.runner import run_eval


def aggregate(all_grades: list[list[dict[str, object]]]) -> dict[tuple[str, str], float]:
    """(ticker, check_name) -> fraction of runs where the check passed.

    Each element of *all_grades* is one run's list of EvalGrade.model_dump() dicts
    (whichever source produced them — a fresh run_eval() call or json.loads on a
    recorded --output file share the same shape).
    """
    pass_counts: dict[tuple[str, str], int] = {}
    total_counts: dict[tuple[str, str], int] = {}
    for grades in all_grades:
        for grade in grades:
            ticker = grade["ticker"]
            checks = grade["checks"]
            assert isinstance(ticker, str)
            assert isinstance(checks, list)
            for check in checks:
                key = (ticker, check["check_name"])
                total_counts[key] = total_counts.get(key, 0) + 1
                if check["passed"]:
                    pass_counts[key] = pass_counts.get(key, 0) + 1
    return {key: pass_counts.get(key, 0) / total for key, total in total_counts.items()}


def _load_json_grades(path: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = json.loads(path.read_text())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--runs", type=int, help="Live mode: replay the golden set N times")
    group.add_argument(
        "--from",
        dest="from_paths",
        nargs="+",
        type=Path,
        help="Offline mode: aggregate these existing eval JSON outputs",
    )
    args = parser.parse_args(argv)

    all_grades: list[list[dict[str, object]]]
    if args.runs:
        load_dotenv()
        client = anthropic.Anthropic()
        all_grades = []
        for i in range(args.runs):
            grades = run_eval(client=client, eval_run_id=f"flaky-{i}")
            all_grades.append([g.model_dump() for g in grades])
    else:
        all_grades = [_load_json_grades(p) for p in args.from_paths]

    rates = aggregate(all_grades)
    n_runs = len(all_grades)
    flaky = {key: rate for key, rate in rates.items() if 0.0 < rate < 1.0}
    if not flaky:
        print(f"No flaky checks across {n_runs} runs.")
        return 0

    print(f"Flaky checks across {n_runs} runs (pass rate < 100%):\n")
    for (ticker, check_name), rate in sorted(flaky.items(), key=lambda kv: kv[1]):
        print(f"  {ticker:8s} {check_name:40s} {rate:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
