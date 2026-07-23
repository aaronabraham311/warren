"""Aggregate a single eval run's failures by check family and by ticker.

    python -m eval.analysis.failures runs/eval-2026-05-10.json

Offline: parses one eval JSON output (the ``list[EvalGrade.model_dump()]`` ``--output``
of ``python -m agent.eval --golden-set``) and groups must-severity failures, with the
judge's / grader's ``actual`` reasoning inline, so a run's failures can be scanned by
check family instead of ticker-by-ticker.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Check-family prefixes, matched longest-first against CheckResult.check_name (which
# for dynamic families like thesis_mentions_{concept} embeds the concept in the name).
CHECK_FAMILIES = [
    "thesis_mentions_",
    "thesis_not_mention_",
    "recommendation_in_allowed",
    "buffett_pros_min_count",
    "buffett_cons_min_count",
    "lynch_pros_min_count",
    "lynch_cons_min_count",
    "key_risks_include_one_of",
    "numerical_grounding",
    "fixture_missing",
]


def check_family(check_name: str) -> str:
    for family in CHECK_FAMILIES:
        if check_name.startswith(family):
            return family
    return check_name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("eval_json", type=Path, help="An eval --output JSON file")
    args = parser.parse_args(argv)

    raw: list[dict[str, object]] = json.loads(args.eval_json.read_text())
    by_family: dict[str, list[tuple[str, dict[str, object]]]] = defaultdict(list)
    for grade in raw:
        ticker = grade["ticker"]
        checks = grade["checks"]
        assert isinstance(ticker, str)
        assert isinstance(checks, list)
        for check in checks:
            if not check["passed"] and check["severity"] == "must":
                by_family[check_family(check["check_name"])].append((ticker, check))

    if not by_family:
        print(f"No must-severity failures in {args.eval_json}")
        return 0

    for family, failures in sorted(by_family.items(), key=lambda kv: -len(kv[1])):
        print(f"\n{family} ({len(failures)} failures)")
        for ticker, check in failures:
            print(f"  {ticker}: expected {check['expected']!r}, got {check['actual']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
