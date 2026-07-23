"""Which tools did the model call per ticker, and did it skip an important one?

    python -m eval.analysis.trace_tools <run_id>

Offline: reads ``logs/runs/{run_id}.jsonl`` (the JSONL-as-WAL trace — see CLAUDE.md's
"Run logging" section) and groups ``tool_call`` events by ticker, the same shape the
CLAUDE.md ``jq`` one-liners parse. Answers "did the model even have the data to satisfy
a check" without hand-jq-ing the trace against the fixtures.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Tools whose absence for *any* analyzed ticker is worth flagging — a small, fixed
# constant, not derived from check names (no stable check<->tool mapping exists).
CORE_COVERAGE_TOOLS = {
    "read_filing",
    "get_capital_allocation",
    "get_quality_metrics",
    "get_insider_activity",
}


def load_tool_calls(run_id: str, logs_dir: Path = Path("logs/runs")) -> dict[str, list[str]]:
    """{ticker: [tool_name, ...]} (in call order) from the run's tool_call events."""
    path = logs_dir / f"{run_id}.jsonl"
    calls: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("event") != "tool_call":
                continue
            ticker = record.get("ticker")
            tool = record.get("tool")
            if ticker is None or tool is None:
                continue
            calls[ticker].append(tool)
    return dict(calls)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id", help="The run_id whose trace to inspect")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/runs"))
    args = parser.parse_args(argv)

    calls = load_tool_calls(args.run_id, args.logs_dir)
    if not calls:
        print(f"No tool_call events found for run_id {args.run_id!r} in {args.logs_dir}")
        return 0

    for ticker, tools in sorted(calls.items()):
        unique_tools = sorted(set(tools))
        print(f"{ticker}: {', '.join(unique_tools)}")
        missing = CORE_COVERAGE_TOOLS - set(tools)
        if missing:
            print(f"  ! never called: {', '.join(sorted(missing))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
