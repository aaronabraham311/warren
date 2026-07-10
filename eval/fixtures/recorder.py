"""Record tool-level eval fixtures by hitting the live APIs once.

    python -m eval.fixtures.recorder AAPL MSFT      # named tickers
    python -m eval.fixtures.recorder                # every golden-set ticker

Each call in :data:`RECORDED_CALLS` is executed through the *real* ``Tool.run`` and handed
to :func:`eval.tool_fixtures.record_tool_result`, which owns the on-disk format and the
path layout. :class:`eval.tool_fixtures.FixtureToolRunner` is the replay side; see
``eval/fixtures/README.md`` for the rotation policy.

A tool that returns ``ToolResultError`` is recorded as an error and reported at the end:
errors are data in this codebase, so a genuinely unavailable data source replays
deterministically rather than leaving a hole that silently falls through to the network.
An *exception* (a bug, not a data-source error) aborts nothing but is reported as failed
and leaves the previous fixture in place.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import dotenv

from agent.budget import Budget, RunContext
from agent.tools import TOOL_REGISTRY
from agent.tools.base import ToolResultError
from eval.golden_set import load_all_examples
from eval.tool_fixtures import FIXTURES_DIR, record_tool_result
from storage.logger import RunLogger

LOG_DIR = Path("logs/runs")


@dataclass(frozen=True)
class RecordedCall:
    """One tool invocation to record per ticker. ``ticker`` is filled in at record time."""

    tool: str
    input: dict[str, object] = field(default_factory=dict)


# The ticker-scoped, network-backed tools an eval run reaches for. Deliberately excludes
# screen_universe / get_holding_context / screen_watchlists (local CSV or entity-scoped).
RECORDED_CALLS: tuple[RecordedCall, ...] = (
    RecordedCall("get_quote"),
    RecordedCall("get_fundamentals"),
    RecordedCall("get_growth_metrics"),
    RecordedCall("get_valuation_multiples"),
    RecordedCall("get_quality_metrics"),
    RecordedCall("get_financial_strength"),
    RecordedCall("get_capital_allocation"),
    RecordedCall("estimate_intrinsic_value"),
    RecordedCall("get_insider_activity"),
    RecordedCall("get_key_persons"),
    RecordedCall("get_news"),
    RecordedCall("get_peer_comparison"),
    RecordedCall("read_filing", {"filing_type": "10-K", "section": "business"}),
    RecordedCall("read_filing", {"filing_type": "10-K", "section": "risk_factors"}),
    RecordedCall("read_filing", {"filing_type": "10-K", "section": "mdna"}),
)


@dataclass
class RecordSummary:
    ticker: str
    ok: int = 0
    errors: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def _context() -> RunContext:
    return RunContext(
        run_id="fixture-record", budget=Budget(), logger=RunLogger("fixture-record", LOG_DIR)
    )


def record_ticker(ticker: str, root: Path = FIXTURES_DIR) -> RecordSummary:
    """Execute every recorded call for *ticker* against live APIs and save the results."""
    ctx = _context()
    summary = RecordSummary(ticker=ticker)

    for call in RECORDED_CALLS:
        tool = TOOL_REGISTRY[call.tool]
        tool_input = tool.input_schema.model_validate({"ticker": ticker, **call.input})
        payload = tool_input.model_dump(mode="json")
        label = f"{ticker}/{call.tool}"

        try:
            result = tool.run(tool_input, ctx)
        except Exception as exc:  # noqa: BLE001 — report and move on; one bad tool ≠ lost run
            summary.failures.append(f"{label}: {exc}")
            print(f"  ✗ {label}: {exc}")
            continue

        path = record_tool_result(ticker, call.tool, payload, result, root)
        if isinstance(result, ToolResultError):
            summary.errors.append(f"{label}: {result.error_code} — {result.message}")
            print(f"  ! {label}: recorded error {result.error_code}")
        else:
            summary.ok += 1
            print(f"  ✓ {path.relative_to(root.parent)}")

    return summary


def main(argv: list[str] | None = None) -> int:
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Tickers to record; defaults to every ticker in the golden set.",
    )
    args = parser.parse_args(argv)

    tickers = args.tickers or [ex.ticker for ex in load_all_examples()]
    summaries = [record_ticker(t) for t in tickers]

    total_ok = sum(s.ok for s in summaries)
    errors = [e for s in summaries for e in s.errors]
    failures = [f for s in summaries for f in s.failures]
    print(f"\nRecorded {total_ok} fixtures across {len(tickers)} tickers.")
    for line in errors:
        print(f"  recorded error: {line}")
    for line in failures:
        print(f"  FAILED: {line}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
