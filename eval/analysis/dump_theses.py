"""Print the full thesis for one or more golden-set tickers.

    python -m eval.analysis.dump_theses AAPL MSFT

Replays each ticker through ``analyze_ticker`` against its recorded tool fixtures
(``FixtureToolRunner``, ``temperature=0.0``) — the same deterministic seam
``eval/runner.py`` uses to grade the golden set. Unlike the eval harness, this prints
the full thesis/recommendation/signals/key_risks instead of grading them, so a failing
check can be understood in context rather than as a one-line judge verdict.

Hits the live agent: one LLM call (well, one agentic loop) per ticker.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from agent.budget import Budget, RunContext
from agent.loop import analyze_ticker
from agent.models import AnalysisOutput
from agent.persona import DefaultPersona
from agent.routing import HardcodedSonnetRouting
from eval.tool_fixtures import FIXTURES_DIR, FixtureToolRunner, has_tool_fixtures
from storage.logger import RunLogger

_LOG_DIR = Path("logs/runs")
_MAX_COST_USD = 5.00
_TEMPERATURE = 0.0


def dump_thesis(
    ticker: str,
    client: anthropic.Anthropic,
    fixtures_root: Path = FIXTURES_DIR,
) -> AnalysisOutput | None:
    """Replay *ticker* via FixtureToolRunner and return its AnalysisOutput.

    Returns None (and prints a notice) when no tool fixtures are recorded for the
    ticker — mirrors the fixture_missing skip in eval/runner.py, without spending an
    LLM call on a run grounded in nothing but not_found errors.
    """
    if not has_tool_fixtures(ticker, fixtures_root):
        print(f"! {ticker}: no recorded tool fixtures under {fixtures_root / ticker / 'tools'}")
        return None

    run_id = f"dump-{ticker}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    run_context = RunContext(
        run_id=run_id,
        budget=Budget(max_cost_usd=_MAX_COST_USD),
        logger=RunLogger(run_id, _LOG_DIR),
    )
    return analyze_ticker(
        ticker=ticker,
        persona=DefaultPersona(),
        routing_policy=HardcodedSonnetRouting(),
        run_context=run_context,
        client=client,
        temperature=_TEMPERATURE,
        tool_runner=FixtureToolRunner(ticker, fixtures_root),
    )


def _print_thesis(ticker: str, result: AnalysisOutput) -> None:
    print(f"\n{'=' * 60}")
    print(f"{ticker}  ->  {result.recommendation.upper()}  (confidence: {result.confidence:.2f})")
    print(f"\nThesis:\n{result.thesis}")
    print(f"\nBuffett pros: {result.buffett_signals.pros}")
    print(f"Buffett cons: {result.buffett_signals.cons}")
    print(f"Lynch pros:   {result.lynch_signals.pros}")
    print(f"Lynch cons:   {result.lynch_signals.cons}")
    print("\nKey risks:")
    for risk in result.key_risks:
        print(f"  - {risk}")
    if result.data_quality_notes:
        print("\nData quality notes:")
        for note in result.data_quality_notes:
            print(f"  - {note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tickers", nargs="+", help="Golden-set tickers, e.g. AAPL MSFT")
    args = parser.parse_args(argv)

    load_dotenv()
    client = anthropic.Anthropic()

    for ticker in args.tickers:
        result = dump_thesis(ticker.upper(), client)
        if result is not None:
            _print_thesis(ticker.upper(), result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
