"""Batch CLI adapter over Warren's shared run service."""

import argparse
import sys
from typing import Literal

import anthropic
from dotenv import load_dotenv

load_dotenv()  # must precede service/storage imports so WARREN_DB is applied once

from agent.budget import Budget  # noqa: E402
from agent.cancellation import CancellationToken  # noqa: E402
from agent.events import EventSink  # noqa: E402
from agent.locking import RunLockHeldError  # noqa: E402
from agent.loop import analyze_ticker  # noqa: E402
from agent.persona import DefaultPersona, DirtPersona  # noqa: E402
from agent.routing import PhaseBasedRouting  # noqa: E402
from agent.service import (  # noqa: E402
    MAX_SCREEN_CANDIDATES,
    RunMode,
    RunRequest,
    build_portfolio_context,
    execute_run,
    sync_input_data,
)
from agent.service import (  # noqa: E402
    _analyze_and_persist as _service_analyze_and_persist,
)
from agent.service import (  # noqa: E402
    resolve_persona as _resolve_persona,
)
from storage.logger import RunLogger  # noqa: E402

_MAX_SCREEN_CANDIDATES = MAX_SCREEN_CANDIDATES
_build_portfolio_context = build_portfolio_context
_sync_input_data = sync_input_data
resolve_persona = _resolve_persona


def _analyze_and_persist(
    ticker: str,
    analysis_type: Literal["holding", "discovery"],
    run_id: str,
    budget: Budget,
    logger: RunLogger,
    persona: DefaultPersona | DirtPersona,
    routing_policy: PhaseBasedRouting,
    client: anthropic.Anthropic,
    portfolio_context: str,
    *,
    cancellation: CancellationToken | None = None,
) -> None:
    """Compatibility seam for tests and integrations that used the former helper."""

    if analysis_type not in ("holding", "discovery"):
        raise ValueError(f"unsupported analysis type: {analysis_type}")
    _service_analyze_and_persist(
        ticker,
        analysis_type,
        run_id,
        budget,
        logger,
        persona,
        routing_policy,
        client,
        portfolio_context,
        cancellation=cancellation,
        analyzer=analyze_ticker,
    )


def build_parser() -> argparse.ArgumentParser:
    """The stable batch CLI surface used by people and scheduler scripts."""

    parser = argparse.ArgumentParser(description="Warren stock analysis agent")
    parser.add_argument(
        "ticker",
        nargs="?",
        default=None,
        help="Ticker to analyse. Omit for nightly mode: screen universe + analyse holdings.",
    )
    parser.add_argument(
        "--skip-ticker-validation",
        action="store_true",
        help="Skip yfinance ticker validation when loading the portfolio (faster startup)",
    )
    parser.add_argument(
        "--persona",
        choices=["default", "dirt"],
        default="default",
        help="Analysis persona: 'default' (Lynch/Buffett) or 'dirt' (deep-value DIRT methodology)",
    )
    parser.add_argument(
        "--gem-hunt",
        action="store_true",
        help=(
            "Opt-in nightly gem-hunt mode: global 3-exchange universe + deep-value screen + "
            "DIRT persona as one switch. The default US GARP nightly is untouched. Forces the "
            "DIRT persona regardless of --persona. Early runs may need --skip-ticker-validation."
        ),
    )
    return parser


def _request_from_args(args: argparse.Namespace) -> RunRequest:
    if args.ticker is not None:
        return RunRequest(
            mode=RunMode.TICKERS,
            tickers=[args.ticker],
            persona="dirt" if args.gem_hunt else args.persona,
            skip_ticker_validation=args.skip_ticker_validation,
        )
    return RunRequest(
        mode=RunMode.GEM_HUNT if args.gem_hunt else RunMode.DISCOVERY,
        persona=args.persona,
        skip_ticker_validation=args.skip_ticker_validation,
    )


def _print_result(result: object) -> None:
    from agent.service import RunResult

    if not isinstance(result, RunResult):
        raise TypeError("unexpected run result")
    if result.screening is not None:
        summary = result.screening
        if summary.gem_hunt:
            print(
                "Gem screen: "
                f"{summary.confirmed_count} confirmed, "
                f"{summary.needs_deeper_fetch_count} need deeper fetch, "
                f"{summary.source_error_count} source errors"
            )
        print(
            f"Screening surfaced {summary.confirmed_count} candidates; "
            f"analysing top {len(summary.selected_candidates)}: "
            f"{list(summary.selected_candidates)}"
        )
    for ticker_result in result.ticker_results:
        if ticker_result.analysis is not None:
            analysis = ticker_result.analysis
            print(
                f"[{ticker_result.ticker}] {analysis.recommendation} "
                f"({analysis.confidence:.2f}): {analysis.thesis[:80]}"
            )
        elif ticker_result.error is not None:
            print(f"[{ticker_result.ticker}] Error: {ticker_result.error}", file=sys.stderr)
    if result.status == "cost_aborted" and result.error_msg:
        print(f"Cost ceiling reached: {result.error_msg}", file=sys.stderr)
    elif result.status == "cancelled":
        print("Run cancelled safely.", file=sys.stderr)
    elif result.status == "failed" and result.error_msg:
        print(f"Run failed: {result.error_msg}", file=sys.stderr)


def main(*, event_sink: EventSink | None = None) -> None:
    args = build_parser().parse_args()
    try:
        result = execute_run(_request_from_args(args), event_sink=event_sink)
    except RunLockHeldError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    _print_result(result)
    if result.status != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
