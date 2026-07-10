import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import anthropic
from dotenv import load_dotenv

from agent.budget import Budget, RunContext
from agent.cooldown import filter_universe_for_cooldown
from agent.loop import CostAbortedError, analyze_ticker
from agent.models import DEFAULT_MODEL_ID
from agent.persona import DefaultPersona, DirtPersona
from agent.portfolio import (
    Holding,
    WatchlistEntry,
    load_portfolio,
    load_watchlist,
    sync_holdings_to_db,
    sync_watchlist_to_db,
)
from agent.routing import PhaseBasedRouting
from agent.screening import run_screening_pass
from agent.tools._clients import yfinance_client
from agent.universe import get_current_universe
from data_sources.yfinance_client import PriceData

load_dotenv()  # must precede storage.engine import so WARREN_DB is applied before engine creation

from storage.engine import (  # noqa: E402
    ensure_prompt_version,
    get_session,
    migrate,
    upsert_analysis,
    write_run_start,
)
from storage.logger import RunLogger  # noqa: E402
from storage.models import AnalysisData, RunStatus  # noqa: E402
from storage.recovery import reconcile_orphans  # noqa: E402

_LOG_DIR = Path("logs/runs")
_PORTFOLIO_FILE = Path("data/portfolio.csv")
_WATCHLIST_FILE = Path("data/watchlist.csv")
_MAX_SCREEN_CANDIDATES = 3


def _build_portfolio_context(portfolio_file: Path) -> str:
    if not portfolio_file.exists():
        return ""
    holdings = load_portfolio(portfolio_file, validate_tickers=False)
    if not holdings:
        return ""
    lines = ["User's current portfolio holdings:"]
    for h in holdings:
        lines.append(
            f"  {h.ticker}: {h.shares:.2f} shares, cost basis ${h.cost_basis:.2f}"
            f" (purchased {h.purchase_date})"
        )
    return "\n".join(lines)


def _sync_input_data(skip_ticker_validation: bool) -> None:
    """Load, validate, and snapshot the portfolio + watchlist into SQLite.

    Fails loudly on malformed input (the whole point of W2) before any analysis runs.
    """
    holdings = load_portfolio(_PORTFOLIO_FILE, validate_tickers=not skip_ticker_validation)
    entries = load_watchlist(_WATCHLIST_FILE) if _WATCHLIST_FILE.exists() else []

    current_prices: dict[str, float] = {}
    for h in holdings:
        price = yfinance_client().get_price(h.ticker)
        if isinstance(price, PriceData) and price.current_price is not None:
            current_prices[h.ticker] = price.current_price

    with get_session() as session:
        sync_holdings_to_db(holdings, session, current_prices)
        sync_watchlist_to_db(entries, session)


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
) -> None:
    """Run analysis for one ticker and persist the result. Raises on hard failure."""
    tokens_before = budget.total_input_tokens + budget.total_output_tokens
    calls_before = budget.total_tool_calls

    run_context = RunContext(run_id=run_id, budget=budget, logger=logger)
    logger.log("ticker_started", ticker=ticker, phase="deep", model=DEFAULT_MODEL_ID)

    result = analyze_ticker(
        ticker=ticker,
        persona=persona,
        routing_policy=routing_policy,
        run_context=run_context,
        client=client,
        portfolio_context=portfolio_context,
    )
    result.analysis_type = analysis_type
    result.tool_calls_made = budget.total_tool_calls - calls_before
    result.tokens_used = (budget.total_input_tokens + budget.total_output_tokens) - tokens_before

    logger.log(
        "ticker_completed",
        ticker=ticker,
        recommendation=result.recommendation,
        confidence=result.confidence,
        iterations=run_context.iterations,
        tokens=result.tokens_used,
        cost_usd=budget.total_cost_usd,
        termination=result.termination_reason,
    )
    upsert_analysis(
        run_id,
        ticker,
        AnalysisData(
            analysis_type=result.analysis_type,
            recommendation=result.recommendation,
            confidence=result.confidence,
            thesis=result.thesis,
            lynch_signals=result.lynch_signals.model_dump(),
            buffett_signals=result.buffett_signals.model_dump(),
            key_risks=result.key_risks,
            data_quality_notes=result.data_quality_notes,
            tool_calls_made=result.tool_calls_made,
            tokens_used=result.tokens_used,
            termination_reason=result.termination_reason,
        ),
    )
    print(f"[{ticker}] {result.recommendation} ({result.confidence:.2f}): {result.thesis[:80]}")


def _run_tickers(
    tickers: list[tuple[str, Literal["holding", "discovery"]]],
    run_id: str,
    budget: Budget,
    logger: RunLogger,
    persona: DefaultPersona | DirtPersona,
    routing_policy: PhaseBasedRouting,
    client: anthropic.Anthropic,
    portfolio_context: str,
) -> tuple[RunStatus, str | None]:
    logger.log("run_started", tickers=[t for t, _ in tickers])

    status: RunStatus = "success"
    error_msg: str | None = None

    for ticker, analysis_type in tickers:
        try:
            _analyze_and_persist(
                ticker,
                analysis_type,
                run_id,
                budget,
                logger,
                persona,
                routing_policy,
                client,
                portfolio_context,
            )
        except CostAbortedError as e:
            status = "cost_aborted"
            error_msg = str(e)
            print(f"Cost ceiling reached after {ticker}: {e}", file=sys.stderr)
            break
        except Exception as e:
            print(f"[{ticker}] Error: {e}", file=sys.stderr)
            logger.log("ticker_failed", ticker=ticker, error=str(e))

    return status, error_msg


def main() -> None:
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
        "--no-batch",
        action="store_true",
        help="Use sequential (non-batch) screening — immediate results, no 50%% discount",
    )
    args = parser.parse_args()

    migrate()
    reconcile_orphans(_LOG_DIR)  # self-heal any run left "running" by a previous crash
    _sync_input_data(args.skip_ticker_validation)

    persona: DefaultPersona | DirtPersona = (
        DirtPersona() if args.persona == "dirt" else DefaultPersona()
    )
    routing_policy = PhaseBasedRouting()

    prompt_version_id = ensure_prompt_version(
        version_tag="v1",
        persona_system_prompt=persona.system_prompt,
        routing_policy_name=type(routing_policy).__name__,
    )

    run_id = str(uuid4())
    started_at = datetime.now(timezone.utc)
    write_run_start(run_id, started_at, prompt_version_id=prompt_version_id)

    budget = Budget()
    logger = RunLogger(run_id, _LOG_DIR)
    client = anthropic.Anthropic()
    portfolio_context = _build_portfolio_context(_PORTFOLIO_FILE)

    holdings: list[Holding] = load_portfolio(_PORTFOLIO_FILE, validate_tickers=False)
    watchlist: list[WatchlistEntry] = (
        load_watchlist(_WATCHLIST_FILE) if _WATCHLIST_FILE.exists() else []
    )

    if args.ticker is not None:
        # Single-ticker deep analysis
        tickers: list[tuple[str, Literal["holding", "discovery"]]] = [
            (args.ticker.upper(), "holding")
        ]
    else:
        # Nightly mode: screen S&P 500 union watchlist, then analyse holdings + top candidates
        watchlist_tickers = [e.ticker for e in watchlist]
        with get_session() as session:
            universe = get_current_universe(session, watchlist_tickers)
            cooldown_result = filter_universe_for_cooldown(universe, session, recent_news={})
        logger.log(
            "discovery_cooldown_applied",
            suppressed_count=len(cooldown_result.suppressed),
            suppressed_tickers=cooldown_result.suppressed,
        )

        screening = run_screening_pass(
            cooldown_result.active,
            persona.system_prompt,
            use_batch_api=not args.no_batch,
            logger=logger,
        )
        candidates = screening.candidates[:_MAX_SCREEN_CANDIDATES]
        print(
            f"Screening surfaced {len(screening.candidates)} candidates; "
            f"analysing top {len(candidates)}: {candidates}"
        )

        tickers = [(h.ticker, "holding") for h in holdings] + [(c, "discovery") for c in candidates]

    status, error_msg = _run_tickers(
        tickers, run_id, budget, logger, persona, routing_policy, client, portfolio_context
    )

    duration_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()
    logger.log(
        "run_completed",
        status=status,
        total_cost_usd=budget.total_cost_usd,
        duration_seconds=duration_seconds,
        error_msg=error_msg,
    )
    with get_session() as session:
        logger.flush_to_db(session)
    logger.close()

    if status not in ("success",):
        sys.exit(1)


if __name__ == "__main__":
    main()
