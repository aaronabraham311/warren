import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import anthropic
from dotenv import load_dotenv

from agent.budget import Budget, RunContext
from agent.loop import CostAbortedError, analyze_ticker
from agent.models import DEFAULT_MODEL_ID
from agent.persona import DefaultPersona
from agent.portfolio import load_portfolio, load_watchlist, sync_holdings_to_db
from agent.routing import HardcodedSonnetRouting

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Warren stock analysis agent")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Ticker symbol to analyse")
    parser.add_argument(
        "--skip-ticker-validation",
        action="store_true",
        help="Skip yfinance ticker validation when loading portfolio (faster for large portfolios)",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    migrate()
    reconcile_orphans(_LOG_DIR)  # self-heal any run left "running" by a previous crash

    portfolio = load_portfolio(validate_tickers=not args.skip_ticker_validation)
    load_watchlist()
    with get_session() as session:
        sync_holdings_to_db(portfolio, session, {})

    persona = DefaultPersona()
    routing_policy = HardcodedSonnetRouting()

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
    run_context = RunContext(run_id=run_id, budget=budget, logger=logger)
    client = anthropic.Anthropic()
    logger.log("run_started", tickers=[ticker])
    logger.log("ticker_started", ticker=ticker, phase="deep", model=DEFAULT_MODEL_ID)

    status: RunStatus = "success"
    error_msg: str | None = None

    try:
        result = analyze_ticker(
            ticker=ticker,
            persona=persona,
            routing_policy=routing_policy,
            run_context=run_context,
            client=client,
        )
        logger.log(
            "ticker_completed",
            ticker=ticker,
            recommendation=result.recommendation,
            confidence=result.confidence,
            iterations=run_context.iterations,
            tokens=budget.total_input_tokens + budget.total_output_tokens,
            cost_usd=budget.total_cost_usd,
            termination="success",
        )
        upsert_analysis(
            run_id,
            ticker,
            AnalysisData(
                analysis_type=result.analysis_type,
                recommendation=result.recommendation,
                confidence=result.confidence,
                thesis=result.thesis,
                lynch_signals=result.lynch_signals,
                buffett_signals=result.buffett_signals,
                key_risks=result.key_risks,
                data_quality_notes=result.data_quality_notes,
                tool_calls_made=budget.total_tool_calls,
                tokens_used=budget.total_input_tokens + budget.total_output_tokens,
            ),
        )
        print(f"Done: {result.recommendation} ({result.confidence:.2f})")
        print(f"  {result.thesis}")
    except CostAbortedError as e:
        status = "cost_aborted"
        error_msg = str(e)
        print(f"Aborted: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        status = "failed"
        error_msg = str(e)
        print(f"Failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        duration_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()
        logger.log(
            "run_completed",
            status=status,
            total_cost_usd=budget.total_cost_usd,
            duration_seconds=duration_seconds,
            error_msg=error_msg,
        )
        # Reconcile the JSONL trace (source of truth) into the runs + tool_calls tables.
        with get_session() as session:
            logger.flush_to_db(session)
        logger.close()


if __name__ == "__main__":
    main()
