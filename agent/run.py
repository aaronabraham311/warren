import argparse
import sys
from datetime import datetime, timezone
from uuid import uuid4

import anthropic
from dotenv import load_dotenv

from agent.budget import Budget, RunContext
from agent.loop import CostAbortedError, analyze_ticker
from agent.persona import DefaultPersona
from agent.routing import HardcodedSonnetRouting

load_dotenv()  # must precede storage.engine import so WARREN_DB is applied before engine creation

from storage.engine import (  # noqa: E402
    ensure_prompt_version,
    migrate,
    upsert_analysis,
    write_run_end,
    write_run_start,
)
from storage.models import AnalysisData, RunStatus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Warren stock analysis agent")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Ticker symbol to analyse")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    migrate()

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
    run_context = RunContext(run_id=run_id, budget=budget)
    client = anthropic.Anthropic()

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
        write_run_end(
            run_id=run_id,
            status=status,
            total_input_tokens=budget.total_input_tokens,
            total_output_tokens=budget.total_output_tokens,
            total_cost_usd=budget.total_cost_usd,
            num_tool_calls=budget.total_tool_calls,
            completed_at=datetime.now(timezone.utc),
            error_msg=error_msg,
        )


if __name__ == "__main__":
    main()
