"""Eval replay command — run the agent over the golden set and grade it.

    python -m agent.eval --golden-set --output runs/eval-2026-05-10.json

Determinism rests on three legs:

1. **Fixture replay.** ``FixtureToolRunner`` serves every tool call from disk, so no
   data-source client is constructed and the network is unreachable by construction.
2. **temperature=0.** Threaded into the Anthropic call. Not bit-exact, but stable enough
   for the keyword-presence assertions the grader makes.
3. **A stable run_id.** ``--eval-run-id`` pins it so two runs can be diffed in
   ``eval_runs``; writes are delete-then-insert, so a re-run overwrites in place.

A ticker with no recorded tool fixtures is failed with a single ``fixture_missing`` check
*without spending an LLM call* — otherwise the command would burn a full Sonnet run per
ticker to produce a guaranteed failure grounded in nothing but ``not_found`` errors.
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from agent.budget import Budget, RunContext
from agent.loop import analyze_ticker
from agent.models import DirtDecisionContract
from agent.persona import DefaultPersona, DirtPersona
from agent.routing import HardcodedSonnetRouting
from data_sources.cache import CacheStore
from data_sources.forensics import ForensicEvidenceBundle
from eval.golden_set import EvalExample, load_all_examples
from eval.grader import EvalGrade, failed_grade, grade_analysis
from eval.judge import SonnetThesisJudge, ThesisJudge
from eval.tool_fixtures import FIXTURES_DIR, FixtureToolRunner, has_tool_fixtures

load_dotenv()  # must precede storage.engine import so WARREN_DB applies before engine creation

from storage.engine import (  # noqa: E402
    ensure_prompt_version,
    ensure_run_started,
    get_session,
    migrate,
    write_eval_run,
    write_run_end,
)
from storage.logger import RunLogger  # noqa: E402

_LOG_DIR = Path("logs/runs")
# Well above the 1.25 nightly ceiling: one eval sweeps the whole golden set in one "run".
_EVAL_MAX_COST_USD = 5.00
_EVAL_TEMPERATURE = 0.0


def _new_eval_run_id() -> str:
    return f"eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"


def resolve_persona(example: EvalExample) -> DefaultPersona | DirtPersona:
    """Pick the persona an example replays under.

    Mirrors ``agent.run.resolve_persona``: ``persona: dirt`` examples exercise the
    deep-value ``DirtPersona``; everything else (the default, unset) stays on
    ``DefaultPersona`` — so existing golden files are unchanged.
    """
    return DirtPersona() if example.persona == "dirt" else DefaultPersona()


def _grade_one(
    example: EvalExample,
    run_id: str,
    budget: Budget,
    logger: RunLogger,
    persona: DefaultPersona | DirtPersona,
    routing_policy: HardcodedSonnetRouting,
    client: anthropic.Anthropic,
    fixtures_root: Path,
    judge: ThesisJudge | None = None,
) -> EvalGrade:
    if not has_tool_fixtures(example.ticker, fixtures_root):
        logger.log("ticker_skipped", ticker=example.ticker, reason="fixture_missing")
        return failed_grade(
            example.ticker,
            check_name="fixture_missing",
            expected=f"recorded tool fixtures under {fixtures_root / example.ticker / 'tools'}",
            actual="none found",
            notes="No tool fixtures recorded for this ticker; skipped without an LLM call.",
        )

    logger.log("ticker_started", ticker=example.ticker, phase="deep")
    run_context = RunContext(run_id=run_id, budget=budget, logger=logger)
    try:
        fixture_runner = FixtureToolRunner(example.ticker, fixtures_root)
        result = analyze_ticker(
            ticker=example.ticker,
            persona=persona,
            routing_policy=routing_policy,
            run_context=run_context,
            client=client,
            temperature=_EVAL_TEMPERATURE,
            tool_runner=fixture_runner,
        )
    except Exception as exc:  # noqa: BLE001 — one bad ticker must not abort the sweep
        logger.log("ticker_failed", ticker=example.ticker, error=str(exc))
        return failed_grade(
            example.ticker,
            check_name="run_completed",
            expected="no exception",
            actual=f"{type(exc).__name__}: {exc}",
            notes=f"Exception: {exc}",
        )

    logger.log(
        "ticker_completed",
        ticker=example.ticker,
        recommendation=result.recommendation,
        confidence=result.confidence,
        iterations=run_context.iterations,
        termination=result.termination_reason,
    )
    forensic_result = fixture_runner.served.get("get_forensic_evidence")
    forensic_evidence = (
        forensic_result.data
        if forensic_result is not None and isinstance(forensic_result.data, ForensicEvidenceBundle)
        else None
    )
    decision_result = fixture_runner.served.get("model_dirt_scenarios")
    served_dirt_decision = (
        decision_result.data
        if decision_result is not None and isinstance(decision_result.data, DirtDecisionContract)
        else None
    )
    return grade_analysis(
        result,
        example,
        judge,
        forensic_evidence,
        served_dirt_decision,
    )


def _print_summary(grades: list[EvalGrade]) -> None:
    passed = [g for g in grades if g.passed]
    failed = [g for g in grades if not g.passed]
    print(f"\n{'=' * 40}")
    print(f"Result: {len(passed)}/{len(grades)} examples passed")
    if failed:
        print("\nFailures:")
        for grade in failed:
            broken = [c for c in grade.checks if not c.passed and c.severity == "must"]
            for check in broken:
                print(
                    f"  {grade.ticker}  {check.check_name}: "
                    f"expected {check.expected}, got {check.actual}"
                )


def run_eval(
    output_path: Path | None = None,
    examples: list[EvalExample] | None = None,
    client: anthropic.Anthropic | None = None,
    eval_run_id: str | None = None,
    fixtures_root: Path = FIXTURES_DIR,
    judge: ThesisJudge | None = None,
) -> list[EvalGrade]:
    """Replay the agent over *examples* against fixtures, grade, persist, and summarise.

    *judge* is injected (not built here) so the harness stays deterministic and offline in
    tests — the CLI ``main()`` supplies the live Sonnet 5 judge; ``judge=None`` grades
    ``thesis_must_mention`` by substring, unchanged.
    """
    if examples is None:
        examples = load_all_examples()
    if client is None:
        client = anthropic.Anthropic()

    routing_policy = HardcodedSonnetRouting()
    # The run-level prompt version is keyed to the default persona; a DIRT example resolves
    # its own persona per-example below (an eval sweep can now mix personas).
    prompt_version_id = ensure_prompt_version(
        version_tag="v1",
        persona_system_prompt=DefaultPersona().system_prompt,
        routing_policy_name=type(routing_policy).__name__,
    )

    run_id = eval_run_id or _new_eval_run_id()
    started_at = datetime.now(timezone.utc)
    ensure_run_started(run_id, started_at, prompt_version_id=prompt_version_id)

    budget = Budget(max_cost_usd=_EVAL_MAX_COST_USD)
    logger = RunLogger(run_id, _LOG_DIR)
    logger.log("run_started", tickers=[e.ticker for e in examples], eval_run_id=run_id)

    grades: list[EvalGrade] = []
    for example in examples:
        persona = resolve_persona(example)
        grade = _grade_one(
            example, run_id, budget, logger, persona, routing_policy, client, fixtures_root, judge
        )
        grades.append(grade)
        write_eval_run(
            run_id,
            grade.ticker,
            grade.passed,
            check_results=json.dumps([c.model_dump() for c in grade.checks]),
            diff_notes=grade.overall_notes,
        )
        n_passed = sum(1 for c in grade.checks if c.passed)
        status = "✅" if grade.passed else "❌"
        print(f"{status} {grade.ticker}: {n_passed}/{len(grade.checks)} checks")

    _print_summary(grades)

    logger.log("run_completed", status="success", total_cost_usd=budget.total_cost_usd)
    write_run_end(
        run_id,
        status="success",
        total_input_tokens=budget.total_input_tokens,
        total_output_tokens=budget.total_output_tokens,
        total_cost_usd=budget.total_cost_usd,
        num_tool_calls=budget.total_tool_calls,
        completed_at=datetime.now(timezone.utc),
    )
    with get_session() as session:
        logger.flush_to_db(session)
    logger.close()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps([g.model_dump() for g in grades], indent=2))
        print(f"Output written to {output_path}")

    return grades


def main() -> None:
    parser = argparse.ArgumentParser(description="Warren eval replay — grade the golden set")
    parser.add_argument(
        "--golden-set",
        action="store_true",
        help="Run every example under eval/examples/ (currently the only mode)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the EvalGrade list as JSON to this path",
    )
    parser.add_argument(
        "--eval-run-id",
        default=None,
        help="Pin the run_id (default: eval-<UTC timestamp>) so two runs can be diffed",
    )
    args = parser.parse_args()

    if not args.golden_set:
        parser.error("nothing to do: pass --golden-set")

    migrate()
    # Semantic thesis grading, pinned to Sonnet 5, with verdicts cached in $WARREN_DB so
    # re-runs are stable and free. Built here (not in run_eval) so tests stay offline.
    client = anthropic.Anthropic()
    cache = CacheStore(sqlite3.connect(os.environ.get("WARREN_DB", "warren.db")))
    judge = SonnetThesisJudge(client, cache)
    grades = run_eval(
        output_path=args.output,
        client=client,
        eval_run_id=args.eval_run_id,
        judge=judge,
    )
    if any(not g.passed for g in grades):
        sys.exit(1)


if __name__ == "__main__":
    main()
