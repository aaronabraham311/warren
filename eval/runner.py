"""Eval replay command — run the agent over the golden set and grade it.

    python -m agent.eval --golden-set --output runs/eval-2026-05-10.json

Determinism rests on three legs:

1. **Fixture replay.** ``FixtureToolRunner`` serves every tool call from disk, so no
   data-source client is constructed and the network is unreachable by construction.
2. **Provider-specific settings.** Anthropic receives temperature 0; alternate providers
   omit it and use the exact requested model, reasoning effort, and service tier.
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

import anthropic
from dotenv import load_dotenv
from google import genai
from openai import OpenAI

from agent.budget import Budget, RunContext
from agent.loop import analyze_ticker
from agent.models import (
    FLEX_PRICING,
    GEMINI_3_6_FLASH,
    LUNA_5_6,
    PRICING,
    SONNET_4_6,
)
from agent.persona import DefaultPersona, DirtPersona
from agent.providers.anthropic import AnthropicProvider
from agent.providers.base import Message, Provider, ReasoningEffort, ServiceTier
from agent.providers.gemini import GeminiProvider
from agent.providers.openai import OpenAIProvider
from data_sources.cache import CacheStore
from data_sources.forensics import ForensicEvidenceBundle
from eval.golden_set import EvalExample, load_all_examples
from eval.grader import EvalGrade, failed_grade, grade_analysis
from eval.judge import SonnetThesisJudge, ThesisJudge
from eval.tool_fixtures import FIXTURES_DIR, FixtureToolRunner, has_tool_fixtures
from eval.usage import write_usage_sidecar

load_dotenv()  # must precede storage.engine import so WARREN_DB applies before engine creation

from storage.engine import (  # noqa: E402
    clear_eval_runs,
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

ProviderName = Literal["anthropic", "openai", "gemini"]
_DEFAULT_MODELS: dict[ProviderName, str] = {
    "anthropic": SONNET_4_6,
    "openai": LUNA_5_6,
    "gemini": GEMINI_3_6_FLASH,
}


@dataclass(frozen=True)
class EvalConfig:
    provider: ProviderName = "anthropic"
    model: str = SONNET_4_6
    service_tier: ServiceTier = "auto"
    reasoning_effort: ReasoningEffort = "none"

    @property
    def temperature(self) -> float | None:
        return _EVAL_TEMPERATURE if self.provider == "anthropic" else None


@dataclass(frozen=True)
class FixedModelRouting:
    model: str

    def select(self, iteration: int, messages: list[Message], ticker: str | None) -> str:
        del iteration, messages, ticker
        return self.model


def _new_eval_run_id() -> str:
    return f"eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"


def resolve_persona(example: EvalExample) -> DefaultPersona | DirtPersona:
    """Pick the persona an example replays under.

    Mirrors ``agent.run.resolve_persona``: ``persona: dirt`` examples exercise the
    deep-value ``DirtPersona``; everything else (the default, unset) stays on
    ``DefaultPersona`` — so existing golden files are unchanged.
    """
    return DirtPersona() if example.persona == "dirt" else DefaultPersona()


def resolve_eval_config(
    *,
    provider: ProviderName = "anthropic",
    model: str | None = None,
    service_tier: ServiceTier = "auto",
    reasoning_effort: ReasoningEffort = "none",
) -> EvalConfig:
    resolved_model = model or _DEFAULT_MODELS[provider]
    expected_prefix = {"anthropic": "claude-", "openai": "gpt-", "gemini": "gemini-"}[provider]
    if not resolved_model.startswith(expected_prefix):
        raise ValueError(f"Model {resolved_model!r} does not belong to provider {provider!r}")
    if resolved_model not in PRICING:
        raise ValueError(f"No pricing configured for model {resolved_model!r}")
    if service_tier == "flex" and resolved_model not in FLEX_PRICING:
        raise ValueError(f"Model {resolved_model!r} does not support Flex pricing")
    if provider == "anthropic" and reasoning_effort != "none":
        raise ValueError("Anthropic evals do not support --reasoning-effort")
    return EvalConfig(provider, resolved_model, service_tier, reasoning_effort)


def validate_api_keys(config: EvalConfig, environ: Mapping[str, str]) -> None:
    required = {"ANTHROPIC_API_KEY"}  # Sonnet 5 remains the semantic thesis judge.
    if config.provider == "openai":
        required.add("OPENAI_API_KEY")
    elif config.provider == "gemini":
        required.add("GEMINI_API_KEY")
    missing = sorted(key for key in required if not environ.get(key, "").strip())
    if missing:
        raise ValueError(f"Missing required API key(s): {', '.join(missing)}")


def create_provider(config: EvalConfig, environ: Mapping[str, str]) -> Provider:
    if config.provider == "anthropic":
        key = _required_key(environ, "ANTHROPIC_API_KEY")
        return AnthropicProvider(anthropic.Anthropic(api_key=key))
    if config.provider == "openai":
        key = _required_key(environ, "OPENAI_API_KEY")
        return OpenAIProvider(OpenAI(api_key=key))
    key = _required_key(environ, "GEMINI_API_KEY")
    return GeminiProvider(genai.Client(api_key=key))


def _required_key(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Missing required API key(s): {name}")
    return value


def _grade_one(
    example: EvalExample,
    run_id: str,
    budget: Budget,
    logger: RunLogger,
    persona: DefaultPersona | DirtPersona,
    routing_policy: FixedModelRouting,
    provider: Provider,
    config: EvalConfig,
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
            provider=provider,
            temperature=config.temperature,
            reasoning_effort=config.reasoning_effort,
            service_tier=config.service_tier,
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
    return grade_analysis(result, example, judge, forensic_evidence)


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
    provider: Provider | None = None,
    config: EvalConfig | None = None,
    eval_run_id: str | None = None,
    fixtures_root: Path = FIXTURES_DIR,
    judge: ThesisJudge | None = None,
) -> list[EvalGrade]:
    """Replay the agent over *examples* against fixtures, grade, persist, and summarise.

    *judge* is injected (not built here) so the harness stays deterministic and offline in
    tests — the CLI ``main()`` supplies the live Sonnet 5 judge; ``judge=None`` grades
    ``thesis_must_mention`` by substring, unchanged.
    """
    supplied_config = config or EvalConfig()
    config = resolve_eval_config(
        provider=supplied_config.provider,
        model=supplied_config.model,
        service_tier=supplied_config.service_tier,
        reasoning_effort=supplied_config.reasoning_effort,
    )
    if examples is None:
        examples = load_all_examples()
    if provider is not None and client is not None:
        raise ValueError("pass provider or client, not both")
    if provider is None:
        if client is not None:
            if config.provider != "anthropic":
                raise ValueError("an Anthropic client cannot run a non-Anthropic eval")
            provider = AnthropicProvider(client)
        else:
            provider = create_provider(config, os.environ)
    if provider.name != config.provider:
        raise ValueError(
            f"Configured provider {config.provider!r} does not match {provider.name!r}"
        )

    routing_policy = FixedModelRouting(config.model)
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
    clear_eval_runs(run_id)

    budget = Budget(max_cost_usd=_EVAL_MAX_COST_USD)
    log_path = _LOG_DIR / f"{run_id}.jsonl"
    # A pinned eval id means replace-in-place. Appending would double the WAL-derived
    # usage/cost while eval_runs rows themselves are overwritten.
    log_path.unlink(missing_ok=True)
    logger = RunLogger(run_id, _LOG_DIR)
    logger.log(
        "run_started",
        tickers=[e.ticker for e in examples],
        eval_run_id=run_id,
        provider=config.provider,
        model=config.model,
        service_tier=config.service_tier,
        reasoning_effort=config.reasoning_effort,
    )

    grades: list[EvalGrade] = []
    for example in examples:
        persona = resolve_persona(example)
        grade = _grade_one(
            example,
            run_id,
            budget,
            logger,
            persona,
            routing_policy,
            provider,
            config,
            fixtures_root,
            judge,
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
        output_path.write_text(
            json.dumps([g.model_dump() for g in grades], indent=2), encoding="utf-8"
        )
        usage_path = write_usage_sidecar(
            output_path=output_path,
            log_path=log_path,
            run_id=run_id,
            provider=config.provider,
            model=config.model,
            service_tier=config.service_tier,
            reasoning_effort=config.reasoning_effort,
            examples=len(grades),
            passed=sum(grade.passed for grade in grades),
        )
        print(f"Output written to {output_path}")
        print(f"Usage written to {usage_path}")

    return grades


def main(argv: Sequence[str] | None = None) -> int:
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
    parser.add_argument(
        "--provider", choices=("anthropic", "openai", "gemini"), default="anthropic"
    )
    parser.add_argument(
        "--model", default=None, help="Provider model id (default: provider baseline)"
    )
    parser.add_argument("--service-tier", choices=("auto", "default", "flex"), default="auto")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high"),
        default="none",
    )
    args = parser.parse_args(argv)

    if not args.golden_set:
        parser.error("nothing to do: pass --golden-set")

    try:
        config = resolve_eval_config(
            provider=cast(ProviderName, args.provider),
            model=cast(str | None, args.model),
            service_tier=cast(ServiceTier, args.service_tier),
            reasoning_effort=cast(ReasoningEffort, args.reasoning_effort),
        )
        validate_api_keys(config, os.environ)
    except ValueError as exc:
        parser.error(str(exc))

    migrate()
    # Semantic thesis grading, pinned to Sonnet 5, with verdicts cached in $WARREN_DB so
    # re-runs are stable and free. Built here (not in run_eval) so tests stay offline.
    judge_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    cache = CacheStore(sqlite3.connect(os.environ.get("WARREN_DB", "warren.db")))
    judge = SonnetThesisJudge(judge_client, cache)
    grades = run_eval(
        output_path=args.output,
        provider=create_provider(config, os.environ),
        config=config,
        eval_run_id=args.eval_run_id,
        judge=judge,
    )
    return 1 if any(not grade.passed for grade in grades) else 0


if __name__ == "__main__":
    sys.exit(main())
