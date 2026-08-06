"""Controlled two-stage eval: fixture-backed evidence collection, then fresh synthesis.

The collection stage deliberately stops on the first provider turn without tool calls. The
observer is invoked by :func:`agent.loop.analyze_ticker` only after that response and its usage
have reached the WAL, but before the loop attempts to parse it as ``AnalysisOutput``. This
keeps collection quality separate from final-answer formatting quality.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, cast

import anthropic
from openai import OpenAI

from agent.budget import Budget, RunContext
from agent.loop import _parse_output, analyze_ticker
from agent.models import AnalysisOutput
from agent.persona import DefaultPersona, DirtPersona
from agent.providers.base import (
    Message,
    Provider,
    ProviderResponse,
    ReasoningEffort,
    ServiceTier,
    TextBlock,
    ToolCallBlock,
)
from agent.tools.base import ToolResult, ToolResultOk
from data_sources.cache import CacheStore
from eval.artifacts import CapturedResponse
from eval.fixture_evidence import CONTROLLED_FOLLOWUP_TICKERS
from eval.golden_set import EvalExample, load_all_examples
from eval.grader import CheckResult, EvalGrade, failed_grade, grade_analysis
from eval.judge import (
    JudgePanel,
    OpenAIThesisJudge,
    SonnetThesisJudge,
    ThesisJudge,
)
from eval.runner import (
    EvalConfig,
    FixedModelRouting,
    ProviderName,
    create_provider,
    resolve_eval_config,
    resolve_persona,
    select_examples,
)
from eval.tool_fixtures import FIXTURES_DIR, FixtureEvidenceIssue, FixtureMiss, FixtureToolRunner
from storage.cost import compute_cost
from storage.logger import RunLogger

DEFAULT_TICKERS = CONTROLLED_FOLLOWUP_TICKERS
_MAX_SYNTHESIS_TOKENS = 8192
_DEFAULT_MAX_COST_USD = 10.0
_MAX_PACKET_STRING_CHARS = 12_000
_MAX_PACKET_LIST_ITEMS = 25

_TOOL_DIMENSIONS: dict[str, str] = {
    "get_quote": "price",
    "get_fundamentals": "fundamentals",
    "get_growth_metrics": "growth",
    "get_valuation_multiples": "valuation",
    "get_valuation_history": "valuation",
    "estimate_intrinsic_value": "valuation",
    "get_quality_metrics": "quality",
    "get_financial_strength": "financial_strength",
    "get_capital_allocation": "capital_allocation",
    "get_insider_activity": "insiders",
    "get_key_persons": "management",
    "get_peer_comparison": "peers",
    "get_news": "news",
    "get_adverse_media": "adverse_media",
    "read_filing": "filings",
}


class EvidenceCollectionComplete(Exception):
    """Internal control-flow signal: collection reached its first non-tool response."""


class _FixtureObservation(Protocol):
    tool_name: str
    canonical_input: dict[str, object]
    input_hash: str
    result: ToolResult


class _ObservableFixtureRunner(Protocol):
    observations: list[_FixtureObservation]


@dataclass(frozen=True)
class StageProviderConfig:
    provider: ProviderName
    model: str
    service_tier: ServiceTier = "default"
    reasoning_effort: ReasoningEffort = "none"

    @property
    def temperature(self) -> float | None:
        return 0.0 if self.provider == "anthropic" else None


@dataclass(frozen=True)
class EvidenceRecord:
    tool_name: str
    canonical_input: dict[str, object]
    input_hash: str
    dimension: str
    data: dict[str, object]


@dataclass(frozen=True)
class PlanningCoverage:
    available_dimensions: tuple[str, ...]
    covered_dimensions: tuple[str, ...]
    coverage_ratio: float | None
    denominator_note: str


@dataclass(frozen=True)
class StageUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    tool_use_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0


@dataclass(frozen=True)
class CollectionStageResult:
    ticker: str
    valid: bool
    evidence: tuple[EvidenceRecord, ...]
    evidence_packet: str | None
    misses: tuple[FixtureMiss, ...]
    evidence_issues: tuple[FixtureEvidenceIssue, ...]
    planning_coverage: PlanningCoverage
    responses: tuple[CapturedResponse, ...]
    usage: StageUsage
    error: str | None = None


@dataclass(frozen=True)
class SynthesisStageResult:
    attempted: bool
    analysis: AnalysisOutput | None
    responses: tuple[CapturedResponse, ...]
    usage: StageUsage
    repair_attempted: bool = False
    error: str | None = None


@dataclass(frozen=True)
class StagedTickerResult:
    ticker: str
    repetition: int
    collector: StageProviderConfig
    synthesizer: StageProviderConfig
    collection: CollectionStageResult
    synthesis: SynthesisStageResult
    grade: EvalGrade | None = None

    @property
    def total_cost_usd(self) -> float:
        return self.collection.usage.cost_usd + self.synthesis.usage.cost_usd

    def to_dict(self) -> dict[str, object]:
        payload = cast(dict[str, object], asdict(self))
        synthesis = cast(dict[str, object], payload["synthesis"])
        if self.synthesis.analysis is not None:
            synthesis["analysis"] = self.synthesis.analysis.model_dump(mode="json")
        if self.grade is not None:
            payload["grade"] = self.grade.model_dump(mode="json")
        payload["total_cost_usd"] = self.total_cost_usd
        mandatory = (
            [check for check in self.grade.checks if check.severity == "must"]
            if self.grade is not None
            else []
        )
        payload["stage_metrics"] = {
            "strict_pass": self.grade.passed if self.grade is not None else None,
            "mandatory_checks_passed": sum(check.passed for check in mandatory),
            "mandatory_checks_total": len(mandatory),
            "schema_valid": self.synthesis.analysis is not None,
            "fixture_miss_count": len(self.collection.misses),
            "fixture_evidence_issue_count": len(self.collection.evidence_issues),
            "planning_coverage_ratio": self.collection.planning_coverage.coverage_ratio,
            "synthesis_attempted": self.synthesis.attempted,
            "synthesis_repair_attempted": self.synthesis.repair_attempted,
            "collector_tokens": (
                self.collection.usage.input_tokens + self.collection.usage.output_tokens
            ),
            "synthesizer_tokens": (
                self.synthesis.usage.input_tokens + self.synthesis.usage.output_tokens
            ),
            "collector_latency_ms": self.collection.usage.latency_ms,
            "synthesizer_latency_ms": self.synthesis.usage.latency_ms,
            "total_cost_usd": self.total_cost_usd,
        }
        return payload


def _stop_before_final_parsing(response: ProviderResponse) -> None:
    if not any(isinstance(block, ToolCallBlock) for block in response.blocks):
        raise EvidenceCollectionComplete


def _dimension(tool_name: str) -> str:
    return _TOOL_DIMENSIONS.get(tool_name, tool_name)


def _available_dimensions(ticker: str, fixtures_root: Path) -> tuple[str, ...]:
    tools_dir = fixtures_root / ticker / "tools"
    if not tools_dir.is_dir():
        return ()
    return tuple(
        sorted({_dimension(path.name) for path in tools_dir.iterdir() if any(path.glob("*.json"))})
    )


def _planning_coverage(
    ticker: str, fixtures_root: Path, evidence: Sequence[EvidenceRecord]
) -> PlanningCoverage:
    available = _available_dimensions(ticker, fixtures_root)
    covered = tuple(sorted({_dimension(item.tool_name) for item in evidence} & set(available)))
    ratio = len(covered) / len(available) if available else None
    all_dimensions = set(_TOOL_DIMENSIONS.values())
    if len(available) < len(all_dimensions):
        note = (
            f"Narrow denominator: {len(available)} fixture-backed dimensions are available; "
            f"the global catalog contains {len(all_dimensions)}."
        )
    else:
        note = "Coverage denominator includes every fixture-backed evidence dimension."
    return PlanningCoverage(available, covered, ratio, note)


def _evidence_records(runner: FixtureToolRunner) -> tuple[EvidenceRecord, ...]:
    observable = cast(_ObservableFixtureRunner, runner)
    observations = getattr(observable, "observations", [])
    by_key: dict[tuple[str, str], EvidenceRecord] = {}
    for observation in observations:
        if not isinstance(observation.result, ToolResultOk):
            continue
        key = (observation.tool_name, observation.input_hash)
        if key in by_key:
            continue
        by_key[key] = EvidenceRecord(
            tool_name=observation.tool_name,
            canonical_input=dict(observation.canonical_input),
            input_hash=observation.input_hash,
            dimension=_dimension(observation.tool_name),
            data=cast(dict[str, object], observation.result.data.model_dump(mode="json")),
        )
    return tuple(by_key[key] for key in sorted(by_key))


def build_evidence_packet(ticker: str, evidence: Sequence[EvidenceRecord]) -> str:
    """Serialize successful observations compactly, without benchmark expectations or labels."""
    deduplicated: dict[tuple[str, str], EvidenceRecord] = {}
    for record in evidence:
        deduplicated.setdefault((record.tool_name, record.input_hash), record)
    payload = {
        "ticker": ticker,
        "evidence": [
            _compact_packet_value(asdict(deduplicated[key])) for key in sorted(deduplicated)
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _compact_packet_value(value: object) -> object:
    """Bound verbose filing/news fields while retaining full observations in the audit result."""
    if isinstance(value, str):
        if len(value) <= _MAX_PACKET_STRING_CHARS:
            return value
        omitted = len(value) - _MAX_PACKET_STRING_CHARS
        return f"{value[:_MAX_PACKET_STRING_CHARS]}… [truncated {omitted} chars]"
    if isinstance(value, list):
        compacted = [_compact_packet_value(item) for item in value[:_MAX_PACKET_LIST_ITEMS]]
        if len(value) > _MAX_PACKET_LIST_ITEMS:
            compacted.append({"_omitted_items": len(value) - _MAX_PACKET_LIST_ITEMS})
        return compacted
    if isinstance(value, dict):
        return {str(key): _compact_packet_value(item) for key, item in value.items()}
    return value


def _budget_usage(budget: Budget, started: float) -> StageUsage:
    return StageUsage(
        input_tokens=budget.total_input_tokens,
        output_tokens=budget.total_output_tokens,
        cache_read_tokens=budget.total_cache_read_tokens,
        cache_write_tokens=budget.total_cache_creation_tokens,
        reasoning_tokens=budget.total_reasoning_tokens,
        tool_use_tokens=budget.total_tool_use_tokens,
        cost_usd=budget.total_cost_usd,
        latency_ms=int((time.monotonic() - started) * 1000),
    )


def _response_usage(
    responses: Sequence[ProviderResponse], config: StageProviderConfig, started: float
) -> StageUsage:
    input_tokens = sum(response.usage.input_tokens for response in responses)
    output_tokens = sum(response.usage.output_tokens for response in responses)
    cache_read = sum(response.usage.cache_read_tokens for response in responses)
    cache_write = sum(response.usage.cache_write_tokens for response in responses)
    reasoning = sum(response.usage.reasoning_tokens or 0 for response in responses)
    tool_use = sum(response.usage.tool_use_tokens for response in responses)
    cost = sum(
        compute_cost(
            config.model,
            input_tokens=response.usage.input_tokens,
            cache_read_tokens=response.usage.cache_read_tokens,
            cache_creation_tokens=response.usage.cache_write_tokens,
            output_tokens=response.usage.output_tokens,
            provider=config.provider,
            service_tier=config.service_tier,
        )
        for response in responses
    )
    return StageUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        reasoning_tokens=reasoning,
        tool_use_tokens=tool_use,
        cost_usd=cost,
        latency_ms=int((time.monotonic() - started) * 1000),
    )


def collect_evidence(
    *,
    ticker: str,
    provider: Provider,
    config: StageProviderConfig,
    run_context: RunContext,
    fixtures_root: Path = FIXTURES_DIR,
    persona: DefaultPersona | DirtPersona | None = None,
    fixture_runner: FixtureToolRunner | None = None,
) -> CollectionStageResult:
    """Run the ordinary agent loop only through evidence gathering."""
    started = time.monotonic()
    runner = fixture_runner or FixtureToolRunner(ticker, fixtures_root)
    before = _budget_usage(run_context.budget, started)
    error: str | None = None
    reached_boundary = False
    responses: list[CapturedResponse] = []

    def observe(response: ProviderResponse) -> None:
        responses.append(CapturedResponse.from_response(response))
        _stop_before_final_parsing(response)

    try:
        analyze_ticker(
            ticker=ticker,
            persona=persona or DefaultPersona(),
            routing_policy=FixedModelRouting(config.model),
            run_context=run_context,
            provider=provider,
            temperature=config.temperature,
            reasoning_effort=config.reasoning_effort,
            service_tier=config.service_tier,
            tool_runner=runner,
            response_observer=observe,
        )
    except EvidenceCollectionComplete:
        reached_boundary = True
    except Exception as exc:  # noqa: BLE001 — retain a per-ticker experimental result
        error = f"{type(exc).__name__}: {exc}"

    evidence = _evidence_records(runner)
    coverage = _planning_coverage(ticker, fixtures_root, evidence)
    after = _budget_usage(run_context.budget, started)
    usage = StageUsage(
        input_tokens=after.input_tokens - before.input_tokens,
        output_tokens=after.output_tokens - before.output_tokens,
        cache_read_tokens=after.cache_read_tokens - before.cache_read_tokens,
        cache_write_tokens=after.cache_write_tokens - before.cache_write_tokens,
        reasoning_tokens=after.reasoning_tokens - before.reasoning_tokens,
        tool_use_tokens=after.tool_use_tokens - before.tool_use_tokens,
        cost_usd=after.cost_usd - before.cost_usd,
        latency_ms=after.latency_ms,
    )
    if runner.misses:
        error = "fixture miss invalidated collection"
    elif runner.evidence_issues:
        error = "fixture evidence issue invalidated collection"
    elif not reached_boundary:
        error = error or "collector ended before the evidence boundary"
    elif not evidence:
        error = "collector produced no successful fixture observations"
    valid = error is None
    return CollectionStageResult(
        ticker=ticker,
        valid=valid,
        evidence=evidence,
        evidence_packet=build_evidence_packet(ticker, evidence) if valid else None,
        misses=tuple(runner.misses),
        evidence_issues=tuple(runner.evidence_issues),
        planning_coverage=coverage,
        responses=tuple(responses),
        usage=usage,
        error=error,
    )


_SYNTHESIS_SYSTEM_PROMPT = (
    "You are an investment research synthesizer. Use only the supplied evidence packet. "
    "Do not assume facts that are absent. Before finalizing, verify that the thesis contains "
    "4–7 substantive bullets, gives each material available evidence dimension its own "
    "analysis, includes at least three numerically grounded bullets when evidence permits, "
    "supports business drivers from the packet, states concrete risks, makes the recommendation "
    "consistent with the evidence, and records unavailable or conflicting data in "
    "data_quality_notes. Return only one JSON object matching this schema: "
    + json.dumps(AnalysisOutput.model_json_schema(), sort_keys=True)
)


def _synthesis_messages(ticker: str, evidence_packet: str) -> list[Message]:
    return [
        Message.text(
            "user",
            f"Synthesize a buy, sell, or hold analysis for {ticker} from this evidence packet:\n"
            f"{evidence_packet}",
        )
    ]


def synthesize_evidence(
    *,
    ticker: str,
    evidence_packet: str,
    provider: Provider,
    config: StageProviderConfig,
    persona: DefaultPersona | DirtPersona | None = None,
) -> SynthesisStageResult:
    """Make a fresh, tool-free synthesis call and allow exactly one schema repair."""
    started = time.monotonic()
    messages = _synthesis_messages(ticker, evidence_packet)
    responses: list[ProviderResponse] = []
    repair_attempted = False
    for attempt in range(2):
        response = provider.complete(
            model=config.model,
            system_prompt=(persona or DefaultPersona()).system_prompt
            + "\n\n━━━━━━━━ CONTROLLED SYNTHESIS STAGE ━━━━━━━━\n"
            + _SYNTHESIS_SYSTEM_PROMPT,
            portfolio_context="",
            messages=messages,
            tools=[],
            max_tokens=_MAX_SYNTHESIS_TOKENS,
            temperature=config.temperature,
            reasoning_effort=config.reasoning_effort,
            service_tier=config.service_tier,
        )
        responses.append(response)
        text = next(
            (block.text for block in reversed(response.blocks) if isinstance(block, TextBlock)),
            "",
        )
        try:
            analysis = _parse_output(text)
        except (ValueError, json.JSONDecodeError) as exc:
            if attempt == 1:
                return SynthesisStageResult(
                    attempted=True,
                    analysis=None,
                    responses=tuple(CapturedResponse.from_response(item) for item in responses),
                    usage=_response_usage(responses, config, started),
                    repair_attempted=True,
                    error=f"schema repair failed: {type(exc).__name__}: {exc}",
                )
            repair_attempted = True
            # Deliberately text-only: collector history and provider replay never cross stages.
            messages.extend(
                [
                    Message.text("assistant", text),
                    Message.text(
                        "user",
                        "Return only valid JSON matching the requested AnalysisOutput schema.",
                    ),
                ]
            )
            continue
        return SynthesisStageResult(
            attempted=True,
            analysis=analysis,
            responses=tuple(CapturedResponse.from_response(item) for item in responses),
            usage=_response_usage(responses, config, started),
            repair_attempted=repair_attempted,
        )
    raise AssertionError("two-attempt synthesis loop did not return")


def run_staged_ticker(
    *,
    ticker: str,
    repetition: int,
    collector_provider: Provider,
    collector_config: StageProviderConfig,
    synthesizer_provider: Provider,
    synthesizer_config: StageProviderConfig,
    run_context: RunContext,
    fixtures_root: Path = FIXTURES_DIR,
    persona: DefaultPersona | DirtPersona | None = None,
    example: EvalExample | None = None,
    judge: ThesisJudge | None = None,
) -> StagedTickerResult:
    collection = collect_evidence(
        ticker=ticker,
        provider=collector_provider,
        config=collector_config,
        run_context=run_context,
        fixtures_root=fixtures_root,
        persona=persona,
    )
    if not collection.valid or collection.evidence_packet is None:
        synthesis = SynthesisStageResult(
            attempted=False,
            analysis=None,
            responses=(),
            usage=StageUsage(),
            error="skipped because collection was invalid",
        )
    else:
        synthesis = synthesize_evidence(
            ticker=ticker,
            evidence_packet=collection.evidence_packet,
            provider=synthesizer_provider,
            config=synthesizer_config,
            persona=persona,
        )
    grade: EvalGrade | None = None
    if example is not None:
        if synthesis.analysis is not None:
            # Gold expectations enter only after generation; neither prompt contains them.
            grade = grade_analysis(synthesis.analysis, example, judge)
        else:
            grade = failed_grade(
                ticker,
                check_name="structured_output_valid",
                expected="valid synthesized AnalysisOutput",
                actual=synthesis.error or collection.error or "no synthesis output",
                notes="Staged synthesis did not produce a gradeable analysis.",
            )
        grade = _append_operational_checks(grade, collection, synthesis)
    return StagedTickerResult(
        ticker=ticker,
        repetition=repetition,
        collector=collector_config,
        synthesizer=synthesizer_config,
        collection=collection,
        synthesis=synthesis,
        grade=grade,
    )


def _append_operational_checks(
    grade: EvalGrade,
    collection: CollectionStageResult,
    synthesis: SynthesisStageResult,
) -> EvalGrade:
    """Use the baseline operational checks so mandatory coverage stays comparable."""
    operational_names = {
        "structured_output_valid",
        "fixture_completeness",
        "fixture_evidence_parity",
    }
    checks = [check for check in grade.checks if check.check_name not in operational_names]
    misses = ", ".join(f"{miss.tool_name}:{miss.input_hash}" for miss in collection.misses)
    issues = ", ".join(
        f"{issue.tool_name}:{issue.input_hash}:{issue.reason}"
        for issue in collection.evidence_issues
    )
    checks.extend(
        [
            CheckResult(
                check_name="structured_output_valid",
                passed=synthesis.analysis is not None,
                expected="valid structured AnalysisOutput",
                actual=(
                    "validated" if synthesis.analysis is not None else synthesis.error or "absent"
                ),
                severity="must",
            ),
            CheckResult(
                check_name="fixture_completeness",
                passed=not collection.misses,
                expected="every requested tool input has an exact offline fixture",
                actual="complete" if not collection.misses else f"missing {misses}",
                severity="must",
            ),
            CheckResult(
                check_name="fixture_evidence_parity",
                passed=not collection.evidence_issues,
                expected="requested mandatory evidence is substantive and concept-bearing",
                actual="complete" if not collection.evidence_issues else issues,
                severity="must",
            ),
        ]
    )
    passed = all(check.passed for check in checks if check.severity == "must")
    n_passed = sum(check.passed for check in checks)
    return EvalGrade(
        ticker=grade.ticker,
        passed=passed,
        checks=checks,
        overall_notes=f"{n_passed}/{len(checks)} checks passed",
    )


def run_staged_eval(
    *,
    examples: Sequence[EvalExample],
    repetitions: int,
    collector_provider: Provider,
    collector_config: StageProviderConfig,
    synthesizer_provider: Provider,
    synthesizer_config: StageProviderConfig,
    run_id: str,
    logger: RunLogger,
    fixtures_root: Path = FIXTURES_DIR,
    judge: ThesisJudge | None = None,
    max_cost_usd: float = _DEFAULT_MAX_COST_USD,
) -> list[StagedTickerResult]:
    """Run comparable staged generations, applying gold checks only after synthesis."""
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    budget = Budget(max_cost_usd=max_cost_usd)
    results: list[StagedTickerResult] = []
    for repetition in range(1, repetitions + 1):
        for example in examples:
            context = RunContext(run_id=run_id, budget=budget, logger=logger)
            results.append(
                run_staged_ticker(
                    ticker=example.ticker,
                    repetition=repetition,
                    collector_provider=collector_provider,
                    collector_config=collector_config,
                    synthesizer_provider=synthesizer_provider,
                    synthesizer_config=synthesizer_config,
                    run_context=context,
                    fixtures_root=fixtures_root,
                    persona=resolve_persona(example),
                    example=example,
                    judge=judge,
                )
            )
    return results


def write_private_results(path: Path, results: Sequence[StagedTickerResult]) -> Path:
    """Persist complete experimental output in an owner-only JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        json.dump([result.to_dict() for result in results], file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    return path


def _stage_config(config: EvalConfig) -> StageProviderConfig:
    return StageProviderConfig(
        provider=config.provider,
        model=config.model,
        service_tier=config.service_tier,
        reasoning_effort=config.reasoning_effort,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the controlled staged eval experiment")
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--agreement-judge",
        choices=("none", "openai"),
        default="openai",
        help="Use Sonnet plus an independently blinded OpenAI judge by default",
    )
    for stage, default_provider in (("collector", "openai"), ("synth", "anthropic")):
        parser.add_argument(
            f"--{stage}-provider",
            choices=("anthropic", "openai", "gemini"),
            default=default_provider,
        )
        parser.add_argument(f"--{stage}-model", default=None)
        parser.add_argument(
            f"--{stage}-service-tier",
            choices=("auto", "default", "flex"),
            default="default",
        )
        parser.add_argument(
            f"--{stage}-reasoning-effort",
            choices=("none", "minimal", "low", "medium", "high"),
            default="medium" if stage == "collector" else "none",
        )
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")

    try:
        collector_eval = resolve_eval_config(
            provider=cast(ProviderName, args.collector_provider),
            model=cast(str | None, args.collector_model),
            service_tier=cast(ServiceTier, args.collector_service_tier),
            reasoning_effort=cast(ReasoningEffort, args.collector_reasoning_effort),
        )
        synth_eval = resolve_eval_config(
            provider=cast(ProviderName, args.synth_provider),
            model=cast(str | None, args.synth_model),
            service_tier=cast(ServiceTier, args.synth_service_tier),
            reasoning_effort=cast(ReasoningEffort, args.synth_reasoning_effort),
        )
        collector = create_provider(collector_eval, os.environ)
        synthesizer = create_provider(synth_eval, os.environ)
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not anthropic_key:
            raise ValueError("Missing required API key(s): ANTHROPIC_API_KEY")
        if args.agreement_judge == "openai" and not os.environ.get("OPENAI_API_KEY", "").strip():
            raise ValueError("Missing required API key(s): OPENAI_API_KEY")
    except ValueError as exc:
        parser.error(str(exc))

    collector_config = _stage_config(collector_eval)
    synth_config = _stage_config(synth_eval)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    run_id = f"eval-staged-{stamp}"
    logger = RunLogger(run_id, args.output_dir / "logs")
    judge_connection = sqlite3.connect(os.environ.get("WARREN_DB", "warren.db"))
    judge_cache = CacheStore(judge_connection)
    sonnet_judge = SonnetThesisJudge(
        anthropic.Anthropic(api_key=anthropic_key),
        judge_cache,
    )
    judge: ThesisJudge = sonnet_judge
    if args.agreement_judge == "openai":
        openai_judge = OpenAIThesisJudge(
            OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
            judge_cache,
        )
        judge = JudgePanel((("sonnet", sonnet_judge), ("openai", openai_judge)))
    try:
        try:
            examples = select_examples(args.tickers, load_all_examples())
        except ValueError as exc:
            parser.error(str(exc))
        results = run_staged_eval(
            examples=examples,
            repetitions=args.repetitions,
            collector_provider=collector,
            collector_config=collector_config,
            synthesizer_provider=synthesizer,
            synthesizer_config=synth_config,
            run_id=run_id,
            logger=logger,
            judge=judge,
        )
    finally:
        logger.close()
        judge_connection.close()

    output_path = write_private_results(args.output_dir / f"{run_id}.json", results)
    print(f"Staged eval output written to {output_path}")
    return 1 if any(item.grade is None or not item.grade.passed for item in results) else 0


if __name__ == "__main__":
    sys.exit(main())
