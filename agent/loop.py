import json
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import anthropic
from pydantic import BaseModel, ValidationError

from agent.budget import RunContext
from agent.models import AnalysisOutput
from agent.providers.anthropic import AnthropicProvider
from agent.providers.base import (
    Message,
    Provider,
    ProviderResponse,
    ReasoningEffort,
    ServiceTier,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from agent.tools import PROVIDER_TOOL_DEFINITIONS, TOOL_REGISTRY
from agent.tools.base import Tool, ToolResult, ToolResultError, ToolResultOk


class _Persona(Protocol):
    @property
    def system_prompt(self) -> str: ...

    @property
    def requires_dirt_decision(self) -> bool: ...


class _RoutingPolicy(Protocol):
    def select(self, iteration: int, messages: list[Message], ticker: str) -> str: ...


class ToolRunner(Protocol):
    """How the loop executes a tool. The one seam the eval harness swaps.

    The default runs the tool for real. ``eval.tool_fixtures.FixtureToolRunner`` reads a
    recorded result from disk instead, so replay never constructs a data-source client.
    """

    def run(self, tool: Tool, tool_input: BaseModel, ctx: RunContext) -> ToolResult: ...


class LiveToolRunner:
    """Production runner — dispatches straight to the tool."""

    def run(self, tool: Tool, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        return tool.run(tool_input, ctx)


_MAX_ITERATIONS = 8
_MAX_DIRT_ITERATIONS = 12
_MAX_TOOL_REPEATS = 3
_ANALYSIS_MAX_TOKENS = 8192

# Max *retries* (not total attempts) per transient error code.
_RETRY_POLICY: dict[str, int] = {"rate_limit": 3, "network": 2}


@dataclass
class _RetryOutcome:
    result: ToolResult
    retry_count: int
    last_retry_error: str | None


def _run_with_retry(
    tool: Tool,
    parsed: BaseModel,
    run_context: RunContext,
    _sleep: Callable[[float], None],
    tool_runner: ToolRunner,
) -> _RetryOutcome:
    """Run the tool, retrying transient errors with exponential backoff."""
    retries = 0
    last_error_code: str | None = None
    while True:
        result = tool_runner.run(tool, parsed, run_context)
        if isinstance(result, ToolResultOk):
            return _RetryOutcome(result, retries, last_error_code)
        if not result.retryable:
            if result.error_code == "unknown":
                print(f"[warren] unknown tool error: {result.message}", file=sys.stderr)
            return _RetryOutcome(result, retries, last_error_code)
        max_retries = _RETRY_POLICY.get(result.error_code, 0)
        if retries >= max_retries:
            return _RetryOutcome(result, retries, last_error_code)
        last_error_code = result.error_code
        _sleep(2.0**retries)
        retries += 1


class CostAbortedError(Exception):
    pass


class SchemaRepairError(Exception):
    pass


class ProviderTerminalError(Exception):
    """A provider stopped without a final answer or executable tool call."""

    def __init__(self, provider: str, model: str, stop_reason: str) -> None:
        self.provider = provider
        self.model = model
        self.stop_reason = stop_reason
        super().__init__(f"{provider} model {model!r} terminated with stop_reason {stop_reason!r}")


def _initial_prompt(ticker: str) -> str:
    return (
        f"Analyze {ticker} and give me a buy, sell, or hold recommendation. "
        f"Use the get_quote tool to fetch current price data, then produce your analysis."
    )


def _extract_json(text: str) -> str:
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Find the outermost JSON object
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _parse_output(text: str) -> AnalysisOutput:
    return AnalysisOutput.model_validate_json(_extract_json(text))


def _schema_repair_prompt(
    persona: _Persona,
    expected_dirt_decision: BaseModel | None,
) -> str:
    if getattr(persona, "requires_dirt_decision", False):
        if expected_dirt_decision is None:
            return (
                "Before final synthesis, call model_dirt_scenarios successfully. Then output "
                "only JSON matching the required schema, with dirt_decision copied exactly "
                "from that tool result."
            )
        return (
            "Output only JSON matching the required schema. dirt_decision must copy the most "
            "recent model_dirt_scenarios result exactly; recommendation must be buy only for "
            "outcome=buy and hold for watchlist/pass."
        )
    return (
        "Output only a valid JSON object matching the required schema exactly, without "
        "markdown fences or explanation."
    )


def _parse_persona_output(
    text: str,
    persona: _Persona,
    expected_dirt_decision: BaseModel | None,
) -> AnalysisOutput:
    """Parse an analysis while keeping the served DIRT contract authoritative.

    The scenario tool computes and validates the decision contract locally. Asking the
    model to reproduce that large object verbatim in its final response adds a fragile
    formatting step and can also change calculated values. Once a contract has been
    served, project it (and its recommendation mapping) into the final analysis before
    Pydantic validation. The narrative and DIRT signals still come from the model.
    """
    if not getattr(persona, "requires_dirt_decision", False) or expected_dirt_decision is None:
        return _parse_output(text)

    payload = json.loads(_extract_json(text))
    if not isinstance(payload, dict):
        raise ValueError("analysis output must be a JSON object")
    payload["dirt_decision"] = expected_dirt_decision.model_dump(mode="json")
    payload["recommendation"] = (
        "buy" if getattr(expected_dirt_decision, "outcome", None) == "buy" else "hold"
    )
    return AnalysisOutput.model_validate(payload)


def _validate_persona_output(
    result: AnalysisOutput,
    persona: _Persona,
    expected_dirt_decision: BaseModel | None,
) -> AnalysisOutput:
    """Enforce persona-only fields and the deterministic DIRT outcome mapping."""
    requires_decision = getattr(persona, "requires_dirt_decision", False)
    decision = getattr(result, "dirt_decision", None)
    if not requires_decision:
        if decision is not None:
            raise ValueError("dirt_decision must be null for the default persona")
        return result
    if result.dirt_signals is None:
        raise ValueError("dirt_signals is required for the DIRT persona")
    if decision is None:
        raise ValueError("dirt_decision is required for the DIRT persona")
    if expected_dirt_decision is None:
        raise ValueError("DIRT persona must call model_dirt_scenarios before synthesis")
    if decision.model_dump(mode="json") != expected_dirt_decision.model_dump(mode="json"):
        raise ValueError("dirt_decision must exactly match model_dirt_scenarios output")
    expected_recommendation = "buy" if decision.outcome == "buy" else "hold"
    if result.recommendation != expected_recommendation:
        raise ValueError("DIRT recommendation must map buy to buy and watchlist/pass to hold")
    return result


def _iteration_limit(persona: _Persona) -> int:
    """Give evidence-heavy DIRT analysis room for its required local decision call."""
    return (
        _MAX_DIRT_ITERATIONS
        if getattr(persona, "requires_dirt_decision", False)
        else _MAX_ITERATIONS
    )


def _last_text(response: ProviderResponse) -> str:
    for block in reversed(response.blocks):
        if isinstance(block, TextBlock):
            return block.text
    return ""


def _force_final_message(label: str) -> Message:
    return Message.text(
        "user",
        f"[{label}] You have reached a stopping condition. "
        "Produce your best analysis now as a JSON object matching the schema "
        "in your instructions. Output ONLY the JSON — no markdown, no explanation.",
    )


def _call_provider(
    provider: Provider,
    model: str,
    persona_prompt: str,
    portfolio_context: str,
    messages: list[Message],
    max_tokens: int,
    temperature: float | None = None,
    reasoning_effort: ReasoningEffort = "none",
    service_tier: ServiceTier = "auto",
) -> ProviderResponse:
    return provider.complete(
        model=model,
        system_prompt=persona_prompt,
        portfolio_context=portfolio_context,
        messages=messages,
        tools=PROVIDER_TOOL_DEFINITIONS,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        service_tier=service_tier,
    )


def _call_and_record(
    provider: Provider,
    model: str,
    persona_prompt: str,
    portfolio_context: str,
    messages: list[Message],
    max_tokens: int,
    run_context: RunContext,
    ticker: str,
    temperature: float | None = None,
    reasoning_effort: ReasoningEffort = "none",
    service_tier: ServiceTier = "auto",
    response_observer: Callable[[ProviderResponse], None] | None = None,
) -> ProviderResponse:
    """Time the call, record token/cost usage, and emit an llm_call WAL event."""
    t0 = time.monotonic()
    response = _call_provider(
        provider,
        model,
        persona_prompt,
        portfolio_context,
        messages,
        max_tokens,
        temperature,
        reasoning_effort,
        service_tier,
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    _record_usage(run_context, response, model, provider.name, service_tier)
    run_context.logger.log_llm_call(
        response,
        ticker=ticker,
        phase="deep",
        model=model,
        latency_ms=latency_ms,
        provider=provider.name,
        service_tier=service_tier,
        reasoning_effort=reasoning_effort,
    )
    if response_observer is not None:
        response_observer(response)
    return response


def analyze_ticker(
    ticker: str,
    persona: _Persona,
    routing_policy: _RoutingPolicy,
    run_context: RunContext,
    client: anthropic.Anthropic | None = None,
    portfolio_context: str = "",
    _sleep: Callable[[float], None] = time.sleep,
    temperature: float | None = None,
    tool_runner: ToolRunner | None = None,
    provider: Provider | None = None,
    reasoning_effort: ReasoningEffort = "none",
    service_tier: ServiceTier = "auto",
    response_observer: Callable[[ProviderResponse], None] | None = None,
) -> AnalysisOutput:
    if provider is not None and client is not None:
        raise ValueError("pass provider or client, not both")
    if provider is None:
        provider = AnthropicProvider(client)
    if tool_runner is None:
        tool_runner = LiveToolRunner()
    messages = [Message.text("user", _initial_prompt(ticker))]
    iteration = 0
    schema_repair_attempt = False
    expected_dirt_decision: BaseModel | None = None
    max_iterations = _iteration_limit(persona)

    while True:
        iteration += 1
        run_context.iterations = iteration

        # ── Cost ceiling (hard abort — no force-final, run is over) ──────────
        if run_context.budget.cost_exceeded():
            raise CostAbortedError(f"Cost ceiling exceeded after {iteration - 1} iterations")

        # ── Iteration / token caps → force a final answer ────────────────────
        force_label: str | None = None
        if iteration > max_iterations:
            force_label = "iteration_capped"
        elif run_context.budget.token_exceeded():
            force_label = "token_capped"

        if force_label:
            messages.append(_force_final_message(force_label))
            model = routing_policy.select(iteration, messages, ticker)
            response = _call_and_record(
                provider,
                model,
                persona.system_prompt,
                portfolio_context,
                messages,
                _ANALYSIS_MAX_TOKENS,
                run_context,
                ticker,
                temperature,
                reasoning_effort,
                service_tier,
                response_observer,
            )
            try:
                result = _validate_persona_output(
                    _parse_persona_output(_last_text(response), persona, expected_dirt_decision),
                    persona,
                    expected_dirt_decision,
                )
                result.termination_reason = force_label  # type: ignore[assignment]
                return result
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                raise SchemaRepairError("Forced-final response was not valid JSON") from exc

        # ── Normal provider call ──────────────────────────────────────────────
        model = routing_policy.select(iteration, messages, ticker)
        response = _call_and_record(
            provider,
            model,
            persona.system_prompt,
            portfolio_context,
            messages,
            # A structured analysis (4–6 thesis bullets + Lynch/Buffett signals + risks +
            # data-quality notes) can exceed 4096 output tokens for verbose names (e.g. a
            # conglomerate). 4096 truncated the final JSON → stop_reason="max_tokens" → crash.
            _ANALYSIS_MAX_TOKENS,
            run_context,
            ticker,
            temperature,
            reasoning_effort,
            service_tier,
            response_observer,
        )

        # ── end_turn → try to parse output ───────────────────────────────────
        if response.stop_reason in {"end_turn", "completed"}:
            text = _last_text(response)
            try:
                result = _validate_persona_output(
                    _parse_persona_output(text, persona, expected_dirt_decision),
                    persona,
                    expected_dirt_decision,
                )
                if schema_repair_attempt:
                    result.termination_reason = "schema_repair_success"
                return result
            except (ValidationError, ValueError, json.JSONDecodeError):
                if schema_repair_attempt:
                    raise SchemaRepairError(
                        "Schema repair failed: Claude produced invalid JSON twice"
                    ) from None
                schema_repair_attempt = True
                messages.append(response.assistant_message())
                messages.append(
                    Message.text(
                        "user",
                        _schema_repair_prompt(persona, expected_dirt_decision),
                    )
                )
                continue

        # ── tool_use → dispatch tools ─────────────────────────────────────────
        if response.stop_reason == "tool_use":
            tool_results: list[ToolResultBlock] = []
            force_tool_loop = False

            for block in response.blocks:
                if not isinstance(block, ToolCallBlock):
                    continue

                call_count = run_context.record_tool_call(block.name, dict(block.arguments))
                if call_count >= _MAX_TOOL_REPEATS:
                    force_tool_loop = True

                tool = TOOL_REGISTRY.get(block.name)
                t0 = time.monotonic()
                error_msg: str | None = None
                cached = False
                retry_count = 0
                last_retry_error: str | None = None
                if tool is None:
                    result_content = json.dumps(
                        {
                            "error_code": "not_found",
                            "message": f"unknown tool '{block.name}'",
                            "retryable": False,
                        }
                    )
                    error_msg = f"unknown tool '{block.name}'"
                else:
                    try:
                        parsed = tool.input_schema.model_validate(dict(block.arguments))
                    except ValidationError as exc:
                        tool_result: ToolResult = ToolResultError(
                            error_code="not_found",
                            message=f"invalid input for {block.name}: {exc}",
                            retryable=False,
                        )
                    else:
                        outcome = _run_with_retry(tool, parsed, run_context, _sleep, tool_runner)
                        tool_result = outcome.result
                        retry_count = outcome.retry_count
                        last_retry_error = outcome.last_retry_error
                    # Serialize the result so the agent sees structured errors as data.
                    if isinstance(tool_result, ToolResultOk):
                        result_content = tool_result.data.model_dump_json()
                        cached = tool_result.cached
                        if block.name == "model_dirt_scenarios":
                            expected_dirt_decision = tool_result.data
                    else:
                        result_content = json.dumps(
                            {
                                "error_code": tool_result.error_code,
                                "message": tool_result.message,
                                "retryable": tool_result.retryable,
                            }
                        )
                        error_msg = tool_result.message
                latency_ms = int((time.monotonic() - t0) * 1000)

                run_context.logger.log_tool_call(
                    tool_name=block.name,
                    tool_input=dict(block.arguments),
                    output=result_content,
                    cached=cached,
                    latency_ms=latency_ms,
                    status="ok" if error_msg is None else "error",
                    ticker=ticker,
                    error_msg=error_msg,
                    retry_count=retry_count,
                    last_retry_error=last_retry_error,
                )

                tool_results.append(
                    ToolResultBlock(
                        call_id=block.id,
                        name=block.name,
                        content=result_content,
                        is_error=error_msg is not None,
                    )
                )

            messages.append(response.assistant_message())
            messages.extend(provider.tool_result_turn(tool_results))

            if force_tool_loop:
                messages.append(_force_final_message("tool_loop_broken"))
                model = routing_policy.select(iteration, messages, ticker)
                response = _call_and_record(
                    provider,
                    model,
                    persona.system_prompt,
                    portfolio_context,
                    messages,
                    _ANALYSIS_MAX_TOKENS,
                    run_context,
                    ticker,
                    temperature,
                    reasoning_effort,
                    service_tier,
                    response_observer,
                )
                try:
                    result = _validate_persona_output(
                        _parse_persona_output(
                            _last_text(response), persona, expected_dirt_decision
                        ),
                        persona,
                        expected_dirt_decision,
                    )
                    result.termination_reason = "tool_loop_broken"
                    return result
                except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                    raise SchemaRepairError("Forced-final response was not valid JSON") from exc

            continue

        raise ProviderTerminalError(provider.name, response.model_id, response.stop_reason)


def _record_usage(
    run_context: RunContext,
    response: ProviderResponse,
    requested_model: str,
    provider: str,
    service_tier: str,
) -> None:
    run_context.budget.record_provider_usage(
        response.usage,
        model=requested_model,
        provider=provider,
        service_tier=service_tier,
    )
