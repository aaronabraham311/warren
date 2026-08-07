import json
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import anthropic
from pydantic import BaseModel, ValidationError

from agent.budget import RunContext
from agent.caching import call_claude_with_caching
from agent.events import LlmCallPurpose
from agent.models import AnalysisOutput
from agent.tools import TOOL_DEFINITIONS, TOOL_REGISTRY
from agent.tools.base import Tool, ToolResult, ToolResultError, ToolResultOk


class _Persona(Protocol):
    @property
    def system_prompt(self) -> str: ...

    @property
    def requires_dirt_decision(self) -> bool: ...


class _RoutingPolicy(Protocol):
    def select(
        self, iteration: int, messages: list[anthropic.types.MessageParam], ticker: str
    ) -> str: ...


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
        run_context.cancellation.raise_if_cancelled()
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


def _last_text(content: list[anthropic.types.ContentBlock]) -> str:
    for block in reversed(content):
        if isinstance(block, anthropic.types.TextBlock):
            return block.text
    return ""


def _force_final_message(label: str) -> anthropic.types.MessageParam:
    return cast(
        anthropic.types.MessageParam,
        {
            "role": "user",
            "content": (
                f"[{label}] You have reached a stopping condition. "
                "Produce your best analysis now as a JSON object matching the schema "
                "in your instructions. Output ONLY the JSON — no markdown, no explanation."
            ),
        },
    )


def _call_claude(
    client: anthropic.Anthropic,
    model: str,
    persona_prompt: str,
    portfolio_context: str,
    messages: list[anthropic.types.MessageParam],
    max_tokens: int,
    temperature: float | None = None,
) -> anthropic.types.Message:
    return call_claude_with_caching(
        client,
        model=model,
        persona_prompt=persona_prompt,
        tool_defs=TOOL_DEFINITIONS,
        portfolio_context=portfolio_context,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _call_and_record(
    client: anthropic.Anthropic,
    model: str,
    persona_prompt: str,
    portfolio_context: str,
    messages: list[anthropic.types.MessageParam],
    max_tokens: int,
    run_context: RunContext,
    ticker: str,
    purpose: LlmCallPurpose,
    temperature: float | None = None,
    response_observer: Callable[[anthropic.types.Message], None] | None = None,
) -> anthropic.types.Message:
    """Durably announce model I/O, then record its completion and usage."""
    run_context.cancellation.raise_if_cancelled()
    run_context.logger.log(
        "llm_call_started",
        ticker=ticker,
        phase="deep",
        model=model,
        purpose=purpose,
        iteration=run_context.iterations,
        tool_count=run_context.budget.total_tool_calls,
    )
    t0 = time.monotonic()
    try:
        response = _call_claude(
            client, model, persona_prompt, portfolio_context, messages, max_tokens, temperature
        )
    except Exception as exc:
        run_context.logger.log(
            "llm_call_failed",
            ticker=ticker,
            phase="deep",
            model=model,
            purpose=purpose,
            iteration=run_context.iterations,
            error_type=type(exc).__name__,
        )
        raise
    latency_ms = int((time.monotonic() - t0) * 1000)
    _record_usage(run_context, response)
    run_context.logger.log_llm_call(
        response,
        ticker=ticker,
        phase="deep",
        model=model,
        latency_ms=latency_ms,
        purpose=purpose,
        iteration=run_context.iterations,
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
    response_observer: Callable[[anthropic.types.Message], None] | None = None,
) -> AnalysisOutput:
    if client is None:
        client = anthropic.Anthropic()
    if tool_runner is None:
        tool_runner = LiveToolRunner()
    messages: list[anthropic.types.MessageParam] = [
        cast(anthropic.types.MessageParam, {"role": "user", "content": _initial_prompt(ticker)})
    ]
    iteration = 0
    schema_repair_attempt = False
    expected_dirt_decision: BaseModel | None = None
    max_iterations = _iteration_limit(persona)

    while True:
        iteration += 1
        run_context.iterations = iteration
        run_context.cancellation.raise_if_cancelled()

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
                client,
                model,
                persona.system_prompt,
                portfolio_context,
                messages,
                _ANALYSIS_MAX_TOKENS,
                run_context,
                ticker,
                "finalizing",
                temperature,
                response_observer,
            )
            try:
                result = _validate_persona_output(
                    _parse_persona_output(
                        _last_text(response.content), persona, expected_dirt_decision
                    ),
                    persona,
                    expected_dirt_decision,
                )
                result.termination_reason = force_label  # type: ignore[assignment]
                return result
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                raise SchemaRepairError("Forced-final response was not valid JSON") from exc

        # ── Normal Claude call ────────────────────────────────────────────────
        model = routing_policy.select(iteration, messages, ticker)
        purpose: LlmCallPurpose
        if schema_repair_attempt:
            purpose = "validation"
        elif iteration == 1 and run_context.budget.total_tool_calls == 0:
            purpose = "planning"
        else:
            purpose = "synthesis"
        response = _call_and_record(
            client,
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
            purpose,
            temperature,
            response_observer,
        )

        # ── end_turn → try to parse output ───────────────────────────────────
        if response.stop_reason == "end_turn":
            text = _last_text(response.content)
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
                messages.append(
                    cast(
                        anthropic.types.MessageParam,
                        {"role": "assistant", "content": response.content},
                    )
                )
                messages.append(
                    cast(
                        anthropic.types.MessageParam,
                        {
                            "role": "user",
                            "content": _schema_repair_prompt(persona, expected_dirt_decision),
                        },
                    )
                )
                continue

        # ── tool_use → dispatch tools ─────────────────────────────────────────
        if response.stop_reason == "tool_use":
            tool_results: list[anthropic.types.ToolResultBlockParam] = []
            force_tool_loop = False

            for block in response.content:
                if not isinstance(block, anthropic.types.ToolUseBlock):
                    continue

                run_context.cancellation.raise_if_cancelled()
                call_count = run_context.record_tool_call(block.name, dict(block.input))
                if call_count >= _MAX_TOOL_REPEATS:
                    force_tool_loop = True

                tool = TOOL_REGISTRY.get(block.name)
                t0 = time.monotonic()
                run_context.logger.log_tool_started(ticker=ticker, tool_name=block.name)
                run_context.cancellation.raise_if_cancelled()
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
                        parsed = tool.input_schema.model_validate(dict(block.input))
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
                    tool_input=dict(block.input),
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
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_content,
                        "is_error": error_msg is not None,
                    }
                )

            messages.append(
                cast(
                    anthropic.types.MessageParam,
                    {"role": "assistant", "content": response.content},
                )
            )
            messages.append(
                cast(
                    anthropic.types.MessageParam,
                    {"role": "user", "content": tool_results},
                )
            )

            if force_tool_loop:
                messages.append(_force_final_message("tool_loop_broken"))
                model = routing_policy.select(iteration, messages, ticker)
                response = _call_and_record(
                    client,
                    model,
                    persona.system_prompt,
                    portfolio_context,
                    messages,
                    _ANALYSIS_MAX_TOKENS,
                    run_context,
                    ticker,
                    "finalizing",
                    temperature,
                    response_observer,
                )
                try:
                    result = _validate_persona_output(
                        _parse_persona_output(
                            _last_text(response.content), persona, expected_dirt_decision
                        ),
                        persona,
                        expected_dirt_decision,
                    )
                    result.termination_reason = "tool_loop_broken"
                    return result
                except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                    raise SchemaRepairError("Forced-final response was not valid JSON") from exc

            continue

        raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason!r}")


def _record_usage(run_context: RunContext, response: anthropic.types.Message) -> None:
    usage = response.usage
    run_context.budget.record_usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
        cache_creation_tokens=usage.cache_creation_input_tokens or 0,
    )
