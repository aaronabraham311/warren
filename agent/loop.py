import json
import re
import sys
import time
from collections.abc import Callable
from typing import Literal, Protocol, cast

import anthropic
from pydantic import BaseModel, Field, ValidationError

from agent.budget import RunContext
from agent.tools import TOOL_DEFINITIONS, TOOL_REGISTRY
from agent.tools.base import Tool, ToolResult, ToolResultError, ToolResultOk


class _Persona(Protocol):
    @property
    def system_prompt(self) -> str: ...


class _RoutingPolicy(Protocol):
    def select(
        self, iteration: int, messages: list[anthropic.types.MessageParam], ticker: str
    ) -> str: ...


_MAX_ITERATIONS = 8
_MAX_TOOL_REPEATS = 3
_FORCE_FINAL_MAX_TOKENS = 2048

# Max *retries* (not total attempts) per transient error code.
_RETRY_POLICY: dict[str, int] = {"rate_limit": 3, "network": 2}


def _run_with_retry(
    tool: Tool,
    parsed: BaseModel,
    run_context: RunContext,
    _sleep: Callable[[float], None],
) -> tuple[ToolResult, int, str | None]:
    """Run tool.run(), retrying transient errors with exponential backoff.

    Returns (result, retry_count, last_retry_error) so callers can annotate the WAL.
    """
    retries = 0
    last_error_code: str | None = None
    while True:
        result = tool.run(parsed, run_context)
        if isinstance(result, ToolResultOk):
            return result, retries, last_error_code
        if not result.retryable:
            if result.error_code == "unknown":
                print(f"[warren] unknown tool error: {result.message}", file=sys.stderr)
            return result, retries, last_error_code
        max_retries = _RETRY_POLICY.get(result.error_code, 0)
        if retries >= max_retries:
            return result, retries, last_error_code
        last_error_code = result.error_code
        _sleep(2.0**retries)
        retries += 1


class AnalysisOutput(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z]{1,5}([.-][A-Z])?$")
    analysis_type: Literal["holding", "discovery"]
    recommendation: Literal["buy", "sell", "hold"]
    confidence: float = Field(ge=0.0, le=1.0)
    thesis: str
    lynch_signals: list[str]
    buffett_signals: list[str]
    key_risks: list[str]
    data_quality_notes: list[str] = Field(default_factory=list)


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
    system: str,
    messages: list[anthropic.types.MessageParam],
    max_tokens: int,
) -> anthropic.types.Message:
    return client.messages.create(
        model=model,
        system=system,
        tools=cast(list[anthropic.types.ToolParam], TOOL_DEFINITIONS),
        messages=messages,
        max_tokens=max_tokens,
    )


def _call_and_record(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    messages: list[anthropic.types.MessageParam],
    max_tokens: int,
    run_context: RunContext,
    ticker: str,
) -> anthropic.types.Message:
    """Time the call, record token/cost usage, and emit an llm_call WAL event."""
    t0 = time.monotonic()
    response = _call_claude(client, model, system, messages, max_tokens)
    latency_ms = int((time.monotonic() - t0) * 1000)
    _record_usage(run_context, response)
    run_context.logger.log_llm_call(
        response, ticker=ticker, phase="deep", model=model, latency_ms=latency_ms
    )
    return response


def analyze_ticker(
    ticker: str,
    persona: _Persona,
    routing_policy: _RoutingPolicy,
    run_context: RunContext,
    client: anthropic.Anthropic | None = None,
    _sleep: Callable[[float], None] = time.sleep,
) -> AnalysisOutput:
    if client is None:
        client = anthropic.Anthropic()
    messages: list[anthropic.types.MessageParam] = [
        cast(anthropic.types.MessageParam, {"role": "user", "content": _initial_prompt(ticker)})
    ]
    iteration = 0
    schema_repair_attempt = False

    while True:
        iteration += 1
        run_context.iterations = iteration

        # ── Cost ceiling (hard abort — no force-final, run is over) ──────────
        if run_context.budget.cost_exceeded():
            raise CostAbortedError(f"Cost ceiling exceeded after {iteration - 1} iterations")

        # ── Iteration / token caps → force a final answer ────────────────────
        force_label: str | None = None
        if iteration > _MAX_ITERATIONS:
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
                messages,
                _FORCE_FINAL_MAX_TOKENS,
                run_context,
                ticker,
            )
            try:
                return _parse_output(_last_text(response.content))
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                raise SchemaRepairError("Forced-final response was not valid JSON") from exc

        # ── Normal Claude call ────────────────────────────────────────────────
        model = routing_policy.select(iteration, messages, ticker)
        response = _call_and_record(
            client, model, persona.system_prompt, messages, 4096, run_context, ticker
        )

        # ── end_turn → try to parse output ───────────────────────────────────
        if response.stop_reason == "end_turn":
            text = _last_text(response.content)
            try:
                return _parse_output(text)
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
                            "content": (
                                "Your response was not valid JSON matching the required schema. "
                                "Please output ONLY a valid JSON object — no markdown fences, "
                                "no explanation — matching the schema in your instructions exactly."
                            ),
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

                call_count = run_context.record_tool_call(block.name, dict(block.input))
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
                        parsed = tool.input_schema.model_validate(dict(block.input))
                    except ValidationError as exc:
                        result: ToolResult = ToolResultError(
                            error_code="not_found",
                            message=f"invalid input for {block.name}: {exc}",
                            retryable=False,
                        )
                    else:
                        result, retry_count, last_retry_error = _run_with_retry(
                            tool, parsed, run_context, _sleep
                        )
                    # Serialize the result so the agent sees structured errors as data.
                    if isinstance(result, ToolResultOk):
                        result_content = result.data.model_dump_json()
                        cached = result.cached
                    else:
                        result_content = json.dumps(
                            {
                                "error_code": result.error_code,
                                "message": result.message,
                                "retryable": result.retryable,
                            }
                        )
                        error_msg = result.message
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
                    messages,
                    _FORCE_FINAL_MAX_TOKENS,
                    run_context,
                    ticker,
                )
                try:
                    return _parse_output(_last_text(response.content))
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
