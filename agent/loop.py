import json
import re
from typing import Any, Literal, cast

import anthropic
from pydantic import BaseModel, Field, ValidationError

from agent.budget import RunContext
from agent.tools import TOOL_DEFINITIONS, TOOL_REGISTRY
from agent.tools.base import ToolResultOk

_MAX_ITERATIONS = 8
_MAX_TOOL_REPEATS = 3
_FORCE_FINAL_MAX_TOKENS = 2048


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


def _last_text(content: list[Any]) -> str:
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
        tools=cast(Any, TOOL_DEFINITIONS),
        messages=messages,
        max_tokens=max_tokens,
    )


def analyze_ticker(
    ticker: str,
    persona: Any,
    routing_policy: Any,
    run_context: RunContext,
) -> AnalysisOutput:
    client = anthropic.Anthropic()
    messages: list[anthropic.types.MessageParam] = [
        cast(anthropic.types.MessageParam, {"role": "user", "content": _initial_prompt(ticker)})
    ]
    iteration = 0
    schema_repair_attempt = False

    while True:
        iteration += 1

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
            response = _call_claude(
                client, model, persona.system_prompt, messages, _FORCE_FINAL_MAX_TOKENS
            )
            _record_usage(run_context, response)
            return _parse_output(_last_text(response.content))

        # ── Normal Claude call ────────────────────────────────────────────────
        model = routing_policy.select(iteration, messages, ticker)
        response = _call_claude(client, model, persona.system_prompt, messages, 4096)
        _record_usage(run_context, response)

        # ── end_turn → try to parse output ───────────────────────────────────
        if response.stop_reason == "end_turn":
            text = _last_text(response.content)
            try:
                return _parse_output(text)
            except (ValidationError, ValueError, json.JSONDecodeError):
                if schema_repair_attempt:
                    raise SchemaRepairError(
                        "Schema repair failed: Claude produced invalid JSON twice"
                    )
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
            tool_results: list[dict[str, Any]] = []
            force_tool_loop = False

            for block in response.content:
                if not isinstance(block, anthropic.types.ToolUseBlock):
                    continue

                call_count = run_context.record_tool_call(block.name, dict(block.input))
                if call_count >= _MAX_TOOL_REPEATS:
                    force_tool_loop = True

                tool = TOOL_REGISTRY.get(block.name)
                if tool is None:
                    result_content = f"ERROR: unknown tool '{block.name}'"
                else:
                    result = tool.run(dict(block.input), run_context)
                    if isinstance(result, ToolResultOk):
                        result_content = result.content
                    else:
                        result_content = f"ERROR: {result.error}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_content,
                })

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
                response = _call_claude(
                    client, model, persona.system_prompt, messages, _FORCE_FINAL_MAX_TOKENS
                )
                _record_usage(run_context, response)
                return _parse_output(_last_text(response.content))

            continue

        raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason!r}")


def _record_usage(run_context: RunContext, response: anthropic.types.Message) -> None:
    usage = response.usage
    run_context.budget.record_usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )
