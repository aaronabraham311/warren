"""OpenAI Responses adapter with stateless encrypted-reasoning replay."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import cast

import openai
from openai import OpenAI
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseInputParam,
    ResponseOutputMessage,
    ResponseReasoningItem,
    ToolParam,
)

from agent.providers.base import (
    JSONObject,
    JSONValue,
    Message,
    ProviderResponse,
    ReasoningBlock,
    ReasoningEffort,
    ServiceTier,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    Usage,
    sanitize_json_schema,
    strip_null_values,
)
from agent.tools.base import ToolDefinition


class OpenAIProvider:
    name = "openai"

    def __init__(self, client: OpenAI | None = None) -> None:
        self._client = client or OpenAI()

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        portfolio_context: str,
        messages: list[Message],
        tools: list[ToolDefinition],
        max_tokens: int,
        temperature: float | None = None,
        reasoning_effort: ReasoningEffort = "none",
        service_tier: ServiceTier = "auto",
    ) -> ProviderResponse:
        api_tools = [_tool(tool) for tool in tools]
        response = self._client.responses.create(
            model=model,
            instructions=system_prompt,
            input=_messages(messages, portfolio_context),
            tools=api_tools,
            max_output_tokens=max_tokens,
            store=False,
            include=["reasoning.encrypted_content"],
            service_tier=service_tier,
            prompt_cache_key=_prompt_cache_key(model, system_prompt, api_tools),
            prompt_cache_options={"mode": "explicit", "ttl": "30m"},
            stream=False,
            reasoning={"effort": reasoning_effort},
            # Reasoning models reject sampling controls unless effort is `none`.
            temperature=(
                temperature
                if reasoning_effort == "none" and temperature is not None
                else openai.omit
            ),
        )

        blocks: list[TextBlock | ReasoningBlock | ToolCallBlock] = []
        replay: list[JSONObject] = []
        refused = False
        for item in response.output:
            replay.append(cast(JSONObject, item.model_dump(mode="json", exclude_none=True)))
            if isinstance(item, ResponseOutputMessage):
                for content in item.content:
                    if content.type == "output_text":
                        blocks.append(TextBlock(content.text))
                    elif content.type == "refusal":
                        refused = True
                        blocks.append(TextBlock(content.refusal))
            elif isinstance(item, ResponseReasoningItem):
                for summary in item.summary:
                    blocks.append(ReasoningBlock(summary.text))
            elif isinstance(item, ResponseFunctionToolCall):
                arguments = _parse_arguments(item.arguments)
                blocks.append(
                    ToolCallBlock(
                        id=item.call_id,
                        name=item.name,
                        arguments=arguments,
                    )
                )

        usage = response.usage
        if usage is None:
            normalized_usage = Usage()
        else:
            cache_read = usage.input_tokens_details.cached_tokens
            cache_write = usage.input_tokens_details.cache_write_tokens
            normalized_usage = Usage(
                input_tokens=max(0, usage.input_tokens - cache_read - cache_write),
                output_tokens=usage.output_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
                total_tokens=usage.total_tokens,
                raw=cast(JSONObject, usage.model_dump(mode="json", exclude_none=True)),
            )
        return ProviderResponse(
            blocks=tuple(blocks),
            stop_reason=_stop_reason(
                response.status,
                response.incomplete_details.reason
                if response.incomplete_details is not None
                else None,
                refused,
                blocks,
            ),
            usage=normalized_usage,
            model_id=str(response.model),
            replay=tuple(replay),
        )

    def tool_result_turn(self, results: list[ToolResultBlock]) -> list[Message]:
        return [Message(role="user", blocks=tuple(results))]


def _tool(tool: ToolDefinition) -> ToolParam:
    return cast(
        ToolParam,
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": sanitize_json_schema(tool.parameters, strict=True),
            "strict": True,
        },
    )


def _messages(messages: list[Message], portfolio_context: str = "") -> ResponseInputParam:
    # The breakpoint comes after the stable instructions/tools request prefix.
    # Everything after it (portfolio and ticker conversation) may vary per run.
    result: list[object] = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "[Warren stable prompt cache boundary]",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        }
    ]
    if portfolio_context:
        result.append({"role": "user", "content": f"Portfolio context:\n{portfolio_context}"})
    for message in messages:
        if message.role == "assistant" and message.replay:
            result.extend(message.replay)
            continue
        text_parts: list[str] = []
        for block in message.blocks:
            if isinstance(block, (TextBlock, ReasoningBlock)):
                text_parts.append(block.text)
            elif isinstance(block, ToolCallBlock):
                result.append(
                    {
                        "type": "function_call",
                        "call_id": block.id,
                        "name": block.name,
                        "arguments": json.dumps(block.arguments),
                    }
                )
            elif isinstance(block, ToolResultBlock):
                result.append(
                    {
                        "type": "function_call_output",
                        "call_id": block.call_id,
                        "output": block.content,
                    }
                )
        if text_parts:
            result.append({"role": message.role, "content": "\n".join(text_parts)})
    return cast(ResponseInputParam, result)


def _prompt_cache_key(model: str, instructions: str, tools: list[ToolParam]) -> str:
    prefix = json.dumps(
        {"model": model, "instructions": instructions, "tools": tools},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"warren-{sha256(prefix.encode()).hexdigest()[:48]}"


def _parse_arguments(arguments: str) -> JSONObject:
    try:
        parsed = _as_json_value(cast(object, json.loads(arguments)))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    without_nulls = strip_null_values(parsed)
    return without_nulls if isinstance(without_nulls, dict) else {}


def _as_json_value(value: object) -> JSONValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_as_json_value(item) for item in cast(list[object], value)]
    if isinstance(value, dict):
        return {
            str(key): _as_json_value(item)
            for key, item in cast(dict[object, object], value).items()
        }
    raise TypeError(f"non-JSON tool argument: {type(value).__name__}")


def _stop_reason(
    status: str | None,
    incomplete_reason: str | None,
    refused: bool,
    blocks: list[TextBlock | ReasoningBlock | ToolCallBlock],
) -> str:
    if any(isinstance(block, ToolCallBlock) for block in blocks):
        return "tool_use"
    if refused:
        return "refusal"
    if incomplete_reason is not None:
        return incomplete_reason
    return status or "completed"
