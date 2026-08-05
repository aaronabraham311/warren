"""Anthropic Messages adapter."""

from __future__ import annotations

from typing import cast

import anthropic

from agent.providers.base import (
    JSONObject,
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
)
from agent.tools.base import ToolDefinition


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self._client = client or anthropic.Anthropic()

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
        del reasoning_effort  # Anthropic model configuration lives in routing/model policy.
        del service_tier  # Anthropic Messages has no synchronous Flex tier.
        api_tools = [_tool(tool) for tool in tools]
        if api_tools:
            api_tools[-1] = cast(
                anthropic.types.ToolParam,
                {
                    **cast(dict[str, object], api_tools[-1]),
                    "cache_control": {"type": "ephemeral"},
                },
            )

        system: list[anthropic.types.TextBlockParam] = [
            cast(
                anthropic.types.TextBlockParam,
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                },
            )
        ]
        if portfolio_context:
            system.append(
                cast(
                    anthropic.types.TextBlockParam,
                    {
                        "type": "text",
                        "text": portfolio_context,
                        "cache_control": {"type": "ephemeral"},
                    },
                )
            )

        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            tools=api_tools,
            system=system,
            messages=_messages(messages),
            temperature=temperature if temperature is not None else anthropic.omit,
        )

        blocks: list[TextBlock | ReasoningBlock | ToolCallBlock] = []
        replay: list[JSONObject] = []
        for block in response.content:
            replay.append(cast(JSONObject, block.model_dump(mode="json", exclude_none=True)))
            if isinstance(block, anthropic.types.TextBlock):
                blocks.append(TextBlock(block.text))
            elif isinstance(block, anthropic.types.ThinkingBlock):
                blocks.append(ReasoningBlock(block.thinking))
            elif isinstance(block, anthropic.types.ToolUseBlock):
                blocks.append(
                    ToolCallBlock(
                        id=block.id,
                        name=block.name,
                        arguments=cast(JSONObject, block.input),
                    )
                )

        usage_raw = cast(JSONObject, response.usage.model_dump(mode="json", exclude_none=True))
        cache_read = response.usage.cache_read_input_tokens or 0
        cache_write = response.usage.cache_creation_input_tokens or 0
        return ProviderResponse(
            blocks=tuple(blocks),
            stop_reason=response.stop_reason or "end_turn",
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                total_tokens=(
                    response.usage.input_tokens
                    + cache_read
                    + cache_write
                    + response.usage.output_tokens
                ),
                raw=usage_raw,
            ),
            model_id=response.model,
            replay=tuple(replay),
        )

    def tool_result_turn(self, results: list[ToolResultBlock]) -> list[Message]:
        return [Message(role="user", blocks=tuple(results))]


def _tool(tool: ToolDefinition) -> anthropic.types.ToolParam:
    return cast(
        anthropic.types.ToolParam,
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": sanitize_json_schema(tool.parameters),
        },
    )


def _messages(messages: list[Message]) -> list[anthropic.types.MessageParam]:
    result: list[anthropic.types.MessageParam] = []
    for message in messages:
        content: list[object] = []
        if message.role == "assistant" and message.replay:
            content.extend(message.replay)
        else:
            for block in message.blocks:
                if isinstance(block, TextBlock):
                    content.append({"type": "text", "text": block.text})
                elif isinstance(block, ReasoningBlock):
                    # A summary is safe as visible text; opaque signed thinking is replayed above.
                    content.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolCallBlock):
                    content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.arguments,
                        }
                    )
                elif isinstance(block, ToolResultBlock):
                    content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.call_id,
                            "content": block.content,
                            "is_error": block.is_error,
                        }
                    )
        result.append(
            cast(
                anthropic.types.MessageParam,
                {"role": message.role, "content": content},
            )
        )
    return _mark_last_user_turn(result)


def _mark_last_user_turn(
    messages: list[anthropic.types.MessageParam],
) -> list[anthropic.types.MessageParam]:
    marked = list(messages)
    for index in range(len(marked) - 1, -1, -1):
        message = cast(dict[str, object], marked[index])
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list) or not content:
            return marked
        new_content = list(content)
        last = new_content[-1]
        if isinstance(last, dict):
            new_content[-1] = {
                **cast(dict[str, object], last),
                "cache_control": {"type": "ephemeral"},
            }
            marked[index] = cast(
                anthropic.types.MessageParam,
                {**message, "content": new_content},
            )
        return marked
    return marked
