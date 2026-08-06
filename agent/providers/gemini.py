"""Gemini Interactions adapter with exact stateless thought-signature replay."""

from __future__ import annotations

from typing import cast

from google import genai
from google.genai import interactions
from google.genai._gaos.types.interactions.interaction import (
    Interaction as GeminiInteraction,
)

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
    combined_system_prompt,
    sanitize_json_schema,
)
from agent.tools.base import ToolDefinition


class GeminiProvider:
    name = "gemini"

    def __init__(self, client: genai.Client | None = None) -> None:
        self._client = client or genai.Client()

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
        del temperature  # Gemini 3.6 deprecates sampling controls.
        generation_config: interactions.GenerationConfigParam = {"max_output_tokens": max_tokens}
        if reasoning_effort != "none":
            generation_config["thinking_level"] = reasoning_effort

        response = cast(
            GeminiInteraction,
            self._client.interactions.create(
                model=model,
                input=_messages(messages),
                store=False,
                service_tier="standard" if service_tier in {"auto", "default"} else "flex",
                system_instruction=combined_system_prompt(system_prompt, portfolio_context),
                tools=[_tool(tool) for tool in tools],
                generation_config=generation_config,
            ),
        )

        blocks: list[TextBlock | ReasoningBlock | ToolCallBlock] = []
        replay: list[JSONObject] = []
        for step in response.steps or []:
            # This exact serialized step includes ThoughtStep.signature. Never
            # reconstruct, reorder, or discard it before the next interaction.
            replay.append(cast(JSONObject, step.model_dump(mode="json", exclude_none=True)))
            if isinstance(step, interactions.ModelOutputStep):
                for content in step.content or []:
                    if isinstance(content, interactions.TextContent):
                        blocks.append(TextBlock(content.text))
            elif isinstance(step, interactions.ThoughtStep):
                for summary in step.summary or []:
                    if isinstance(summary, interactions.TextContent):
                        blocks.append(ReasoningBlock(summary.text))
            elif isinstance(step, interactions.FunctionCallStep):
                blocks.append(
                    ToolCallBlock(
                        id=step.id,
                        name=step.name,
                        arguments=cast(JSONObject, step.arguments),
                    )
                )

        usage = response.usage
        if usage is None:
            normalized_usage = Usage()
        else:
            cache_read = usage.total_cached_tokens or 0
            normalized_usage = Usage(
                input_tokens=max(0, (usage.total_input_tokens or 0) - cache_read),
                output_tokens=usage.total_output_tokens or 0,
                cache_read_tokens=cache_read,
                reasoning_tokens=usage.total_thought_tokens or 0,
                tool_use_tokens=usage.total_tool_use_tokens or 0,
                total_tokens=usage.total_tokens or 0,
                raw=cast(JSONObject, usage.model_dump(mode="json", exclude_none=True)),
            )
        return ProviderResponse(
            blocks=tuple(blocks),
            stop_reason=_stop_reason(str(response.status), blocks),
            usage=normalized_usage,
            model_id=str(response.model or model),
            replay=tuple(replay),
        )

    def tool_result_turn(self, results: list[ToolResultBlock]) -> list[Message]:
        return [Message(role="user", blocks=tuple(results))]


def _tool(tool: ToolDefinition) -> interactions.ToolParam:
    return cast(
        interactions.ToolParam,
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": sanitize_json_schema(tool.parameters),
        },
    )


def _messages(messages: list[Message]) -> interactions.InteractionsInputParam:
    history: list[object] = []
    for message in messages:
        if message.role == "assistant" and message.replay:
            history.extend(message.replay)
            continue

        text_parts: list[dict[str, str]] = []
        for block in message.blocks:
            if isinstance(block, (TextBlock, ReasoningBlock)):
                text_parts.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolCallBlock):
                history.append(
                    {
                        "type": "function_call",
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.arguments,
                    }
                )
            elif isinstance(block, ToolResultBlock):
                history.append(
                    {
                        "type": "function_result",
                        "name": block.name,
                        "call_id": block.call_id,
                        "result": [{"type": "text", "text": block.content}],
                        "is_error": block.is_error,
                    }
                )
        if text_parts:
            step_type = "user_input" if message.role == "user" else "model_output"
            history.append({"type": step_type, "content": text_parts})
    return cast(interactions.InteractionsInputParam, history)


def _stop_reason(
    status: str,
    blocks: list[TextBlock | ReasoningBlock | ToolCallBlock],
) -> str:
    if any(isinstance(block, ToolCallBlock) for block in blocks):
        return "tool_use"
    return status
