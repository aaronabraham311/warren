from typing import cast
from unittest.mock import MagicMock

import anthropic

from agent.providers.anthropic import AnthropicProvider
from agent.providers.base import Message, TextBlock, ToolCallBlock
from agent.tools.base import ToolDefinition


def test_anthropic_normalizes_usage_and_sets_explicit_cache_breakpoints() -> None:
    client = MagicMock()
    client.messages.create.return_value = anthropic.types.Message(
        id="msg_1",
        content=[
            anthropic.types.TextBlock(type="text", text="checking"),
            anthropic.types.ToolUseBlock(
                type="tool_use", id="call_1", name="quote", input={"ticker": "AAPL"}
            ),
        ],
        model="claude-sonnet-4-6",
        role="assistant",
        stop_reason="tool_use",
        stop_sequence=None,
        type="message",
        usage=anthropic.types.Usage(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=400,
            cache_creation_input_tokens=50,
        ),
    )
    provider = AnthropicProvider(cast(anthropic.Anthropic, client))

    response = provider.complete(
        model="claude-sonnet-4-6",
        system_prompt="system",
        portfolio_context="portfolio",
        messages=[Message.text("user", "analyze")],
        tools=[ToolDefinition("quote", "Quote", {"type": "object"})],
        max_tokens=1000,
    )

    assert response.usage.input_tokens == 100
    assert response.usage.cache_read_tokens == 400
    assert response.usage.cache_write_tokens == 50
    assert response.usage.reasoning_tokens is None
    assert response.usage.total_tokens == 570
    assert isinstance(response.blocks[0], TextBlock)
    assert isinstance(response.blocks[1], ToolCallBlock)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["system"][1]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
