from typing import cast
from unittest.mock import MagicMock

from openai import OpenAI
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_reasoning_item import Summary
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)

from agent.providers.base import Message, ToolCallBlock
from agent.providers.openai import OpenAIProvider
from agent.tools.base import ToolDefinition


def _client() -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.output = [
        ResponseReasoningItem(
            id="rs_1",
            type="reasoning",
            encrypted_content="opaque",
            summary=[Summary(type="summary_text", text="summary")],
        ),
        ResponseFunctionToolCall(
            type="function_call",
            call_id="call_1",
            name="quote",
            arguments='{"ticker":"AAPL"}',
        ),
        ResponseOutputMessage(
            id="msg_1",
            type="message",
            role="assistant",
            status="completed",
            content=[
                ResponseOutputText(type="output_text", text="checking", annotations=[], logprobs=[])
            ],
        ),
    ]
    response.usage = ResponseUsage(
        input_tokens=1000,
        input_tokens_details=InputTokensDetails(cached_tokens=700, cache_write_tokens=100),
        output_tokens=80,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=30),
        total_tokens=1080,
    )
    response.status = "completed"
    response.incomplete_details = None
    response.model = "gpt-5.6-terra"
    client.responses.create.return_value = response
    return client


def test_openai_replays_encrypted_reasoning_and_partitions_cached_input() -> None:
    client = _client()
    provider = OpenAIProvider(cast(OpenAI, client))
    tool = ToolDefinition("quote", "Quote", {"type": "object", "properties": {}})

    response = provider.complete(
        model="gpt-5.6-terra",
        system_prompt="system",
        portfolio_context="portfolio",
        messages=[Message.text("user", "analyze")],
        tools=[tool],
        max_tokens=1000,
        temperature=0,
        reasoning_effort="none",
        service_tier="flex",
    )

    assert response.usage.input_tokens == 200
    assert response.usage.cache_read_tokens == 700
    assert response.usage.cache_write_tokens == 100
    assert response.usage.reasoning_tokens == 30
    assert response.replay[0]["encrypted_content"] == "opaque"
    assert isinstance(response.blocks[1], ToolCallBlock)
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["reasoning"] == {"effort": "none"}
    assert kwargs["temperature"] == 0
    assert kwargs["service_tier"] == "flex"
    assert kwargs["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert kwargs["instructions"] == "system"
    assert kwargs["input"][0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert kwargs["input"][1]["content"].startswith("Portfolio context:")
    assert kwargs["prompt_cache_key"].startswith("warren-")
    first_cache_key = kwargs["prompt_cache_key"]

    provider.complete(
        model="gpt-5.6-terra",
        system_prompt="system",
        portfolio_context="a changed per-run portfolio",
        messages=[response.assistant_message()],
        tools=[tool],
        max_tokens=1000,
        reasoning_effort="high",
        service_tier="flex",
    )
    replay_input = client.responses.create.call_args.kwargs["input"]
    assert any(item.get("encrypted_content") == "opaque" for item in replay_input)
    assert client.responses.create.call_args.kwargs["prompt_cache_key"] == first_cache_key
