from typing import cast
from unittest.mock import MagicMock

from google import genai
from google.genai import interactions
from google.genai._gaos.types.interactions.interaction import Interaction

from agent.providers.base import Message, ToolCallBlock, ToolResultBlock
from agent.providers.gemini import GeminiProvider
from agent.tools.base import ToolDefinition


def test_gemini_replays_signed_steps_and_partitions_cached_input() -> None:
    client = MagicMock()
    interaction = Interaction(
        status="completed",
        model="gemini-3.6-flash",
        steps=[
            interactions.ThoughtStep(type="thought", signature="signed-thought"),
            interactions.FunctionCallStep(
                type="function_call",
                id="call_1",
                name="quote",
                arguments={"ticker": "AAPL"},
            ),
        ],
        usage=interactions.Usage(
            total_input_tokens=1000,
            total_cached_tokens=700,
            total_output_tokens=80,
            total_thought_tokens=30,
            total_tool_use_tokens=20,
            total_tokens=1110,
        ),
    )
    client.interactions.create.return_value = interaction
    provider = GeminiProvider(cast(genai.Client, client))
    tool = ToolDefinition("quote", "Quote", {"type": "object", "properties": {}})

    response = provider.complete(
        model="gemini-3.6-flash",
        system_prompt="system",
        portfolio_context="portfolio",
        messages=[Message.text("user", "analyze")],
        tools=[tool],
        max_tokens=1000,
        temperature=0,
        reasoning_effort="high",
        service_tier="flex",
    )

    assert response.usage.input_tokens == 300
    assert response.usage.cache_read_tokens == 700
    assert response.usage.output_tokens == 110
    assert response.usage.reasoning_tokens == 30
    assert response.usage.tool_use_tokens == 20
    assert response.replay[0]["signature"] == "signed-thought"
    assert isinstance(response.blocks[0], ToolCallBlock)
    kwargs = client.interactions.create.call_args.kwargs
    assert kwargs["service_tier"] == "flex"
    assert "temperature" not in kwargs["generation_config"]

    result = ToolResultBlock("call_1", "quote", '{"price": 10}')
    next_messages = [
        Message.text("user", "analyze"),
        response.assistant_message(),
        *provider.tool_result_turn([result]),
    ]
    provider.complete(
        model="gemini-3.6-flash",
        system_prompt="system",
        portfolio_context="portfolio",
        messages=next_messages,
        tools=[tool],
        max_tokens=1000,
        service_tier="flex",
    )
    history = client.interactions.create.call_args.kwargs["input"]
    assert history[1]["signature"] == "signed-thought"
    assert history[-1] == {
        "type": "function_result",
        "name": "quote",
        "call_id": "call_1",
        "result": [{"type": "text", "text": '{"price": 10}'}],
        "is_error": False,
    }
