"""Unit tests for agent.caching — prompt-cache breakpoint layout and TTL ordering."""

from typing import cast
from unittest.mock import MagicMock

import anthropic
import pytest

from agent.caching import (
    _HAIKU_MIN_CACHE_TOKENS,
    _mark_last_user_turn,
    build_claude_request,
    extract_cache_breakpoints,
    validate_haiku_caching_threshold,
)
from agent.persona import SYSTEM_PROMPT
from agent.tools import TOOL_DEFINITIONS

_PORTFOLIO_CTX = "Holdings: AAPL 10 shares, MSFT 5 shares"
_MESSAGES: list[anthropic.types.MessageParam] = [
    cast(anthropic.types.MessageParam, {"role": "user", "content": "Analyze AAPL"}),
]


def _build_request(portfolio_context: str = _PORTFOLIO_CTX) -> dict[str, object]:
    return build_claude_request(
        model="claude-sonnet-4-6",
        persona_prompt=SYSTEM_PROMPT,
        tool_defs=TOOL_DEFINITIONS,
        portfolio_context=portfolio_context,
        messages=_MESSAGES,
        max_tokens=4096,
    )


# ── TTL ordering ──────────────────────────────────────────────────────────────


def test_cache_breakpoint_ttl_ordering() -> None:
    req = _build_request()
    bps = extract_cache_breakpoints(req)
    positions = [bp["position"] for bp in bps]
    assert positions == [
        "tools",
        "system_persona",
        "system_portfolio",
        "messages_last_user",
    ], f"Expected TTL-ordered breakpoints, got: {positions}"


def test_persona_precedes_portfolio_in_system_blocks() -> None:
    req = _build_request()
    system = cast(list[dict[str, object]], req["system"])
    # BP2 must be the persona prompt; BP3 must be the portfolio context
    assert system[0]["text"] == SYSTEM_PROMPT, "BP2 (system[0]) must be the persona prompt"
    assert system[1]["text"] == _PORTFOLIO_CTX, "BP3 (system[1]) must be the portfolio context"


# ── Breakpoint count ──────────────────────────────────────────────────────────


def test_four_breakpoints_with_portfolio_context() -> None:
    req = _build_request()
    bps = extract_cache_breakpoints(req)
    assert len(bps) == 4


def test_three_breakpoints_without_portfolio_context() -> None:
    req = _build_request(portfolio_context="")
    bps = extract_cache_breakpoints(req)
    positions = [bp["position"] for bp in bps]
    assert positions == ["tools", "system_persona", "messages_last_user"]


# ── BP1: tools ────────────────────────────────────────────────────────────────


def test_cache_control_only_on_last_tool() -> None:
    req = _build_request()
    tools = cast(list[dict[str, object]], req["tools"])
    # Only the last tool should carry cache_control
    for tool in tools[:-1]:
        assert "cache_control" not in tool
    assert "cache_control" in tools[-1]


# ── BP4: _mark_last_user_turn ─────────────────────────────────────────────────


def test_mark_last_user_turn_string_content() -> None:
    messages: list[anthropic.types.MessageParam] = [
        cast(anthropic.types.MessageParam, {"role": "user", "content": "hello"}),
    ]
    result = _mark_last_user_turn(messages)
    last_msg = cast(dict[str, object], result[-1])
    content = cast(list[dict[str, object]], last_msg["content"])
    assert isinstance(content, list)
    assert content[-1].get("cache_control") == {"type": "ephemeral"}
    assert content[-1].get("text") == "hello"


def test_mark_last_user_turn_list_content() -> None:
    tool_result: dict[str, object] = {
        "type": "tool_result",
        "tool_use_id": "tu_01",
        "content": "price: $150",
        "is_error": False,
    }
    messages: list[anthropic.types.MessageParam] = [
        cast(anthropic.types.MessageParam, {"role": "user", "content": "start"}),
        cast(anthropic.types.MessageParam, {"role": "assistant", "content": "thinking"}),
        cast(anthropic.types.MessageParam, {"role": "user", "content": [tool_result]}),
    ]
    result = _mark_last_user_turn(messages)
    last_msg = cast(dict[str, object], result[-1])
    content = cast(list[dict[str, object]], last_msg["content"])
    assert content[-1].get("cache_control") == {"type": "ephemeral"}


def test_mark_last_user_turn_does_not_mutate_input() -> None:
    messages: list[anthropic.types.MessageParam] = [
        cast(anthropic.types.MessageParam, {"role": "user", "content": "hello"}),
    ]
    original_content = cast(dict[str, object], messages[0])["content"]
    _mark_last_user_turn(messages)
    assert cast(dict[str, object], messages[0])["content"] == original_content


def test_mark_last_user_turn_skips_assistant_messages() -> None:
    messages: list[anthropic.types.MessageParam] = [
        cast(anthropic.types.MessageParam, {"role": "user", "content": "first"}),
        cast(anthropic.types.MessageParam, {"role": "assistant", "content": "response"}),
    ]
    result = _mark_last_user_turn(messages)
    # Only the user message should be modified, not the trailing assistant message
    last_assistant = cast(dict[str, object], result[-1])
    assert "cache_control" not in last_assistant
    last_user = cast(dict[str, object], result[0])
    content = cast(list[dict[str, object]], last_user["content"])
    assert content[-1].get("cache_control") == {"type": "ephemeral"}


# ── validate_haiku_caching_threshold ─────────────────────────────────────────


def test_validate_haiku_threshold_passes() -> None:
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.count_tokens.return_value = MagicMock(
        input_tokens=_HAIKU_MIN_CACHE_TOKENS + 1000
    )
    validate_haiku_caching_threshold(mock_client)  # should not raise


def test_validate_haiku_threshold_fails_below_minimum() -> None:
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.count_tokens.return_value = MagicMock(
        input_tokens=_HAIKU_MIN_CACHE_TOKENS - 1
    )
    with pytest.raises(ValueError, match="below Haiku"):
        validate_haiku_caching_threshold(mock_client)


def test_validate_haiku_threshold_uses_defaults_when_args_omitted() -> None:
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.count_tokens.return_value = MagicMock(input_tokens=9999)
    validate_haiku_caching_threshold(mock_client)
    call_kwargs = mock_client.messages.count_tokens.call_args.kwargs
    assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
    # System and tools should be populated from module defaults
    assert len(call_kwargs["system"]) > 0
    assert len(call_kwargs["tools"]) > 0
