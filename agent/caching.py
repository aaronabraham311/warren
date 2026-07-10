"""Prompt caching — 4 breakpoints in correct TTL order.

Anthropic silently breaks caching (no error, just no hit) if shorter-TTL breakpoints
precede longer-TTL ones.  Breakpoint layout (must stay in this order):

  BP1 — last tool definition    (1h, stable across runs)
  BP2 — persona system prompt   (1h, stable across runs)
  BP3 — portfolio context       (5m, same within one run; omitted if empty)
  BP4 — last user message turn  (5m, accumulates per-ticker turn)

All four use {"type": "ephemeral"}; the "TTL" names describe ordering intent, not a
separate API parameter.  The Anthropic maximum is 4 breakpoints per request.
"""

from typing import cast

import anthropic

from agent.models import HAIKU_4_5
from agent.persona import SYSTEM_PROMPT
from agent.tools import TOOL_DEFINITIONS

_HAIKU_MIN_CACHE_TOKENS = 4096
_CACHE_CONTROL: anthropic.types.CacheControlEphemeralParam = {"type": "ephemeral"}


def build_claude_request(
    *,
    model: str,
    persona_prompt: str,
    tool_defs: list[dict[str, object]],
    portfolio_context: str,
    messages: list[anthropic.types.MessageParam],
    max_tokens: int,
    temperature: float | None = None,
) -> dict[str, object]:
    """Return the full API request dict with cache breakpoints in correct TTL order.

    ``temperature`` is omitted from the request entirely when None, leaving the SDK
    default in place. The eval harness passes 0.0 for reproducible grading.
    """
    # BP1: cache_control on the last tool definition
    tools: list[dict[str, object]] = [
        {**t, "cache_control": _CACHE_CONTROL} if i == len(tool_defs) - 1 else dict(t)
        for i, t in enumerate(tool_defs)
    ]

    # BP2 + optional BP3: system as a list of structured text blocks
    system: list[dict[str, object]] = [
        {"type": "text", "text": persona_prompt, "cache_control": _CACHE_CONTROL},  # BP2
    ]
    if portfolio_context:
        system.append(
            {"type": "text", "text": portfolio_context, "cache_control": _CACHE_CONTROL}  # BP3
        )

    # BP4: cache_control on last content block of the last user message
    marked_messages = _mark_last_user_turn(messages)

    req: dict[str, object] = {
        "model": model,
        "max_tokens": max_tokens,
        "tools": tools,
        "system": system,
        "messages": marked_messages,
    }
    if temperature is not None:
        req["temperature"] = temperature
    return req


def call_claude_with_caching(
    client: anthropic.Anthropic,
    *,
    model: str,
    persona_prompt: str,
    tool_defs: list[dict[str, object]],
    portfolio_context: str,
    messages: list[anthropic.types.MessageParam],
    max_tokens: int,
    temperature: float | None = None,
) -> anthropic.types.Message:
    """Make an Anthropic API call with 4-breakpoint prompt caching."""
    req = build_claude_request(
        model=model,
        persona_prompt=persona_prompt,
        tool_defs=tool_defs,
        portfolio_context=portfolio_context,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    # anthropic.omit is the SDK's "parameter absent" sentinel — passing it leaves the
    # server-side default in place, exactly as if temperature were never named.
    return client.messages.create(
        model=cast(str, req["model"]),
        max_tokens=cast(int, req["max_tokens"]),
        tools=cast(list[anthropic.types.ToolParam], req["tools"]),
        system=cast(list[anthropic.types.TextBlockParam], req["system"]),
        messages=cast(list[anthropic.types.MessageParam], req["messages"]),
        temperature=temperature if temperature is not None else anthropic.omit,
    )


def extract_cache_breakpoints(request: dict[str, object]) -> list[dict[str, str]]:
    """Return [{position}] for every cache breakpoint in a request dict.

    Used in tests to assert TTL ordering without live API calls.
    Positions: "tools", "system_persona", "system_portfolio", "messages_last_user".
    """
    breakpoints: list[dict[str, str]] = []

    for tool in cast(list[dict[str, object]], request.get("tools") or []):
        if "cache_control" in tool:
            breakpoints.append({"position": "tools"})

    system = cast(list[dict[str, object]], request.get("system") or [])
    for i, block in enumerate(system):
        if "cache_control" in block:
            breakpoints.append({"position": "system_persona" if i == 0 else "system_portfolio"})

    for msg in reversed(cast(list[dict[str, object]], request.get("messages") or [])):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for block in cast(list[dict[str, object]], content):
                if "cache_control" in block:
                    breakpoints.append({"position": "messages_last_user"})
        break

    return breakpoints


def validate_haiku_caching_threshold(
    client: anthropic.Anthropic,
    persona_prompt: str | None = None,
    tool_defs: list[dict[str, object]] | None = None,
) -> None:
    """Raise ValueError if BP1+BP2 token count is below Haiku's 4096-token minimum.

    Run once at startup or in CI to catch prompt regressions before they silently
    disable caching on the cheapest model.
    """
    if persona_prompt is None:
        persona_prompt = SYSTEM_PROMPT
    if tool_defs is None:
        tool_defs = TOOL_DEFINITIONS

    result = client.messages.count_tokens(
        model=HAIKU_4_5,
        system=[cast(anthropic.types.TextBlockParam, {"type": "text", "text": persona_prompt})],
        tools=cast(list[anthropic.types.ToolParam], tool_defs),
        messages=[cast(anthropic.types.MessageParam, {"role": "user", "content": "x"})],
    )
    total = result.input_tokens
    if total < _HAIKU_MIN_CACHE_TOKENS:
        raise ValueError(
            f"Tools + persona = {total} tokens — below Haiku's {_HAIKU_MIN_CACHE_TOKENS}-token "
            "minimum. Expand the screening rubric in the persona prompt."
        )


def _mark_last_user_turn(
    messages: list[anthropic.types.MessageParam],
) -> list[anthropic.types.MessageParam]:
    """Return a copy of messages with cache_control on the last user message's last content block.

    Does not mutate the input list.  Handles both string content (converts to a text
    block) and list content (marks the last block).
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = cast(dict[str, object], messages[i])
        if msg.get("role") != "user":
            continue

        content = msg.get("content")
        result: list[anthropic.types.MessageParam] = list(messages)
        new_msg = dict(msg)

        if isinstance(content, str):
            new_msg["content"] = [
                {"type": "text", "text": content, "cache_control": _CACHE_CONTROL}
            ]
        elif isinstance(content, list) and content:
            new_content = [dict(b) if isinstance(b, dict) else b for b in content]
            last = dict(cast(dict[str, object], new_content[-1]))
            last["cache_control"] = _CACHE_CONTROL
            new_content[-1] = last
            new_msg["content"] = new_content

        result[i] = cast(anthropic.types.MessageParam, new_msg)
        return result

    return list(messages)
