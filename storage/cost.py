"""Per-model LLM cost computation.

Pricing lives in ``agent/models.py`` (the single source of truth per CLAUDE.md);
this module consumes it. Cache-read and cache-creation tokens are billed at
distinct rates and must be passed separately — they are *not* folded into
``input_tokens``.
"""

from agent.models import FLEX_PRICING, PRICING


def compute_cost(
    model: str,
    input_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
    output_tokens: int,
    *,
    provider: str = "anthropic",
    service_tier: str = "default",
) -> float:
    """Return the USD cost of one LLM call.

    ``input_tokens`` is the count of *non-cached* input tokens. Tokens served
    from cache (``cache_read_tokens``) and written to cache
    (``cache_creation_tokens``, billed at the 5-minute TTL tier) are priced
    separately.

    Raises ``ValueError`` for an unknown model rather than silently pricing at $0.
    """
    if min(input_tokens, cache_read_tokens, cache_creation_tokens, output_tokens) < 0:
        raise ValueError("Token counts must be non-negative")

    expected_provider = _provider_for_model(model)
    if provider != expected_provider:
        raise ValueError(
            f"Model {model!r} belongs to provider {expected_provider!r}, not {provider!r}"
        )

    resolved_tier = "default" if service_tier == "auto" else service_tier
    if resolved_tier == "default":
        table = PRICING
    elif resolved_tier == "flex":
        table = FLEX_PRICING
    else:
        raise ValueError(f"Unknown service tier {service_tier!r}")

    try:
        rates = table[model]
    except KeyError:
        if resolved_tier == "default":
            raise ValueError(
                f"No pricing for model {model!r}; add it to agent.models.PRICING"
            ) from None
        raise ValueError(f"No {resolved_tier} pricing for {provider} model {model!r}") from None

    cache_write_rate = rates["cache_write_5m"]
    if cache_creation_tokens and cache_write_rate is None:
        raise ValueError(f"No cache-write pricing for {provider} model {model!r}")

    return (
        input_tokens * rates["input"]
        + cache_read_tokens * rates["cache_read"]
        + cache_creation_tokens * (cache_write_rate or 0.0)
        + output_tokens * rates["output"]
    )


def _provider_for_model(model: str) -> str:
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gpt-"):
        return "openai"
    if model.startswith("gemini-"):
        return "gemini"
    raise ValueError(f"No pricing for model {model!r}; add it to agent.models.PRICING")
