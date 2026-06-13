"""Per-model LLM cost computation.

Pricing lives in ``agent/models.py`` (the single source of truth per CLAUDE.md);
this module consumes it. Cache-read and cache-creation tokens are billed at
distinct rates and must be passed separately — they are *not* folded into
``input_tokens``.
"""

from agent.models import PRICING


def compute_cost(
    model: str,
    input_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
    output_tokens: int,
) -> float:
    """Return the USD cost of one LLM call.

    ``input_tokens`` is the count of *non-cached* input tokens. Tokens served
    from cache (``cache_read_tokens``) and written to cache
    (``cache_creation_tokens``, billed at the 5-minute TTL tier) are priced
    separately.

    Raises ``ValueError`` for an unknown model rather than silently pricing at $0.
    """
    try:
        rates = PRICING[model]
    except KeyError:
        raise ValueError(
            f"No pricing for model {model!r}; add it to agent.models.PRICING"
        ) from None

    return (
        input_tokens * rates["input"]
        + cache_read_tokens * rates["cache_read"]
        + cache_creation_tokens * rates["cache_write_5m"]
        + output_tokens * rates["output"]
    )
