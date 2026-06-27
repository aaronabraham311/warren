# Single source of truth for model identifiers and per-token pricing (USD).
# All other modules must import from here rather than hardcoding strings.

from typing import Final, TypedDict

# ── Model identifiers ────────────────────────────────────────────────────────
# Marked Final so mypy infers literal types (e.g. Literal["claude-haiku-4-5-20251001"]),
# letting agent/routing.py return them where the ModelID Literal is expected.
HAIKU_4_5: Final = "claude-haiku-4-5-20251001"
SONNET_4_6: Final = "claude-sonnet-4-6"
OPUS_4_7: Final = "claude-opus-4-7"

DEFAULT_MODEL_ID: Final = SONNET_4_6


class ModelPricing(TypedDict):
    """Per-token (USD) rates for one model. Cache write uses the 5-minute TTL tier."""

    input: float
    output: float
    cache_read: float
    cache_write_5m: float


def _per_mtok(rate: float) -> float:
    return rate / 1_000_000


# Per-token pricing keyed by model id. Source: Tech Spec §8 pricing tiers.
#   Haiku 4.5: $1/$5 in/out, cache read $0.10, cache write 5m $1.25
#   Sonnet 4.6: $3/$15 in/out, cache read $0.30, cache write 5m = 1.25× input
#   Opus 4.7:  $5/$25 in/out, cache read $0.50, cache write 5m = 1.25× input
PRICING: dict[str, ModelPricing] = {
    HAIKU_4_5: {
        "input": _per_mtok(1.0),
        "output": _per_mtok(5.0),
        "cache_read": _per_mtok(0.10),
        "cache_write_5m": _per_mtok(1.25),
    },
    SONNET_4_6: {
        "input": _per_mtok(3.0),
        "output": _per_mtok(15.0),
        "cache_read": _per_mtok(0.30),
        "cache_write_5m": _per_mtok(3.0) * 1.25,
    },
    OPUS_4_7: {
        "input": _per_mtok(5.0),
        "output": _per_mtok(25.0),
        "cache_read": _per_mtok(0.50),
        "cache_write_5m": _per_mtok(5.0) * 1.25,
    },
}

# Legacy flat constants for the default model (Sonnet 4.6). Derived from PRICING
# so there is a single source of truth; agent/budget.py uses these for in-run
# cost ceilings. Per-call/per-run cost is computed per-model via storage/cost.py.
PRICE_INPUT_PER_TOKEN = PRICING[DEFAULT_MODEL_ID]["input"]
PRICE_OUTPUT_PER_TOKEN = PRICING[DEFAULT_MODEL_ID]["output"]
PRICE_CACHE_READ_PER_TOKEN = PRICING[DEFAULT_MODEL_ID]["cache_read"]
PRICE_CACHE_CREATION_PER_TOKEN = PRICING[DEFAULT_MODEL_ID]["cache_write_5m"]
