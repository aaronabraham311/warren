# Single source of truth for model identifiers and per-token pricing (USD).
# All other modules must import from here rather than hardcoding strings.

DEFAULT_MODEL_ID = "claude-sonnet-4-6"

# USD per token for DEFAULT_MODEL_ID (Sonnet 4.6)
PRICE_INPUT_PER_TOKEN = 3.0 / 1_000_000
PRICE_OUTPUT_PER_TOKEN = 15.0 / 1_000_000
PRICE_CACHE_READ_PER_TOKEN = 0.30 / 1_000_000
# Cache creation billed at 1.25× the input rate (5-min TTL)
PRICE_CACHE_CREATION_PER_TOKEN = PRICE_INPUT_PER_TOKEN * 1.25
