# Single source of truth for model identifiers, per-token pricing (USD), and
# the AnalysisOutput schema shared between the loop and storage layers.

from typing import Final, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

# ── Model identifiers ────────────────────────────────────────────────────────
# Marked Final so mypy infers literal types (e.g. Literal["claude-haiku-4-5-20251001"]),
# letting agent/routing.py return them where the ModelID Literal is expected.
HAIKU_4_5: Final = "claude-haiku-4-5-20251001"
SONNET_4_6: Final = "claude-sonnet-4-6"
SONNET_5: Final = "claude-sonnet-5"
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
#   Sonnet 5:  $3/$15 in/out, cache read $0.30, cache write 5m = 1.25× input
#              (standard sticker; the intro $2/$10 rate lapses 2026-08-31)
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
    SONNET_5: {
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


# ── Analysis output schema ───────────────────────────────────────────────────


class LynchBuffettSignals(BaseModel):
    pros: list[str]
    cons: list[str]


class DirtSignals(BaseModel):
    ev_ebit: float | None = None
    price_to_ncav: float | None = None
    ncav_discount_pct: float | None = None
    net_cash_positive: bool | None = None
    consecutive_profit_years: int | None = None
    buyback_active: bool | None = None
    insider_sentiment: Literal["positive", "negative", "neutral"] | None = None
    analyst_coverage_count: int | None = None
    aggregator_discrepancies_found: bool = False


TerminationReason = Literal[
    "success",
    "schema_repair_success",
    "schema_repair_failed",
    "iteration_capped",
    "token_capped",
    "tool_loop_broken",
]


class AnalysisOutput(BaseModel):
    model_config = ConfigDict(frozen=False)

    ticker: str = Field(pattern=r"^[A-Z]{1,5}([.-][A-Z])?$")
    analysis_type: Literal["holding", "discovery"]
    recommendation: Literal["buy", "sell", "hold"]
    confidence: float = Field(ge=0.0, le=1.0)
    thesis: str = Field(min_length=10)
    lynch_signals: LynchBuffettSignals
    buffett_signals: LynchBuffettSignals
    key_risks: list[str] = Field(min_length=1)
    data_quality_notes: list[str] = Field(default_factory=list)
    tool_calls_made: int = Field(ge=0, default=0)
    tokens_used: int = Field(ge=0, default=0)
    termination_reason: TerminationReason = "success"
    dirt_signals: DirtSignals | None = None
