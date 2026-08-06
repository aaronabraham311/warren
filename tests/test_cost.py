import pytest

from agent.models import (
    GEMINI_3_6_FLASH,
    HAIKU_4_5,
    LUNA_5_6,
    OPUS_4_7,
    SONNET_4_6,
    SONNET_5,
    TERRA_5_6,
)
from storage.cost import compute_cost


def test_sonnet_cost_with_cache_breakdown() -> None:
    # Sonnet 4.6: $3/$15 in/out, cache read $0.30, cache write 5m = 1.25× input ($3.75)
    cost = compute_cost(
        SONNET_4_6,
        input_tokens=1000,
        cache_read_tokens=200,
        cache_creation_tokens=100,
        output_tokens=500,
    )
    expected = (1000 * 3.0 + 200 * 0.30 + 100 * 3.75 + 500 * 15.0) / 1_000_000
    assert cost == pytest.approx(expected, rel=1e-9)


def test_haiku_cost_tier() -> None:
    # Haiku 4.5: $1/$5 in/out, cache read $0.10, cache write 5m $1.25
    cost = compute_cost(
        HAIKU_4_5,
        input_tokens=1000,
        cache_read_tokens=1000,
        cache_creation_tokens=1000,
        output_tokens=1000,
    )
    expected = (1000 * 1.0 + 1000 * 0.10 + 1000 * 1.25 + 1000 * 5.0) / 1_000_000
    assert cost == pytest.approx(expected, rel=1e-9)


def test_opus_cost_tier() -> None:
    # Opus 4.7: $5/$25 in/out, cache read $0.50
    cost = compute_cost(
        OPUS_4_7,
        input_tokens=2000,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        output_tokens=400,
    )
    expected = (2000 * 5.0 + 400 * 25.0) / 1_000_000
    assert cost == pytest.approx(expected, rel=1e-9)


def test_sonnet_5_introductory_cost_tier() -> None:
    cost = compute_cost(SONNET_5, 1000, 1000, 1000, 1000)
    expected = (1000 * 2.0 + 1000 * 0.20 + 1000 * 2.50 + 1000 * 10.0) / 1_000_000
    assert cost == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    ("model", "service_tier", "rates"),
    [
        (LUNA_5_6, "default", (0.20, 0.02, 0.25, 1.20)),
        (LUNA_5_6, "flex", (0.10, 0.01, 0.125, 0.60)),
        (TERRA_5_6, "default", (2.0, 0.20, 2.50, 12.0)),
        (TERRA_5_6, "flex", (1.0, 0.10, 1.25, 6.0)),
    ],
)
def test_openai_cost_tiers(
    model: str, service_tier: str, rates: tuple[float, float, float, float]
) -> None:
    cost = compute_cost(
        model,
        input_tokens=1000,
        cache_read_tokens=2000,
        cache_creation_tokens=3000,
        output_tokens=4000,
        provider="openai",
        service_tier=service_tier,
    )
    expected = (1000 * rates[0] + 2000 * rates[1] + 3000 * rates[2] + 4000 * rates[3]) / 1_000_000
    assert cost == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    ("service_tier", "rates"),
    [
        ("default", (1.50, 0.15, 7.50)),
        ("flex", (0.75, 0.075, 3.75)),
    ],
)
def test_gemini_cost_tiers(service_tier: str, rates: tuple[float, float, float]) -> None:
    cost = compute_cost(
        GEMINI_3_6_FLASH,
        input_tokens=1000,
        cache_read_tokens=2000,
        cache_creation_tokens=0,
        output_tokens=3000,
        provider="gemini",
        service_tier=service_tier,
    )
    expected = (1000 * rates[0] + 2000 * rates[1] + 3000 * rates[2]) / 1_000_000
    assert cost == pytest.approx(expected, rel=1e-9)


def test_gemini_unverified_cache_write_fails_loudly() -> None:
    with pytest.raises(ValueError, match="No cache-write pricing"):
        compute_cost(
            GEMINI_3_6_FLASH,
            0,
            cache_read_tokens=0,
            cache_creation_tokens=1,
            output_tokens=0,
            provider="gemini",
        )


def test_reasoning_is_billed_once_as_part_of_output() -> None:
    # The cost API intentionally receives total billable output only. A provider adapter
    # records reasoning separately for observability, but must not add it to this count.
    cost = compute_cost(
        TERRA_5_6,
        input_tokens=0,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        output_tokens=500,
        provider="openai",
    )
    assert cost == pytest.approx(500 * 12.0 / 1_000_000)


def test_cache_read_and_creation_priced_separately() -> None:
    # cache_read is far cheaper than cache_creation — the two must not be conflated.
    read_only = compute_cost(
        SONNET_4_6, 0, cache_read_tokens=1000, cache_creation_tokens=0, output_tokens=0
    )
    creation_only = compute_cost(
        SONNET_4_6, 0, cache_read_tokens=0, cache_creation_tokens=1000, output_tokens=0
    )
    assert creation_only > read_only


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="No pricing for model"):
        compute_cost("claude-imaginary-9", 100, 0, 0, 50)


def test_provider_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="belongs to provider"):
        compute_cost(LUNA_5_6, 100, 0, 0, 50, provider="anthropic")


def test_unsupported_flex_tier_raises() -> None:
    with pytest.raises(ValueError, match="No flex pricing"):
        compute_cost(SONNET_4_6, 100, 0, 0, 50, service_tier="flex")
