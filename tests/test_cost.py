import pytest

from agent.models import HAIKU_4_5, OPUS_4_7, SONNET_4_6
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
