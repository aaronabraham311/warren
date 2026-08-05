import pytest

from agent.budget import Budget
from agent.models import TERRA_5_6
from agent.providers.base import Usage
from storage.cost import compute_cost


def test_budget_uses_provider_model_and_tier_pricing() -> None:
    budget = Budget()
    budget.record_usage(
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=200,
        cache_creation_tokens=100,
        reasoning_tokens=250,
        tool_use_tokens=100,
        model=TERRA_5_6,
        provider="openai",
        service_tier="flex",
    )

    assert budget.total_input_tokens == 1000
    assert budget.total_output_tokens == 500
    assert budget.total_cache_read_tokens == 200
    assert budget.total_cache_creation_tokens == 100
    assert budget.total_reasoning_tokens == 250
    assert budget.total_tool_use_tokens == 100
    assert budget.total_cost_usd == pytest.approx(
        compute_cost(TERRA_5_6, 1000, 200, 100, 500, provider="openai", service_tier="flex")
    )


def test_reasoning_and_tool_use_are_subsets_not_extra_billable_tokens() -> None:
    baseline = Budget()
    observed = Budget()
    baseline.record_usage(
        input_tokens=1000,
        output_tokens=500,
        model=TERRA_5_6,
        provider="openai",
    )
    observed.record_usage(
        input_tokens=1000,
        output_tokens=500,
        reasoning_tokens=400,
        tool_use_tokens=700,
        model=TERRA_5_6,
        provider="openai",
    )

    assert observed.total_cost_usd == baseline.total_cost_usd


def test_normalized_usage_records_through_the_same_accounting_path() -> None:
    budget = Budget()
    usage = Usage(
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=200,
        cache_write_tokens=100,
        reasoning_tokens=250,
        tool_use_tokens=100,
    )
    budget.record_provider_usage(usage, model=TERRA_5_6, provider="openai", service_tier="flex")

    assert budget.total_reasoning_tokens == 250
    assert budget.total_tool_use_tokens == 100
    assert budget.total_cost_usd == compute_cost(
        TERRA_5_6, 1000, 200, 100, 500, provider="openai", service_tier="flex"
    )


@pytest.mark.parametrize(
    ("reasoning_tokens", "tool_use_tokens", "message"),
    [(501, 0, "reasoning_tokens"), (0, 1001, "tool_use_tokens")],
)
def test_usage_subdivision_cannot_exceed_parent_total(
    reasoning_tokens: int, tool_use_tokens: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Budget().record_usage(
            input_tokens=1000,
            output_tokens=500,
            reasoning_tokens=reasoning_tokens,
            tool_use_tokens=tool_use_tokens,
        )


def test_cached_tool_use_is_valid_and_pricing_failure_is_atomic() -> None:
    budget = Budget()
    budget.record_usage(
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=90,
        tool_use_tokens=80,
    )
    snapshot = vars(budget).copy()

    with pytest.raises(ValueError):
        budget.record_usage(
            input_tokens=1,
            output_tokens=1,
            model="unknown-model",
            provider="openai",
        )

    assert vars(budget) == snapshot


@pytest.mark.parametrize("input_tokens", [-1, 1])
def test_invalid_usage_does_not_partially_mutate(input_tokens: int) -> None:
    budget = Budget()
    snapshot = vars(budget).copy()

    with pytest.raises(ValueError):
        budget.record_usage(
            input_tokens=input_tokens,
            output_tokens=1,
            model=TERRA_5_6,
            provider="openai",
            service_tier="unsupported",
        )

    assert vars(budget) == snapshot
