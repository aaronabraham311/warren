import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent.models import DEFAULT_MODEL_ID
from agent.providers.base import Usage
from storage.cost import compute_cost

if TYPE_CHECKING:
    from storage.logger import RunLogger


@dataclass
class Budget:
    max_input_tokens: int = 1_500_000
    max_output_tokens: int = 200_000
    max_cost_usd: float = 1.25
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_tool_use_tokens: int = 0
    total_cost_usd: float = 0.0
    total_tool_calls: int = 0

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        reasoning_tokens: int = 0,
        tool_use_tokens: int = 0,
        *,
        model: str = DEFAULT_MODEL_ID,
        provider: str = "anthropic",
        service_tier: str = "default",
    ) -> None:
        values = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "reasoning_tokens": reasoning_tokens,
            "tool_use_tokens": tool_use_tokens,
        }
        if any(value < 0 for value in values.values()):
            raise ValueError("token counts cannot be negative")
        if reasoning_tokens > output_tokens:
            raise ValueError("reasoning_tokens must be a subset of output_tokens")
        total_input_tokens = input_tokens + cache_read_tokens + cache_creation_tokens
        if tool_use_tokens > total_input_tokens:
            raise ValueError("tool_use_tokens must be a subset of total input tokens")
        cost = compute_cost(
            model,
            input_tokens=input_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            output_tokens=output_tokens,
            provider=provider,
            service_tier=service_tier,
        )
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cache_read_tokens += cache_read_tokens
        self.total_cache_creation_tokens += cache_creation_tokens
        self.total_reasoning_tokens += reasoning_tokens
        self.total_tool_use_tokens += tool_use_tokens
        self.total_cost_usd += cost

    def record_provider_usage(
        self,
        usage: Usage,
        *,
        model: str,
        provider: str,
        service_tier: str = "default",
    ) -> None:
        """Record one normalized provider usage object without double-counting subsets."""
        self.record_usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_creation_tokens=usage.cache_write_tokens,
            reasoning_tokens=usage.reasoning_tokens or 0,
            tool_use_tokens=usage.tool_use_tokens,
            model=model,
            provider=provider,
            service_tier=service_tier,
        )

    def token_exceeded(self) -> bool:
        return (
            self.total_input_tokens >= self.max_input_tokens
            or self.total_output_tokens >= self.max_output_tokens
        )

    def cost_exceeded(self) -> bool:
        return self.total_cost_usd >= self.max_cost_usd


@dataclass
class RunContext:
    run_id: str
    budget: Budget
    # The run's JSONL write-ahead log; every run has one (see storage.logger).
    logger: "RunLogger"
    # Number of agent iterations spent on the current ticker (for ticker_completed).
    iterations: int = 0
    # key = (tool_name, input_hash_prefix) → call count; for tool-loop detection
    tool_call_counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def record_tool_call(self, tool_name: str, tool_input: dict[str, object]) -> int:
        digest = hashlib.sha256(json.dumps(tool_input, sort_keys=True).encode()).hexdigest()[:8]
        key = (tool_name, digest)
        self.tool_call_counts[key] = self.tool_call_counts.get(key, 0) + 1
        self.budget.total_tool_calls += 1
        return self.tool_call_counts[key]
