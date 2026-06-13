import hashlib
import json
from dataclasses import dataclass, field

# Sonnet 4.6 pricing (USD per token)
_SONNET_INPUT = 3.0 / 1_000_000
_SONNET_OUTPUT = 15.0 / 1_000_000
_SONNET_CACHE_READ = 0.30 / 1_000_000


def compute_cost(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    return (
        input_tokens * _SONNET_INPUT
        + output_tokens * _SONNET_OUTPUT
        + cache_read_tokens * _SONNET_CACHE_READ
        # cache_creation billed at 1.25x input rate (5-min TTL)
        + cache_creation_tokens * _SONNET_INPUT * 1.25
    )


@dataclass
class Budget:
    max_input_tokens: int = 1_500_000
    max_output_tokens: int = 200_000
    max_cost_usd: float = 1.25
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_tool_calls: int = 0

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += compute_cost(
            input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens
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
    # key = (tool_name, input_hash_prefix) → call count; for tool-loop detection
    tool_call_counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def record_tool_call(self, tool_name: str, tool_input: dict[str, object]) -> int:
        digest = hashlib.sha256(json.dumps(tool_input, sort_keys=True).encode()).hexdigest()[:8]
        key = (tool_name, digest)
        self.tool_call_counts[key] = self.tool_call_counts.get(key, 0) + 1
        self.budget.total_tool_calls += 1
        return self.tool_call_counts[key]
