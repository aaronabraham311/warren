import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent.cancellation import NeverCancelToken
from agent.events import NullEventSink
from agent.models import (
    PRICE_CACHE_CREATION_PER_TOKEN,
    PRICE_CACHE_READ_PER_TOKEN,
    PRICE_INPUT_PER_TOKEN,
    PRICE_OUTPUT_PER_TOKEN,
)

if TYPE_CHECKING:
    from agent.cancellation import CancellationToken
    from agent.events import EventSink
    from storage.logger import RunLogger


def compute_cost(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    return (
        input_tokens * PRICE_INPUT_PER_TOKEN
        + output_tokens * PRICE_OUTPUT_PER_TOKEN
        + cache_read_tokens * PRICE_CACHE_READ_PER_TOKEN
        + cache_creation_tokens * PRICE_CACHE_CREATION_PER_TOKEN
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
    # The run's JSONL write-ahead log; every run has one (see storage.logger).
    logger: "RunLogger"
    # Cooperative cancellation and safe display progress. Defaults retain the exact
    # behavior of existing batch/eval callers.
    cancellation: "CancellationToken" = field(default_factory=NeverCancelToken)
    event_sink: "EventSink" = field(default_factory=NullEventSink)
    # Number of agent iterations spent on the current ticker (for ticker_completed).
    iterations: int = 0
    # key = (tool_name, input_hash_prefix) → call count; for tool-loop detection
    tool_call_counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # RunLogger owns durable-before-display fan-out. Preserve the public
        # RunContext(event_sink=...) boundary for callers that construct the logger
        # separately, while leaving an explicitly configured logger sink untouched.
        self.logger.set_event_sink_if_unset(self.event_sink)

    def record_tool_call(self, tool_name: str, tool_input: dict[str, object]) -> int:
        digest = hashlib.sha256(json.dumps(tool_input, sort_keys=True).encode()).hexdigest()[:8]
        key = (tool_name, digest)
        self.tool_call_counts[key] = self.tool_call_counts.get(key, 0) + 1
        self.budget.total_tool_calls += 1
        return self.tool_call_counts[key]
