from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from agent.budget import Budget, RunContext
from agent.loop import ProviderTerminalError, analyze_ticker
from agent.models import GEMINI_3_6_FLASH, SONNET_4_6, TERRA_5_6
from agent.persona import DefaultPersona
from agent.providers.base import (
    JSONObject,
    Message,
    ProviderResponse,
    ReasoningEffort,
    ServiceTier,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    Usage,
)
from agent.tools.base import ToolDefinition, ToolResultOk
from data_sources.yfinance_client import PriceData
from eval.tool_fixtures import FixtureToolRunner, record_tool_result
from storage.cost import compute_cost
from storage.logger import RunLogger
from tests.conftest import VALID_ANALYSIS_JSON


@dataclass
class _Routing:
    model: str

    def select(self, iteration: int, messages: list[Message], ticker: str) -> str:
        del iteration, messages, ticker
        return self.model


@dataclass
class _NormalizedProvider:
    name: str
    model: str
    responses: list[ProviderResponse]
    calls: list[list[Message]] = field(default_factory=list)
    result_turns: list[list[ToolResultBlock]] = field(default_factory=list)

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        portfolio_context: str,
        messages: list[Message],
        tools: list[ToolDefinition],
        max_tokens: int,
        temperature: float | None = None,
        reasoning_effort: ReasoningEffort = "none",
        service_tier: ServiceTier = "auto",
    ) -> ProviderResponse:
        del (
            system_prompt,
            portfolio_context,
            tools,
            max_tokens,
            temperature,
            reasoning_effort,
            service_tier,
        )
        assert model == self.model
        self.calls.append(list(messages))
        return self.responses.pop(0)

    def tool_result_turn(self, results: list[ToolResultBlock]) -> list[Message]:
        self.result_turns.append(list(results))
        return [Message(role="user", blocks=tuple(results))]


@dataclass
class _WalEvent:
    event: str
    cost_usd: float | None = None


_CASES = [
    pytest.param("anthropic", SONNET_4_6, "default", id="anthropic"),
    pytest.param("openai", TERRA_5_6, "flex", id="openai"),
    pytest.param("gemini", GEMINI_3_6_FLASH, "flex", id="gemini"),
]


@pytest.mark.parametrize(("provider_name", "model", "service_tier"), _CASES)
def test_normalized_provider_tool_loop_preserves_replay_and_cost_agrees_with_wal(
    provider_name: str,
    model: str,
    service_tier: ServiceTier,
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixtures"
    price = PriceData(
        ticker="AAPL",
        current_price=180.0,
        previous_close=178.0,
        day_change_pct=1.12,
        volume=50_000_000,
        as_of=datetime.now(timezone.utc),
        data_age_hours=0,
    )
    record_tool_result(
        "AAPL",
        "get_quote",
        {"ticker": "AAPL"},
        ToolResultOk(data=price),
        fixture_root,
    )

    opaque: JSONObject = {
        "type": "opaque_replay",
        "provider": provider_name,
        "token": "signed",
    }
    final_replay: JSONObject = {"type": "final", "provider": provider_name}
    usage = Usage(input_tokens=100, output_tokens=20, total_tokens=120)
    provider = _NormalizedProvider(
        name=provider_name,
        model=model,
        responses=[
            ProviderResponse(
                blocks=(
                    ToolCallBlock(id="call_1", name="get_quote", arguments={"ticker": "AAPL"}),
                ),
                stop_reason="tool_use",
                usage=usage,
                model_id=model,
                replay=(opaque,),
            ),
            ProviderResponse(
                blocks=(TextBlock(VALID_ANALYSIS_JSON),),
                stop_reason="completed" if provider_name != "anthropic" else "end_turn",
                usage=usage,
                model_id=model,
                replay=(final_replay,),
            ),
        ],
    )
    logger = RunLogger(f"run-{provider_name}", tmp_path / "logs")
    budget = Budget(max_cost_usd=10.0)
    context = RunContext(run_id=f"run-{provider_name}", budget=budget, logger=logger)

    result = analyze_ticker(
        "AAPL",
        DefaultPersona(),
        _Routing(model),
        context,
        provider=provider,
        service_tier=service_tier,
        tool_runner=FixtureToolRunner("AAPL", fixture_root),
    )

    assert result.ticker == "AAPL"
    assert result.recommendation == "hold"
    assert len(provider.calls) == 2
    assert provider.calls[1][1].role == "assistant"
    assert provider.calls[1][1].replay == (opaque,)
    assert provider.result_turns[0][0].call_id == "call_1"
    assert provider.result_turns[0][0].content == price.model_dump_json()

    expected_call_cost = compute_cost(
        model,
        input_tokens=usage.input_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_creation_tokens=usage.cache_write_tokens,
        output_tokens=usage.output_tokens,
        provider=provider_name,
        service_tier=service_tier,
    )
    events = [
        TypeAdapter(_WalEvent).validate_json(line)
        for line in logger.path.read_text(encoding="utf-8").splitlines()
    ]
    llm_events = [event for event in events if event.event == "llm_call"]
    assert len(llm_events) == 2
    assert all(event.cost_usd is not None for event in llm_events)
    wal_cost = sum(event.cost_usd or 0.0 for event in llm_events)
    assert budget.total_cost_usd == pytest.approx(expected_call_cost * 2)
    assert wal_cost == pytest.approx(budget.total_cost_usd)


@pytest.mark.parametrize(
    "stop_reason",
    ["refusal", "incomplete", "max_tokens", "max_output_tokens", "failed"],
)
def test_terminal_provider_stop_reason_raises_typed_error(stop_reason: str, tmp_path: Path) -> None:
    provider = _NormalizedProvider(
        name="anthropic",
        model=SONNET_4_6,
        responses=[
            ProviderResponse(
                blocks=(),
                stop_reason=stop_reason,
                usage=Usage(),
                model_id=SONNET_4_6,
            )
        ],
    )
    context = RunContext(
        run_id="terminal",
        budget=Budget(),
        logger=RunLogger("terminal", tmp_path / "logs"),
    )

    with pytest.raises(ProviderTerminalError) as exc_info:
        analyze_ticker(
            "AAPL",
            DefaultPersona(),
            _Routing(SONNET_4_6),
            context,
            provider=provider,
        )

    assert exc_info.value.provider == "anthropic"
    assert exc_info.value.model == SONNET_4_6
    assert exc_info.value.stop_reason == stop_reason
