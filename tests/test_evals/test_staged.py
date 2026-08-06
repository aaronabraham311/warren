import json
import stat
from collections.abc import Callable, Iterator
from datetime import date, datetime, timezone
from pathlib import Path
from typing import cast

import pytest

from agent.budget import Budget, RunContext
from agent.models import SONNET_4_6
from agent.persona import DirtPersona
from agent.providers.base import (
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
from eval.golden_set import EvalExample, EvalExpectations, RecommendationExpectation
from eval.staged import (
    EvidenceRecord,
    StageProviderConfig,
    build_evidence_packet,
    collect_evidence,
    run_staged_ticker,
    synthesize_evidence,
    write_private_results,
)
from eval.tool_fixtures import FixtureMiss, FixtureObservation, FixtureToolRunner
from storage.logger import RunLogger

_VALID_ANALYSIS = """{
  "ticker": "AAPL",
  "analysis_type": "holding",
  "recommendation": "hold",
  "confidence": 0.7,
  "thesis": "Revenue grew 8%, free cash flow yield is 5%, and leverage is below 1.0x.",
  "lynch_signals": {"pros": ["steady growth"], "cons": []},
  "buffett_signals": {"pros": ["cash generation"], "cons": []},
  "key_risks": ["valuation could compress by 20%"],
  "data_quality_notes": []
}"""


class FakeProvider:
    name = "anthropic"

    def __init__(self, responses: list[ProviderResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

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
        self.calls.append(
            {
                "model": model,
                "system_prompt": system_prompt,
                "portfolio_context": portfolio_context,
                "messages": messages,
                "tools": tools,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
                "service_tier": service_tier,
            }
        )
        return self.responses.pop(0)

    def tool_result_turn(self, results: list[ToolResultBlock]) -> list[Message]:
        del results
        return []


def _response(text: str, *, tool_call: bool = False) -> ProviderResponse:
    blocks = (
        (ToolCallBlock(id="call-1", name="get_quote", arguments={"ticker": "AAPL"}),)
        if tool_call
        else (TextBlock(text),)
    )
    return ProviderResponse(
        blocks=blocks,
        stop_reason="tool_use" if tool_call else "end_turn",
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        model_id=SONNET_4_6,
    )


def _price() -> PriceData:
    return PriceData(
        ticker="AAPL",
        current_price=190.0,
        previous_close=189.0,
        day_change_pct=0.5,
        volume=1_000_000,
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        data_age_hours=1.0,
    )


def _config() -> StageProviderConfig:
    return StageProviderConfig(provider="anthropic", model=SONNET_4_6)


@pytest.fixture()
def context(tmp_path: Path) -> Iterator[RunContext]:
    logger = RunLogger("staged-test", tmp_path / "logs")
    context = RunContext("staged-test", Budget(max_cost_usd=10), logger)
    yield context
    logger.close()


def _record_success(runner: FixtureToolRunner) -> None:
    runner.observations.append(
        FixtureObservation(
            tool_name="get_quote",
            canonical_input={"ticker": "AAPL"},
            input_hash="abc12345",
            result=ToolResultOk(data=_price()),
        )
    )


def test_collection_observer_stops_before_final_schema_parsing(
    tmp_path: Path,
    context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures = tmp_path / "fixtures"
    quote_dir = fixtures / "AAPL" / "tools" / "get_quote"
    quote_dir.mkdir(parents=True)
    (quote_dir / "fixture.json").write_text("{}")
    runner = FixtureToolRunner("AAPL", fixtures)
    reached_after_observer = False

    def fake_analyze(**kwargs: object) -> None:
        nonlocal reached_after_observer
        _record_success(cast(FixtureToolRunner, kwargs["tool_runner"]))
        observer = cast(Callable[[ProviderResponse], None], kwargs["response_observer"])
        observer(_response("this is deliberately not JSON"))
        reached_after_observer = True

    monkeypatch.setattr("eval.staged.analyze_ticker", fake_analyze)
    result = collect_evidence(
        ticker="AAPL",
        provider=FakeProvider([]),
        config=_config(),
        run_context=context,
        fixtures_root=fixtures,
        fixture_runner=runner,
    )

    assert result.valid
    assert not reached_after_observer
    assert result.responses[0].final_text == "this is deliberately not JSON"
    assert json.loads(cast(str, result.evidence_packet))["ticker"] == "AAPL"


def test_synthesis_is_fresh_tool_free_and_contains_no_gold_labels() -> None:
    provider = FakeProvider([_response(_VALID_ANALYSIS)])
    packet = '{"ticker":"AAPL","evidence":[{"data":{"price":190}}]}'

    result = synthesize_evidence(
        ticker="AAPL", evidence_packet=packet, provider=provider, config=_config()
    )

    assert result.analysis is not None
    call = provider.calls[0]
    assert call["tools"] == []
    messages = cast(list[Message], call["messages"])
    assert all(not message.replay for message in messages)
    rendered = " ".join(
        block.text
        for message in messages
        for block in message.blocks
        if isinstance(block, TextBlock)
    )
    assert packet in rendered
    assert "allowed recommendation" not in rendered
    assert "thesis_must_mention" not in rendered


def test_fixture_miss_invalidates_collection_and_skips_synthesis(
    tmp_path: Path,
    context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = FixtureToolRunner("AAPL", tmp_path / "fixtures")

    def fake_analyze(**kwargs: object) -> None:
        fixture_runner = cast(FixtureToolRunner, kwargs["tool_runner"])
        fixture_runner.misses.append(FixtureMiss("get_news", "deadbeef"))
        observer = cast(Callable[[ProviderResponse], None], kwargs["response_observer"])
        observer(_response("done"))

    monkeypatch.setattr("eval.staged.analyze_ticker", fake_analyze)
    monkeypatch.setattr(
        "eval.staged.FixtureToolRunner",
        lambda ticker, root: runner,
    )
    synthesizer = FakeProvider([_response(_VALID_ANALYSIS)])
    result = run_staged_ticker(
        ticker="AAPL",
        repetition=1,
        collector_provider=FakeProvider([]),
        collector_config=_config(),
        synthesizer_provider=synthesizer,
        synthesizer_config=_config(),
        run_context=context,
        fixtures_root=tmp_path / "fixtures",
    )

    assert not result.collection.valid
    assert not result.synthesis.attempted
    assert synthesizer.calls == []


def test_schema_repair_succeeds_once_without_provider_replay() -> None:
    provider = FakeProvider([_response("not json"), _response(_VALID_ANALYSIS)])

    result = synthesize_evidence(
        ticker="AAPL", evidence_packet="{}", provider=provider, config=_config()
    )

    assert result.analysis is not None
    assert result.repair_attempted
    assert len(provider.calls) == 2
    repair_messages = cast(list[Message], provider.calls[1]["messages"])
    assert all(not message.replay for message in repair_messages)


def test_synthesis_preserves_persona_without_collector_history() -> None:
    provider = FakeProvider([_response(_VALID_ANALYSIS)])

    synthesize_evidence(
        ticker="CIRSA.MC",
        evidence_packet="{}",
        provider=provider,
        config=_config(),
        persona=DirtPersona(),
    )

    system_prompt = cast(str, provider.calls[0]["system_prompt"])
    assert "DIRT universe: US small-caps (Russell 2000)" in system_prompt
    assert "CONTROLLED SYNTHESIS STAGE" in system_prompt


def test_schema_repair_failure_is_explicit() -> None:
    provider = FakeProvider([_response("not json"), _response("still not json")])

    result = synthesize_evidence(
        ticker="AAPL", evidence_packet="{}", provider=provider, config=_config()
    )

    assert result.analysis is None
    assert result.repair_attempted
    assert "schema repair failed" in cast(str, result.error)


def test_evidence_packet_is_sorted_deduplicated_and_label_free() -> None:
    price = EvidenceRecord("get_quote", {"ticker": "AAPL"}, "b", "price", {"price": 190})
    news = EvidenceRecord("get_news", {"days": 7}, "a", "news", {"items": []})

    first = build_evidence_packet("AAPL", [price, news, price])
    second = build_evidence_packet("AAPL", [news, price])

    assert first == second
    payload = json.loads(first)
    assert len(payload["evidence"]) == 2
    assert "expectations" not in first
    assert "preferred" not in first


def test_evidence_packet_compacts_verbose_fields_without_mutating_audit_record() -> None:
    verbose = "evidence " * 3_000
    record = EvidenceRecord(
        "read_filing",
        {"ticker": "AAPL"},
        "filing",
        "filings",
        {"text": verbose, "items": list(range(40))},
    )

    packet = build_evidence_packet("AAPL", [record])
    data = json.loads(packet)["evidence"][0]["data"]

    assert "[truncated " in data["text"]
    assert data["items"][-1] == {"_omitted_items": 15}
    assert record.data["text"] == verbose


def test_planning_coverage_uses_only_available_fixture_dimensions(
    tmp_path: Path,
    context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures = tmp_path / "fixtures"
    for tool in ("get_quote", "get_news"):
        path = fixtures / "AAPL" / "tools" / tool
        path.mkdir(parents=True)
        (path / "fixture.json").write_text("{}")
    runner = FixtureToolRunner("AAPL", fixtures)

    def fake_analyze(**kwargs: object) -> None:
        _record_success(cast(FixtureToolRunner, kwargs["tool_runner"]))
        observer = cast(Callable[[ProviderResponse], None], kwargs["response_observer"])
        observer(_response("done"))

    monkeypatch.setattr("eval.staged.analyze_ticker", fake_analyze)
    result = collect_evidence(
        ticker="AAPL",
        provider=FakeProvider([]),
        config=_config(),
        run_context=context,
        fixtures_root=fixtures,
        fixture_runner=runner,
    )

    coverage = result.planning_coverage
    assert coverage.available_dimensions == ("news", "price")
    assert coverage.covered_dimensions == ("price",)
    assert coverage.coverage_ratio == 0.5
    assert coverage.denominator_note.startswith("Narrow denominator")


def test_grade_is_applied_only_after_synthesis(
    tmp_path: Path,
    context: RunContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures = tmp_path / "fixtures"
    path = fixtures / "AAPL" / "tools" / "get_quote"
    path.mkdir(parents=True)
    (path / "fixture.json").write_text("{}")

    def fake_analyze(**kwargs: object) -> None:
        _record_success(cast(FixtureToolRunner, kwargs["tool_runner"]))
        observer = cast(Callable[[ProviderResponse], None], kwargs["response_observer"])
        observer(_response("not a final answer"))

    monkeypatch.setattr("eval.staged.analyze_ticker", fake_analyze)
    example = EvalExample(
        ticker="AAPL",
        notes="SECRET_GOLD_LABEL",
        last_curated=date(2026, 1, 1),
        expectations=EvalExpectations(
            recommendation=RecommendationExpectation(allowed=["hold"]),
        ),
    )
    synthesizer = FakeProvider([_response(_VALID_ANALYSIS)])
    result = run_staged_ticker(
        ticker="AAPL",
        repetition=1,
        collector_provider=FakeProvider([]),
        collector_config=_config(),
        synthesizer_provider=synthesizer,
        synthesizer_config=_config(),
        run_context=context,
        fixtures_root=fixtures,
        example=example,
    )

    assert result.grade is not None
    assert result.grade.passed
    operational = {
        check.check_name: check.passed
        for check in result.grade.checks
        if check.check_name.startswith(("structured_", "fixture_"))
    }
    assert operational == {
        "structured_output_valid": True,
        "fixture_completeness": True,
        "fixture_evidence_parity": True,
    }
    metrics = cast(dict[str, object], result.to_dict()["stage_metrics"])
    assert metrics["strict_pass"] is True
    assert metrics["schema_valid"] is True
    assert metrics["fixture_miss_count"] == 0
    rendered_call = repr(synthesizer.calls[0])
    assert "SECRET_GOLD_LABEL" not in rendered_call


def test_private_results_are_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "staged.json"
    path.write_text("old")
    path.chmod(0o644)
    write_private_results(path, [])

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text()) == []
