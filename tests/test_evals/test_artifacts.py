import json
import stat
from pathlib import Path

from agent.providers.base import ProviderResponse, TextBlock, Usage
from eval.artifacts import CapturedResponse, EvalArtifactRecord, EvalArtifactWriter, content_hash


def test_captured_response_preserves_raw_final_text_and_usage() -> None:
    response = ProviderResponse(
        blocks=(TextBlock('{"ticker":"AAPL"}'),),
        stop_reason="completed",
        usage=Usage(input_tokens=10, output_tokens=4, raw={"provider": "raw"}),
        model_id="test-model",
        replay=({"opaque": "provider-state"},),
    )

    captured = CapturedResponse.from_response(response)

    assert captured.final_text == '{"ticker":"AAPL"}'
    assert captured.usage["input_tokens"] == 10
    assert captured.usage["raw"] == {"provider": "raw"}
    assert captured.replay == [{"opaque": "provider-state"}]


def test_artifact_writer_is_owner_only_and_flushes_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "run.audit.jsonl"
    record = EvalArtifactRecord(
        run_id="eval-1",
        ticker="AAPL",
        provider="openai",
        model="gpt-test",
        service_tier="default",
        reasoning_effort="medium",
        persona="DefaultPersona",
        prompt_hash=content_hash("private prompt"),
        fixture_set_id="fixture-hash",
        tool_trace="logs/runs/eval-1.jsonl",
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
        fixture_misses=[],
        fixture_evidence_issues=[],
        responses=[],
        analysis_output=None,
        raw_final_content="invalid raw output",
        failure={"type": "SchemaRepairError", "message": "invalid"},
        grade={"ticker": "AAPL", "passed": False},
    )

    with EvalArtifactWriter(path) as writer:
        writer.write(record)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1
    assert payload["raw_final_content"] == "invalid raw output"
    assert payload["prompt_hash"] != "private prompt"
