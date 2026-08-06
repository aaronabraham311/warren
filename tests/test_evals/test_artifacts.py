import json
import stat
from pathlib import Path

from eval.artifacts import CapturedResponse, EvalArtifactRecord, EvalArtifactWriter, content_hash
from tests.conftest import make_end_turn


def test_captured_response_preserves_raw_final_text_and_usage() -> None:
    response = make_end_turn('{"ticker":"AAPL"}', input_tokens=10, output_tokens=4)

    captured = CapturedResponse.from_response(response)

    assert captured.final_text == '{"ticker":"AAPL"}'
    assert captured.usage["input_tokens"] == 10
    assert captured.usage["output_tokens"] == 4
    assert captured.model_id == "claude-sonnet-4-6"


def test_artifact_writer_is_owner_only_and_flushes_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "run.audit.jsonl"
    record = EvalArtifactRecord(
        run_id="eval-1",
        ticker="AAPL",
        provider="anthropic",
        model="claude-sonnet-4-6",
        service_tier="default",
        reasoning_effort="none",
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
