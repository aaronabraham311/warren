import json
from pathlib import Path

import pytest

from eval.usage import usage_sidecar_path, write_usage_sidecar


def _write_events(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


def test_usage_sidecar_sums_wal_metrics_without_double_counting_subsets(tmp_path: Path) -> None:
    log_path = tmp_path / "eval-provider.jsonl"
    _write_events(
        log_path,
        [
            {
                "event": "llm_call",
                "input_tokens": 100,
                "cache_read_tokens": 300,
                "cache_creation_tokens": 50,
                "output_tokens": 80,
                "reasoning_tokens": 30,
                "tool_use_tokens": 20,
                "latency_ms": 100,
                "cost_usd": 0.01,
                "raw_usage": {"input_tokens": 450, "reasoning_tokens": 30},
            },
            {
                "event": "llm_call",
                "input_tokens": 20,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
                "output_tokens": 20,
                "reasoning_tokens": 0,
                "tool_use_tokens": 0,
                "latency_ms": 300,
                "cost_usd": 0.002,
            },
            {"event": "tool_call", "input_tokens": 999_999},
        ],
    )
    output = tmp_path / "eval-provider.json"

    sidecar = write_usage_sidecar(
        output_path=output,
        log_path=log_path,
        run_id="eval-provider",
        provider="openai",
        model="gpt-5.6-luna",
        service_tier="flex",
        reasoning_effort="medium",
        examples=2,
        passed=1,
    )

    payload = json.loads(sidecar.read_text())
    assert payload["config"] == {
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "service_tier": "flex",
        "reasoning_effort": "medium",
    }
    metrics = payload["metrics"]
    assert metrics["pass_rate"] == 0.5
    assert metrics["input_tokens"] == 120
    assert metrics["cache_read_tokens"] == 300
    assert metrics["cache_creation_tokens"] == 50
    assert metrics["output_tokens"] == 100
    assert metrics["visible_output_tokens"] == 70
    assert metrics["reasoning_tokens"] == 30
    assert metrics["tool_use_tokens"] == 20
    assert metrics["cost_usd"] == pytest.approx(0.012)
    assert metrics["prompt_cache_hit_rate"] == pytest.approx(300 / 470)
    assert metrics["reasoning_token_ratio"] == pytest.approx(0.3)
    assert metrics["latency_ms"] == {"mean": 200.0, "p50": 100, "p95": 300, "max": 300}


def test_unavailable_reasoning_and_empty_denominators_are_null(tmp_path: Path) -> None:
    log_path = tmp_path / "empty.jsonl"
    _write_events(log_path, [{"event": "llm_call", "reasoning_tokens": None}])
    output = tmp_path / "empty.json"

    sidecar = write_usage_sidecar(
        output_path=output,
        log_path=log_path,
        run_id="empty",
        provider="anthropic",
        model="claude-sonnet-4-6",
        service_tier="auto",
        reasoning_effort="none",
        examples=0,
        passed=0,
    )

    metrics = json.loads(sidecar.read_text())["metrics"]
    assert metrics["pass_rate"] is None
    assert metrics["reasoning_tokens"] is None
    assert metrics["prompt_cache_hit_rate"] is None
    assert metrics["reasoning_token_ratio"] is None
    assert metrics["visible_output_tokens"] == 0
    assert metrics["latency_ms"] is None


def test_usage_sidecar_does_not_match_grade_json_glob(tmp_path: Path) -> None:
    output = tmp_path / "eval-a.json"
    output.write_text("[]", encoding="utf-8")
    sidecar = usage_sidecar_path(output)
    sidecar.write_text("{}", encoding="utf-8")

    assert list(tmp_path.glob("*.json")) == [output]


def test_torn_final_wal_line_is_ignored(tmp_path: Path) -> None:
    log_path = tmp_path / "torn.jsonl"
    log_path.write_text(
        '{"event":"llm_call","input_tokens":10,"cost_usd":0.1}\n{"event":',
        encoding="utf-8",
    )
    output = tmp_path / "torn.json"

    sidecar = write_usage_sidecar(
        output_path=output,
        log_path=log_path,
        run_id="torn",
        provider="anthropic",
        model="claude-sonnet-4-6",
        service_tier="auto",
        reasoning_effort="none",
        examples=1,
        passed=1,
    )

    assert json.loads(sidecar.read_text())["metrics"]["input_tokens"] == 10
