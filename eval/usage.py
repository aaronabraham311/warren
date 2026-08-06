"""WAL-derived usage artifact for provider-comparison eval runs."""

from __future__ import annotations

import json
import math
from pathlib import Path


def usage_sidecar_path(output_path: Path) -> Path:
    """Return a sidecar path excluded by the existing ``runs/*.json`` grade glob."""
    return Path(f"{output_path}.usage")


def write_usage_sidecar(
    *,
    output_path: Path,
    log_path: Path,
    run_id: str,
    provider: str,
    model: str,
    service_tier: str,
    reasoning_effort: str,
    examples: int,
    passed: int,
) -> Path:
    events = _read_events(log_path)
    calls = [event for event in events if event.get("event") == "llm_call"]

    input_tokens = _sum_int(calls, "input_tokens")
    cache_read_tokens = _sum_int(calls, "cache_read_tokens")
    cache_creation_tokens = _sum_int(calls, "cache_creation_tokens")
    output_tokens = _sum_int(calls, "output_tokens")
    tool_use_tokens = _sum_int(calls, "tool_use_tokens")
    reasoning_values = [_as_int(event.get("reasoning_tokens")) for event in calls]
    observed_reasoning = [value for value in reasoning_values if value is not None]
    reasoning_tokens = sum(observed_reasoning) if observed_reasoning else None
    visible_output_tokens = max(0, output_tokens - (reasoning_tokens or 0))
    latencies = sorted(
        value for event in calls if (value := _as_int(event.get("latency_ms"))) is not None
    )
    total_cost_usd = sum(_as_float(event.get("cost_usd")) for event in calls)

    cache_denominator = input_tokens + cache_read_tokens + cache_creation_tokens
    artifact: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "config": {
            "provider": provider,
            "model": model,
            "service_tier": service_tier,
            "reasoning_effort": reasoning_effort,
        },
        "metrics": {
            "examples": examples,
            "passed": passed,
            "pass_rate": passed / examples if examples else None,
            "input_tokens": input_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "output_tokens": output_tokens,
            "visible_output_tokens": visible_output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "tool_use_tokens": tool_use_tokens,
            "cost_usd": total_cost_usd,
            "prompt_cache_hit_rate": (
                cache_read_tokens / cache_denominator if cache_denominator else None
            ),
            "reasoning_token_ratio": (
                reasoning_tokens / output_tokens
                if reasoning_tokens is not None and output_tokens
                else None
            ),
            "latency_ms": (
                None
                if not latencies
                else {
                    "mean": sum(latencies) / len(latencies),
                    "p50": _nearest_rank(latencies, 0.50),
                    "p95": _nearest_rank(latencies, 0.95),
                    "max": latencies[-1],
                }
            ),
        },
    }
    sidecar = usage_sidecar_path(output_path)
    sidecar.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return sidecar


def _read_events(path: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _sum_int(events: list[dict[str, object]], key: str) -> int:
    return sum(value for event in events if (value := _as_int(event.get(key))) is not None)


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _nearest_rank(sorted_values: list[int], percentile: float) -> int:
    index = max(0, math.ceil(percentile * len(sorted_values)) - 1)
    return sorted_values[index]
