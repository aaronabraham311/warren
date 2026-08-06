from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

from agent.events import (
    LlmCallStarted,
    RunCompleted,
    RunStarted,
    TickerStarted,
    ToolCallCompleted,
)
from agent.terminal.health import HealthMonitor, HealthThresholds
from agent.terminal.reliability import FakeClock
from agent.terminal.renderer import TerminalRenderer


def test_external_model_wait_has_slow_and_prolonged_levels(tmp_path: Path) -> None:
    clock = FakeClock()
    monitor = HealthMonitor(clock=clock, log_dir=tmp_path)
    monitor.observe_event(RunStarted("run-1", tickers=("AMD",)))
    monitor.frame_rendered()
    monitor.observe_event(LlmCallStarted("run-1", "AMD", "sonnet", "synthesis", 4, 7))
    monitor.frame_rendered()

    clock.advance(15)
    slow = monitor.snapshot()
    assert slow.health == "external_wait"
    assert slow.wait_level == "slow"
    assert slow.operation == "synthesis"
    assert slow.trace_path == tmp_path / "run-1.jsonl"

    clock.advance(45)
    assert monitor.snapshot().wait_level == "prolonged"


def test_health_classifies_renderer_agent_and_trace_stalls() -> None:
    thresholds = HealthThresholds(local_seconds=1.0, trace_seconds=1.0)

    renderer_clock = FakeClock()
    renderer = HealthMonitor(clock=renderer_clock, thresholds=thresholds)
    renderer.observe_event(RunStarted("run-renderer"))
    renderer.frame_rendered()
    renderer_clock.advance(1)
    renderer.observe_event(TickerStarted("run-renderer", "AMD"))
    renderer_clock.advance(2)
    assert renderer.snapshot().health == "renderer_stall"

    agent_clock = FakeClock()
    agent = HealthMonitor(clock=agent_clock, thresholds=thresholds)
    agent.observe_event(RunStarted("run-agent"))
    agent.frame_rendered()
    agent_clock.advance(2)
    assert agent.snapshot().health == "agent_stall"

    trace_clock = FakeClock()
    trace = HealthMonitor(clock=trace_clock, thresholds=thresholds)
    trace.observe_event(RunStarted("run-trace"))
    trace.frame_rendered()
    trace_clock.advance(1)
    trace.observe_event(TickerStarted("run-trace", "AMD"), persisted=False)
    trace.frame_rendered()
    trace_clock.advance(2)
    assert trace.snapshot().health == "trace_stall"


def test_recovery_transition_is_emitted_once() -> None:
    clock = FakeClock()
    monitor = HealthMonitor(clock=clock)
    monitor.observe_event(RunStarted("run-1"))
    monitor.frame_rendered()
    initial = monitor.sample_transition()
    assert initial is not None and initial.current == "healthy"

    clock.advance(2)
    stalled = monitor.sample_transition()
    assert stalled is not None and stalled.current == "agent_stall"
    assert stalled.recovered is False

    monitor.observe_event(RunCompleted("run-1", "success", 0.0, 2.0))
    monitor.frame_rendered()
    recovered = monitor.sample_transition()
    assert recovered is not None and recovered.current == "healthy"
    assert recovered.recovered is True
    assert monitor.sample_transition() is None


def test_metric_dimensions_are_bounded_and_exclude_run_or_ticker() -> None:
    clock = FakeClock()
    monitor = HealthMonitor(clock=clock)
    monitor.observe_event(RunStarted("secret-run-id", tickers=("SECRET",)))
    monitor.observe_event(TickerStarted("secret-run-id", "SECRET"))
    monitor.frame_rendered()
    clock.advance(0.5)

    samples = monitor.metric_samples()
    serialized = repr(samples)
    assert "secret-run-id" not in serialized
    assert "SECRET" not in serialized
    assert {key for sample in samples for key, _value in sample.attributes} <= {
        "health",
        "operation",
    }


def test_renderer_failure_leaves_queryable_stall_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    renderer = TerminalRenderer(
        stdout=StringIO(),
        stderr=StringIO(),
        color="never",
        animation=False,
        clock=clock,
    )
    renderer.start_activity("Preparing analysis…")
    clock.advance(1)

    def fail_render(_event: object) -> None:
        raise RuntimeError("renderer broke")

    monkeypatch.setattr(renderer, "_emit_event", fail_render)
    with pytest.raises(RuntimeError, match="renderer broke"):
        renderer.emit(RunStarted("run-renderer-failure"))
    clock.advance(2)

    assert renderer.diagnostics.run_id == "run-renderer-failure"
    assert renderer.diagnostics.health == "renderer_stall"


def test_renderer_diagnostics_include_safe_counts() -> None:
    renderer = TerminalRenderer(
        stdout=StringIO(),
        stderr=StringIO(),
        color="never",
        animation=False,
    )
    renderer.emit(RunStarted("run-counts", tickers=("AMD",)))
    renderer.emit(ToolCallCompleted("run-counts", "AMD", "get_quote", "ok", False, 10, 2))

    assert renderer.diagnostics.completed_tools == 1
    assert renderer.diagnostics.retry_count == 2
