"""Local-first active-run diagnostics derived from the safe lifecycle stream."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from agent.activity import ActivityState, reduce_activity
from agent.events import RunEvent

HealthClassification: TypeAlias = Literal[
    "healthy",
    "external_wait",
    "renderer_stall",
    "agent_stall",
    "trace_stall",
]
WaitLevel: TypeAlias = Literal["normal", "slow", "prolonged"]


@dataclass(frozen=True, slots=True)
class HealthThresholds:
    local_seconds: float = 1.0
    model_slow_seconds: float = 15.0
    model_prolonged_seconds: float = 60.0
    trace_seconds: float = 1.0


@dataclass(frozen=True, slots=True)
class HealthTransition:
    previous: HealthClassification | None
    current: HealthClassification
    recovered: bool
    observed_at: float


@dataclass(frozen=True, slots=True)
class MetricSample:
    name: str
    value: float
    attributes: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class DiagnosticsSnapshot:
    run_id: str | None
    ticker: str | None
    operation: str | None
    operation_age_seconds: float | None
    semantic_event_age_seconds: float | None
    renderer_frame_age_seconds: float | None
    trace_fsync_age_seconds: float | None
    external_wait: bool
    wait_level: WaitLevel
    completed_models: int
    completed_tools: int
    retry_count: int
    health: HealthClassification
    trace_path: Path | None


class HealthMonitor:
    """Track health samples without persisting spinner ticks or private content."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        thresholds: HealthThresholds | None = None,
        log_dir: Path | None = None,
    ) -> None:
        self._clock = clock
        self._thresholds = thresholds or HealthThresholds()
        self._log_dir = log_dir
        self._activity = ActivityState()
        self._last_frame_at: float | None = None
        self._last_fsync_at: float | None = None
        self._retry_count = 0
        self._classification: HealthClassification | None = None

    @property
    def activity(self) -> ActivityState:
        return self._activity

    def observe_event(self, event: RunEvent, *, persisted: bool = True) -> None:
        now = self._clock()
        self._activity = reduce_activity(self._activity, event, now=now)
        if persisted:
            self._last_fsync_at = now

    def frame_rendered(self) -> None:
        self._last_frame_at = self._clock()

    def trace_persisted(self) -> None:
        self._last_fsync_at = self._clock()

    def record_retries(self, count: int) -> None:
        self._retry_count += max(0, count)

    def _trace_path(self) -> Path | None:
        if self._activity.run_id is None:
            return None
        log_dir = self._log_dir or Path(os.environ.get("WARREN_LOGS_DIR", "logs/runs"))
        return log_dir / f"{self._activity.run_id}.jsonl"

    @staticmethod
    def _age(now: float, then: float | None) -> float | None:
        return None if then is None else max(0.0, now - then)

    def snapshot(self) -> DiagnosticsSnapshot:
        now = self._clock()
        operation_age = self._activity.operation_age(now)
        semantic_age = self._activity.semantic_event_age(now)
        frame_age = self._age(now, self._last_frame_at)
        fsync_age = self._age(now, self._last_fsync_at)
        if operation_age is None or self._activity.operation != "model":
            wait_level: WaitLevel = "normal"
        elif operation_age >= self._thresholds.model_prolonged_seconds:
            wait_level = "prolonged"
        elif operation_age >= self._thresholds.model_slow_seconds:
            wait_level = "slow"
        else:
            wait_level = "normal"

        event_after_fsync = (
            self._activity.last_event_at is not None
            and self._last_fsync_at is not None
            and self._activity.last_event_at > self._last_fsync_at
        )
        event_after_frame = (
            self._activity.last_event_at is not None
            and self._last_frame_at is not None
            and self._activity.last_event_at > self._last_frame_at
        )
        if (
            event_after_fsync
            and fsync_age is not None
            and fsync_age > self._thresholds.trace_seconds
        ):
            health: HealthClassification = "trace_stall"
        elif (
            event_after_frame
            and frame_age is not None
            and frame_age > self._thresholds.local_seconds
        ):
            health = "renderer_stall"
        elif self._activity.external_wait:
            health = "external_wait"
        elif (
            self._activity.outcome is None
            and semantic_age is not None
            and semantic_age > self._thresholds.local_seconds
        ):
            health = "agent_stall"
        else:
            health = "healthy"

        return DiagnosticsSnapshot(
            run_id=self._activity.run_id,
            ticker=self._activity.ticker,
            operation=self._activity.operation_name,
            operation_age_seconds=operation_age,
            semantic_event_age_seconds=semantic_age,
            renderer_frame_age_seconds=frame_age,
            trace_fsync_age_seconds=fsync_age,
            external_wait=self._activity.external_wait,
            wait_level=wait_level,
            completed_models=self._activity.completed_models,
            completed_tools=self._activity.completed_tools,
            retry_count=self._retry_count,
            health=health,
            trace_path=self._trace_path(),
        )

    def sample_transition(self) -> HealthTransition | None:
        snapshot = self.snapshot()
        previous = self._classification
        if snapshot.health == previous:
            return None
        self._classification = snapshot.health
        return HealthTransition(
            previous=previous,
            current=snapshot.health,
            recovered=previous not in {None, "healthy", "external_wait"}
            and snapshot.health == "healthy",
            observed_at=self._clock(),
        )

    def metric_samples(self) -> tuple[MetricSample, ...]:
        """Return bounded-cardinality samples suitable for an optional OTel adapter."""

        snapshot = self.snapshot()
        operation = self._activity.operation or "idle"
        health_attributes = (("health", snapshot.health), ("operation", operation))
        samples = [MetricSample("terminal.health", 1.0, health_attributes)]
        if snapshot.renderer_frame_age_seconds is not None:
            samples.append(
                MetricSample(
                    "terminal.renderer.frame_age",
                    snapshot.renderer_frame_age_seconds,
                    (("operation", operation),),
                )
            )
        if snapshot.trace_fsync_age_seconds is not None:
            samples.append(
                MetricSample(
                    "terminal.trace.fsync_age",
                    snapshot.trace_fsync_age_seconds,
                    (("operation", operation),),
                )
            )
        return tuple(samples)
