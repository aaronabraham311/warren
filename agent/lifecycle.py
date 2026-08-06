"""Validation and diagnostics derived from Warren's versioned lifecycle trace."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from storage.logger import TRACE_SCHEMA_VERSION

_START_EVENTS = {
    "run_started": "run",
    "phase_started": "setup",
    "ticker_started": "ticker",
    "llm_call_started": "model",
    "tool_call_started": "tool",
}
_OUTCOME_EVENTS = {
    "run_completed": "run",
    "run_failed": "run",
    "run_cancelled": "run",
    "phase_completed": "setup",
    "phase_failed": "setup",
    "ticker_completed": "ticker",
    "ticker_failed": "ticker",
    "llm_call": "model",
    "llm_call_failed": "model",
    "tool_call": "tool",
}


@dataclass(frozen=True, slots=True)
class IntegrityIssue:
    code: str
    sequence: int | None
    detail: str


@dataclass(frozen=True, slots=True)
class SlowOperation:
    kind: str
    name: str
    duration_ms: int


@dataclass(frozen=True, slots=True)
class LifecycleSummary:
    verdict: str
    current_or_final_phase: str
    issues: tuple[IntegrityIssue, ...]
    unmatched_starts: int
    retries: int
    failures: int
    slowest: tuple[SlowOperation, ...]


def _int(record: Mapping[str, object], key: str) -> int | None:
    value = record.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _str(record: Mapping[str, object], key: str) -> str | None:
    value = record.get(key)
    return value if isinstance(value, str) else None


def _operation_name(record: Mapping[str, object], kind: str) -> str:
    if kind == "tool":
        return _str(record, "tool") or "tool"
    if kind == "model":
        return _str(record, "model") or "model"
    if kind == "setup":
        return _str(record, "phase") or "setup"
    if kind == "ticker":
        return _str(record, "ticker") or "ticker"
    return "run"


def validate_trace(records: Iterable[Mapping[str, object]]) -> LifecycleSummary:
    """Validate supported lifecycle envelopes while safely ignoring future events."""

    issues: list[IntegrityIssue] = []
    starts: dict[str, tuple[str, int | None]] = {}
    completed: set[str] = set()
    known_operations: set[str] = set()
    slowest_by_kind: dict[str, SlowOperation] = {}
    expected_sequence = 1
    retries = 0
    failures = 0
    phase = "unknown"
    saw_run_outcome = False

    for record in records:
        sequence = _int(record, "sequence")
        schema_version = _int(record, "schema_version")
        if sequence is not None:
            if sequence != expected_sequence:
                issues.append(
                    IntegrityIssue(
                        "sequence_gap",
                        sequence,
                        f"expected sequence {expected_sequence}, observed {sequence}",
                    )
                )
            expected_sequence = sequence + 1
        if schema_version is not None and schema_version > TRACE_SCHEMA_VERSION:
            continue

        event = _str(record, "event")
        if event is None:
            continue
        if event in {"run_completed", "run_failed", "run_cancelled"}:
            phase = _str(record, "status") or event.removeprefix("run_")
            saw_run_outcome = True
        elif event.endswith("_started"):
            phase = _str(record, "phase") or event.removesuffix("_started")

        operation_id = _str(record, "operation_id")
        parent_id = _str(record, "parent_operation_id")
        if parent_id is not None and parent_id not in known_operations:
            issues.append(
                IntegrityIssue(
                    "invalid_parent",
                    sequence,
                    f"parent operation {parent_id!r} was not started earlier",
                )
            )

        start_kind = _START_EVENTS.get(event)
        if start_kind is not None and operation_id is not None:
            if operation_id in starts:
                issues.append(
                    IntegrityIssue("duplicate_start", sequence, f"duplicate {operation_id}")
                )
            starts[operation_id] = (event, sequence)
            known_operations.add(operation_id)

        outcome_kind = _OUTCOME_EVENTS.get(event)
        if outcome_kind is not None:
            if operation_id is None or operation_id not in starts:
                issues.append(
                    IntegrityIssue(
                        "orphan_outcome",
                        sequence,
                        f"{event} has no matching start",
                    )
                )
            elif operation_id in completed:
                issues.append(
                    IntegrityIssue(
                        "duplicate_outcome",
                        sequence,
                        f"operation {operation_id} already ended",
                    )
                )
            else:
                completed.add(operation_id)

            duration = _int(record, "latency_ms")
            if duration is None:
                seconds = record.get("duration_seconds")
                if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
                    duration = int(float(seconds) * 1000)
            if duration is not None:
                if duration < 0:
                    issues.append(
                        IntegrityIssue(
                            "negative_duration",
                            sequence,
                            f"{event} duration is negative",
                        )
                    )
                else:
                    candidate = SlowOperation(
                        outcome_kind,
                        _operation_name(record, outcome_kind),
                        duration,
                    )
                    previous = slowest_by_kind.get(outcome_kind)
                    if previous is None or candidate.duration_ms > previous.duration_ms:
                        slowest_by_kind[outcome_kind] = candidate

        retries += max(0, _int(record, "retry_count") or 0)
        if event.endswith("failed") or record.get("status") == "error":
            failures += 1

    unmatched = sorted(set(starts) - completed)
    for operation_id in unmatched:
        start_event, sequence = starts[operation_id]
        issues.append(
            IntegrityIssue(
                "unmatched_start",
                sequence,
                f"{start_event} operation {operation_id} has no terminal outcome",
            )
        )
    if starts and not saw_run_outcome:
        issues.append(IntegrityIssue("missing_run_outcome", None, "run has no terminal outcome"))

    return LifecycleSummary(
        verdict="healthy" if not issues else "degraded",
        current_or_final_phase=phase,
        issues=tuple(issues),
        unmatched_starts=len(unmatched),
        retries=retries,
        failures=failures,
        slowest=tuple(slowest_by_kind[kind] for kind in sorted(slowest_by_kind)),
    )
