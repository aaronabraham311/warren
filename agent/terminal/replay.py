"""Offline validation, semantic replay, and privacy-safe terminal failure bundles."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from agent.events import RunEvent, event_from_wal_record
from agent.lifecycle import LifecycleSummary, validate_trace
from agent.terminal.reliability import FakeClock, ScreenSnapshot, TerminalScenario
from agent.terminal.renderer import ColorMode, sanitize_terminal_text

ReplayMode: TypeAlias = Literal["tty", "pipe", "no_color", "dumb"]

_SAFE_EVENT_FIELDS = (
    "kind",
    "run_id",
    "ticker",
    "mode",
    "tickers",
    "total",
    "completed",
    "rank",
    "model",
    "purpose",
    "iteration",
    "tool_count",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "cache_read_tokens",
    "cache_creation_tokens",
    "tool_name",
    "status",
    "cached",
    "retry_count",
    "error_summary",
    "error_type",
    "recommendation",
    "confidence",
    "total_cost_usd",
    "duration_seconds",
    "timestamp",
)
_SAFE_SOURCE_FIELDS = {
    "schema_version",
    "sequence",
    "ts",
    "monotonic_ms",
    "run_id",
    "event",
    "operation_id",
    "parent_operation_id",
    "outcome",
    "ticker",
    "phase",
    "tool",
    "model",
    "status",
    "cached",
    "retry_count",
    "latency_ms",
    "input_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "output_tokens",
    "cost_usd",
    "duration_seconds",
    "purpose",
    "iteration",
    "tool_count",
    "error_type",
    "recommendation",
    "confidence",
    "mode",
    "tickers",
    "total",
    "completed",
    "rank",
}
_REJECTED_KEYS = {
    "authorization",
    "cookie",
    "headers",
    "input",
    "output",
    "prompt",
    "reasoning",
    "secret",
    "token",
    "tool_input",
    "tool_output",
}
_SECRET_TEXT = re.compile(
    r"(?i)\b(authorization|cookie|password|secret|token|x-api-key)\b(\s*[:=]\s*)(\S+)"
)
_HOME_PATH = re.compile(r"(?<!\w)(?:/Users|/home)/[^/\s]+")


@dataclass(frozen=True, slots=True)
class SanitizedEvent:
    fields: dict[str, object]
    dropped_field_count: int


@dataclass(frozen=True, slots=True)
class ReplayResult:
    snapshot: ScreenSnapshot
    integrity: LifecycleSummary
    events: tuple[SanitizedEvent, ...]


def _safe_text(value: str) -> str:
    sanitized = sanitize_terminal_text(value)
    sanitized = _SECRET_TEXT.sub(r"\1\2[redacted]", sanitized)
    return _HOME_PATH.sub("<home>", sanitized)[:512]


def _safe_value(value: object) -> object | None:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, (tuple, list)):
        return tuple(_safe_value(item) for item in value[:50])
    return None


def sanitize_event(event: RunEvent, source: dict[str, object]) -> SanitizedEvent:
    """Apply a central allow-list after projection through the typed safe boundary."""

    safe: dict[str, object] = {}
    for name in _SAFE_EVENT_FIELDS:
        if name.casefold() in _REJECTED_KEYS or not hasattr(event, name):
            continue
        value = _safe_value(getattr(event, name))
        if value is not None:
            safe[name] = value
    dropped = sum(
        1 for key in source if key not in _SAFE_SOURCE_FIELDS or key.casefold() in _REJECTED_KEYS
    )
    return SanitizedEvent(safe, dropped)


def _load_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"malformed trace record at line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"non-object trace record at line {line_number}")
        records.append(record)
    return records


def _mode_options(mode: ReplayMode) -> tuple[bool, ColorMode, str]:
    if mode == "tty":
        return True, "always", "xterm-256color"
    if mode == "no_color":
        return True, "never", "xterm-256color"
    if mode == "dumb":
        return True, "never", "dumb"
    return False, "never", "dumb"


def replay_trace(
    path: Path,
    *,
    width: int = 80,
    height: int = 24,
    mode: ReplayMode = "tty",
    playback_speed: float = 1.0,
    through_sequence: int | None = None,
) -> ReplayResult:
    """Replay sanitized lifecycle events with a virtual clock and no network calls."""

    if width <= 0 or height <= 0:
        raise ValueError("terminal dimensions must be positive")
    if playback_speed <= 0:
        raise ValueError("playback speed must be positive")
    records = _load_records(path)
    clock = FakeClock()
    tty, color, terminal_type = _mode_options(mode)
    scenario = TerminalScenario(
        width=width,
        height=height,
        tty=tty,
        color=color,
        clock=clock,
        terminal_type=terminal_type,
    ).start()
    safe_events: list[SanitizedEvent] = []
    last_monotonic_ms: int | None = None
    try:
        for fallback_sequence, record in enumerate(records, 1):
            sequence = record.get("sequence")
            safe_sequence = (
                sequence
                if isinstance(sequence, int) and not isinstance(sequence, bool)
                else fallback_sequence
            )
            if through_sequence is not None and safe_sequence > through_sequence:
                break
            monotonic_ms = record.get("monotonic_ms")
            if isinstance(monotonic_ms, int) and not isinstance(monotonic_ms, bool):
                if last_monotonic_ms is not None:
                    delta = max(0, monotonic_ms - last_monotonic_ms) / 1000 / playback_speed
                    scenario.advance(delta)
                last_monotonic_ms = monotonic_ms
            event = event_from_wal_record(record)
            if event is None:
                continue
            safe_events.append(sanitize_event(event, record))
            scenario.emit(event)
        if through_sequence is None:
            scenario.close()
        snapshot = scenario.checkpoint(
            "final" if through_sequence is None else f"sequence-{through_sequence}"
        )
    finally:
        scenario.close()
    return ReplayResult(snapshot, validate_trace(records), tuple(safe_events))


def _summary_json(summary: LifecycleSummary) -> dict[str, object]:
    return {
        "verdict": summary.verdict,
        "phase": summary.current_or_final_phase,
        "unmatched_starts": summary.unmatched_starts,
        "retries": summary.retries,
        "failures": summary.failures,
        "issues": [
            {"code": issue.code, "sequence": issue.sequence, "detail": issue.detail}
            for issue in summary.issues
        ],
    }


def export_failure_bundle(
    trace_path: Path,
    output_path: Path,
    *,
    width: int = 80,
    height: int = 24,
    mode: ReplayMode = "tty",
) -> Path:
    """Write a 0600 metadata-only bundle that is never exported automatically."""

    result = replay_trace(trace_path, width=width, height=height, mode=mode)
    payload = {
        "schema_version": 1,
        "source": trace_path.name,
        "terminal": {"width": width, "height": height, "mode": mode},
        "integrity": _summary_json(result.integrity),
        "events": [
            {**item.fields, "dropped_field_count": item.dropped_field_count}
            for item in result.events
        ],
        "screen": {
            "cells": tuple(_safe_text(line) for line in result.snapshot.cells),
            "cursor": result.snapshot.cursor,
            "cursor_visible": result.snapshot.cursor_visible,
            "scrollback": tuple(_safe_text(line) for line in result.snapshot.scrollback),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    output_path.chmod(0o600)
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and replay Warren terminal traces")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate a lifecycle trace")
    validate.add_argument("trace", type=Path)
    replay = subparsers.add_parser("replay", help="render a semantic terminal checkpoint")
    replay.add_argument("trace", type=Path)
    replay.add_argument("--width", type=int, default=80)
    replay.add_argument("--height", type=int, default=24)
    replay.add_argument("--mode", choices=("tty", "pipe", "no_color", "dumb"), default="tty")
    replay.add_argument("--speed", type=float, default=1.0)
    replay.add_argument("--sequence", type=int)
    bundle = subparsers.add_parser("bundle", help="write a private local failure bundle")
    bundle.add_argument("trace", type=Path)
    bundle.add_argument("output", type=Path)
    bundle.add_argument("--width", type=int, default=80)
    bundle.add_argument("--height", type=int, default=24)
    bundle.add_argument("--mode", choices=("tty", "pipe", "no_color", "dumb"), default="tty")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate":
        summary = validate_trace(_load_records(args.trace))
        print(json.dumps(_summary_json(summary), indent=2, sort_keys=True))
        return 0 if summary.verdict == "healthy" else 1
    if args.command == "replay":
        result = replay_trace(
            args.trace,
            width=args.width,
            height=args.height,
            mode=args.mode,
            playback_speed=args.speed,
            through_sequence=args.sequence,
        )
        print(
            json.dumps(
                {
                    "checkpoint": result.snapshot.name,
                    "cells": result.snapshot.cells,
                    "cursor": result.snapshot.cursor,
                    "cursor_visible": result.snapshot.cursor_visible,
                    "integrity": _summary_json(result.integrity),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    export_failure_bundle(
        args.trace,
        args.output,
        width=args.width,
        height=args.height,
        mode=args.mode,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
