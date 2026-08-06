"""Reconcile JSONL run traces into the SQLite projection.

:func:`reconcile_run` rebuilds the ``runs`` aggregate row and ``tool_calls`` rows for a
single run from its ``logs/runs/{run_id}.jsonl`` trace. It is idempotent
(delete-then-insert, like ``upsert_analysis``), so it is safe to replay — this is what
makes the DB a disposable, rebuildable cache.

:func:`reconcile_orphans` is the startup recovery sweep: any run left ``status="running"``
with a trace on disk (a crashed or never-flushed run) is reconciled into the DB.
"""

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from storage.engine import get_session
from storage.models import Run, RunStatus, ToolCall


def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _output_file_path(stored_output: str | None) -> str | None:
    """If a tool_call output is a truncation marker, return the sidecar file path."""
    if not stored_output:
        return None
    try:
        marker = json.loads(stored_output)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(marker, dict) and marker.get("truncated") is True:
        path = marker.get("path")
        return path if isinstance(path, str) else None
    return None


def _read_events(jsonl_path: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for raw_line in jsonl_path.read_bytes().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            # A hard crash may leave the final record incomplete or mid-codepoint.
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def reconcile_run(session: Session, run_id: str, jsonl_path: Path) -> None:
    """Rebuild the runs row + tool_calls rows for ``run_id`` from its trace, then commit.

    Idempotent: existing tool_calls for the run are deleted and re-inserted.
    """
    events = _read_events(jsonl_path)

    llm_calls = [e for e in events if e.get("event") == "llm_call"]
    tool_calls = [e for e in events if e.get("event") == "tool_call"]
    started = next((e for e in events if e.get("event") == "run_started"), None)
    completed = next((e for e in events if e.get("event") == "run_completed"), None)

    total_input = sum(_as_int(e.get("input_tokens")) or 0 for e in llm_calls)
    total_output = sum(_as_int(e.get("output_tokens")) or 0 for e in llm_calls)
    total_cost = sum(_as_float(e.get("cost_usd")) for e in llm_calls)

    # A trace with no run_completed event is a crashed/incomplete run.
    status: RunStatus = "failed"
    if completed is not None:
        completed_status = _as_str(completed.get("status"))
        if completed_status in ("success", "cost_aborted", "failed", "running", "cancelled"):
            status = completed_status  # type: ignore[assignment]  # narrowed above

    run = session.get(Run, run_id)
    if run is None:
        run = Run(id=run_id)
        session.add(run)
    run.status = status
    run.total_input_tokens = total_input
    run.total_output_tokens = total_output
    run.total_cost_usd = total_cost
    run.num_tool_calls = len(tool_calls)
    if started is not None:
        run.started_at = _parse_ts(started.get("ts"))
    if completed is not None:
        run.completed_at = _parse_ts(completed.get("ts"))
        run.error_msg = _as_str(completed.get("error_msg"))
    else:
        run.completed_at = None
        run.error_msg = "Run did not complete (no run_completed event in trace)"

    session.execute(delete(ToolCall).where(ToolCall.run_id == run_id))
    for e in tool_calls:
        stored_output = _as_str(e.get("output"))
        tool_input = e.get("input")
        session.add(
            ToolCall(
                run_id=run_id,
                tool_name=_as_str(e.get("tool")),
                input_json=json.dumps(tool_input) if tool_input is not None else None,
                output_json=stored_output,
                output_file_path=_output_file_path(stored_output),
                latency_ms=_as_int(e.get("latency_ms")),
                cached=bool(e.get("cached")),
                error_msg=_as_str(e.get("error_msg")),
                created_at=_parse_ts(e.get("ts")),
            )
        )
    session.commit()


def reconcile_orphans(log_dir: Path = Path("logs/runs")) -> int:
    """Reconcile every run left ``status="running"`` that has a trace on disk.

    Returns the number of runs reconciled. Idempotent and safe to call on every startup.
    """
    reconciled = 0
    with get_session() as session:
        orphan_ids = list(session.scalars(select(Run.id).where(Run.status == "running")))
        for run_id in orphan_ids:
            jsonl_path = log_dir / f"{run_id}.jsonl"
            if jsonl_path.exists():
                reconcile_run(session, run_id, jsonl_path)
                reconciled += 1
    return reconciled
