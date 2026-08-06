"""Per-run JSONL event logging — the durable write-ahead log (WAL) for a run.

The ``logs/runs/{run_id}.jsonl`` trace is the source of truth: one JSON object per
line, ``flush()``-ed and ``os.fsync()``-ed immediately so a crash can never leave a
partial line. The ``runs`` and ``tool_calls`` SQLite tables are a *derived projection*
of this trace, rebuilt by :func:`storage.recovery.reconcile_run` (via
:meth:`RunLogger.flush_to_db`) — never written incrementally during the loop.

Wired event types (single-ticker loop): ``run_started``, ``ticker_started``,
``llm_call_started``, ``llm_call``, ``tool_call``, ``ticker_completed``,
``run_completed``. The generic
:meth:`log` also supports ``phase_started`` / ``phase_completed`` for the future
screening orchestrator.

Most useful debug query — every ticker where the agent immediately reached for a
different tool after the first one (signal that it didn't trust the first answer)::

    cat logs/runs/*.jsonl \\
      | jq -c 'select(.event=="tool_call") | {ticker, tool}' \\
      | awk -F'"' '{print $4}' | uniq -c
"""

import json
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import anthropic
from sqlalchemy.orm import Session

from storage.cost import compute_cost
from storage.engine import truncate_tool_output

if TYPE_CHECKING:
    from agent.events import EventSink, LlmCallPurpose


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


TRACE_SCHEMA_VERSION = 1
_START_KIND = {
    "run_started": "run",
    "phase_started": "phase",
    "ticker_started": "ticker",
    "llm_call_started": "model",
    "tool_call_started": "tool",
}
_OUTCOME_KIND = {
    "run_completed": "run",
    "run_failed": "run",
    "run_cancelled": "run",
    "phase_completed": "phase",
    "phase_failed": "phase",
    "ticker_completed": "ticker",
    "ticker_failed": "ticker",
    "llm_call": "model",
    "llm_call_failed": "model",
    "tool_call": "tool",
}


class RunLogger:
    def __init__(
        self,
        run_id: str,
        log_dir: Path,
        event_sink: "EventSink | None" = None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.run_id = run_id
        self.log_dir = log_dir
        self.path = log_dir / f"{run_id}.jsonl"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._tool_seq = 0
        self._event_sink = event_sink
        self._monotonic = monotonic
        self._sequence = self._last_sequence()
        self._open_operations: dict[tuple[str, str | None, str | None], list[str]] = {}

    def _last_sequence(self) -> int:
        """Continue a run-local sequence when a recovered process appends a trace."""

        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return 0
        for line in reversed(lines):
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(record, dict):
                sequence = record.get("sequence")
                if isinstance(sequence, int) and not isinstance(sequence, bool):
                    return sequence
        return 0

    @staticmethod
    def _operation_key(
        kind: str,
        event: str,
        fields: dict[str, object],
    ) -> tuple[str, str | None, str | None]:
        ticker = fields.get("ticker")
        safe_ticker = ticker if isinstance(ticker, str) else None
        if kind == "phase":
            phase = fields.get("phase")
            return kind, safe_ticker, phase if isinstance(phase, str) else None
        if kind == "ticker":
            return kind, safe_ticker, None
        if kind == "model":
            model = fields.get("model")
            purpose = fields.get("purpose")
            iteration = fields.get("iteration")
            identity = f"{model}:{purpose}:{iteration}"
            return kind, safe_ticker, identity
        if kind == "tool":
            tool = fields.get("tool")
            return kind, safe_ticker, tool if isinstance(tool, str) else None
        del event
        return kind, None, None

    def _parent_operation(self, ticker: str | None) -> str | None:
        for key in (("ticker", ticker, None), ("run", None, None)):
            active = self._open_operations.get(key)
            if active:
                return active[-1]
        return None

    def log(self, event: str, **fields: object) -> None:
        """Append one versioned, sequenced JSON line and fsync it before fan-out."""

        self._sequence += 1
        record = dict(fields)
        start_kind = _START_KIND.get(event)
        outcome_kind = _OUTCOME_KIND.get(event)
        if start_kind is not None:
            key = self._operation_key(start_kind, event, record)
            operation_id = record.get("operation_id")
            if not isinstance(operation_id, str):
                operation_id = f"{self.run_id}:{self._sequence}"
            record["operation_id"] = operation_id
            ticker = record.get("ticker")
            parent = self._parent_operation(ticker if isinstance(ticker, str) else None)
            if parent is not None and "parent_operation_id" not in record:
                record["parent_operation_id"] = parent
            self._open_operations.setdefault(key, []).append(operation_id)
        elif outcome_kind is not None:
            key = self._operation_key(outcome_kind, event, record)
            active = self._open_operations.get(key)
            if active:
                record["operation_id"] = active.pop()
            record["outcome"] = (
                "failed"
                if event.endswith("failed") or record.get("status") == "error"
                else "cancelled"
                if event.endswith("cancelled") or record.get("status") == "cancelled"
                else "completed"
            )
        record.update(
            {
                "schema_version": TRACE_SCHEMA_VERSION,
                "sequence": self._sequence,
                "ts": _utc_now_iso(),
                "monotonic_ms": int(self._monotonic() * 1000),
                "run_id": self.run_id,
                "event": event,
            }
        )
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._emit_display_event(record)

    def log_tool_started(self, *, ticker: str | None, tool_name: str) -> None:
        """Persist a safe tool start before validation or dispatch can block/fail."""

        self.log("tool_call_started", ticker=ticker, tool=tool_name)

    def _emit_display_event(self, record: dict[str, object]) -> None:
        """Fan out a redacted event only after its authoritative WAL record is durable."""
        if self._event_sink is None:
            return
        from agent.events import event_from_wal_record

        display_event = event_from_wal_record(record)
        if display_event is None:
            return
        try:
            self._event_sink.emit(display_event)
        except Exception as exc:
            # Rendering is subordinate to persistence and must never fail a run.
            print(
                f"[warren] display event sink failed ({type(exc).__name__})",
                file=sys.stderr,
            )

    def log_llm_call(
        self,
        response: anthropic.types.Message,
        *,
        ticker: str | None,
        phase: str,
        model: str,
        latency_ms: int,
        purpose: "LlmCallPurpose" = "synthesis",
        iteration: int = 0,
    ) -> None:
        usage = response.usage
        cache_read = usage.cache_read_input_tokens or 0
        cache_creation = usage.cache_creation_input_tokens or 0
        cost_usd = compute_cost(
            model,
            input_tokens=usage.input_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
            output_tokens=usage.output_tokens,
        )
        self.log(
            "llm_call",
            ticker=ticker,
            phase=phase,
            model=model,
            input_tokens=usage.input_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
            output_tokens=usage.output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            purpose=purpose,
            iteration=iteration,
        )

    def log_tool_call(
        self,
        *,
        tool_name: str,
        tool_input: dict[str, object],
        output: str,
        cached: bool,
        latency_ms: int,
        status: str,
        ticker: str | None,
        error_msg: str | None,
        retry_count: int = 0,
        last_retry_error: str | None = None,
    ) -> None:
        # Reuse the engine's >8 KB sidecar truncation so the JSONL line stays small
        # while the full payload remains recoverable from logs/runs/{run_id}/tool_outputs/.
        stored_output = truncate_tool_output(
            output,
            self.run_id,
            self._tool_seq,
            base_dir=self.log_dir,
        )
        self._tool_seq += 1
        self.log(
            "tool_call",
            ticker=ticker,
            tool=tool_name,
            input=tool_input,
            output=stored_output,
            cached=cached,
            latency_ms=latency_ms,
            status=status,
            error_msg=error_msg,
            retry_count=retry_count,
            last_retry_error=last_retry_error,
        )

    def flush_to_db(self, session: Session) -> None:
        """Reconcile this run's trace into the runs + tool_calls tables (idempotent)."""
        from storage.recovery import reconcile_run

        reconcile_run(session, self.run_id, self.path)

    def close(self) -> None:
        self._fh.close()
