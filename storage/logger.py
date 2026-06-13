"""Per-run JSONL event logging — the durable write-ahead log (WAL) for a run.

The ``logs/runs/{run_id}.jsonl`` trace is the source of truth: one JSON object per
line, ``flush()``-ed and ``os.fsync()``-ed immediately so a crash can never leave a
partial line. The ``runs`` and ``tool_calls`` SQLite tables are a *derived projection*
of this trace, rebuilt by :func:`storage.recovery.reconcile_run` (via
:meth:`RunLogger.flush_to_db`) — never written incrementally during the loop.

Wired event types (single-ticker loop): ``run_started``, ``ticker_started``,
``llm_call``, ``tool_call``, ``ticker_completed``, ``run_completed``. The generic
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
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from sqlalchemy.orm import Session

from storage.cost import compute_cost
from storage.engine import truncate_tool_output


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class RunLogger:
    def __init__(self, run_id: str, log_dir: Path) -> None:
        self.run_id = run_id
        self.log_dir = log_dir
        self.path = log_dir / f"{run_id}.jsonl"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._tool_seq = 0

    def log(self, event: str, **fields: object) -> None:
        """Append one JSON line atomically: ts + run_id + event + fields, then fsync."""
        record: dict[str, object] = {"ts": _utc_now_iso(), "run_id": self.run_id, "event": event}
        record.update(fields)
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def log_llm_call(
        self,
        response: anthropic.types.Message,
        *,
        ticker: str | None,
        phase: str,
        model: str,
        latency_ms: int,
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
    ) -> None:
        # Reuse the engine's >8 KB sidecar truncation so the JSONL line stays small
        # while the full payload remains recoverable from logs/runs/{run_id}/tool_outputs/.
        stored_output = truncate_tool_output(output, self.run_id, self._tool_seq)
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
        )

    def flush_to_db(self, session: Session) -> None:
        """Reconcile this run's trace into the runs + tool_calls tables (idempotent)."""
        from storage.recovery import reconcile_run

        reconcile_run(session, self.run_id, self.path)

    def close(self) -> None:
        self._fh.close()
