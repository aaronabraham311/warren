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

from agent.providers.base import ProviderResponse
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
        response: ProviderResponse | anthropic.types.Message,
        *,
        ticker: str | None,
        phase: str,
        model: str | None = None,
        latency_ms: int,
        provider: str = "anthropic",
        service_tier: str = "default",
        reasoning_effort: str = "none",
    ) -> None:
        if isinstance(response, ProviderResponse):
            normalized_usage = response.usage
            response_model = response.model_id
            resolved_model = model or response_model
            input_tokens = normalized_usage.input_tokens
            output_tokens = normalized_usage.output_tokens
            cache_read = normalized_usage.cache_read_tokens
            cache_creation = normalized_usage.cache_write_tokens
            reasoning_tokens = normalized_usage.reasoning_tokens
            tool_use_tokens = normalized_usage.tool_use_tokens
            total_tokens = normalized_usage.total_tokens
            raw_usage = normalized_usage.raw
        else:
            anthropic_usage = response.usage
            resolved_model = model or response.model
            response_model = response.model
            input_tokens = anthropic_usage.input_tokens
            output_tokens = anthropic_usage.output_tokens
            cache_read = anthropic_usage.cache_read_input_tokens or 0
            cache_creation = anthropic_usage.cache_creation_input_tokens or 0
            reasoning_tokens = None
            tool_use_tokens = 0
            total_tokens = input_tokens + cache_read + cache_creation + output_tokens
            raw_usage = anthropic_usage.model_dump(mode="json")

        cost_usd = compute_cost(
            resolved_model,
            input_tokens=input_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
            output_tokens=output_tokens,
            provider=provider,
            service_tier=service_tier,
        )
        self.log(
            "llm_call",
            ticker=ticker,
            phase=phase,
            provider=provider,
            model=resolved_model,
            response_model=response_model,
            service_tier=service_tier,
            reasoning_effort=reasoning_effort,
            input_tokens=input_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            tool_use_tokens=tool_use_tokens,
            total_tokens=total_tokens,
            raw_usage=raw_usage,
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
        retry_count: int = 0,
        last_retry_error: str | None = None,
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
            retry_count=retry_count,
            last_retry_error=last_retry_error,
        )

    def flush_to_db(self, session: Session) -> None:
        """Reconcile this run's trace into the runs + tool_calls tables (idempotent)."""
        from storage.recovery import reconcile_run

        reconcile_run(session, self.run_id, self.path)

    def close(self) -> None:
        self._fh.close()
