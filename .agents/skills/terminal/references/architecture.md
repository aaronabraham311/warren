# Terminal architecture

## Public flow

```text
prompt-toolkit input
  → agent.terminal.commands / agent.requests (deterministic parsing)
  → agent.terminal.app.TerminalApp (session and command routing)
  → agent.service.execute_run (shared batch + interactive orchestration)
  → RunLogger fsyncs JSONL
  → safe RunEvent projection
  → agent.terminal.renderer.TerminalRenderer
```

The interactive terminal was introduced by Codex task
`019fd537-7408-7532-b431-cd7de83730ca` as a client of the same service used by batch
and scheduled runs. Preserve that architecture: never shell from the terminal into
`agent.run`, and never move orchestration or data-source calls into `agent/terminal/`.

## Module ownership

- `agent/terminal/app.py`: REPL lifecycle, injected input/parser/executor seams, settings,
  slash-command dispatch, recent context, SIGINT coordination, startup recovery.
- `agent/terminal/commands.py`: typed, deterministic slash-command grammar and validation.
- `agent/terminal/completion.py`: local completion only; no network or database access.
- `agent/terminal/renderer.py`: the only Rich presentation/event-sink boundary. Owns
  TTY detection, theme, sanitization, live activity, tool transcript, results, and
  plain fallbacks.
- `agent/terminal/reliability.py`: injected monotonic clock and semantic VT
  checkpoints. It uses `pyte`; do not grow a Warren terminal
  emulator or infer state by scraping rendered text.
- `agent/activity.py`: renderer-independent activity state and reducer.
- `agent/terminal/health.py`: metadata-only active-run diagnostics, bounded metrics,
  and external-wait/renderer/agent/trace stall classification.
- `agent/terminal/queries.py`: read-only persisted run/trace/portfolio/watchlist views.
- `agent/terminal/settings.py`: versioned local terminal preferences and history paths.
- `agent/service.py`: migration/recovery, input sync, target selection, run locking,
  agent execution, persistence, and final typed `RunResult`.
- `agent/events.py`: redacted display-event contract. It must not contain prompt text,
  tool input/output, secrets, or hidden reasoning.
- `agent/lifecycle.py`: pure validator and derived trace summary for sequences,
  operation-parent links, paired outcomes, retries, failures, and slow operations.
- `storage/logger.py`: durable JSONL chokepoint. Display events derived from WAL records
  emit only after the record is flushed and fsynced. Every new record carries the
  supported schema version, run-local sequence, monotonic time, and operation linkage.
- `agent/cancellation.py` and `agent/locking.py`: shared cooperative cancellation and
  non-overlap contract for interactive, batch, and scheduled runs.

## Output contract

- stdout: durable analyses, comparisons, command results, final run summaries.
- stderr: transient activity and actionable diagnostics; completed tool rows are durable
  stderr transcript because they describe execution rather than the analysis payload.
- TTY: one Rich `Live`/`Progress` owner; printing through the same Console commits lines
  above it safely.
- non-TTY: newline-delimited deterministic text, no prompt bytes, ANSI, spinner frames,
  cursor controls, or full-screen behavior.

## Event/status gotchas

- `RunLogger.log_tool_call()` records successful tools with status `ok`, not `success`.
- `ToolCallStarted` is emitted directly before dispatch; `ToolCallCompleted` is projected
  from the durable WAL record.
- Event rendering is best effort and must never abort or roll back a run.
- `migrate(quiet=True)` suppresses normal Alembic INFO only. Migration failures remain
  exceptions and become concise terminal diagnostics.
- Cooperative cancellation cannot forcibly interrupt every blocking provider call.
  Make `Stopping…` durable immediately and tell the user the second Ctrl-C stops now.

## Safety boundary

Use human tool labels and aggregate metadata (status, cached, latency, retry count).
Sanitize all external text. Keep raw evidence in JSONL/sidecars and expose it through
the existing trace query, never the live status line.
