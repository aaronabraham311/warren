# Terminal testing and E2E verification

## Test matrix

| Surface | Required checks |
|---|---|
| Renderer unit | safe sanitization; theme/forced color; `NO_COLOR`; immediate activity; `ok` and error tool rows; duration/cache/retry; live cleanup; cancelled footer |
| App unit | typed routing; structured help; styled prompt; missing-key path; first/second SIGINT; startup activity before recovery |
| Integration | piped transcript order; no prompt/ANSI when redirected; generic startup; no Alembic INFO; no runtime state for `--help` |
| Restart | settings/history/recent stored-run behavior across process restart |
| Real PTY | spinner/elapsed; durable tool order; cursor restoration; prompt after success/cancel; narrow width; user-visible color |

Run the focused automated surface:

```bash
uv run pytest -q \
  tests/test_terminal_renderer.py \
  tests/test_terminal_app.py \
  tests/test_terminal_integration.py \
  tests/test_terminal_restart_integration.py
```

## Deterministic startup PTY smoke

Run the bundled script from the repository root. It uses an isolated temporary DB,
logs, and state directory; it does not call the model or mutate user runtime data.

```bash
uv run python .agents/skills/terminal/scripts/pty_smoke.py
uv run python .agents/skills/terminal/scripts/pty_smoke.py --no-color --columns 60
```

This is deliberately a startup/help/quit smoke, not the full agent lifecycle test. It
must observe startup, welcome, prompt, structured help, clean exit, and absence of
Alembic/traceback output. With color enabled it requires ANSI; with `--no-color` it
rejects ANSI color SGR codes while permitting prompt-toolkit's cursor/redraw controls.

For a deterministic offline check of tool order, status `ok`, cancellation transitions,
and final summaries, run:

```bash
uv run pytest -q \
  tests/test_terminal_renderer.py \
  tests/test_terminal_app.py \
  tests/test_terminal_integration.py
```

Those tests inject executors and typed events, so they do not require credentials or
network access. They complement rather than replace the startup PTY smoke and the
manual provider-backed E2E below.

## Manual live-agent E2E

Use a fresh runtime root; never point experiments at the user's `warren.db` or logs.

```bash
terminal_e2e_dir=$(mktemp -d)
WARREN_DB="$terminal_e2e_dir/warren.db" \
WARREN_LOGS_DIR="$terminal_e2e_dir/logs/runs" \
WARREN_STATE_DIR="$terminal_e2e_dir/state" \
uv run warren
```

Exercise this sequence in a real PTY:

1. Confirm `Starting Warren…` appears before any wait and no migration revision appears.
2. Run `/help`; confirm grouped rows and a clean restored `warren ›` prompt.
3. Run `Analyze AAPL`. Confirm `Preparing analysis…` appears immediately.
4. Observe at least one tool: start is named, completion persists once with `✓`/`✗`
   and duration, and successful runtime status `ok` does not render as failure.
5. Press Ctrl-C during model/tool work. Confirm durable `Stopping…` guidance immediately.
   If the provider call is still blocking, press Ctrl-C again and confirm clean unwind.
6. Run `/trace` or `/history`; confirm durable state is queryable and no secret/raw
   payload appeared in ordinary scrollback.
7. Exit with `/quit`; confirm cursor, echo, and shell prompt are normal.

If credentials/network are unavailable, record that limitation and still run the
deterministic executor/event tests. Do not substitute a claim that live E2E passed.

## Plain and narrow checks

```bash
printf '/help\n/quit\n' | NO_COLOR=1 uv run warren
TERM=dumb NO_COLOR=1 uv run warren
```

For an interactive narrow check, run `stty cols 60` in a disposable PTY before launch.
Verify no command/status becomes an unreadable wrapped fragment. Restore terminal size
afterward.

## Independent critique prompt

For substantial UI work, give a fresh subagent only the branch/worktree and this task:

> Run `uv run warren` in a PTY. Exercise startup, help, one analysis if credentials
> permit, tool progress, Ctrl-C, return to prompt, narrow mode, and `NO_COLOR`. Do not
> edit files. Report verified defects with severity, reproduction, and a proposed fix;
> separate environment limitations from product bugs.

Iterate on verified P0/P1 findings, then rerun focused tests and the full definition of done.
