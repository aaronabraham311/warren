---
name: terminal
description: Use whenever a task changes Warren's interactive terminal, `uv run warren`, `agent/terminal/`, slash commands, Rich rendering or themes, prompt-toolkit behavior, run progress/tool visibility, cancellation, terminal settings, or terminal unit/integration/E2E tests.
---

# Warren terminal skill

Build and verify Warren's transcript-first Rich + prompt-toolkit interface. Load this
before changing terminal behavior; visual polish, event safety, batch compatibility,
and real PTY behavior are one contract.

## Start here

1. Inspect `git status --short --branch` and preserve runtime artifacts.
2. Read [references/architecture.md](references/architecture.md) for service/event
   boundaries before changing control flow, startup, tools, or cancellation.
3. Read [references/design.md](references/design.md) for any visual, progress, prompt,
   help, result, or tool-transcript change.
4. Read [references/testing.md](references/testing.md) before implementing so the
   acceptance criteria include TTY, plain, narrow, and interruption paths.
5. Keep the change in the correct gh-stack layer. Put reusable service/event changes
   below presentation consumers; keep renderer-only changes in the terminal layer.

## Non-negotiable behavior

- Preserve scrollback. Use one transient live region for current activity; commit
  completed tools and final outcomes exactly once as durable lines.
- Show life immediately, before migration, input sync, provider construction, model
  calls, or tools can block. Startup is generic; active analysis names safe phases.
- Treat runtime tool status `ok` as success. Do not invent a second status vocabulary.
- Keep stdout for durable analysis results and stderr for progress/diagnostics so
  piping remains predictable.
- Never display prompts, hidden reasoning, secrets, raw tool arguments, or raw tool
  results. The safe typed `RunEvent` stream drives UI; `/trace` is the detail surface.
- Centralize semantic styles. Inherit the terminal background, use navy/blue hierarchy
  for brand and activity, and retain words/glyphs when hue is unavailable.
- Honor `NO_COLOR`, redirected output, `TERM=dumb`, animation settings, narrow widths,
  and cursor restoration. Never make ANSI or Unicode a correctness dependency.
- First Ctrl-C must immediately say `Stopping…`; always close live output in `finally`.
  A second Ctrl-C may unwind immediately. Persist the eventual interruption outcome.
- Keep startup concise and `/help` structured by user intent. Do not print a command
  wall, migration revisions, or internal logger output.
- Do not convert Warren into a full-screen Textual app without an explicit product
  decision. The REPL, shell scrollback, redirection, and restart continuity are features.

## Implementation workflow

1. Reproduce in a real PTY and capture the exact transcript/state that is wrong.
2. Trace input → `TerminalApp` → `execute_run` → `RunLogger`/`RunEvent` →
   `TerminalRenderer`. Fix the earliest correct boundary, not a downstream string hack.
3. Map every visible state: startup, preparing, screening, ticker, tool start, tool
   completion, model, success, error, stopping, cancelled.
4. Add focused tests first. Use `StringIO` for deterministic transcripts and a fake TTY
   stream for Rich live cleanup/cursor behavior. For viewport regressions, drive
   `TerminalScenario` with a `FakeClock` and assert the normalized `pyte` cell grid.
5. Run the smoke script and manual live checks in [references/testing.md](references/testing.md).
6. For a substantive UX change, ask an independent subagent (when available) to use
   the real terminal and critique it without seeing the intended fix. Iterate on verified
   P0/P1 findings; mark provider/network limitations explicitly.
7. Run Warren's full definition of done before committing.

## Fast verification

```bash
uv run pytest -q tests/test_terminal_renderer.py tests/test_terminal_app.py \
  tests/test_terminal_integration.py tests/test_terminal_restart_integration.py \
  tests/test_terminal_reliability.py
uv run python .agents/skills/terminal/scripts/pty_smoke.py
uv run python .agents/skills/terminal/scripts/pty_smoke.py --no-color --columns 60
uv run ruff check . && uv run ruff format --check . && uv run mypy . && uv run pytest -q
```

Do not claim the UI is finished from unit tests alone. A real PTY pass is required.
