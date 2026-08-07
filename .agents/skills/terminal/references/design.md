# Terminal design principles

## Reference patterns

- [Hermes Agent](https://github.com/NousResearch/hermes-agent/blob/main/cli.py#L11040-L11147):
  current tool activity is transient; completion with duration persists. Its source
  explicitly documents why transient-only tool output is a bug.
- [Aider streaming](https://github.com/Aider-AI/aider/blob/main/aider/mdstream.py#L92-L223):
  commit stable lines to scrollback and repaint only the unstable tail. Its
  [waiting UI](https://github.com/Aider-AI/aider/blob/main/aider/waiting.py#L23-L204)
  restores the cursor, truncates to width, and has plain/ASCII fallbacks.
- [DVC UI](https://github.com/treeverse/dvc/blob/main/dvc/ui/__init__.py): one facade
  coordinates stdout, stderr, status, prompts, progress-safe writes, and plain output.
- [Posting themes](https://github.com/darrenburns/posting/blob/main/src/posting/themes.py):
  semantic tokens and a navy surface ramp. Warren borrows the hierarchy, not full-screen
  painted backgrounds.
- [gptme states](https://github.com/gptme/gptme/blob/master/gptme/tui/app.py#L407-L449):
  thinking, generation, tool, interruption, and idle are explicit states.
- [Toolong progress](https://github.com/Textualize/toolong/blob/main/src/toolong/scan_progress_bar.py):
  use determinate progress only when a real total exists.

## Warren visual language

- Transcript-first, compact, calm, financially serious.
- Inherit the user's background and default foreground for body text.
- Use semantic theme roles shared by Rich and prompt-toolkit. Current navy roles live
  in `agent/terminal/renderer.py`; do not scatter new hex literals.
- Use bright blue for the prompt/current activity, muted blue for metadata/borders, and
  restrained amber/red only for warnings/errors. Success remains legible through `✓`
  and wording even in monochrome.
- Prefer whitespace and alignment. Reserve panels for a final investment summary or an
  actionable failure; reserve tables for genuinely comparable fields.
- Use one concise welcome line and group `/help` by intent: analyze, research, runs,
  holdings, session, reference.

## Activity language

Use active, safe, human labels:

```text
Starting Warren…
Preparing analysis…
Screening: 34/412 ACME
Using: Regulatory filing
✓ Regulatory filing  ·  1.4s
Analyzing AAPL…
■ Stopping… press Ctrl-C again to stop immediately
■ Interrupted · 7 tools · 99.9s · $0.0987 · Run …
```

Startup may be generic because schema/recovery details are internal. Once a request is
accepted, name the phase or curated tool without exposing inputs. Show a spinner for
unknown-duration work and `n/total` only for known screening totals.

## Anti-patterns

- No full-screen Textual rewrite, alternate-buffer takeover, or transcript repaint.
- No spinner that vanishes without durable tool/outcome history.
- No raw function payloads, model prompts, JSON dumps, schema revisions, or secret-like text.
- No nested Rich live contexts or competing prompt/output owners.
- No large banner, box around every line, rainbow semantics, or dark-only background panel.
- No generic `Loading…` throughout a run, silent provider wait, silent Ctrl-C, or swallowed error.
- No color-only status and no animation in redirected or dumb-terminal output.

## Review questions

1. Is something visible within 100 ms of accepting work?
2. Can the user tell whether Warren is preparing, screening, calling a tool, using the
   model, stopping, or done?
3. After live output clears, can scrollback explain every completed tool and outcome?
4. Does Ctrl-C visibly change state immediately and restore a clean prompt?
5. Does the same interaction remain understandable with ANSI removed and at 60 columns?
6. Could any visible text leak tool inputs/results, secrets, or hidden reasoning?
