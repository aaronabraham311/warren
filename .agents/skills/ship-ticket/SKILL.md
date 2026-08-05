---
name: ship-ticket
description: Use when taking a Warren ticket from implementation through verification, commit, pull request, and CI follow-up.
---

# Warren ship-ticket skill

End-to-end autonomous flow for taking a ticket (usually a Notion task) all the way
to a green PR: **understand → plan (HTML, auto-opened) → implement → verify → PR →
watch CI → self-fix until green**. The goal is *minimal human involvement* — there is
exactly **one** approval gate (the HTML plan). Everything after it runs unattended.

## How to invoke

Codex loads this skill automatically for matching ticket implementation and delivery tasks. You can also invoke it explicitly as `$ship-ticket`.

---

## The three failure modes this skill exists to prevent

Read these first — they are the mistakes that cost human round-trips on past runs.

1. **HTML plan must exist *before* you ask for approval.** Do not request plan approval
   (or otherwise request sign-off) until the HTML plan is written to `local/` **and
   opened in the browser**. A plan the human can't see is not a plan they can approve.

2. **Always branch from a freshly-fetched `origin/main`, never from whatever HEAD you
   happen to be on.** Branching off a feature branch bundles its unmerged commits into
   your PR, which then **conflicts with `main`**. GitHub silently refuses to schedule
   the `pull_request` workflow while the merge commit is unresolvable — so CI shows
   *"no checks reported"* and looks stuck forever. See Phase 0 and Phase 6's diagnosis.

3. **Use an isolated Codex worktree, not `git checkout -b`.** Using `git checkout` switches the
   *shared* main checkout to your branch, stomping any other agent working there.
   An isolated worktree keeps the session's edits separate, so
   all tool calls (Read, Edit, Write, Bash) target your private tree. **Never skip this
   even for a "quick" ticket** — the main checkout is always someone else's floor.

---

## Phase 0 — Worktree setup (do this FIRST, always)

Each ticket gets its own isolated Git worktree. Start from the latest remote default branch; do not branch from the current feature branch or a dirty checkout:

```bash
git fetch origin main -q
git worktree add .worktrees/<slug> -b codex/<slug> origin/main
```

Continue all edits and commands from the new worktree. Never switch the shared checkout onto the ticket branch. If the Codex app provides a native isolated-worktree action, it is also acceptable, but verify that it is based on `origin/main`.

## Phase 1 — Understand the ticket & explore

1. Fetch the ticket with the Notion tool (`notion-fetch` — load via ToolSearch if
   deferred). Extract: goal, acceptance criteria, named files/modules, dependencies,
   and any code snippets/schemas the ticket dictates.
2. Explore the codebase for the patterns to mirror — prefer **one `Explore` subagent**
   for a focused sweep, or read the obvious sibling files directly (e.g. for a new
   `data_sources/*_client.py`, read `yfinance_client.py`, `edgar_client.py`,
   `cache.py`, `errors.py`, an existing test file, `conftest.py`, `pyproject.toml`).
3. Note reusable utilities so the plan says *reuse X*, not *write new X*.
4. **Resolve genuine forks with the user now** (e.g. "SDK vs raw HTTP?"), not
   later — one batched question up front beats interrupting mid-implementation.

---

## Phase 2 — HTML plan + auto-open + the one approval gate

1. Write a plan to **`local/<slug>-plan.html`** (the `local/` dir is gitignored scratch).
   Use the template at the bottom of this file. It must cover: **Context** (why),
   **Reuse** (existing utils to lean on), **Files** (new/edit, with code sketches),
   **Acceptance criteria → tests**, and **Verification** commands.
2. **Open it in the default browser immediately:**
   ```bash
   open "local/<slug>-plan.html"            # macOS
   # xdg-open on Linux, start "" on Windows
   ```
3. Also write the same content to the plan-mode plan file if in plan mode.
4. Present a short text summary in chat, then request approval:
   - In plan mode: request plan approval only now — the HTML exists and is open.
   - Not in plan mode: ask once, concisely, for go/no-go.
   - With `--no-gate`: skip approval and proceed directly to Phase 3.
5. Incorporate any feedback into the HTML before continuing.

---

## Phase 3 — Implement

- Follow `AGENTS.md` conventions exactly (empty `__init__.py`, import from submodules,
  SQLAlchemy 2.x style, all external calls in `data_sources/`, `os.environ[...]` not
  repeated `load_dotenv()`, no committing `.env`/`warren.db`/`logs/`).
- Mirror the sibling files found in Phase 1 (error handling, typing, `_sleep`
  injection for testability, `source: Literal[...]`, object→scalar helpers to keep
  mypy strict-clean — no explicit `Any`).
- Add the new module to the `[[tool.mypy.overrides]]` list in `pyproject.toml` if it
  defines pydantic `BaseModel`s (known explicit-Any false positive).
- Add new deps with `uv add <pkg>` (updates `pyproject.toml` + `uv.lock`).
- Write tests that map 1:1 to the ticket's acceptance criteria; mock all network.
  Add fixtures to `tests/conftest.py` next to the existing ones.
- Update `AGENTS.md` (directory map, conventions) in the same change if structure changed; preserve `CLAUDE.md` compatibility when applicable.

---

## Phase 4 — Definition of Done (must all pass before pushing)

```bash
uv run ruff check . && uv run ruff format . && uv run mypy . && uv run pytest -q
```

Fix anything red and re-run until clean. Do not proceed to Phase 5 with a failing DoD.

---

## Phase 5 — Commit, push, open PR

1. **Verify the diff is scoped to this ticket** (catches a wrong-base branch early; run
   from `<worktree>`):
   ```bash
   git diff --name-only origin/main...HEAD
   ```
   If it lists files unrelated to the ticket, you branched off the wrong base — go fix
   it per Phase 6's rebase recipe **before** pushing.
2. Commit (end the message with the trailer):
   ```
   Co-Authored-By: Codex <noreply@openai.com>
   ```
3. `git push -u origin <branch>`.
4. `gh pr create` with a **descriptive** body: What & why, design decisions (and any
   user-choice outcomes), the public API, an acceptance-criteria checklist, and
   the DoD result. End PR bodies with:
   ```
   🤖 Generated with Codex
   ```

---

## Phase 6 — Watch CI and self-fix until green

After opening the PR, arm a recurring Codex monitor (or the app's monitoring/automation mechanism) to poll the PR every minute so this survives idle turns. The monitor should:

1. Run `gh pr checks <N>` and report one compact line while checks are pending.
2. If a check fails, run `gh run view --log-failed`, diagnose and fix the issue, rerun the DoD locally, commit, and push.
3. Stop itself when all checks pass or no further action remains.

Each tick:

1. `gh pr checks <N>`.
2. **If it says *"no checks reported"* — do not just wait. Diagnose:**
   ```bash
   gh pr view <N> --json mergeable,headRefOid
   ```
   - `mergeable: "CONFLICTING"` → the merge commit can't be computed, so CI never
     schedules. **Rebase onto main** (this is the wrong-base / unmerged-dependency case):
     ```bash
     git fetch origin main -q
     git rebase --onto origin/main <first-commit-NOT-yours>   # drops bundled commits
     # or, if your work is one clean commit on a stale base: git rebase origin/main
     ```
     Resolve conflicts, re-run the DoD, `git push --force-with-lease`, re-check mergeable.
   - `mergeable: "MERGEABLE"` but still no run after a minute → confirm Actions is on
     (`gh api repos/<owner>/<repo>/actions/permissions`) and that the workflow triggers
     on `pull_request`; otherwise nudge with an empty commit.
3. **If a check fails:** `gh run view --log-failed`, fix the code, re-run the full DoD
   locally, `git push`. Let the next tick re-verify.
4. **When all checks pass — verify mergeability before declaring done:**
   ```bash
   gh pr view <N> --json mergeable,mergeStateStatus
   ```
   - `"mergeable":"MERGEABLE"` → proceed to step 5.
   - `"mergeable":"CONFLICTING"` → CI passed but the branch can't merge cleanly. This
     means something merged into main after your branch was cut. Diagnose with:
     ```bash
     git log --oneline origin/main...HEAD
     ```
     Commits that aren't yours were bundled — your branch was based on a stale
     `origin/main`. Fix with a rebase:
     ```bash
     git fetch origin main -q
     git rebase origin/main
     ```
     **Resolving conflicts when both sides are additive** (the common case — two
     independent tickets each added to the same file, e.g. `CLAUDE.md`, `pyproject.toml`,
     a test registry set): keep *both* sides. Accept all changes from `HEAD` (theirs,
     which is `origin/main`) and then re-add your additions below them. Do **not** discard
     either side. After resolving:
     ```bash
     git add <conflicted-files>
     git rebase --continue
     ```
     Re-run the full DoD (`ruff check . && ruff format . && mypy . && pytest`), then:
     ```bash
     git push --force-with-lease
     ```
     Wait one minute and re-check `gh pr view <N> --json mergeable,mergeStateStatus`
     before proceeding.
5. **Confirmed MERGEABLE — wrap up:** `MonitorDelete` the job, clean up the worktree, and
   post a one-line ✅ summary:
   ```bash
   # Run from the main repo (../warren), not the worktree
   git worktree remove ../warren-<slug>
   git branch -d <type>/<slug>   # safe: branch is merged / on remote
   ```
   If `git branch -d` refuses (unmerged locally), use `-D` only after confirming the PR
   is merged on GitHub.

Only involve the human if you hit something you genuinely cannot resolve (e.g. a
required secret is missing, or the fix needs a product decision) — otherwise drive it
to green autonomously.

---

## HTML plan template

Dark-themed, scannable, renders standalone. Fill the `<!-- … -->` slots.

```html
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title><!-- TICKET ID --> — Implementation Plan</title>
<style>
  :root{--bg:#0f1115;--panel:#171a21;--ink:#e6e9ef;--muted:#9aa3b2;--accent:#4f9cf9;
        --green:#46c08a;--amber:#e0a13b;--border:#262b35;--code:#0b0d11;}
  *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  .wrap{max-width:920px;margin:0 auto;padding:40px 24px 80px}
  h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:36px 0 12px;
    padding-bottom:6px;border-bottom:1px solid var(--border)}
  h3{font-size:16px;margin:22px 0 8px;color:var(--accent)}.sub{color:var(--muted);margin:0 0 24px}
  .panel{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:16px 20px;margin:14px 0}
  code{background:var(--code);padding:1px 6px;border-radius:4px;
    font-family:ui-monospace,Menlo,monospace;font-size:13px}
  pre{background:var(--code);border:1px solid var(--border);border-radius:8px;
    padding:14px 16px;overflow-x:auto;font-size:13px;line-height:1.5}pre code{background:none;padding:0}
  .pill{display:inline-block;font-size:11px;font-weight:600;padding:2px 9px;border-radius:999px}
  .pill.new{background:rgba(70,192,138,.15);color:var(--green);border:1px solid rgba(70,192,138,.4)}
  .pill.edit{background:rgba(224,161,59,.15);color:var(--amber);border:1px solid rgba(224,161,59,.4)}
  table{width:100%;border-collapse:collapse;margin:10px 0;font-size:14px}
  th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--border);vertical-align:top}
  th{color:var(--muted)}.muted{color:var(--muted)}.note{border-left:3px solid var(--accent);
    padding-left:14px;color:var(--muted)}
  .check{list-style:none;padding-left:0}.check li::before{content:"\2610 ";color:var(--accent);font-weight:700}
</style></head><body><div class="wrap">
  <h1><!-- Ticket title --></h1>
  <p class="sub"><!-- one-line subtitle · branch name --></p>
  <h2>Context</h2><div class="panel"><!-- why this change; problem & intended outcome --></div>
  <h2>Reuse — don't reinvent</h2>
  <table><tr><th>Existing</th><th>What it gives us</th></tr>
    <!-- <tr><td>path</td><td>…</td></tr> --></table>
  <h2>Files</h2>
  <h3><span class="pill new">NEW</span> &nbsp;<!-- path --></h3>
  <pre><code><!-- code sketch --></code></pre>
  <h3><span class="pill edit">EDIT</span> &nbsp;<!-- path --></h3>
  <ul><!-- bullet edits --></ul>
  <h2>Acceptance criteria &rarr; tests</h2>
  <ul class="check"><!-- <li>criterion</li> --></ul>
  <h2>Verification</h2>
  <pre><code>uv run ruff check . &amp;&amp; uv run ruff format --check . &amp;&amp; uv run mypy . &amp;&amp; uv run pytest</code></pre>
</div></body></html>
```
