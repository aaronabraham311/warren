---
name: streamlit
description: Use whenever a task changes Warren's dashboard, Streamlit pages or components, AppTest coverage, dashboard data access, or Streamlit runtime behavior.
---

# Warren Streamlit skill

How to build, test end-to-end, run, and debug Warren's Streamlit dashboard
(`dashboard/`). **Load this whenever the task touches Streamlit** — adding or editing
a `dashboard/` page/component, writing `AppTest` integration tests, launching the app,
seeding demo data, or debugging a "blank page / stale data / no reasoning trace" report.
It encodes the conventions and the specific gotchas that have already cost debugging
round-trips, so read it before touching `dashboard/`.

## How to invoke

Codex loads this skill automatically for matching dashboard tasks. You can also invoke it explicitly as `$streamlit`.

---

## Architecture — three layers, keep them separate

The dashboard is **strictly read-only**: it queries `warren.db` (via the ORM) and the
JSONL run logs, and **never** triggers an analysis or writes to the DB. Mirror the
existing structure so new pages slot in with zero rework:

- **`dashboard/data.py`** — pure data access over `storage.models` + JSONL logs. **No
  `streamlit` import here.** This is what unit/integration tests exercise directly.
- **`dashboard/components/*.py`** — reusable render functions (`st.*`). Take ORM rows /
  parsed events as input; don't query the DB inside a component.
- **`dashboard/pages/*.py`** — thin wiring: fetch via `data.py` → render via components.
- **`dashboard/app.py`** — the multi-page entrypoint (`st.navigation([st.Page(...)])`).

Hard rules learned the hard way:

1. **`st.set_page_config` is called exactly once, in `app.py` only.** Pages must not
   call it (Streamlit raises if it runs twice or after another `st.*` call).
2. **Each page + the entrypoint need a `sys.path` bootstrap** so `streamlit run
   dashboard/app.py` from the repo root can resolve `dashboard.*` / `storage.*`
   (Streamlit puts the entrypoint's own dir on `sys.path`, not the repo root):
   ```python
   import sys
   from pathlib import Path
   _REPO_ROOT = str(Path(__file__).resolve().parents[2])  # parent.parent in app.py
   if _REPO_ROOT not in sys.path:
       sys.path.insert(0, _REPO_ROOT)
   # then: import streamlit as st  # noqa: E402  (and the dashboard.* imports)
   ```
   Tests don't need it (pytest already has the repo root on the path), but real
   `streamlit run` does.
3. **Reuse `storage.engine.get_session`** (it honours `$WARREN_DB`); don't open a raw
   `sqlite3.connect("warren.db")`. Make the JSONL log dir configurable via
   `WARREN_LOGS_DIR` (default `logs/runs`) so tests can point it at a tmp dir.
4. **Nested expanders bottom out at two levels.** A card expander containing a
   "Reasoning trace" expander is fine; do **not** add a third expander inside that.
   Render detail with `st.json(...)`, `st.code(...)`, `st.caption(...)` instead.
5. **JSON columns come back already parsed** (the ORM `JSON` type returns Python
   lists/dicts) — don't `json.loads` them again. JSONL log lines *are* strings, so those
   you parse yourself.

---

## End-to-end testing with `streamlit.testing.v1.AppTest`

This is how we test pages headlessly — it runs the page script in-process and gives an
element tree to assert on. No browser, no server. Put tests next to the others in
`tests/` and rely on the same offline guards.

```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(str(PAGE_PATH)).run()   # PAGE_PATH absolute, via Path(__file__)
assert not at.exception                         # page rendered without raising
```

Wiring a temp DB + logs into a page run (file-backed, see the FK note below):

```python
@pytest.fixture()
def today_env(tmp_path, monkeypatch):
    db_path = tmp_path / "warren.db"
    logs_dir = tmp_path / "logs"; logs_dir.mkdir()
    monkeypatch.setenv("WARREN_DB", str(db_path))
    monkeypatch.setenv("WARREN_LOGS_DIR", str(logs_dir))
    monkeypatch.setattr(eng, "engine", None)    # force get_engine() to rebuild on the new path
    engine = eng.get_engine()
    Base.metadata.create_all(engine)
    yield SimpleNamespace(engine=engine, logs_dir=logs_dir)
    eng.engine = None                            # don't leak the engine into the next test
```

Element accessors that matter (Streamlit 1.58):

- `at.expander` → list of `Expander`; `.label` is the header, **`.proto.expanded`** is
  the open/closed flag (there is no `.expanded` attribute — that one bites).
- `at.markdown`, `at.header`, `at.caption`, `at.info`, `at.warning`, `at.json`,
  `at.code`, `at.metric`, `at.divider` → lists of the matching elements.
- Elements inside a **collapsed** expander still appear in the tree — `AppTest` builds
  the whole script's output regardless of UI expand state, so you can assert on trace
  contents without "opening" anything.

Gotchas that produce confusing failures:

- **File DB enforces foreign keys; the in-memory `db_engine` fixture does not.** Our
  engine attaches `PRAGMA foreign_keys=ON`, so when seeding a run + its analyses into a
  file DB, insert/`flush()` the `Run` row **before** the `Analysis` rows or you get
  `FOREIGN KEY constraint failed`. (The `db_engine` fixture skips the pragma, so the
  same code can pass there and fail under `AppTest` — test against the file DB.)
- **`:memory:` SQLite is per-connection.** Don't share an in-memory DB across the test
  thread and a page run — use a `tmp_path` file DB for `AppTest`.
- Map each ticket acceptance criterion to one assertion (sort order, auto-expand via
  `.proto.expanded`, ⚠️ badge in the expander label, empty-DB `st.info` + `st.stop`,
  full reasoning trace shows args + outputs).

Run just the dashboard tests fast: `uv run pytest tests/test_dashboard_*.py -q`.

---

## Running the app live + seeding demo data

For a live walkthrough you need data. If no real run exists (or you want a full spread
of buy/sell/hold + a ⚠️ card + complete traces), seed a demo run:

```bash
uv run python -m dashboard.seed_demo                    # -> $WARREN_DB / $WARREN_LOGS_DIR
uv run python -m dashboard.seed_demo --db demo.db --logs-dir demo_logs   # isolated demo
```

`seed_demo` is idempotent (merges a fixed run id) and writes a **sequential** JSONL
trace per ticker (`plan → get_quote → get_fundamentals → synthesise`) with real tool
args + outputs, so the reasoning trace renders fully.

Launch (background, fixed port, isolated demo data so you never clobber real `warren.db`):

```bash
export WARREN_DB=demo.db WARREN_LOGS_DIR=demo_logs
nohup uv run streamlit run dashboard/app.py \
  --server.port 8533 --server.headless true --browser.gatherUsageStats false \
  > streamlit.log 2>&1 &
```

Then drive it with the **Codex browser tooling** MCP (`navigate` → `computer` screenshot;
expand a card's "🔍 Reasoning trace" and an output `st.json` to confirm payloads). If
the Chrome extension isn't connected, fall back to `open http://localhost:8533`.

### Verify the UI in a browser — required for any page/widget you add or change

`AppTest` proves the *logic*; it does **not** prove the page actually renders or that a
widget is wired to the right behaviour. For every UI element you add or modify, drive it
with the browser automation available in the current Codex session and **screenshot each state** — don't just load the page
once and declare it done. Exercise each interactive control and confirm the result
changes as expected:

1. **Initial load** — screenshot the default state; confirm the result count, headers,
   and that no error/traceback banner is showing.
2. **Each input** — type into text inputs, add/remove `multiselect` chips, drag sliders,
   pick dates — and screenshot after each, asserting the visible output changed (e.g. the
   result count drops, rows narrow to the filtered set, a badge/label appears).
3. **Each expander/button/action** — click it and screenshot the opened/triggered state
   (e.g. a card's "🔍 Reasoning trace" expands to show the LLM-turn / tool-call steps).
4. **Edge states** — drive filters to an empty result and confirm the `st.info` +
   `st.stop` path renders, not a blank page or a stack trace.
5. **Console + server are clean** — `read_console_messages(onlyErrors=true)` and
   `grep -iE 'error|traceback|exception' streamlit.log` should both come back empty.

Save the key screenshots (`save_to_disk=true`) so the verification is reviewable.

**Gotcha — Streamlit reruns asynchronously, so the screenshot can race the rerun.**
A `multiselect` removal or a slider drag commits over the websocket and the page reruns a
beat later; a screenshot taken immediately can still show the *old* result count. If the
count looks wrong, take one more screenshot (or a no-op scroll) and re-check before
concluding the widget is broken — the value usually settles on the next frame. Sliders in
particular commit on handle *release* (use `left_click_drag`), not while dragging.

> Want true cross-browser / visual-regression e2e (real widget interactions, CSS, layout)?
> Streamlit renders an ordinary web app, so **Playwright works** — target the stable
> `data-testid` hooks (`stSidebar`, `stExpander`, `stMarkdownContainer`, …) and
> `get_by_text`/`get_by_role`, never the hashed CSS classes; wait on elements (or network
> idle), never fixed sleeps, because of the async reruns above. Streamlit's own repo uses
> Playwright for its e2e suite. We don't depend on it today — `AppTest` covers logic and
> the Codex browser tooling screenshot pass above covers rendering/interaction — so add
> Playwright only if we need automated visual regression in CI (it's a real dep: browser
> binaries, a running server, slower/flakier runs).

---

## Common debugging

**"No reasoning trace" / stale data / blank-looking page — check the server first, not
the code.** The usual cause is a zombie/stale Streamlit process, not a bug:

- **Killing `uv run streamlit ...` leaves the child `streamlit` holding the port.** The
  relaunch then logs `Port 8533 is not available`, picks another port (or fails), and
  you keep talking to the **old** server. Always kill by the listening PID:
  ```bash
  lsof -ti tcp:8533 | xargs kill -9        # not `kill <uv-wrapper-pid>`
  pkill -9 -f "streamlit run dashboard/app.py"   # belt-and-suspenders
  lsof -nP -iTCP:8533 -sTCP:LISTEN          # confirm 0 listeners before relaunch
  ```
- **Never delete/recreate the DB file under a running server.** Its pooled SQLite
  connection keeps the *deleted inode* open, so the page shows the old run (and, if its
  JSONL is gone, "No reasoning trace found"). Re-seed into the *same* file, or restart
  the server after swapping data.
- After changing data on disk, **the browser must re-run** — refresh the tab. Source
  edits trigger Streamlit's own rerun; data changes do not.
- To confirm it's environmental, bypass the UI and check the data layer directly:
  ```bash
  WARREN_LOGS_DIR=demo_logs uv run python -c \
    "from dashboard.data import read_reasoning_trace as r; print(len(r('demo-run','AAPL')))"
  ```
  If that prints events but the page says none, it's a stale server — restart cleanly.
- Inspect what Streamlit actually did: `grep -iE 'URL|Port|error' streamlit.log`.

---

## Definition of done (same bar as the rest of the repo)

```bash
uv run ruff check . && uv run ruff format . && uv run mypy . && uv run pytest -q
```

For a UI change, also do the **browser verification pass** above — launch the app and
screenshot every state of each control you added/changed (Codex browser tooling), since
`AppTest` covers logic but not rendering, layout, or that a widget is wired correctly.
Update `AGENTS.md` (directory map / commands / env vars) in the same change if structure
moved; preserve `CLAUDE.md` compatibility when applicable.
