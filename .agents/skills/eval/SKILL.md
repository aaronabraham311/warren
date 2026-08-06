---
name: eval
description: Use whenever a task touches Warren's eval harness, golden-set replay, graders, tool fixtures, eval analysis CLIs, or python -m agent.eval.
---

# Warren eval skill

How the eval replay harness works, how to run it, how to record the fixtures it needs, and
how to add a new check. **Load this whenever the task touches `eval/`** — the runner, the
grader, the golden set, tool fixtures, the `eval/analysis/` debugging CLIs, or the
`uv run python -m agent.eval` command. It encodes the determinism invariants that are easy to break
silently: a broken one doesn't crash, it just makes the eval stop detecting regressions.

## How to invoke

Codex loads this skill automatically for matching `eval/` tasks. You can also invoke it explicitly as `$eval`.

---

## What the command does

```bash
uv run python -m agent.eval --golden-set                              # grade every golden example
uv run python -m agent.eval --golden-set --output runs/eval-2026-05-10.json
uv run python -m agent.eval --golden-set --eval-run-id eval-baseline  # pin the id so two runs diff
```

`uv run python -m eval.runner` is the same entrypoint; `agent/eval.py` is a shim so the command
reads the way the ticket specified. It exits **1** if any example failed, so CI can gate on it.

For each `eval/examples/{ticker}.yaml` the runner replays `analyze_ticker` against recorded
tool outputs, grades the result into an `EvalGrade`, writes one `eval_runs` row, and prints a
per-ticker line. It never triggers a real analysis run and never writes `analyses`.
Any fixture miss or invalid evidence fixture is a mandatory run-invalidating failure.

---

## The three legs of determinism

Every one of these is load-bearing. Breaking one degrades the eval into noise.

**1. Fixture replay — offline by construction.**
`eval.tool_fixtures.FixtureToolRunner` implements the `agent.loop.ToolRunner` protocol and is
injected into `analyze_ticker(tool_runner=...)`. It reads
`eval/fixtures/{TICKER}/tools/{tool_name}/{input_hash}.json` and **never calls `tool.run()`**,
so no `data_sources` client is ever constructed. The network is unreachable because nothing
that could reach it is instantiated — *not* because a socket guard blocks it. Don't "fix" a
missing fixture by falling back to the live tool; that silently reintroduces network variance.

**2. `temperature=0`.**
Threaded `analyze_ticker` → `_call_and_record` → `call_claude_with_caching` →
`build_claude_request`. When `None` it is omitted from the request entirely (via
`anthropic.omit`), so the nightly run keeps the SDK default. Only the eval passes `0.0`.
This gets keyword-presence stability, not bit-exactness — grade on membership, never equality.

**3. A pinned `--eval-run-id`.**
Both `write_eval_run` and `ensure_run_started` upsert, so re-running under the same id
overwrites its rows in place rather than duplicating them. `eval_runs.run_id` is an FK to
`runs.id`, so the parent `runs` row must exist first — that row also carries
`prompt_version_id`, which is what makes a grade attributable to a persona/routing version.

---

## Grading: envelopes, not answers

`eval/golden_set.py` describes the *envelope* of acceptable output (which recommendations are
allowed, which topics the thesis must engage, how many signals, which semantic risk concepts).
`grade_analysis` asserts membership in that envelope and separately checks whether the thesis'
evidence supports its recommendation. It never asserts a single expected answer — a prompt
change that moves an ambiguous ticker from `hold` to `buy` should be *visible*, not fatal.

Severity decides what a failure means:

| Severity | Effect on `grade.passed` | Use it for |
|---|---|---|
| `must` | any failure sets `passed = False` | recommendation envelope, forbidden terms, required keywords, required risks, numerical grounding |
| `should` | recorded, but does not fail the example | signal counts (`{buffett,lynch}_{pros,cons}_min_count`) |

The signal counts are `should` on purpose: how many pros a model surfaces is a stylistic
choice that drifts across prompt versions without indicating a regression. A forbidden term
or an out-of-envelope recommendation is not.

**Adding a check:** append a `CheckResult` in `grade_analysis`, emit it *only when the YAML
actually sets the expectation* (an always-on check that nothing configures is an assertion
that never fails), and default to `should` unless a failure genuinely means the analysis is
wrong. `EvalExpectations` sets `extra="forbid"`, so a misspelled YAML key fails loudly.

---

## Recording tool fixtures

Note the two distinct fixture trees under `eval/fixtures/{TICKER}/`:

- `{client}/{method}/{hash}.json` — **raw upstream payloads** (yfinance `.info`, EDGAR HTML).
  Consumed by `eval.fixtures.load_fixture` in the data-fetcher tests, which exercise the real
  parsing path. Recorded by `uv run python -m eval.fixtures --record AAPL`.
- `tools/{tool_name}/{hash}.json` — **serialized `ToolResult`s**, i.e. exactly what the loop
  feeds back to Claude. Consumed by the eval replay. Written by
  `eval.tool_fixtures.record_tool_result`, which owns the on-disk format — call it rather than
  hand-rolling the path or the JSON.

`{hash}` is `sha256(json.dumps(tool_input, sort_keys=True))[:8]` in both trees. Files are
written with `sort_keys=True`, so a fixture diff is always a real change, never key reordering.
Tool keys are semantic: explicit DCF behavioral defaults collapse to the omitted-default key,
while genuinely different news windows remain distinct.

Record all supported calls with `uv run python -m eval.fixtures.recorder AAPL`, or repair only
the filing sources backing curated expectations with
`uv run python -m eval.fixtures.recorder --mandatory-evidence-only SBUX LUMN`. The recorder
validates output schemas and filing substance before overwriting an existing fixture.

---

## Gotchas that have already bitten

- **A missing fixture must be `retryable=False`.** `agent.loop._RETRY_POLICY` retries
  `network` and `rate_limit` with exponential backoff. A retryable miss turns a 13-ticker eval
  into a multi-minute stall. `FixtureToolRunner` returns `not_found`, which is non-retryable.
- **Never spend an LLM call on a ticker with no fixtures.** `has_tool_fixtures()` gates this;
  without it the command burns a full Sonnet run per ticker to produce a guaranteed failure
  grounded in nothing but `not_found` errors. The `fixture_missing` check is a `must` failure,
  so the exit code stays honest.
- **`SignalsExpectation` is a pydantic model, not a dict.** Use attribute access
  (`expectations.buffett_signals.pros`), not `.get("pros", {})`.
- **An empty `tools/` directory is not coverage.** `has_tool_fixtures` globs for actual files.
- **The eval writes a `runs` row.** It shows up in the dashboard's run list like any other run.
  Filter on the `eval-` id prefix if that's noise.
- **Semantic judging is blinded and batched.** Gold identifiers and labels are not sent to the
  judge. `JudgePanel` records disagreement/unavailability explicitly and can combine the
  pinned Sonnet judge with imported human verdicts.
- **Keep output compatibility.** `--output` is a top-level grade list. Usage totals come from
  the run WAL rather than the public grade list. A concise `<output>.report` summarizes strict
  pass rate, mandatory coverage, failure families,
  fixture parity, schema failures, and judge disagreement. Full prompts, raw model blocks,
  finals, validated outputs/failures, fixture diagnostics, and grades go to the owner-only
  private JSONL audit companion; do not publish or commit it.

---

## Analysis toolkit — debugging a golden-set run

`eval/analysis/` holds five thin CLIs over the logic above, for the parts of debugging an eval
run that used to mean a throwaway script or hand-`jq`-ing a trace. None of them reimplement
grading/diffing/replay — they call straight into `analyze_ticker`+`FixtureToolRunner`,
`dashboard.data.diff_eval_runs`, or `eval.runner.run_eval`.

```bash
uv run python -m eval.analysis.dump_theses AAPL MSFT
```
Replays each ticker via `FixtureToolRunner` + `analyze_ticker(temperature=0.0)` — the exact
seam `eval/runner.py` uses — and prints the full thesis/recommendation/confidence/signals/
key_risks instead of grading them. **Hits the live agent** (one loop per ticker); skips a
ticker with no recorded fixtures rather than burning an LLM call on it.

```bash
uv run python -m eval.analysis.diff_runs runs/eval-before.json runs/eval-after.json
```
Pure, offline. Parses two `--output` JSON files into the same
`{ticker: {check_name: EvalCheckResult}}` shape `dashboard.data.load_eval_grades` builds from
the DB, then hands them to the same `diff_eval_runs` the dashboard's Eval page uses — green
`+` for a fix, red `-` for a regression, plus the net delta.

```bash
uv run python -m eval.analysis.flakiness --runs 5           # live: N fresh replays of the golden set
uv run python -m eval.analysis.flakiness --from runs/*.json # offline: aggregate N existing outputs
```
`temperature=0` is not bit-exact on Sonnet and the LLM judge adds its own stochasticity, so a
single failure doesn't tell you whether a check is really broken or just flaky. Live mode calls
`eval.runner.run_eval` N times (**N× the golden-set API spend** — reach for offline mode when
the JSON outputs already exist); either mode reports each `(ticker, check_name)`'s pass rate
across the N runs, so `flaky` is quantified rather than eyeballed.

```bash
uv run python -m eval.analysis.trace_tools <run_id>
```
Offline — reads `logs/runs/{run_id}.jsonl` and lists the tools called per ticker, the same
`tool_call` events the project-context `jq` one-liners parse. Flags a small fixed
`CORE_COVERAGE_TOOLS` set (`read_filing`, `get_capital_allocation`, `get_quality_metrics`,
`get_insider_activity`) if any went uncalled for a ticker — answers "did the model even have
the data" without cross-referencing the trace against fixtures by hand.

```bash
uv run python -m eval.analysis.failures runs/eval-2026-05-10.json
```
Offline — parses one `--output` JSON and groups **must**-severity failures by check family
(`thesis_mentions_*`, `numerical_grounding`, etc. — `should` failures are excluded, since they
don't fail the example) then by ticker, with the grader's `expected`/`actual` inline, so a run's
failures can be scanned by family instead of ticker-by-ticker.

Tests live under `tests/test_evals/test_analysis/`, one file per script, following the same
`mock_claude` + `record_tool_result`-into-`tmp_path` pattern as `tests/test_evals/test_runner.py`.

---

## Testing the harness itself

- `tests/test_evals/test_grader.py` — each check family in isolation; the `must`/`should` rule.
- `tests/test_evals/test_tool_fixtures.py` — record→replay round-trip, the miss path, and
  `test_fixture_runner_never_calls_the_real_tool`, which fails loudly if replay regresses into
  dispatching a live tool.
- `tests/test_evals/test_runner.py` — end-to-end over a `tmp_path` fixture tree with
  `mock_claude`, asserting console output, `--output` JSON, the `.report` sidecar, private
  audit records, `eval_runs` rows, and that `temperature=0.0` reaches `messages.create`.

The acceptance criterion "same recommendation for ≥12/13 tickers across two runs" needs 26 live
Sonnet calls and **is not CI-testable**. Offline we assert `temperature=0.0` reaches the API and
that two runs over identical mocked responses produce identical grades. Verify the real thing
manually with a pinned `--eval-run-id` and diff the two `--output` files.
