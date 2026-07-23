# Warren — Claude Code context

Warren is an AI stock-analysis agent. The user asks a natural-language question; an agentic loop built on the Anthropic `anthropic` SDK selects Claude tools (quote, fundamentals, filings, news, …) to gather data, synthesises an answer, and persists the run to SQLite. A Streamlit dashboard exposes history and evaluation results.

## Tech stack

- **Python 3.13**, managed with **uv** (`uv sync` to install, `uv add <pkg>` to add dependencies)
- **Anthropic SDK** — tool-use / agentic loop lives in `agent/`
- **SQLAlchemy 2.x** with a local SQLite file `warren.db`
- **ruff** for linting (`ruff check .` / `ruff format .`)
- **mypy** for type checking
- **pytest** + **pytest-recording** for tests

## Directory map

```
agent/
  run.py          # CLI entrypoint: python -m agent.run [TICKER] [--skip-ticker-validation]
  cooldown.py     # Discovery cooldown — filter_universe_for_cooldown(), has_material_event(), set_cooldown(), get_cooldown_entry(), clear_cooldown(); 7-day dedup with material-news override
  portfolio.py    # load_portfolio/load_watchlist (validated) + sync_*_to_db snapshots
  universe.py     # get_current_universe(session, watchlist) → sorted S&P 500 ∪ watchlist for the Haiku screening prefix. Weekly-refreshed (UniverseSnapshot single-row table); SP500Client fetch with data/sp500.csv fallback. Deterministic sort keeps the cache prefix byte-stable.
  screening.py    # run_screening_pass(universe, system_prompt, ...) → ScreeningResult — Haiku PASS/FAIL filter over the universe. Batch API path (async, 50% discount) and sequential path (immediate, for local dev). Cooldown filtering is the caller's responsibility.
  loop.py         # Main agentic loop — sends messages, handles tool calls
  persona.py      # System prompt / persona definition
  routing.py      # RoutingPolicy Protocol + strategy objects: PhaseBasedRouting (screen→Haiku, deep→Sonnet, synthesize→Opus via DefaultOpusTrigger's 3 independent §4.2 conditions) and HardcodedSonnetRouting (eval baseline). Swappable into analyze_ticker() with zero loop changes.
  budget.py       # Token / cost budget tracking
  models.py       # Model IDs + per-model PRICING table (single source of truth)
  tools/          # One file per Claude tool; __init__.py holds the registry
    base.py       # Tool ABC, ToolResult (ToolResultOk/ToolResultError), error_from_data_source()
    _clients.py   # Lazy data-source client singletons (yfinance/edgar/finnhub) + reset_clients()
    quote.py      # get_quote          → YFinanceClient.get_price → PriceData
    fundamentals.py  # get_fundamentals → yfinance, Finnhub fallback when stale → FundamentalsData
    growth.py     # get_growth_metrics → YFinanceClient.get_growth_metrics → GrowthData
    filings.py    # read_filing        → EDGARClient.get_filing_section → FilingSection
    news.py       # get_news           → FinnhubClient.get_news → NewsResult
    screen.py     # screen_universe    → portfolio∪watchlist filtered on fundamentals → ScreenResult
    holdings.py   # get_holding_context→ portfolio.csv + get_price → HoldingContext
    valuation.py  # get_valuation_multiples → YFinanceClient.get_valuation_multiples → ValuationData
    quality.py    # get_quality_metrics → YFinanceClient.get_quality_metrics → QualityData (ROIC, ROA, gross-margin stability, cash conversion)
    insider.py    # get_insider_activity → FinnhubClient.get_insider_transactions + YFinanceClient.get_ownership → InsiderActivity
    peers.py      # get_peer_comparison → fundamentals+valuation per peer → PeerComparison (rank/percentile)
    financial_strength.py  # get_financial_strength → YFinanceClient.get_financial_strength → FinancialStrengthData
    intrinsic_value.py  # estimate_intrinsic_value → owner-earnings DCF off get_financials + get_price → IntrinsicValue (intrinsic value/share, margin of safety, reverse-DCF implied growth)
    capital_allocation.py  # get_capital_allocation → YFinanceClient.get_capital_allocation → CapitalAllocation (share-count CAGR, buyback/dividend/shareholder yield, dividend growth streak, payout ratio, net-debt trajectory — Buffett/Munger management-quality lens)
    persons.py    # get_key_persons → YFinanceClient.get_key_persons (companyOfficers + institutional_holders) + EDGARClient.get_sc13_holders (EFTS SC 13G/D) → KeyPersonsData (persons list with name/role/ownership_pct/source, controlling_holder_identified flag, source_notes)
    adverse_media.py  # get_adverse_media → GDELTClient.get_adverse_articles → AdverseMediaResult (ranked, categorised negative-tone hits across 65 languages; keyword + GKG-theme classification; dual-query dedup)
    watchlists.py # screen_watchlists → OFACClient.search_entity → WatchlistResult (matches with score, risk_categories=["sanction"], datasets=OFAC programs; asymmetry note always present)

data_sources/
  cache.py             # CacheStore (shared SQLite cache) + make_key(tool_name, *parts)
  errors.py            # DataSourceError(error_code, message) — shared by all clients
  yfinance_client.py   # Wraps yfinance for price quotes and fundamentals; get_financials → FinancialsHistory (multi-year income/balance/cash-flow rows) is the shared foundation that get_growth_metrics / get_quality_metrics / get_financial_strength all compute off via _build_financials + _series
  edgar_client.py      # EDGARClient — SEC 10-K/10-Q/8-K/DEF 14A filing sections + get_sc13_holders (EFTS SC 13G/D beneficial owner search) (cached, polite)
  finnhub_client.py    # FinnhubClient — news + fundamentals fallback (cached, rate-limited)
  gdelt_client.py      # GDELTClient — GDELT DOC 2.0 ArtList adverse news (keyless, 7d cache). Note: ArtList does not return tone/themes in the response; tone<-2 is a server-side filter only.
  ofac_client.py       # OFACClient — OFAC SDN free public API (no key), 7-day cache; US sanctions only
  sp500_client.py      # SP500Client — keyless Wikipedia scrape of S&P 500 constituents → list[str] | DataSourceError (no cache; the UniverseSnapshot table is the weekly cache)

storage/
  models.py       # ORM models (Base + all table classes + indexes) — no I/O
  engine.py       # Engine, WAL/FK pragmas, get_session, migrate(), helper fns
  logger.py       # RunLogger — JSONL per-run trace (the durable WAL) + flush_to_db
  cost.py         # compute_cost(model, …) — per-model LLM cost from models.PRICING
  recovery.py     # reconcile_run / reconcile_orphans — rebuild DB rows from a trace
  migrations/     # Alembic migration scripts

eval/
  golden_set.py   # EvalExample / EvalExpectations pydantic models + load_eval_example() / load_all_examples()
  runner.py       # run_eval() + CLI — replays the agent over the golden set, grades, persists to eval_runs
  grader.py       # grade_analysis() → EvalGrade / CheckResult; must-vs-should severity model
  tool_fixtures.py # FixtureToolRunner (agent.loop.ToolRunner) — serves recorded ToolResults; record_tool_result()
  examples/       # Hand-curated golden expectations, one YAML per ticker (Tech Spec §6.2)
    {ticker}.yaml # e.g. aapl.yaml, brk_b.yaml — dotted tickers use an underscore stem
  analysis/       # CLI helpers for debugging golden-set runs (see /eval skill for the full rundown)
  fixtures/
    README.md     # The two fixture kinds + the quarterly rotation policy
    __init__.py   # CLIENT-level: load_fixture(ticker, client, method, name=) + record_fixtures()
    __main__.py   # CLI: python -m eval.fixtures --record AAPL MSFT GOOG
    recorder.py   # CLI: python -m eval.fixtures.recorder [TICKER...] (default: whole golden set)
                  #   Hits live APIs once and writes via eval.tool_fixtures.record_tool_result
    {TICKER}/{client}/{method}/{hash}.json   # raw upstream payloads — for the data-fetcher tests
    {TICKER}/tools/{tool_name}/{hash}.json   # serialized ToolResults — for eval replay

dashboard/        # Streamlit read-only dashboard — one exception: Today's "Run now" button (see below)
  app.py          # Multi-page entrypoint — `streamlit run dashboard/app.py`; st.navigation + set_page_config
  data.py         # Pure data access over the ORM + JSONL logs (no Streamlit) — get_latest_run, get_analyses_for_run (§9.Q3 sort), search_analyses (History filters), read_reasoning_trace, cooldown_suppressed_count, previous_recommendation, MONTHLY_WARNING_THRESHOLD_USD; Eval page: eval_run_summaries (per-run pass rate + version join), load_eval_grades (parse check_results JSON → {ticker: {check: EvalCheckResult}}), diff_eval_runs (structured fix/regression diff)
  pages/today.py  # Today page: run-metadata header (incl. cooldown-suppressed count + budget/status banners) + holdings/discovery cards (prior-call delta) + sidebar "Run now" button (`subprocess.run(["python", "-m", "agent.run"])`, the one deliberate exception to the read-only rule — a human-clicked dev convenience, not an automated write path)
  pages/history.py  # History page: sidebar filters (ticker/recommendation/date/confidence) → searchable recommendation archive
  pages/eval.py   # Eval page: pass-rate-by-prompt-version chart (altair line + dataframe) + side-by-side diff of any two eval runs (net-change banner + per-ticker expanders, green=fix / red=regression, with expected/actual)
  components/analysis_card.py  # render_analysis_card (optional prompt_version label) / render_reasoning_trace; reused by Today + History
  seed_demo.py    # Dev tool: `python -m dashboard.seed_demo` seeds a demo run + history runs (prompt versions) + JSONL trace for a dashboard walkthrough

tests/
  conftest.py     # Shared fixtures (db_engine, db_session, mock_claude,
                  #   edgar_fixture, finnhub_fixture, gdelt_fixture, autouse _no_live_network guard)
  test_agent_loop.py
  test_dashboard_today.py  # data-layer + headless streamlit AppTest integration
  test_dashboard_history.py  # search_analyses filters + History page AppTest integration
  test_{yfinance,edgar,finnhub,gdelt}_client.py
  test_adverse_media.py
  test_screening.py
  test_evals/     # All eval/ package tests, grouped
    test_grader.py, test_judge.py, test_recorder.py, test_runner.py, test_tool_fixtures.py
    test_analysis/  # eval/analysis/ CLI tests, one file per script
      test_dump_theses.py, test_diff_runs.py, test_flakiness.py, test_trace_tools.py, test_failures.py

data/
  portfolio.csv   # ticker, shares, cost_basis, purchase_date
  watchlist.csv   # ticker, notes
  sp500.csv       # bootstrap S&P 500 constituents — fallback for agent/universe.py when the live Wikipedia fetch fails

main.py           # Thin wrapper: delegates to agent.run.main
agent/eval.py     # Entrypoint shim: `python -m agent.eval` → eval.runner.main
logs/runs/        # Per-run JSONL traces (gitignored)
```

## Eval replay

`python -m agent.eval --golden-set` replays the agent over the golden set against recorded
tool fixtures and grades the results into `eval_runs`. **Working on `eval/`? Use the `/eval`
skill** — it covers the three determinism invariants (fixture replay, `temperature=0`, a
pinned `--eval-run-id`), the `must`/`should` severity model, how to record tool fixtures, and
the gotchas that have already bitten.

## Run logging — JSONL-as-WAL

Each run writes a structured event trace to `logs/runs/{run_id}.jsonl` via
`storage.logger.RunLogger` (one `flush()`+`fsync()`'d JSON line per event, so a crash
never leaves a partial line). **The trace is the source of truth**; the `runs` and
`tool_calls` SQLite tables are a *derived projection* of it — a queryable cache for the
dashboard, not written incrementally during the loop.

- The loop only appends events (`log_llm_call` / `log_tool_call`); it does **not** write
  `tool_calls` rows directly. The `tool_call` event carries the tool input/output (big
  payloads sidecar to `logs/runs/{run_id}/tool_outputs/`) so every DB column is rebuildable.
- `RunLogger.flush_to_db(session)` reconciles the trace into the DB at run end
  (`storage.recovery.reconcile_run`, idempotent delete-then-insert).
- `storage.recovery.reconcile_orphans()` runs on `agent.run` startup: any run left
  `status="running"` with a trace on disk (a crash) is reconciled — the DB self-heals.
- Wired events (single-ticker loop): `run_started`, `ticker_started`, `llm_call`,
  `tool_call`, `ticker_completed`, `run_completed`. Nightly mode additionally logs
  `discovery_cooldown_applied` (with `suppressed_count`/`suppressed_tickers`) right after
  cooldown filtering — the Today page's suppressed-count metric reads it back via
  `dashboard.data.cooldown_suppressed_count`. `phase_started`/`phase_completed` are
  supported by `RunLogger.log()` but unwired until the screening orchestrator exists.

Debug the trace with `jq`, e.g. per-call cost breakdown:

```bash
jq -c 'select(.event=="llm_call") | {model,input_tokens,cache_read_tokens,cost_usd}' logs/runs/*.jsonl
```

Surface retry pressure (tool calls that required at least one retry):

```bash
jq -c 'select(.event=="tool_call" and .retry_count > 0) | {ticker,tool,retry_count,last_retry_error,latency_ms}' logs/runs/*.jsonl
```

## Test fixtures — recorded, never live

Data-fetcher tests never hit the network. Each client (`yfinance` / `edgar` / `finnhub`)
is mocked at its upstream boundary and fed a **recorded payload** committed under
`eval/fixtures/{TICKER}/{client}/{method}/{hash}.json`, where `hash` is
`sha256(json.dumps(input, sort_keys=True))[:8]` (error cases use names like
`error_not_found.json`). Fixtures store the *raw upstream payload* the client parses,
not the model output — tests exercise the real parsing path. Load them with
`eval.fixtures.load_fixture(...)`.

- An autouse `_no_live_network` guard in `conftest.py` blocks `socket.connect`, so any
  unmocked live call fails loudly. Keep the suite offline.
- Fixtures are committed and **never regenerated in CI**. To refresh them from live APIs
  (requires network; Finnhub also needs `FINNHUB_API_KEY`, else it's skipped):
  `python -m eval.fixtures --record AAPL`.

**Eval fixtures are a second, separate kind** (`eval/fixtures/README.md`). They live under
`{TICKER}/tools/{tool_name}/{hash}.json` and store the *serialized `ToolResult`* a tool
returned, keyed by its validated input. `eval.tool_fixtures` owns that format — reading it
(`FixtureToolRunner`) and writing it (`record_tool_result`). Record with
`python -m eval.fixtures.recorder AAPL` (hits live APIs, overwrites in place; no args
records the whole golden set). Fixtures rot — a `recorded_at` older than 90 days warns on
load; refresh quarterly.

## Code conventions

- `__init__.py` files are empty, with one deliberate exception: `agent/tools/__init__.py` holds the `TOOL_REGISTRY` / `TOOL_DEFINITIONS`. Everywhere else import directly from the submodule, e.g. `from storage.engine import get_session`. Do not re-export through `__init__.py`.
- **Tools return errors as data, never raise** (Tech Spec §5). A `Tool.run(tool_input, ctx)` returns `ToolResultOk(data=<BaseModel>)` or `ToolResultError(error_code, message, retryable)`; `error_code ∈ {rate_limit, not_found, stale_data, network, unknown}`. Map a `DataSourceError` with `error_from_data_source()` (in `agent/tools/base.py`); wrap the body in `try/except` so any stray exception becomes `ToolResultError(error_code="unknown")`. The loop validates `block.input` against the tool's `input_schema` and serializes the result back to the agent (ok → `data.model_dump_json()`; error → `{error_code,message,retryable}` with `is_error=True`).
- Tools reach data-source clients only via the lazy singletons in `agent/tools/_clients.py` (bound to the `$WARREN_DB` cache); tests reset them with the autouse `_reset_tool_clients` fixture (which also points `WARREN_DB` at `:memory:`). The §5.4 loop-level retry/backoff is intentionally **not** in the loop yet — it's a separate W3 ticket.
- Use SQLAlchemy 2.x style (`with Session(engine) as s:`, not legacy `Session()` context).
- All external API calls live in `data_sources/`; `agent/tools/` calls into `data_sources/`, never directly into yfinance/finnhub.
- Environment variables are loaded once at startup via `dotenv.load_dotenv()` in `agent/run.py`. Everywhere else, read with `os.environ["KEY"]` — no repeated `load_dotenv()` calls.
- Do not commit `.env`, `warren.db`, or anything under `logs/`.
- `local/` is a gitignored scratch dir for review notes, findings, and throwaway artifacts — never shipped or imported by code.
- **Working on the Streamlit dashboard (`dashboard/`)? Use the `/streamlit` skill** — it covers the three-layer structure, `AppTest` e2e testing, running the app, `python -m dashboard.seed_demo`, and the stale-server/port debugging gotchas.
- **Working on the eval harness (`eval/`)? Use the `/eval` skill** — it covers the determinism invariants, the `must`/`should` grading model, recording tool fixtures, and `python -m agent.eval`.
- **Prefer `@dataclass` over plain tuples for multi-value returns** (named fields, no positional unpacking drift). Use `NamedTuple` only for genuinely tuple-like data (e.g. a coordinate pair). Raw tuples are fine for single-purpose, immediately-unpacked returns.

## Common commands

```bash
uv sync                        # install / sync deps
python -m agent.run AAPL       # single ticker deep analysis
python -m agent.run            # nightly mode: screen universe → deep-analyse top 3 candidates
python -m agent.run --no-batch # nightly mode with sequential (non-batch) screening
ruff check .                   # lint
ruff format .                  # format
mypy .                         # type check
pytest                         # run tests
python -m agent.eval --golden-set --output runs/eval-2026-05-10.json  # eval replay (exits 1 on any failure)
streamlit run dashboard/app.py # launch the read-only dashboard (Today page)
python -m eval.fixtures.recorder        # re-record eval fixtures for the whole golden set
python -m eval.fixtures.recorder AAPL   # …or just one ticker (overwrites in place)
python -m dashboard.seed_demo  # seed a demo run + trace to view the dashboard without a real run
```

## Definition of done

Before claiming a task complete, run and pass `ruff check .`, `mypy .`, and `pytest` — then state the result. If the change touched conventions, structure, or commands, update this file in the same change.

## Environment variables (see .env.example)

| Variable | Used by |
|---|---|
| `ANTHROPIC_API_KEY` | `agent/run.py` — constructs the Anthropic client |
| `WARREN_DB` | `storage/engine.py` — SQLite path override (default: `warren.db`) |
| `WARREN_LOGS_DIR` | `dashboard/data.py` — JSONL run-log dir for the reasoning trace (default: `logs/runs`) |
