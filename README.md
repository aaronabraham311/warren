# Warren

An AI-powered stock analysis agent built on Claude. Warren answers natural-language questions about equities — quotes, fundamentals, filings, news, and portfolio health — through a conversational agentic loop backed by a persistent SQLite store and a Streamlit dashboard.

> **What this project actually is:** Warren is mostly vibe-coded, built as a personal exercise in designing an *agent harness* — tool-use loops, routing, eval scaffolding, run logging, recovery — rather than as a serious or reliable stock-picking engine. Treat its analysis as a demo of the plumbing, not as investment advice. Expect rough edges, and don't trade on anything it tells you.

## Quick start

```bash
# 1. Install dependencies
uv sync

# 2. Set up environment variables
cp .env.example .env
# Fill in ANTHROPIC_API_KEY and FINNHUB_API_KEY in .env

# 3. Run a single-ticker analysis
python -m agent.run AAPL

# 4. Nightly mode: screen the S&P 500 + watchlist for candidates, then
#    deep-analyse your holdings plus the top 3 candidates (more API calls/cost)
python -m agent.run

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

## Stack

| Layer | Tech |
|---|---|
| Language model | Anthropic Claude (via `anthropic` SDK) |
| Market data | yfinance, Finnhub |
| SEC filings | EDGAR |
| News / sanctions | GDELT (adverse media), OFAC (sanctions screening) |
| Storage | JSONL run traces (source of truth) + SQLite projection via SQLAlchemy 2.x |
| Dashboard | Streamlit |
| Package manager | uv |

## Capabilities

The agent picks from 17 Claude tools per turn (`agent/tools/`), grouped by what they cover:

| Area | Tools |
|---|---|
| Price / valuation | `get_quote`, `get_fundamentals`, `get_valuation_multiples`, `estimate_intrinsic_value` (owner-earnings DCF + reverse-DCF) |
| Growth / quality | `get_growth_metrics`, `get_quality_metrics` (ROIC, ROA, margin stability, cash conversion), `get_financial_strength` |
| Management & capital allocation | `get_capital_allocation` (buybacks, dividends, net-debt trajectory), `get_key_persons`, `get_insider_activity` |
| Competitive context | `get_peer_comparison` (rank/percentile vs. peers) |
| Filings / news / risk | `read_filing` (SEC 10-K/10-Q/8-K/DEF 14A), `get_news`, `get_adverse_media` (GDELT negative-tone screening), `screen_watchlists` (OFAC sanctions) |
| Portfolio / discovery | `get_holding_context`, `screen_universe` (Haiku PASS/FAIL screen over the S&P 500 + watchlist) |

Model routing is phase-based (`agent/routing.py`): cheap Haiku screening → Sonnet for deep tool-use → Opus only for final synthesis, gated by explicit trigger conditions rather than always escalating. Every run is budget-capped (`agent/budget.py`) and aborts cleanly if a cost ceiling is hit mid-run.

## Evals

`eval/` is a golden-set harness, not a benchmark suite with a published score — it exists to catch regressions as the prompt/routing/tools change, not to certify investment quality. It replays the agent against **recorded tool fixtures** (no live network, fully deterministic) over a hand-curated set of tickers (`eval/examples/*.yaml`) and grades each run on:

- **Recommendation correctness** — is the buy/hold/sell call within the allowed set for that ticker
- **Thesis content** — does the writeup engage the topics a domain expert would expect (checked via an LLM-as-judge pass, not just keyword matching) and avoid forbidden claims (e.g. "guaranteed," "risk-free")
- **Signal counts** — minimum Lynch/Buffett pros and cons cited
- **Numerical grounding** — the thesis cites real numbers pulled from tool output rather than hallucinating figures

Checks carry a `must`/`should` severity — a `must` failure fails the run, a `should` failure is logged but doesn't. Results persist to the `eval_runs` table and are browsable in the dashboard's Eval page (pass-rate-by-prompt-version chart + run-to-run diff view). Pass rates move with prompt/persona changes and haven't stabilized into a fixed number worth quoting here — run it yourself for the current state:

```bash
python -m agent.eval --golden-set --output runs/eval-$(date +%F).json
```

## Directory layout

```
warren/
├── agent/          # Agentic loop, routing, persona, budget, and tools
├── data_sources/   # Thin clients for yfinance, EDGAR, Finnhub, GDELT, OFAC
├── storage/        # SQLAlchemy models, schema, migrations
├── eval/           # Golden-set evaluation harness
├── dashboard/      # Streamlit multi-page app
├── scripts/        # Nightly scheduler install/uninstall (launchd + cron)
├── docs/           # PRD and tech spec
├── data/           # Portfolio and watchlist CSVs
└── logs/runs/      # JSONL run traces
```

## Nightly Scheduler

Warren can run autonomously at 2 AM via the OS scheduler. The process is short-lived (not a daemon) — it starts, runs, and exits.

### macOS (launchd)

```bash
# Install — validates .env has ANTHROPIC_API_KEY; the agent reads .env itself at
# startup, so no keys are copied into the plist or the launchd environment
bash scripts/install_scheduler.sh

# Verify the job is registered
launchctl list | grep warren

# Check logs
tail -f logs/launchd_stdout.log
tail -f logs/launchd_stderr.log

# Manual trigger (runs immediately, same as the scheduled run)
python -m agent.run

# Uninstall
bash scripts/uninstall_scheduler.sh
```

### Linux (cron)

```bash
# Install — validates .env has ANTHROPIC_API_KEY; the agent reads .env itself at
# startup, so no keys go into the crontab line. Runs are flock-guarded so a slow
# run never overlaps the next night's trigger.
bash scripts/install_cron.sh

# Verify the entry
crontab -l | grep warren

# Check logs
tail -f logs/cron.log

# Manual trigger
python -m agent.run

# Uninstall — remove the warren line from your crontab
crontab -e
```

## Environment variables

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API access |
| `FINNHUB_API_KEY` | Real-time quotes and news |
| `WARREN_DB` | Optional: override the SQLite path (default `warren.db`) |
| `WARREN_LOGS_DIR` | Optional: override the JSONL run-log dir the dashboard reads (default `logs/runs`) |

Never commit `.env`. Use `.env.example` as the template.
