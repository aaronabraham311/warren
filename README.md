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

# 3. Open the interactive terminal
uv run warren

# 4. Or run a single batch analysis
uv run python -m agent.run AAPL

# 5. Nightly mode: screen the S&P 500 + watchlist for candidates, then
#    deep-analyse your holdings plus the top 3 candidates (more API calls/cost)
uv run python -m agent.run

# 6. Launch the dashboard
uv run streamlit run dashboard/app.py
```

## Interactive terminal

`uv run warren` opens a terminal transcript backed by the same run service, JSONL
traces, and SQLite projection as the batch CLI. It accepts a deliberately small,
deterministic natural-language grammar, so unsupported or ambiguous requests do not
silently start an analysis.

Examples:

```text
Analyze AAPL
Compare COST with WMT
Review my portfolio
Run discovery
Run gem hunt
Show the risks
```

The terminal supports these slash commands:

| Command | Behavior |
|---|---|
| `/help` | Show commands and request examples. |
| `/new` | Clear transient recent-result context without deleting history. |
| `/history [ticker]` | List recent analyses, optionally for one ticker. |
| `/show RUN_ID` | Render analyses from a completed or partial run. |
| `/trace [RUN_ID]` | Show the latest or selected run's tool and model events. |
| `/portfolio` | Show current holdings and snapshot prices. |
| `/watchlist` | Show current watchlist entries and notes. |
| `/discover` | Run the existing US GARP discovery workflow. |
| `/gem-hunt` | Run global deep-value discovery with the DIRT persona. |
| `/persona [default\|dirt]` | Show or set the default analysis persona. |
| `/budget [USD]` | Show or set the per-run cost ceiling, greater than $0 and at most $10. |
| `/tools` | List Warren's available financial tools by category. |
| `/quit` | Exit cleanly. |

Input history and non-secret preferences persist under `.warren/` by default (or the
directory selected by `WARREN_STATE_DIR`):

- `.warren/history` stores prompt history.
- `.warren/settings.json` stores persona, budget, and display preferences.
- `.warren/active-run.lock` prevents interactive, batch, and scheduled analyses from
  overlapping.

These files are gitignored, and Warren never writes API configuration to them. Prompt
history contains the text you enter, so do not paste credentials at the prompt. JSONL
files under `logs/runs/` remain the authoritative run traces; SQLite remains the query
projection used by history views. The terminal can use read-only commands without
`ANTHROPIC_API_KEY`, but it rejects analysis commands before creating a run when the
key is absent.

Ctrl-C during a run requests cooperative cancellation; a second Ctrl-C may exit
immediately. EOF or `/quit` exits cleanly. Piped and `NO_COLOR` output is stable and
line-oriented.

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
| Filings / news / risk | `read_filing` (SEC sections or verified stored regional PDFs with page citations and explicit OCR/translation coverage), `get_news`, `get_adverse_media` (GDELT negative-tone screening), `screen_watchlists` (OFAC sanctions) |
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
├── agent/          # Shared run service, interactive terminal, loop, routing, and tools
├── data_sources/   # Thin clients for yfinance, EDGAR, Finnhub, GDELT, OFAC
├── storage/        # SQLAlchemy models, schema, migrations
├── eval/           # Golden-set evaluation harness
├── dashboard/      # Streamlit multi-page app
├── scripts/        # Weekly scheduler install/uninstall (launchd + cron)
├── docs/           # PRD and tech spec
├── data/           # Portfolio and watchlist CSVs
└── logs/runs/      # JSONL run traces
```

## Weekly Scheduler

Warren can run autonomously every Sunday at 7 AM ET via the OS scheduler. The process is short-lived (not a daemon) — it starts, runs, and exits.

The scheduled run is **gem-hunt mode** (`python -m agent.run --gem-hunt`): the global three-exchange universe (Euronext Growth Milan `.MI`, Bolsa de Madrid `.MC`, GPW Warsaw `.WA`), the deep-value screen with score-based ranking, and the DIRT persona. The US GARP nightly is still there on demand — run `python -m agent.run` with no flags. Re-running an installer over an already-installed job replaces it, so switching modes is just a matter of editing the command and re-installing.

### macOS (launchd)

`launchd` uses the Mac's system timezone. Set the Mac to Eastern Time for the scheduled run to occur at 7 AM ET.

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
uv run python -m agent.run --gem-hunt

# Uninstall
bash scripts/uninstall_scheduler.sh
```

### Linux (cron)

The installer sets `CRON_TZ=America/New_York`, so the Sunday trigger follows Eastern Time across daylight-saving changes.

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
uv run python -m agent.run --gem-hunt

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
| `WARREN_STATE_DIR` | Optional: override terminal settings, prompt history, and active-run lock directory (default `.warren`) |
| `WARREN_FILINGS_DIR` | Optional: content-addressed raw filing/text artifacts (default `local/filings`; excluded from Git) |
| `WARREN_TRANSLATION_MODEL` | Optional: explicit Anthropic model for filing-page translation; unset keeps translation fail-closed |
| `WARREN_TRANSLATION_INPUT_USD_PER_MILLION_TOKENS` | Required with a translation model: current input-token price used by the estimated cost admission budget |
| `WARREN_TRANSLATION_OUTPUT_USD_PER_MILLION_TOKENS` | Required with a translation model: current output-token price used by the estimated cost admission budget |

Never commit `.env`. Use `.env.example` as the template.
