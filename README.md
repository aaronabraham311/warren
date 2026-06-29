# Warren

An AI-powered stock analysis agent built on Claude. Warren answers natural-language questions about equities — quotes, fundamentals, filings, news, and portfolio health — through a conversational agentic loop backed by a persistent SQLite store and a Streamlit dashboard.

## Quick start

```bash
# 1. Install dependencies
uv sync

# 2. Set up environment variables
cp .env.example .env
# Fill in ANTHROPIC_API_KEY and FINNHUB_API_KEY in .env

# 3. Run a single-ticker analysis
python -m agent.run AAPL

# 4. Run a full portfolio sweep
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
| Storage | JSONL run traces (source of truth) + SQLite projection via SQLAlchemy 2.x |
| Dashboard | Streamlit |
| Package manager | uv |

## Directory layout

```
warren/
├── agent/          # Agentic loop, routing, persona, budget, and tools
├── data_sources/   # Thin clients for yfinance, EDGAR, Finnhub
├── storage/        # SQLAlchemy models, schema, migrations
├── eval/           # Golden-set evaluation harness
├── dashboard/      # Streamlit multi-page app
├── data/           # Portfolio and watchlist CSVs
└── logs/runs/      # JSONL run traces
```

## Nightly Scheduler

Warren can run autonomously at 2 AM via the OS scheduler. The process is short-lived (not a daemon) — it starts, runs, and exits.

### macOS (launchd)

```bash
# Install — reads API keys from .env and injects them via PlistBuddy (never stored in the template)
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
# Install — reads API keys from .env and injects them into the cron entry
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

Never commit `.env`. Use `.env.example` as the template.
