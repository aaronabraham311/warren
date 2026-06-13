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
| Storage | SQLite via SQLAlchemy 2.x |
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

## Environment variables

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API access |
| `FINNHUB_API_KEY` | Real-time quotes and news |

Never commit `.env`. Use `.env.example` as the template.
