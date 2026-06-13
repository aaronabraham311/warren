# Warren — Claude Code context

Warren is an AI stock-analysis agent. The user asks a natural-language question; an agentic loop built on the Anthropic `anthropic` SDK selects Claude tools (quote, fundamentals, filings, news, …) to gather data, synthesises an answer, and persists the run to SQLite. A Streamlit dashboard exposes history and evaluation results.

## Tech stack

- **Python 3.13**, managed with **uv** (`uv sync` to install, `uv add <pkg>` to add dependencies)
- **Anthropic SDK** — tool-use / agentic loop lives in `agent/`
- **SQLAlchemy 2.x** with a local SQLite file `warren.db`
- **Streamlit** for the dashboard
- **ruff** for linting (`ruff check .` / `ruff format .`)
- **mypy** for type checking
- **pytest** + **pytest-recording** for tests (recorded HTTP fixtures in `eval/fixtures/`)

## Directory map

```
agent/
  run.py          # CLI entrypoint: python -m agent.run [TICKER]
  loop.py         # Main agentic loop — sends messages, handles tool calls
  persona.py      # System prompt / persona definition
  routing.py      # Decides which tools to invoke for a given query
  budget.py       # Token / cost budget tracking
  tools/          # One file per Claude tool; __init__.py holds the registry
    quote.py
    fundamentals.py
    growth.py
    filings.py
    news.py
    screen.py
    holdings.py

data_sources/
  yfinance_client.py   # Wraps yfinance for price history, fundamentals
  edgar_client.py      # Wraps SEC EDGAR full-text search + filing fetch
  finnhub_client.py    # Wraps Finnhub REST API (quotes, news)

storage/
  db.py           # SQLAlchemy engine, session factory (get_session)
  schema.sql      # Raw DDL for reference
  migrations/     # Schema migration scripts

eval/
  golden_set.py   # Loads YAML golden examples from eval/examples/
  runner.py       # Runs the agent against golden set, scores outputs
  examples/       # YAML files: input query + expected answer fields
  fixtures/       # Recorded HTTP responses for offline tests

dashboard/
  app.py          # Streamlit entrypoint
  pages/
    today.py      # Today's run results
    history.py    # Historical run browser
    eval.py       # Eval harness results

data/
  portfolio.csv   # ticker, shares, cost_basis, purchase_date
  watchlist.csv   # ticker, notes

logs/runs/        # Per-run JSONL traces (gitignored)
```

## Code conventions

- `__init__.py` files are empty. Import directly from the submodule, e.g. `from storage.db import get_session`. Do not re-export through `__init__.py`.
- Use SQLAlchemy 2.x style (`with Session(engine) as s:`, not legacy `Session()` context).
- All external API calls live in `data_sources/`; `agent/tools/` calls into `data_sources/`, never directly into yfinance/finnhub.
- Environment variables are loaded once at startup via `dotenv.load_dotenv()` in `agent/run.py`. Everywhere else, read with `os.environ["KEY"]` — no repeated `load_dotenv()` calls.
- Do not commit `.env`, `warren.db`, or anything under `logs/`.

## Common commands

```bash
uv sync                        # install / sync deps
python -m agent.run AAPL       # single ticker
python -m agent.run            # full portfolio run
streamlit run dashboard/app.py # launch dashboard
ruff check .                   # lint
ruff format .                  # format
mypy .                         # type check
pytest                         # run tests
```

## Environment variables (see .env.example)

| Variable | Used by |
|---|---|
| `ANTHROPIC_API_KEY` | `agent/loop.py` — Claude API |
| `FINNHUB_API_KEY` | `data_sources/finnhub_client.py` |
