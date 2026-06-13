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
  run.py          # CLI entrypoint: python -m agent.run [TICKER]
  loop.py         # Main agentic loop — sends messages, handles tool calls
  persona.py      # System prompt / persona definition
  routing.py      # Decides which model to route each call to
  budget.py       # Token / cost budget tracking
  models.py       # Model ID and per-token pricing constants (single source of truth)
  tools/          # One file per Claude tool; __init__.py holds the registry
    base.py       # Tool ABC, ToolResult, ToolResultOk, ToolResultError
    quote.py      # get_quote tool (current price via yfinance)
    # planned: fundamentals, growth, filings, news, screen, holdings

data_sources/
  yfinance_client.py   # Wraps yfinance for price quotes and fundamentals
  # planned: edgar_client.py, finnhub_client.py

storage/
  models.py       # ORM models (Base + all table classes + indexes) — no I/O
  engine.py       # Engine, WAL/FK pragmas, get_session, migrate(), helper fns
  migrations/     # Alembic migration scripts

tests/
  conftest.py     # Shared fixtures (db_engine, db_session, mock_claude)
  test_agent_loop.py

data/
  portfolio.csv   # ticker, shares, cost_basis, purchase_date
  watchlist.csv   # ticker, notes

main.py           # Thin wrapper: delegates to agent.run.main
logs/runs/        # Per-run JSONL traces (gitignored)
```

## Code conventions

- `__init__.py` files are empty. Import directly from the submodule, e.g. `from storage.engine import get_session` or `from storage.models import Run`. Do not re-export through `__init__.py`.
- Use SQLAlchemy 2.x style (`with Session(engine) as s:`, not legacy `Session()` context).
- All external API calls live in `data_sources/`; `agent/tools/` calls into `data_sources/`, never directly into yfinance/finnhub.
- Environment variables are loaded once at startup via `dotenv.load_dotenv()` in `agent/run.py`. Everywhere else, read with `os.environ["KEY"]` — no repeated `load_dotenv()` calls.
- Do not commit `.env`, `warren.db`, or anything under `logs/`.
- `local/` is a gitignored scratch dir for review notes, findings, and throwaway artifacts — never shipped or imported by code.

## Common commands

```bash
uv sync                        # install / sync deps
python -m agent.run AAPL       # single ticker
python -m agent.run            # full portfolio run
ruff check .                   # lint
ruff format .                  # format
mypy .                         # type check
pytest                         # run tests
```

## Definition of done

Before claiming a task complete, run and pass `ruff check .`, `mypy .`, and `pytest` — then state the result. If the change touched conventions, structure, or commands, update this file in the same change.

## Environment variables (see .env.example)

| Variable | Used by |
|---|---|
| `ANTHROPIC_API_KEY` | `agent/run.py` — constructs the Anthropic client |
| `WARREN_DB` | `storage/engine.py` — SQLite path override (default: `warren.db`) |
