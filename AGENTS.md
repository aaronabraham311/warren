# Warren — Codex repository instructions

Warren is a Python 3.13 stock-analysis agent. The runtime agent uses Anthropic's SDK; Codex is the coding agent working on this repository. Refer to Warren's runtime capabilities as “agent tools,” not “Codex tools.”

## Project map

- `agent/`: provider-normalized agent loop, provider adapters, routing, personas, budgets, portfolio/universe screening, and typed tools.
- `data_sources/`: all external API clients and cache behavior, including typed
  junior-market identity sources used by universe refresh and regional filings.
- `storage/`: SQLAlchemy models, SQLite engine, Alembic migrations, JSONL WAL recovery.
- `eval/`: deterministic golden-set replay, graders, fixtures, and analysis helpers.
- `dashboard/`: Streamlit read-only dashboard and its data layer.
- `tests/`: offline unit and integration tests.
- `.agents/skills/`: Codex-native project skills. Load the matching skill before working on evals, migrations, Streamlit, or shipping workflows.
- `.claude/commands/`: Claude Code compatibility commands; they are not Codex skills.

## Working rules

- Inspect `git status --short --branch` before editing and preserve unrelated changes.
- Use `apply_patch` for file edits. Do not use shell redirection or ad-hoc scripts to write source files.
- Keep external API calls in `data_sources/`; `agent/tools/` calls clients through `agent/tools/_clients.py`.
- Treat TradingView's keyless scanner as an unofficial NewConnect source: its terms/source risk require repository-owner approval before merge; committed snapshots remain the runtime fallback.
- Keep LLM SDK wire formats inside `agent/providers/`; production defaults to Anthropic while evals may inject another provider.
- Tools return typed `ToolResultOk` or `ToolResultError` data; they do not raise for expected data-source failures.
- Keep tests offline. Do not fall back from missing eval fixtures to live tools.
- Use SQLAlchemy 2.x style and Alembic batch mode for SQLite schema alterations.
- Load environment variables once at startup. Never commit `.env`, API keys, `warren.db`, `logs/`, caches, or generated worktrees.
- Prefer dataclasses for multi-value returns and keep public boundaries typed and validated.
- Do not push, open PRs, modify external systems, or use destructive Git commands unless the user asks.

## Common commands

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest -q
uv run python -m agent.run AAPL
uv run python -m agent.eval --golden-set
uv run python -m agent.eval --golden-set --provider openai --service-tier flex --reasoning-effort medium
uv run python -m eval.staged --output-dir local/eval-followup --repetitions 3
uv run streamlit run dashboard/app.py
```

Provider comparison evals keep the top-level grade-list JSON unchanged and write WAL-derived
usage metrics beside it as `<output>.usage`. Non-Anthropic evals still require
`ANTHROPIC_API_KEY` because the primary semantic judge remains Sonnet; pass
`--agreement-judge openai` to measure independently blinded inter-rater agreement.

For a normal code change, run focused tests first and then the full checks above before claiming completion. If a check is slow or interrupted, report that rather than implying it passed.

## Definition of done

Before completion, verify the diff is scoped, run Ruff, mypy, and pytest, and summarize changed files plus verification results. If commands or repository structure changed, update this file and the relevant Codex skill in the same change.
