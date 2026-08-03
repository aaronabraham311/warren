#!/bin/bash
# Install a cron entry that runs Warren nightly at 2 AM (Linux).
set -euo pipefail

if [[ "$(uname)" != "Linux" ]]; then
  echo "error: install_cron.sh is for Linux only. Use install_scheduler.sh on macOS." >&2
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

if [[ ! -f "$VENV_PYTHON" ]]; then
  echo "error: virtualenv not found at $VENV_PYTHON — run 'uv sync' first." >&2
  exit 1
fi

ENV_FILE="$PROJECT_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "error: .env not found at $ENV_FILE — copy .env.example and fill in API keys." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "error: ANTHROPIC_API_KEY is not set in $ENV_FILE." >&2
  exit 1
fi

FLOCK_BIN="$(command -v flock || true)"
if [[ -z "$FLOCK_BIN" ]]; then
  echo "error: flock not found — install util-linux (e.g. 'apt install util-linux')." >&2
  exit 1
fi

mkdir -p "$PROJECT_DIR/logs"

# No API keys on the cron line: agent/run.py's load_dotenv() reads $PROJECT_DIR/.env
# directly (resolved once cwd is the project dir — see the `cd` below), so nothing
# needs to be injected into the environment here. Keeping secrets out of the crontab
# line also keeps them out of `ps`/`/proc/<pid>/environ`, which is world-readable on
# most systems.
# `cd` is required: `python -m agent.run` resolves the module against cwd, and cron's
# default cwd is the user's home directory, not this repo — without it the job fails
# every night with "No module named 'agent'".
# flock -n skips this run instead of stacking a second one if the prior run is still going.
CRON_ENTRY="0 2 * * * cd $PROJECT_DIR && $FLOCK_BIN -n $PROJECT_DIR/logs/.nightly.lock $VENV_PYTHON -m agent.run >> $PROJECT_DIR/logs/cron.log 2>&1"

# Idempotent: skip if an entry for this project already exists.
if crontab -l 2>/dev/null | grep -qF "$VENV_PYTHON -m agent.run"; then
  echo "Warren cron entry already installed — no changes made."
  exit 0
fi

(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -
echo "Warren cron entry installed. Next run: tonight at 2 AM."
echo "Logs: $PROJECT_DIR/logs/cron.log"
