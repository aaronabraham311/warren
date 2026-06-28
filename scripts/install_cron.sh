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

mkdir -p "$PROJECT_DIR/logs"

CRON_ENTRY="0 2 * * * ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY FINNHUB_API_KEY=${FINNHUB_API_KEY:-} $VENV_PYTHON -m agent.run >> $PROJECT_DIR/logs/cron.log 2>&1"

# Idempotent: skip if an entry for this project already exists.
if crontab -l 2>/dev/null | grep -qF "$VENV_PYTHON -m agent.run"; then
  echo "Warren cron entry already installed — no changes made."
  exit 0
fi

(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -
echo "Warren cron entry installed. Next run: tonight at 2 AM."
echo "Logs: $PROJECT_DIR/logs/cron.log"
