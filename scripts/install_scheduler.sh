#!/bin/bash
# Install Warren as a launchd LaunchAgent that runs the weekly gem hunt Sunday at 7 AM
# in the Mac's system timezone (macOS only).
set -euo pipefail

if [[ "$(uname)" != "Darwin" ]]; then
  echo "error: install_scheduler.sh is for macOS only. Use install_cron.sh on Linux." >&2
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
PLIST_NAME="com.warren.agent"
PLIST_DEST="$HOME/Library/LaunchAgents/$PLIST_NAME.plist"
PLIST_TEMPLATE="$PROJECT_DIR/scripts/$PLIST_NAME.plist.template"

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

mkdir -p "$HOME/Library/LaunchAgents"
mkdir -p "$PROJECT_DIR/logs"

# No API keys go into the plist: agent/run.py's load_dotenv() reads $PROJECT_DIR/.env
# directly (resolved from agent/run.py's own path, not cwd or the launchd environment),
# so nothing needs to be injected here. This also avoids leaving a second plaintext
# copy of the keys sitting in ~/Library/LaunchAgents.
sed \
  -e "s|/path/to/warren|$PROJECT_DIR|g" \
  -e "s|/path/to/.venv/bin/python|$VENV_PYTHON|g" \
  "$PLIST_TEMPLATE" > "$PLIST_DEST"

# Unload first so re-running this installer actually applies the new plist. launchd keeps
# the argv it loaded until the label is unloaded, so a bare `load` over an already-registered
# job silently keeps running the old command (e.g. one installed before --gem-hunt).
launchctl unload "$PLIST_DEST" 2>/dev/null || true
launchctl load "$PLIST_DEST"
echo "Warren scheduler installed. Runs Sundays at 7 AM system time (gem-hunt mode)."
echo "Logs: $PROJECT_DIR/logs/launchd_stdout.log and launchd_stderr.log"
