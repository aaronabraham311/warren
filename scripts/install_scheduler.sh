#!/bin/bash
# Install Warren as a launchd LaunchAgent that runs nightly at 2 AM (macOS only).
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

# Substitute path placeholders; API keys are injected separately via PlistBuddy
# so they never appear in the template file.
sed \
  -e "s|/path/to/warren|$PROJECT_DIR|g" \
  -e "s|/path/to/.venv/bin/python|$VENV_PYTHON|g" \
  "$PLIST_TEMPLATE" > "$PLIST_DEST"

/usr/libexec/PlistBuddy -c "Set :EnvironmentVariables:ANTHROPIC_API_KEY $ANTHROPIC_API_KEY" "$PLIST_DEST"
/usr/libexec/PlistBuddy -c "Set :EnvironmentVariables:FINNHUB_API_KEY ${FINNHUB_API_KEY:-}" "$PLIST_DEST"

launchctl load "$PLIST_DEST"
echo "Warren scheduler installed. Next run: tonight at 2 AM."
echo "Logs: $PROJECT_DIR/logs/launchd_stdout.log and launchd_stderr.log"
