#!/bin/bash
# Install a cron entry that runs Warren's weekly gem hunt Sunday at 7 AM ET (Linux).
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
# --gem-hunt is the scheduled mode: global 3-exchange universe + deep-value screen + Dirt
# persona. Bare `python -m agent.run` (US GARP nightly) stays available on demand.
CRON_TIMEZONE="CRON_TZ=America/New_York"
CRON_ENTRY="0 7 * * 0 cd $PROJECT_DIR && $FLOCK_BIN -n $PROJECT_DIR/logs/.nightly.lock $VENV_PYTHON -m agent.run --gem-hunt >> $PROJECT_DIR/logs/cron.log 2>&1"

# Identifies this project's entry regardless of which flags it was installed with.
CRON_MATCH="$VENV_PYTHON -m agent.run"
EXISTING="$(crontab -l 2>/dev/null || true)"

if printf '%s\n' "$EXISTING" | grep -qF "$CRON_ENTRY" \
  && printf '%s\n' "$EXISTING" | grep -qF "$CRON_TIMEZONE"; then
  echo "Warren cron entry already installed — no changes made."
  exit 0
fi

WITHOUT_WARREN="$(printf '%s\n' "$EXISTING" | grep -vF "$CRON_MATCH" || true)"

if printf '%s\n' "$EXISTING" | grep -qF "$CRON_MATCH"; then
  # An entry from an earlier install (e.g. before --gem-hunt) — replace it rather than
  # skipping, otherwise re-running this installer would leave the stale command in place.
  (printf '%s\n' "$WITHOUT_WARREN"; echo "$CRON_TIMEZONE"; echo "$CRON_ENTRY") | crontab -
  echo "Warren cron entry updated — runs Sundays at 7 AM ET (gem-hunt mode)."
else
  (printf '%s\n' "$EXISTING"; echo "$CRON_TIMEZONE"; echo "$CRON_ENTRY") | crontab -
  echo "Warren cron entry installed. Runs Sundays at 7 AM ET (gem-hunt mode)."
fi
echo "Logs: $PROJECT_DIR/logs/cron.log"
