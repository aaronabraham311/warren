#!/bin/bash
# Remove the Warren launchd LaunchAgent (macOS only).
set -euo pipefail

if [[ "$(uname)" != "Darwin" ]]; then
  echo "error: uninstall_scheduler.sh is for macOS only." >&2
  exit 1
fi

PLIST_PATH="$HOME/Library/LaunchAgents/com.warren.agent.plist"

if [[ ! -f "$PLIST_PATH" ]]; then
  echo "error: plist not found at $PLIST_PATH — scheduler may not be installed." >&2
  exit 1
fi

launchctl unload "$PLIST_PATH"
rm "$PLIST_PATH"
echo "Warren scheduler removed."
