#!/usr/bin/env bash
set -euo pipefail

PIDS_FILE="/tmp/pipeline.pids"

if [ ! -f "$PIDS_FILE" ]; then
    echo "No PID file found at $PIDS_FILE — nothing to stop."
    exit 0
fi

read -r DJANGO_PID WATCHER_PID < "$PIDS_FILE"

for PID in "$DJANGO_PID" "$WATCHER_PID"; do
    if kill -0 "$PID" 2>/dev/null; then
        kill -TERM "$PID" && echo "Sent SIGTERM to PID $PID"
    else
        echo "PID $PID is not running — skipping."
    fi
done

rm -f "$PIDS_FILE"
echo "Done."
