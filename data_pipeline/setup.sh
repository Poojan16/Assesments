#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/venv"
PIDS_FILE="/tmp/pipeline.pids"

# ── 1. Virtual environment ────────────────────────────────────────────────────
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ── 2. Install dependencies ───────────────────────────────────────────────────
pip install --upgrade pip --quiet
pip install -r "$SCRIPT_DIR/requirements.txt" --quiet

# ── 3. Copy .env if missing ───────────────────────────────────────────────────
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo ".env created from .env.example — fill in your secrets before continuing."
fi

# ── 4. Django migrations ──────────────────────────────────────────────────────
python "$SCRIPT_DIR/dashboard/manage.py" migrate

# ── 5. Start Django dev server (background) ───────────────────────────────────
python "$SCRIPT_DIR/dashboard/manage.py" runserver 0.0.0.0:8000 &
DJANGO_PID=$!

# ── 6. Start FTP watcher (background) ────────────────────────────────────────
python "$SCRIPT_DIR/watcher/ftp_watcher.py" &
WATCHER_PID=$!

# ── 7. Persist PIDs ──────────────────────────────────────────────────────────
echo "$DJANGO_PID $WATCHER_PID" > "$PIDS_FILE"
echo "Started Django (PID $DJANGO_PID) and watcher (PID $WATCHER_PID)."
echo "PIDs saved to $PIDS_FILE"
