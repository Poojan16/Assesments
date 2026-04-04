#!/usr/bin/env bash
# stop.sh — gracefully stop Django web server and FTP/SFTP watcher
# Usage: ./stop.sh [--help]
set -euo pipefail

# ── Colour codes ──────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'

ts()    { date '+%Y-%m-%d %H:%M:%S'; }
info()  { echo -e "${CYAN}[$(ts)] INFO${NC}  $*"; }
warn()  { echo -e "${YELLOW}[$(ts)] WARN${NC}  $*" >&2; }
error() { echo -e "${RED}[$(ts)] ERROR${NC} $*" >&2; }

usage() {
  cat <<EOF
Usage: $(basename "$0") [--help]

Gracefully stops the Django web server and FTP/SFTP watcher started by start.sh.
Sends SIGTERM; escalates to SIGKILL after \$STOP_TIMEOUT seconds if needed.

Key env vars:
  PID_DIR       Directory containing PID files  (default: \$PROJECT_ROOT/run)
  STOP_TIMEOUT  Seconds to wait before SIGKILL  (default: 10)
EOF
}

[[ "${1:-}" == "--help" ]] && { usage; exit 0; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${PROJECT_ROOT}/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -o allexport
  # shellcheck disable=SC1090
  source <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$ENV_FILE" | sed 's/[[:space:]]*#.*//')
  set +o allexport
fi

PID_DIR="${PID_DIR:-${PROJECT_ROOT}/run}"
STOP_TIMEOUT="${STOP_TIMEOUT:-10}"

# ── stop_service <name> <pid_file> ────────────────────────────────────────────
stop_service() {
  local name="$1" pid_file="$2"

  if [[ ! -f "$pid_file" ]]; then
    warn "${name}: no PID file at ${pid_file} — already stopped?"
    return
  fi

  local pid
  pid=$(cat "$pid_file")

  if ! kill -0 "$pid" 2>/dev/null; then
    warn "${name}: PID ${pid} not running — cleaning up stale PID file."
    rm -f "$pid_file"
    return
  fi

  info "${name}: sending SIGTERM to PID ${pid}…"
  kill -TERM "$pid"

  # Wait up to STOP_TIMEOUT seconds for the process to exit
  local elapsed=0
  while kill -0 "$pid" 2>/dev/null && (( elapsed < STOP_TIMEOUT )); do
    sleep 1
    (( elapsed++ ))
  done

  if kill -0 "$pid" 2>/dev/null; then
    warn "${name}: still alive after ${STOP_TIMEOUT}s — sending SIGKILL."
    kill -KILL "$pid"
  fi

  rm -f "$pid_file"
  echo -e "  ${GREEN}✔${NC} ${name} stopped."
}

echo ""
stop_service "web"     "${PID_DIR}/web.pid"
stop_service "watcher" "${PID_DIR}/watcher.pid"
echo ""
info "All services stopped."
