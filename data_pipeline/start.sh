#!/usr/bin/env bash
# start.sh — start Django web server + FTP/SFTP watcher with PID-file tracking
# Usage: ./start.sh [--help]
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

Starts the Django web server and the FTP/SFTP watcher as background processes.
PIDs are written to \$PID_DIR. Logs go to \$LOG_DIR/<service>.log.
Run stop.sh to shut both processes down gracefully.

Key env vars (all have defaults):
  APP_PORT    Django listen port          (default: 8000)
  LOG_DIR     Directory for log files     (default: \$PROJECT_ROOT/logs)
  PID_DIR     Directory for PID files     (default: \$PROJECT_ROOT/run)
  VENV_DIR    Virtual-environment path    (default: \$PROJECT_ROOT/venv)
  DJANGO_SETTINGS_MODULE                  (default: dashboard.settings)
EOF
}

[[ "${1:-}" == "--help" ]] && { usage; exit 0; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Load .env ─────────────────────────────────────────────────────────────────
ENV_FILE="${PROJECT_ROOT}/.env"
[[ -f "$ENV_FILE" ]] || { error ".env not found. Run setup.sh first."; exit 1; }
set -o allexport
# shellcheck disable=SC1090
source <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$ENV_FILE" | sed 's/[[:space:]]*#.*//')
set +o allexport

# ── Resolve paths / defaults ──────────────────────────────────────────────────
APP_PORT="${APP_PORT:-8000}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
PID_DIR="${PID_DIR:-${PROJECT_ROOT}/run}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/venv}"
export DJANGO_SETTINGS_MODULE="${DJANGO_SETTINGS_MODULE:-dashboard.settings}"

mkdir -p "$LOG_DIR" "$PID_DIR"

WEB_PID_FILE="${PID_DIR}/web.pid"
WATCHER_PID_FILE="${PID_DIR}/watcher.pid"
WEB_LOG="${LOG_DIR}/web.log"
WATCHER_LOG="${LOG_DIR}/watcher.log"

# ── Activate venv ─────────────────────────────────────────────────────────────
[[ -f "${VENV_DIR}/bin/activate" ]] || { error "venv not found at $VENV_DIR. Run setup.sh first."; exit 1; }
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ── Helper: check if a PID file's process is still alive ─────────────────────
pid_alive() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

# ── Guard: refuse to double-start ─────────────────────────────────────────────
if pid_alive "$WEB_PID_FILE" || pid_alive "$WATCHER_PID_FILE"; then
  error "Services appear to be running already. Run stop.sh first."
  exit 1
fi

# ── Start Django web server ───────────────────────────────────────────────────
info "Starting Django on port ${APP_PORT}…"
# nohup keeps the process alive after the shell exits; stdout+stderr → log file
nohup python "${PROJECT_ROOT}/dashboard/manage.py" runserver "0.0.0.0:${APP_PORT}" \
  >> "$WEB_LOG" 2>&1 &
echo $! > "$WEB_PID_FILE"   # $! is the PID of the last backgrounded command

# ── Start FTP/SFTP watcher ────────────────────────────────────────────────────
info "Starting FTP/SFTP watcher (mode: ${WATCHER_MODE:-ftp})…"
nohup python "${PROJECT_ROOT}/watcher/ftp_watcher.py" \
  >> "$WATCHER_LOG" 2>&1 &
echo $! > "$WATCHER_PID_FILE"

# ── Brief pause so processes can fail fast if misconfigured ──────────────────
sleep 1

# ── Startup summary ───────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}  data_pipeline — startup summary${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"

for svc in web watcher; do
  pid_file="${PID_DIR}/${svc}.pid"
  log_file="${LOG_DIR}/${svc}.log"
  if pid_alive "$pid_file"; then
    pid=$(cat "$pid_file")
    echo -e "  ${GREEN}✔${NC} ${svc}     PID ${pid}   log → ${log_file}"
  else
    echo -e "  ${RED}✘${NC} ${svc}     FAILED to start — check ${log_file}"
  fi
done

echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "  Dashboard : http://localhost:${APP_PORT}/dashboard/"
echo -e "  Stop      : ./stop.sh"
echo ""
