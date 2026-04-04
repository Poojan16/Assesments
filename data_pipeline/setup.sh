#!/usr/bin/env bash
# setup.sh — one-time environment bootstrap for data_pipeline
# Usage: ./setup.sh [--help]
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

One-time bootstrap: creates venv, installs deps, runs migrations, validates
env vars, and seeds initial data.

Required env vars (set in .env):
  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
  WATCHER_MODE  (ftp | sftp)

Optional env vars:
  VENV_DIR        Path for the virtual environment  (default: \$PROJECT_ROOT/venv)
  SEED_ROWS       Rows to generate for seed data    (default: 5000)
  SEED_OUTPUT     Seed file path                    (default: \$PROJECT_ROOT/data/sample_transactions.xlsx)
  DJANGO_SETTINGS_MODULE                            (default: dashboard.settings)
EOF
}

[[ "${1:-}" == "--help" ]] && { usage; exit 0; }

# ── Resolve project root (directory containing this script) ───────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Load .env if present ──────────────────────────────────────────────────────
ENV_FILE="${PROJECT_ROOT}/.env"
if [[ -f "$ENV_FILE" ]]; then
  info "Loading environment from $ENV_FILE"
  # Export every non-comment, non-blank line; strip inline comments after value
  set -o allexport
  # shellcheck disable=SC1090
  source <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$ENV_FILE" | sed 's/[[:space:]]*#.*//')
  set +o allexport
else
  warn ".env not found — copying from .env.example"
  cp "${PROJECT_ROOT}/.env.example" "$ENV_FILE"
  error "Fill in ${ENV_FILE} with real credentials, then re-run setup.sh."
  exit 1
fi

# ── Validate required env vars ────────────────────────────────────────────────
REQUIRED_VARS=(DB_HOST DB_PORT DB_NAME DB_USER DB_PASSWORD WATCHER_MODE)
missing=()
for var in "${REQUIRED_VARS[@]}"; do
  [[ -z "${!var:-}" ]] && missing+=("$var")
done
if (( ${#missing[@]} > 0 )); then
  error "Missing required env vars: ${missing[*]}"
  error "Set them in ${ENV_FILE} and re-run."
  exit 1
fi
info "All required env vars present."

# ── Resolve configurable paths ────────────────────────────────────────────────
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/venv}"
SEED_ROWS="${SEED_ROWS:-5000}"
SEED_OUTPUT="${SEED_OUTPUT:-${PROJECT_ROOT}/data/sample_transactions.xlsx}"
export DJANGO_SETTINGS_MODULE="${DJANGO_SETTINGS_MODULE:-dashboard.settings}"

# ── 1. Virtual environment ────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
  info "Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
else
  info "Virtual environment already exists at $VENV_DIR — skipping creation."
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
info "Activated venv: $(python --version)"

# ── 2. Install dependencies ───────────────────────────────────────────────────
info "Upgrading pip…"
pip install --upgrade pip --quiet

info "Installing dependencies from requirements.txt…"
pip install -r "${PROJECT_ROOT}/requirements.txt" --quiet
info "Dependencies installed."

# ── 3. Database migrations ────────────────────────────────────────────────────
info "Running Django migrations…"
python "${PROJECT_ROOT}/dashboard/manage.py" migrate --noinput
info "Migrations complete."

# ── 4. Env-var validation already done above (step order per spec) ────────────

# ── 5. Seed initial data ──────────────────────────────────────────────────────
if [[ -f "$SEED_OUTPUT" ]]; then
  info "Seed file already exists at $SEED_OUTPUT — skipping generation."
else
  info "Generating ${SEED_ROWS} rows of seed data → ${SEED_OUTPUT}…"
  mkdir -p "$(dirname "$SEED_OUTPUT")"
  python "${PROJECT_ROOT}/generate_data.py" --rows "$SEED_ROWS" --output "$SEED_OUTPUT"
  info "Seed data written to $SEED_OUTPUT."
fi

echo -e "\n${GREEN}[$(ts)] Setup complete.${NC} Run ./start.sh to launch services.\n"
