#!/usr/bin/env bash
# ops.sh — archive processed files, compress, and purge old archives
#
# Cron example (run daily at 02:00):
#   0 2 * * * /path/to/data_pipeline/ops.sh >> /var/log/data_pipeline_ops.log 2>&1
#
# Usage: ./ops.sh [--help]
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

Archives processed Excel files from \$DATA_DIR into a timestamped tar.gz under
\$ARCHIVE_DIR, then deletes originals and purges archives older than
\$ARCHIVE_RETAIN_DAYS days.

Key env vars:
  DATA_DIR              Source directory of processed .xlsx files  (default: \$PROJECT_ROOT/data)
  ARCHIVE_DIR           Destination for compressed archives        (default: \$PROJECT_ROOT/archives)
  LOG_DIR               Directory for ops log                      (default: \$PROJECT_ROOT/logs)
  ARCHIVE_RETAIN_DAYS   Days to keep archives before deletion      (default: 30)
  DATA_RETAIN_DAYS      Days after which originals are deleted     (default: 7)
EOF
}

[[ "${1:-}" == "--help" ]] && { usage; exit 0; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Load .env ─────────────────────────────────────────────────────────────────
ENV_FILE="${PROJECT_ROOT}/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -o allexport
  # shellcheck disable=SC1090
  source <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$ENV_FILE" | sed 's/[[:space:]]*#.*//')
  set +o allexport
fi

# ── Resolve paths / defaults ──────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${PROJECT_ROOT}/archives}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
ARCHIVE_RETAIN_DAYS="${ARCHIVE_RETAIN_DAYS:-30}"
DATA_RETAIN_DAYS="${DATA_RETAIN_DAYS:-7}"

mkdir -p "$ARCHIVE_DIR" "$LOG_DIR"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
ARCHIVE_NAME="processed_files_${TIMESTAMP}.tar.gz"
ARCHIVE_PATH="${ARCHIVE_DIR}/${ARCHIVE_NAME}"
STAGING_DIR=$(mktemp -d)          # temp dir to collect files before archiving
# Ensure staging dir is removed on exit (even on error)
trap 'rm -rf "$STAGING_DIR"' EXIT

# ── Counters for summary ──────────────────────────────────────────────────────
files_archived=0
bytes_archived=0
archives_purged=0
bytes_freed=0

# ── 1. Collect .xlsx files older than DATA_RETAIN_DAYS ───────────────────────
info "Scanning ${DATA_DIR} for .xlsx files older than ${DATA_RETAIN_DAYS} days…"

# -mtime +N means strictly older than N days
while IFS= read -r -d '' file; do
  cp "$file" "$STAGING_DIR/"
  size=$(stat -c '%s' "$file")    # file size in bytes (GNU stat, Linux)
  (( files_archived++ ))
  (( bytes_archived += size ))
done < <(find "$DATA_DIR" -maxdepth 1 -name '*.xlsx' -mtime "+${DATA_RETAIN_DAYS}" -print0)

# ── 2. Create archive (only if there is something to archive) ─────────────────
if (( files_archived == 0 )); then
  info "No files eligible for archiving — nothing to do."
else
  info "Archiving ${files_archived} file(s) ($(( bytes_archived / 1024 )) KB) → ${ARCHIVE_PATH}"
  # -C changes into staging dir so the archive contains bare filenames, not full paths
  tar -czf "$ARCHIVE_PATH" -C "$STAGING_DIR" .

  # ── 3. Delete originals that were successfully archived ───────────────────
  info "Removing archived originals from ${DATA_DIR}…"
  find "$DATA_DIR" -maxdepth 1 -name '*.xlsx' -mtime "+${DATA_RETAIN_DAYS}" -delete
  info "Originals removed."
fi

# ── 4. Purge archives older than ARCHIVE_RETAIN_DAYS ─────────────────────────
info "Purging archives older than ${ARCHIVE_RETAIN_DAYS} days from ${ARCHIVE_DIR}…"
while IFS= read -r -d '' old_archive; do
  size=$(stat -c '%s' "$old_archive")
  (( archives_purged++ ))
  (( bytes_freed += size ))
  rm -f "$old_archive"
done < <(find "$ARCHIVE_DIR" -maxdepth 1 -name '*.tar.gz' -mtime "+${ARCHIVE_RETAIN_DAYS}" -print0)

# ── 5. Human-readable summary ─────────────────────────────────────────────────
echo ""
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}  ops.sh — archive summary  ${TIMESTAMP}${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
printf "  %-28s %s\n" "Files archived:"        "${files_archived}"
printf "  %-28s %s\n" "Data compressed:"       "$(( bytes_archived / 1024 )) KB"
printf "  %-28s %s\n" "Archive written:"       "${files_archived > 0 && echo "$ARCHIVE_PATH" || echo "n/a"}"
printf "  %-28s %s\n" "Old archives purged:"   "${archives_purged}"
printf "  %-28s %s\n" "Disk space freed:"      "$(( bytes_freed / 1024 )) KB"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo ""
