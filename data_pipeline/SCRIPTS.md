# Bash Scripts Reference

Production-quality bash scripts for `data_pipeline` on Ubuntu/Debian Linux.

---

## Quick Start

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with real credentials

# 2. Bootstrap (once)
./setup.sh

# 3. Start services
./start.sh

# 4. Stop services
./stop.sh

# 5. Archive maintenance (cron)
./ops.sh
```

---

## setup.sh

**Purpose:** One-time environment bootstrap.

**Steps:**
1. Create and activate Python venv
2. Install dependencies from `requirements.txt`
3. Run Django migrations
4. Validate required env vars
5. Seed initial database data

**Required env vars:**
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- `WATCHER_MODE` (ftp | sftp)

**Optional env vars:**
- `VENV_DIR` (default: `./venv`)
- `SEED_ROWS` (default: `5000`)
- `SEED_OUTPUT` (default: `./data/sample_transactions.xlsx`)
- `DJANGO_SETTINGS_MODULE` (default: `dashboard.settings`)

**Usage:**
```bash
./setup.sh
./setup.sh --help
```

---

## start.sh

**Purpose:** Start Django web server and FTP/SFTP watcher as background processes.

**Features:**
- PID-file process tracking (`$PID_DIR/web.pid`, `$PID_DIR/watcher.pid`)
- Per-service log files (`$LOG_DIR/web.log`, `$LOG_DIR/watcher.log`)
- Startup summary showing process status
- Refuses to double-start if services are already running

**Key env vars:**
- `APP_PORT` (default: `8000`)
- `LOG_DIR` (default: `./logs`)
- `PID_DIR` (default: `./run`)
- `VENV_DIR` (default: `./venv`)
- `DJANGO_SETTINGS_MODULE` (default: `dashboard.settings`)

**Usage:**
```bash
./start.sh
./start.sh --help

# Tail logs
tail -f logs/web.log
tail -f logs/watcher.log
```

---

## stop.sh

**Purpose:** Gracefully stop both services started by `start.sh`.

**Features:**
- Sends SIGTERM to each process
- Waits up to `$STOP_TIMEOUT` seconds for graceful exit
- Escalates to SIGKILL if process is still alive
- Cleans up PID files

**Key env vars:**
- `PID_DIR` (default: `./run`)
- `STOP_TIMEOUT` (default: `10` seconds)

**Usage:**
```bash
./stop.sh
./stop.sh --help
```

---

## ops.sh

**Purpose:** Archive processed files, compress, and purge old archives.

**Steps:**
1. Find `.xlsx` files in `$DATA_DIR` older than `$DATA_RETAIN_DAYS` days
2. Copy them to a staging directory
3. Create a timestamped `tar.gz` archive in `$ARCHIVE_DIR`
4. Delete the original `.xlsx` files
5. Purge archives older than `$ARCHIVE_RETAIN_DAYS` days
6. Print a human-readable summary (files archived, bytes freed, etc.)

**Key env vars:**
- `DATA_DIR` (default: `./data`)
- `ARCHIVE_DIR` (default: `./archives`)
- `LOG_DIR` (default: `./logs`)
- `DATA_RETAIN_DAYS` (default: `7`)
- `ARCHIVE_RETAIN_DAYS` (default: `30`)

**Usage:**
```bash
./ops.sh
./ops.sh --help

# Cron example (daily at 02:00)
0 2 * * * /path/to/data_pipeline/ops.sh >> /var/log/data_pipeline_ops.log 2>&1
```

---

## Common Features (All Scripts)

- `set -euo pipefail` at the top (fail fast, no unset vars, catch pipe failures)
- Colored, timestamped log output (`info`, `warn`, `error` functions)
- `usage()` function and `--help` flag
- All paths read from env vars (no hardcoded paths)
- `.env` file auto-loaded if present

---

## Environment Variables

See `.env.example` for the complete list with inline descriptions. Key categories:

| Category | Variables |
|---|---|
| **PostgreSQL** | `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` |
| **FTP/SFTP** | `FTP_HOST`, `FTP_PORT`, `FTP_USER`, `FTP_PASSWORD`, `FTP_REMOTE_DIR`, `SFTP_*`, `WATCHER_MODE` |
| **Django** | `DJANGO_SECRET_KEY`, `DJANGO_DEBUG`, `DJANGO_ALLOWED_HOSTS`, `DJANGO_SETTINGS_MODULE` |
| **AI** | `LANGCHAIN_LLM_PROVIDER`, `OPENAI_API_KEY` |
| **Scripts** | `VENV_DIR`, `APP_PORT`, `LOG_DIR`, `PID_DIR`, `DATA_DIR`, `ARCHIVE_DIR`, retention days, etc. |

---

## Troubleshooting

**"Missing required env vars"**
- Fill in all required variables in `.env` (see `.env.example`)

**"venv not found"**
- Run `./setup.sh` first

**"Services appear to be running already"**
- Run `./stop.sh` first, or manually remove stale PID files from `$PID_DIR`

**"FAILED to start" in startup summary**
- Check the service log file in `$LOG_DIR` for error details

**ops.sh reports "No files eligible for archiving"**
- No `.xlsx` files older than `$DATA_RETAIN_DAYS` days exist in `$DATA_DIR`
