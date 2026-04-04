# SFTP File Listener

Production-ready SFTP directory watcher with SQLite idempotency tracking, structured logging, and a configurable webhook trigger.

---

## Quick Start

```bash
# 1. Copy and fill in credentials
cp watcher/.env.sftp.example .env

# 2. Install dependencies (paramiko + python-dotenv already in requirements.txt)
pip install -r requirements.txt

# 3. Run
python watcher/sftp_watcher.py

# 4. Stop cleanly
kill -TERM <pid>   # or Ctrl-C
```

---

## Environment Variables

All variables are read from `.env` (or the real environment). Copy `.env.sftp.example` to `.env` and fill in real values.

| Variable | Required | Default | Description |
|---|---|---|---|
| `SFTP_HOST` | **Yes** | — | SFTP server hostname or IP |
| `SFTP_PORT` | No | `22` | SFTP port |
| `SFTP_USER` | **Yes** | — | SSH username |
| `SFTP_PASSWORD` | One of these | — | Password authentication |
| `SFTP_KEY_PATH` | One of these | — | Absolute path to SSH private key (RSA or Ed25519) |
| `SFTP_REMOTE_DIR` | No | `/incoming` | Remote directory to watch |
| `SFTP_FILE_PATTERN` | No | `*.xlsx` | Glob pattern for files to process |
| `POLL_INTERVAL_SECONDS` | No | `60` | Seconds between polls |
| `STAGING_DIR` | No | `tmp/incoming` | Local download directory |
| `DB_PATH` | No | `watcher/sftp_seen.db` | SQLite idempotency database path |
| `WEBHOOK_URL` | No | `""` (disabled) | HTTP(S) endpoint called on new file |
| `WEBHOOK_TIMEOUT_SECONDS` | No | `10` | Webhook POST timeout |
| `LOG_LEVEL` | No | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `LOG_FORMAT` | No | `text` | `text` (dev) or `json` (production) |

At least one of `SFTP_PASSWORD` or `SFTP_KEY_PATH` must be set. If both are provided, key auth takes precedence.

---

## How It Works

### Polling loop

```
while not shutdown:
    list remote files matching SFTP_FILE_PATTERN
    for each file:
        if (filename, mtime) in SQLite DB → skip          ← idempotency gate
        download to STAGING_DIR
        POST webhook
        INSERT (filename, mtime) into SQLite DB           ← mark as seen
    sleep POLL_INTERVAL_SECONDS
```

### Idempotency design

The idempotency key is `(filename, last_modified_unix_timestamp)`.

- **Why mtime, not a content checksum?** The check happens *before* downloading, so no bytes are transferred for already-seen files. A checksum would require downloading the file first.
- **Why not filename alone?** If the same filename is re-uploaded with new content, the mtime changes and the file is processed again — correct behaviour.
- **Crash safety:** A file is only marked as seen *after* both the download and the webhook succeed. If the process crashes mid-file, the next poll retries it.
- **Restart safety:** The SQLite database persists across restarts. `INSERT OR IGNORE` prevents duplicate rows even if two processes race on the same file.

### Webhook payload

```json
{
  "event":       "file_arrived",
  "filename":    "transactions_2024_06_01.xlsx",
  "mtime":       1717200000,
  "size_bytes":  204800,
  "local_path":  "/app/tmp/incoming/transactions_2024_06_01.xlsx",
  "detected_at": "2024-06-01T10:00:00+00:00"
}
```

The Django trigger endpoint (`POST /dashboard/trigger/?file=<filename>`) is a drop-in target for `WEBHOOK_URL`.

---

## Running the Tests

```bash
# All watcher tests (no network, no SFTP server needed)
pytest tests/test_sftp_watcher.py -v

# Full test suite
pytest tests/ -v
```

Test coverage:

| Test class | What it covers |
|---|---|
| `TestSeenFilesDB` | SQLite schema creation, is_seen/mark_seen, idempotent inserts, persistence across restarts |
| `TestPollOnceIdempotency` | Already-seen skip, new file download+webhook, failure → not marked seen, two-poll scenario, shutdown mid-cycle |
| `TestConfigValidation` | Missing required vars, invalid port, valid minimal config |
| `TestCallWebhook` | No-op when URL empty, correct payload shape |

---

## Extending the Processing Trigger

The watcher calls `call_webhook()` after every successful download. To replace or augment it:

**Option 1 — Change the webhook target**
Set `WEBHOOK_URL` to any HTTP endpoint. The existing Django trigger endpoint works out of the box:
```
WEBHOOK_URL=http://localhost:8000/dashboard/trigger/
```

**Option 2 — Run the pipeline inline (no HTTP hop)**
Replace the `call_webhook(cfg, rf, local_path)` call in `poll_once()` with a direct function call:
```python
from dashboard.dashboard.pipeline_runner import run_pipeline_for_file
run_pipeline_for_file(local_path)
```

**Option 3 — Enqueue to Celery / Redis**
```python
from myapp.tasks import process_file
process_file.delay(str(local_path))
```

**Option 4 — Publish to an SNS / SQS topic**
```python
import boto3
boto3.client("sns").publish(
    TopicArn=os.environ["SNS_TOPIC_ARN"],
    Message=json.dumps({"local_path": str(local_path)}),
)
```

All four options require only a one-line change inside `poll_once()` — the idempotency, download, and retry logic remain unchanged.

---

## Files

| File | Purpose |
|---|---|
| `watcher/sftp_watcher.py` | Main watcher: Config, SeenFilesDB, poll loop, webhook |
| `watcher/.env.sftp.example` | Template listing every supported env var |
| `tests/test_sftp_watcher.py` | Offline unit + integration tests |
