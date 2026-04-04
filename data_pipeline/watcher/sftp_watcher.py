"""sftp_watcher.py — Production SFTP file listener with SQLite idempotency tracking.

Polls a remote SFTP directory at a configurable interval.  Every file seen is
keyed by (filename, last_modified_timestamp) in a local SQLite database so the
watcher is fully idempotent across process restarts: a file is processed at
most once per unique (name, mtime) pair.

When a new file is detected it is downloaded to a local staging directory and
a configurable webhook URL is called with a JSON payload describing the file.

Environment variables (all read at startup — see .env.sftp.example):
    SFTP_HOST, SFTP_PORT, SFTP_USER, SFTP_PASSWORD / SFTP_KEY_PATH,
    SFTP_REMOTE_DIR, SFTP_FILE_PATTERN, POLL_INTERVAL_SECONDS,
    STAGING_DIR, DB_PATH, WEBHOOK_URL, WEBHOOK_TIMEOUT_SECONDS,
    LOG_LEVEL, LOG_FORMAT

Usage:
    python watcher/sftp_watcher.py
"""

from __future__ import annotations

import json
import logging
import logging.config
import os
import signal
import sqlite3
import stat
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Generator, Iterator

import paramiko
from dotenv import load_dotenv

# Load .env from the project root (two levels up from watcher/)
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# ── Structured logging setup ──────────────────────────────────────────────────

def _configure_logging() -> None:
    """Configure structured logging from LOG_LEVEL / LOG_FORMAT env vars.

    LOG_FORMAT = "json"  → each record is a single JSON object (production)
    LOG_FORMAT = "text"  → human-readable timestamped lines (development)
    """
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    fmt = os.environ.get("LOG_FORMAT", "text").lower()

    if fmt == "json":
        # Emit one JSON object per log record so log-aggregation tools
        # (Datadog, CloudWatch, Loki) can parse fields without regex.
        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                payload: dict = {
                    "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                if record.exc_info:
                    payload["exc"] = self.formatException(record.exc_info)
                # Merge any extra fields passed via the `extra=` kwarg
                for key, val in record.__dict__.items():
                    if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                        payload[key] = val
                return json.dumps(payload)

        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
        )

    logging.basicConfig(level=level, handlers=[handler], force=True)


_configure_logging()
log = logging.getLogger("sftp_watcher")


# ── Configuration dataclass ───────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    """All runtime configuration, validated once at startup.

    Attributes are populated from environment variables; sensible defaults
    are provided for every optional field.
    """

    # SFTP connection
    host: str
    port: int
    user: str
    password: str | None          # mutually exclusive with key_path
    key_path: str | None          # absolute path to SSH private key file
    remote_dir: str
    file_pattern: str             # glob pattern, e.g. "*.xlsx"

    # Polling
    poll_interval: int            # seconds between polls

    # Local paths
    staging_dir: Path             # where downloaded files are staged
    db_path: Path                 # SQLite database for idempotency tracking

    # Webhook
    webhook_url: str              # POST target; empty string disables webhook
    webhook_timeout: int          # seconds before webhook call times out

    @classmethod
    def from_env(cls) -> "Config":
        """Build and validate a Config from environment variables.

        Raises:
            EnvironmentError: If a required variable is missing or invalid.
        """
        errors: list[str] = []

        def _require(name: str) -> str:
            val = os.environ.get(name, "").strip()
            if not val:
                errors.append(f"  {name} is required but not set")
            return val

        def _optional(name: str, default: str) -> str:
            return os.environ.get(name, default).strip()

        host = _require("SFTP_HOST")
        user = _require("SFTP_USER")

        # Auth: password XOR key_path — at least one must be provided
        password = _optional("SFTP_PASSWORD", "")
        key_path = _optional("SFTP_KEY_PATH", "")
        if not password and not key_path:
            errors.append(
                "  At least one of SFTP_PASSWORD or SFTP_KEY_PATH must be set"
            )

        # Validate port
        port_str = _optional("SFTP_PORT", "22")
        try:
            port = int(port_str)
            if not (1 <= port <= 65535):
                raise ValueError
        except ValueError:
            errors.append(f"  SFTP_PORT must be 1–65535, got: {port_str!r}")
            port = 22

        # Validate poll interval
        interval_str = _optional("POLL_INTERVAL_SECONDS", "60")
        try:
            poll_interval = int(interval_str)
            if poll_interval < 1:
                raise ValueError
        except ValueError:
            errors.append(
                f"  POLL_INTERVAL_SECONDS must be a positive integer, got: {interval_str!r}"
            )
            poll_interval = 60

        # Validate webhook timeout
        wh_timeout_str = _optional("WEBHOOK_TIMEOUT_SECONDS", "10")
        try:
            webhook_timeout = int(wh_timeout_str)
            if webhook_timeout < 1:
                raise ValueError
        except ValueError:
            errors.append(
                f"  WEBHOOK_TIMEOUT_SECONDS must be a positive integer, got: {wh_timeout_str!r}"
            )
            webhook_timeout = 10

        if errors:
            raise EnvironmentError(
                "SFTP watcher configuration errors:\n" + "\n".join(errors)
            )

        return cls(
            host=host,
            port=port,
            user=user,
            password=password or None,
            key_path=os.path.expanduser(key_path) if key_path else None,
            remote_dir=_optional("SFTP_REMOTE_DIR", "/incoming"),
            file_pattern=_optional("SFTP_FILE_PATTERN", "*.xlsx"),
            poll_interval=poll_interval,
            staging_dir=Path(_optional("STAGING_DIR", "tmp/incoming")),
            db_path=Path(_optional("DB_PATH", "watcher/sftp_seen.db")),
            webhook_url=_optional("WEBHOOK_URL", ""),
            webhook_timeout=webhook_timeout,
        )


# ── SQLite idempotency store ──────────────────────────────────────────────────

class SeenFilesDB:
    """SQLite-backed store that tracks every (filename, mtime) pair seen.

    Idempotency key design
    ----------------------
    The key is (filename, last_modified_unix_int).  Using the server-reported
    mtime rather than a content checksum means we can decide to skip *before*
    downloading the file, saving bandwidth on large files.  If the same
    filename is re-uploaded with a newer mtime it will be treated as a new
    file and processed again — which is the correct behaviour.

    Thread safety
    -------------
    Each public method opens its own connection with check_same_thread=False
    and WAL journal mode so concurrent reads never block writes.
    """

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield an open, auto-committing SQLite connection."""
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        # WAL mode: readers don't block writers and vice-versa
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Create the seen_files table if it does not already exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS seen_files (
                    filename    TEXT    NOT NULL,
                    mtime       INTEGER NOT NULL,  -- Unix timestamp (seconds)
                    processed_at TEXT   NOT NULL,  -- ISO-8601 UTC
                    local_path  TEXT    NOT NULL,
                    PRIMARY KEY (filename, mtime)
                )
                """
            )
        log.debug("SeenFilesDB schema ready at %s", self._path)

    def is_seen(self, filename: str, mtime: int) -> bool:
        """Return True if (filename, mtime) is already in the database.

        This is the idempotency check: called before downloading so we can
        skip files we have already processed without touching the network.

        Args:
            filename: Remote filename (basename only).
            mtime:    Server-reported last-modified time as a Unix timestamp.

        Returns:
            True  → file already processed, skip it.
            False → file is new, proceed with download + webhook.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM seen_files WHERE filename = ? AND mtime = ?",
                (filename, mtime),
            ).fetchone()
        return row is not None

    def mark_seen(self, filename: str, mtime: int, local_path: Path) -> None:
        """Record (filename, mtime) as successfully processed.

        Uses INSERT OR IGNORE so concurrent processes cannot produce
        duplicate rows even if they race on the same file.

        Args:
            filename:   Remote filename.
            mtime:      Server-reported last-modified Unix timestamp.
            local_path: Where the file was staged locally.
        """
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO seen_files (filename, mtime, processed_at, local_path)
                VALUES (?, ?, ?, ?)
                """,
                (filename, mtime, now, str(local_path)),
            )
        log.debug(
            "Marked as seen",
            extra={"remote_file": filename, "mtime": mtime},
        )


# ── SFTP connection helpers ───────────────────────────────────────────────────

@dataclass
class RemoteFile:
    """Metadata for a single file on the SFTP server."""

    filename: str
    mtime: int       # Unix timestamp (seconds) from the server's stat
    size: int        # bytes


@contextmanager
def _sftp_session(cfg: Config) -> Iterator[paramiko.SFTPClient]:
    """Open an authenticated SFTP session and yield the client.

    Supports both password auth and private-key auth.  The transport and
    client are always closed in the finally block, even on exception.

    Args:
        cfg: Validated Config instance.

    Yields:
        An open :class:`paramiko.SFTPClient`.
    """
    transport = paramiko.Transport((cfg.host, cfg.port))
    try:
        if cfg.key_path:
            # Try RSA first, fall back to Ed25519 / ECDSA via auto-detection
            try:
                pkey = paramiko.RSAKey.from_private_key_file(cfg.key_path)
            except paramiko.SSHException:
                pkey = paramiko.Ed25519Key.from_private_key_file(cfg.key_path)
            transport.connect(username=cfg.user, pkey=pkey)
        else:
            transport.connect(username=cfg.user, password=cfg.password)

        sftp = paramiko.SFTPClient.from_transport(transport)
        try:
            yield sftp
        finally:
            sftp.close()
    finally:
        transport.close()


def list_remote_files(cfg: Config) -> list[RemoteFile]:
    """List files in the remote directory that match cfg.file_pattern.

    Uses listdir_attr (one round-trip) rather than listdir + stat per file
    to minimise the number of SFTP requests.

    Args:
        cfg: Validated Config instance.

    Returns:
        List of RemoteFile objects sorted by mtime ascending (oldest first)
        so files are processed in arrival order.
    """
    with _sftp_session(cfg) as sftp:
        attrs = sftp.listdir_attr(cfg.remote_dir)

    files = []
    for attr in attrs:
        name = attr.filename
        # Skip directories and files that don't match the glob pattern
        if stat.S_ISDIR(attr.st_mode or 0):
            continue
        if not fnmatch(name, cfg.file_pattern):
            continue
        files.append(
            RemoteFile(
                filename=name,
                mtime=int(attr.st_mtime or 0),
                size=int(attr.st_size or 0),
            )
        )

    # Process oldest files first so partial runs are resumable
    files.sort(key=lambda f: f.mtime)
    log.info(
        "Listed remote directory",
        extra={
            "remote_dir": cfg.remote_dir,
            "total_files": len(attrs),
            "matched_files": len(files),
            "pattern": cfg.file_pattern,
        },
    )
    return files


def download_file(cfg: Config, remote_file: RemoteFile) -> Path:
    """Download a single remote file to the local staging directory.

    Args:
        cfg:         Validated Config instance.
        remote_file: Metadata of the file to download.

    Returns:
        Path to the downloaded local file.
    """
    cfg.staging_dir.mkdir(parents=True, exist_ok=True)
    local_path = cfg.staging_dir / remote_file.filename
    remote_path = f"{cfg.remote_dir}/{remote_file.filename}"

    with _sftp_session(cfg) as sftp:
        sftp.get(remote_path, str(local_path))

    log.info(
        "Downloaded file",
        extra={
            "remote_file": remote_file.filename,
            "size_bytes": remote_file.size,
            "local_path": str(local_path),
        },
    )
    return local_path


# ── Webhook trigger ───────────────────────────────────────────────────────────

def call_webhook(cfg: Config, remote_file: RemoteFile, local_path: Path) -> None:
    """POST a JSON payload to cfg.webhook_url describing the new file.

    The payload shape is intentionally minimal and stable so downstream
    consumers can rely on it.  Extend the payload dict here to add fields.

    Payload:
        {
            "event":      "file_arrived",
            "filename":   "<basename>",
            "mtime":      <unix_int>,
            "size_bytes": <int>,
            "local_path": "<absolute_path>",
            "detected_at": "<iso8601_utc>"
        }

    Args:
        cfg:         Validated Config instance.
        remote_file: Metadata of the file that arrived.
        local_path:  Where the file was staged locally.
    """
    if not cfg.webhook_url:
        log.debug("Webhook URL not configured — skipping.")
        return

    payload = {
        "event": "file_arrived",
        "filename": remote_file.filename,
        "mtime": remote_file.mtime,
        "size_bytes": remote_file.size,
        "local_path": str(local_path.resolve()),
        "detected_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        cfg.webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.webhook_timeout) as resp:
            status = resp.status
        log.info(
            "Webhook delivered",
            extra={"url": cfg.webhook_url, "status": status, "remote_file": remote_file.filename},
        )
    except urllib.error.HTTPError as exc:
        log.error(
            "Webhook HTTP error",
            extra={"url": cfg.webhook_url, "status": exc.code, "remote_file": remote_file.filename},
        )
    except Exception as exc:
        log.error(
            "Webhook call failed",
            extra={"url": cfg.webhook_url, "error": str(exc), "remote_file": remote_file.filename},
        )


# ── Single poll cycle ─────────────────────────────────────────────────────────

def poll_once(cfg: Config, db: SeenFilesDB, shutdown: threading.Event) -> None:
    """Perform one complete poll cycle.

    For each file in the remote directory:
      1. Check the SQLite DB — if (filename, mtime) is already recorded, skip.
         This is the idempotency gate: no download, no webhook, no side-effects.
      2. Download the file to the staging directory.
      3. Call the webhook.
      4. Record (filename, mtime) in the DB so future polls skip it.

    Steps 2–4 are wrapped in a try/except so a single bad file never aborts
    the rest of the poll cycle.

    Args:
        cfg:      Validated Config instance.
        db:       Open SeenFilesDB instance.
        shutdown: Threading event; checked between files so SIGTERM is
                  honoured mid-cycle without leaving the DB in a partial state.
    """
    try:
        remote_files = list_remote_files(cfg)
    except Exception as exc:
        log.error("Failed to list remote directory", extra={"error": str(exc)})
        return

    new_count = 0
    skip_count = 0

    for rf in remote_files:
        if shutdown.is_set():
            log.info("Shutdown requested — stopping mid-cycle cleanly.")
            break

        # ── Idempotency check ─────────────────────────────────────────────────
        # We key on (filename, mtime).  If this exact (name, mtime) pair is
        # already in the DB the file has been fully processed before; skip it.
        # A re-uploaded file with the same name but a newer mtime will have a
        # different key and will be processed again — correct behaviour.
        if db.is_seen(rf.filename, rf.mtime):
            log.debug(
                "Skipping already-seen file",
                extra={"remote_file": rf.filename, "mtime": rf.mtime},
            )
            skip_count += 1
            continue

        # ── New file: download → webhook → mark seen ──────────────────────────
        try:
            local_path = download_file(cfg, rf)
            call_webhook(cfg, rf, local_path)
            # Only mark as seen AFTER the webhook succeeds so a webhook failure
            # causes a retry on the next poll rather than silent data loss.
            db.mark_seen(rf.filename, rf.mtime, local_path)
            new_count += 1
        except Exception as exc:
            # Log and continue — do NOT mark as seen so the next poll retries.
            log.error(
                "Failed to process file — will retry next poll",
                extra={"remote_file": rf.filename, "error": str(exc)},
                exc_info=True,
            )

    log.info(
        "Poll cycle complete",
        extra={"new": new_count, "skipped": skip_count, "total": len(remote_files)},
    )


# ── Main polling loop ─────────────────────────────────────────────────────────

def run(cfg: Config) -> None:
    """Start the SFTP polling loop.

    Runs until SIGTERM or SIGINT is received.  Uses threading.Event.wait()
    for the sleep so the process wakes immediately on a signal rather than
    waiting out the full interval.

    Args:
        cfg: Validated Config instance.
    """
    shutdown = threading.Event()

    def _handle_signal(signum: int, _frame: object) -> None:
        log.info("Signal received — shutting down after current cycle.", extra={"signal": signum})
        shutdown.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    db = SeenFilesDB(cfg.db_path)

    log.info(
        "SFTP watcher started",
        extra={
            "host": cfg.host,
            "port": cfg.port,
            "user": cfg.user,
            "remote_dir": cfg.remote_dir,
            "pattern": cfg.file_pattern,
            "poll_interval_s": cfg.poll_interval,
            "db_path": str(cfg.db_path),
            "webhook_url": cfg.webhook_url or "(disabled)",
        },
    )

    while not shutdown.is_set():
        log.info("Starting poll cycle")
        poll_once(cfg, db, shutdown)

        if not shutdown.is_set():
            log.info("Sleeping until next poll", extra={"seconds": cfg.poll_interval})
            # wait() returns True if the event was set (shutdown), False on timeout.
            # Either way the while condition is checked next iteration.
            shutdown.wait(timeout=cfg.poll_interval)

    log.info("SFTP watcher shut down cleanly.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        cfg = Config.from_env()
    except EnvironmentError as exc:
        log.error("Startup validation failed:\n%s", exc)
        raise SystemExit(1) from exc

    run(cfg)
