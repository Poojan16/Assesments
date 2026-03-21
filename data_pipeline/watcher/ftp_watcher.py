"""ftp_watcher.py — FTP/SFTP directory watcher that drives the full pipeline."""

import ftplib
import hashlib
import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import paramiko
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Local imports ─────────────────────────────────────────────────────────────
from ingestion.pandas_ingestor import ingest
from analytics.polars_analytics import run_analytics
from persistence.pickle_store import save as pickle_save
from persistence.pg_store import get_connection, ensure_schema, insert_dataframe

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE = Path(__file__).parent.parent
_INCOMING_DIR = _BASE / "tmp" / "incoming"
_STATE_FILE = Path(__file__).parent / "processed_files.json"

# ── Shutdown flag ─────────────────────────────────────────────────────────────
_shutdown = False


def _handle_signal(signum: int, frame: object) -> None:
    """Set the global shutdown flag on SIGTERM or SIGINT.

    Args:
        signum: Signal number received.
        frame: Current stack frame (unused).
    """
    global _shutdown
    log.info("Signal %d received — finishing current file then exiting.", signum)
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ── State helpers ─────────────────────────────────────────────────────────────

def _load_state() -> dict[str, Any]:
    """Load the processed-files state from disk.

    Returns:
        Dict mapping filename → ``{"checksum": str, "processed_at": str}``.
        Returns an empty dict if the state file does not exist.
    """
    if _STATE_FILE.exists():
        return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_state(state: dict[str, Any]) -> None:
    """Persist the processed-files state to disk.

    Args:
        state: Current state dict to serialise.
    """
    _STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _file_checksum(path: Path) -> str:
    """Compute the SHA-256 hex digest of a local file.

    Args:
        path: Path to the file.

    Returns:
        Lowercase hex SHA-256 string.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def _already_processed(state: dict[str, Any], filename: str, checksum: str) -> bool:
    """Return True if *filename* with *checksum* has already been processed.

    Args:
        state: Current state dict.
        filename: Remote filename.
        checksum: SHA-256 of the downloaded file.

    Returns:
        ``True`` if the file should be skipped.
    """
    entry = state.get(filename)
    return entry is not None and entry.get("checksum") == checksum


def _mark_processed(state: dict[str, Any], filename: str, checksum: str) -> None:
    """Record *filename* as successfully processed in *state*.

    Args:
        state: State dict to mutate in-place.
        filename: Remote filename.
        checksum: SHA-256 of the processed file.
    """
    state[filename] = {
        "checksum": checksum,
        "processed_at": datetime.now(tz=timezone.utc).isoformat(),
    }


# ── FTP / SFTP download ───────────────────────────────────────────────────────

def _list_ftp() -> list[str]:
    """List ``.xlsx`` files in the FTP remote directory.

    Returns:
        List of filenames (not full paths).
    """
    with ftplib.FTP() as ftp:
        ftp.connect(
            host=os.environ["FTP_HOST"],
            port=int(os.environ.get("FTP_PORT", 21)),
        )
        ftp.login(
            user=os.environ["FTP_USER"],
            passwd=os.environ["FTP_PASSWORD"],
        )
        ftp.cwd(os.environ["FTP_REMOTE_DIR"])
        files = [f for f in ftp.nlst() if f.endswith(".xlsx")]
    log.info("FTP: found %d .xlsx file(s).", len(files))
    return files


def _download_ftp(filename: str, dest: Path) -> None:
    """Download *filename* from the FTP remote directory to *dest*.

    Args:
        filename: Name of the remote file.
        dest: Local destination path.
    """
    with ftplib.FTP() as ftp:
        ftp.connect(
            host=os.environ["FTP_HOST"],
            port=int(os.environ.get("FTP_PORT", 21)),
        )
        ftp.login(
            user=os.environ["FTP_USER"],
            passwd=os.environ["FTP_PASSWORD"],
        )
        ftp.cwd(os.environ["FTP_REMOTE_DIR"])
        with dest.open("wb") as fh:
            ftp.retrbinary(f"RETR {filename}", fh.write)
    log.info("FTP: downloaded '%s' → %s", filename, dest)


def _list_sftp() -> list[str]:
    """List ``.xlsx`` files in the SFTP remote directory.

    Returns:
        List of filenames (not full paths).
    """
    key_path = os.path.expanduser(os.environ["SFTP_KEY_PATH"])
    pkey = paramiko.RSAKey.from_private_key_file(key_path)
    transport = paramiko.Transport((
        os.environ["SFTP_HOST"],
        int(os.environ.get("SFTP_PORT", 22)),
    ))
    transport.connect(username=os.environ["SFTP_USER"], pkey=pkey)
    sftp = paramiko.SFTPClient.from_transport(transport)
    try:
        remote_dir = os.environ["SFTP_REMOTE_DIR"]
        files = [f for f in sftp.listdir(remote_dir) if f.endswith(".xlsx")]
    finally:
        sftp.close()
        transport.close()
    log.info("SFTP: found %d .xlsx file(s).", len(files))
    return files


def _download_sftp(filename: str, dest: Path) -> None:
    """Download *filename* from the SFTP remote directory to *dest*.

    Args:
        filename: Name of the remote file.
        dest: Local destination path.
    """
    key_path = os.path.expanduser(os.environ["SFTP_KEY_PATH"])
    pkey = paramiko.RSAKey.from_private_key_file(key_path)
    transport = paramiko.Transport((
        os.environ["SFTP_HOST"],
        int(os.environ.get("SFTP_PORT", 22)),
    ))
    transport.connect(username=os.environ["SFTP_USER"], pkey=pkey)
    sftp = paramiko.SFTPClient.from_transport(transport)
    try:
        remote_path = f"{os.environ['SFTP_REMOTE_DIR']}/{filename}"
        sftp.get(remote_path, str(dest))
    finally:
        sftp.close()
        transport.close()
    log.info("SFTP: downloaded '%s' → %s", filename, dest)


# ── Pipeline trigger ──────────────────────────────────────────────────────────

def _run_pipeline(local_path: Path) -> None:
    """Run the full ingestion pipeline for a single downloaded file.

    Steps: ingest → analytics → pickle_save → pg insert.

    Args:
        local_path: Path to the downloaded ``.xlsx`` file.
    """
    log.info("Pipeline start: %s", local_path.name)

    df = ingest(local_path)
    log.info("Ingestion complete — %d rows.", len(df))

    run_analytics(df)
    log.info("Analytics complete.")

    pkl_path = pickle_save(df, version=1)
    log.info("Pickle saved → %s", pkl_path)

    conn = get_connection()
    try:
        ensure_schema(conn)
        insert_dataframe(df, conn)
    finally:
        conn.close()

    log.info("Pipeline complete: %s", local_path.name)


def _run_with_retry(local_path: Path, max_attempts: int = 3) -> bool:
    """Run the pipeline with exponential backoff on failure.

    Waits 2 s, 4 s, 8 s between attempts.

    Args:
        local_path: Path to the downloaded ``.xlsx`` file.
        max_attempts: Maximum number of attempts before giving up.

    Returns:
        ``True`` if the pipeline succeeded, ``False`` after all retries failed.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            _run_pipeline(local_path)
            return True
        except Exception as exc:
            wait = 2 ** attempt
            log.warning(
                "Attempt %d/%d failed for '%s': %s. "
                "%s",
                attempt, max_attempts, local_path.name, exc,
                f"Retrying in {wait}s …" if attempt < max_attempts else "Giving up.",
            )
            if attempt < max_attempts:
                time.sleep(wait)
    return False


# ── Main poll loop ────────────────────────────────────────────────────────────

def _poll_once(mode: str, state: dict[str, Any]) -> None:
    """Perform one poll cycle: list remote files, download new ones, run pipeline.

    Args:
        mode: ``"ftp"`` or ``"sftp"``.
        state: Mutable processed-files state dict (updated in-place on success).
    """
    list_fn = _list_ftp if mode == "ftp" else _list_sftp
    download_fn = _download_ftp if mode == "ftp" else _download_sftp

    try:
        remote_files = list_fn()
    except Exception as exc:
        log.error("Failed to list remote directory: %s", exc)
        return

    _INCOMING_DIR.mkdir(parents=True, exist_ok=True)

    for filename in remote_files:
        if _shutdown:
            break

        dest = _INCOMING_DIR / filename
        try:
            download_fn(filename, dest)
        except Exception as exc:
            log.error("Failed to download '%s': %s — skipping.", filename, exc)
            continue

        checksum = _file_checksum(dest)
        if _already_processed(state, filename, checksum):
            log.info("'%s' already processed (checksum match) — skipping.", filename)
            continue

        success = _run_with_retry(dest)
        if success:
            _mark_processed(state, filename, checksum)
            _save_state(state)
        else:
            log.error("'%s' failed after all retries — marked as skipped.", filename)


def run() -> None:
    """Start the watcher poll loop.

    Reads ``WATCHER_MODE`` (default ``ftp``) and
    ``WATCHER_POLL_INTERVAL`` (default ``30`` seconds) from env vars.
    Loops until a SIGTERM or SIGINT is received.
    """
    mode = os.environ.get("WATCHER_MODE", "ftp").lower()
    interval = int(os.environ.get("WATCHER_POLL_INTERVAL", 30))

    if mode not in ("ftp", "sftp"):
        raise ValueError(f"WATCHER_MODE must be 'ftp' or 'sftp', got: {mode!r}")

    log.info("Watcher starting — mode=%s, interval=%ds", mode, interval)
    state = _load_state()

    while not _shutdown:
        log.info("Polling remote directory …")
        _poll_once(mode, state)
        if not _shutdown:
            log.info("Sleeping %ds until next poll.", interval)
            time.sleep(interval)

    log.info("Watcher shut down cleanly.")


if __name__ == "__main__":
    run()
