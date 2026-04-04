"""test_sftp_watcher.py — unit and integration tests for sftp_watcher.py.

All tests run fully offline: no real SFTP server, no network calls.
Paramiko and urllib are patched at the boundary so only the watcher's own
logic is exercised.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to build test objects without touching the network or filesystem
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, webhook_url: str = "") -> "Config":
    """Return a Config wired to tmp_path with no real SFTP credentials."""
    from watcher.sftp_watcher import Config

    return Config(
        host="sftp.test",
        port=22,
        user="testuser",
        password="testpass",
        key_path=None,
        remote_dir="/incoming",
        file_pattern="*.xlsx",
        poll_interval=30,
        staging_dir=tmp_path / "staging",
        db_path=tmp_path / "seen.db",
        webhook_url=webhook_url,
        webhook_timeout=5,
    )


def _make_remote_file(filename: str = "txns.xlsx", mtime: int = 1_700_000_000, size: int = 1024):
    from watcher.sftp_watcher import RemoteFile
    return RemoteFile(filename=filename, mtime=mtime, size=size)


# ── SeenFilesDB tests ─────────────────────────────────────────────────────────

class TestSeenFilesDB:
    """Tests for the SQLite idempotency store."""

    def test_new_file_is_not_seen(self, tmp_path: Path) -> None:
        """A (filename, mtime) pair not yet recorded must return False."""
        from watcher.sftp_watcher import SeenFilesDB

        db = SeenFilesDB(tmp_path / "seen.db")
        assert db.is_seen("txns.xlsx", 1_700_000_000) is False

    def test_file_seen_after_mark(self, tmp_path: Path) -> None:
        """After mark_seen, is_seen must return True for the same key."""
        from watcher.sftp_watcher import SeenFilesDB

        db = SeenFilesDB(tmp_path / "seen.db")
        db.mark_seen("txns.xlsx", 1_700_000_000, tmp_path / "txns.xlsx")
        assert db.is_seen("txns.xlsx", 1_700_000_000) is True

    def test_same_filename_new_mtime_is_not_seen(self, tmp_path: Path) -> None:
        """Same filename but a newer mtime must be treated as a new file."""
        from watcher.sftp_watcher import SeenFilesDB

        db = SeenFilesDB(tmp_path / "seen.db")
        db.mark_seen("txns.xlsx", 1_700_000_000, tmp_path / "txns.xlsx")

        # Re-uploaded file: same name, newer mtime → must NOT be skipped
        assert db.is_seen("txns.xlsx", 1_700_000_001) is False

    def test_mark_seen_is_idempotent(self, tmp_path: Path) -> None:
        """Calling mark_seen twice for the same key must not raise or duplicate rows."""
        from watcher.sftp_watcher import SeenFilesDB

        db = SeenFilesDB(tmp_path / "seen.db")
        db.mark_seen("txns.xlsx", 1_700_000_000, tmp_path / "txns.xlsx")
        db.mark_seen("txns.xlsx", 1_700_000_000, tmp_path / "txns.xlsx")  # second call

        conn = sqlite3.connect(str(tmp_path / "seen.db"))
        count = conn.execute(
            "SELECT COUNT(*) FROM seen_files WHERE filename='txns.xlsx' AND mtime=1700000000"
        ).fetchone()[0]
        conn.close()
        assert count == 1, "Duplicate row inserted by second mark_seen call"

    def test_schema_created_on_init(self, tmp_path: Path) -> None:
        """SeenFilesDB.__init__ must create the seen_files table."""
        from watcher.sftp_watcher import SeenFilesDB

        db_path = tmp_path / "seen.db"
        SeenFilesDB(db_path)

        conn = sqlite3.connect(str(db_path))
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "seen_files" in tables

    def test_persists_across_restarts(self, tmp_path: Path) -> None:
        """State must survive process restart (new SeenFilesDB instance)."""
        from watcher.sftp_watcher import SeenFilesDB

        db_path = tmp_path / "seen.db"

        # First "process"
        db1 = SeenFilesDB(db_path)
        db1.mark_seen("txns.xlsx", 1_700_000_000, tmp_path / "txns.xlsx")

        # Second "process" — new instance, same DB file
        db2 = SeenFilesDB(db_path)
        assert db2.is_seen("txns.xlsx", 1_700_000_000) is True


# ── poll_once idempotency tests ───────────────────────────────────────────────

class TestPollOnceIdempotency:
    """Tests that poll_once skips already-seen files and processes new ones."""

    def test_already_seen_file_is_skipped(self, tmp_path: Path) -> None:
        """Core requirement: a file already in the DB must not be downloaded again."""
        from watcher.sftp_watcher import SeenFilesDB, poll_once

        cfg = _make_config(tmp_path)
        db = SeenFilesDB(cfg.db_path)
        rf = _make_remote_file()

        # Pre-populate the DB as if this file was processed in a previous run
        db.mark_seen(rf.filename, rf.mtime, cfg.staging_dir / rf.filename)

        shutdown = threading.Event()

        with (
            patch("watcher.sftp_watcher.list_remote_files", return_value=[rf]),
            patch("watcher.sftp_watcher.download_file") as mock_dl,
            patch("watcher.sftp_watcher.call_webhook") as mock_wh,
        ):
            poll_once(cfg, db, shutdown)

        # Neither download nor webhook should have been called
        mock_dl.assert_not_called()
        mock_wh.assert_not_called()

    def test_new_file_is_downloaded_and_webhook_called(self, tmp_path: Path) -> None:
        """A file not yet in the DB must be downloaded and the webhook called."""
        from watcher.sftp_watcher import SeenFilesDB, poll_once

        cfg = _make_config(tmp_path, webhook_url="http://localhost/hook")
        db = SeenFilesDB(cfg.db_path)
        rf = _make_remote_file()
        local_path = cfg.staging_dir / rf.filename
        shutdown = threading.Event()

        with (
            patch("watcher.sftp_watcher.list_remote_files", return_value=[rf]),
            patch("watcher.sftp_watcher.download_file", return_value=local_path) as mock_dl,
            patch("watcher.sftp_watcher.call_webhook") as mock_wh,
        ):
            poll_once(cfg, db, shutdown)

        mock_dl.assert_called_once_with(cfg, rf)
        mock_wh.assert_called_once_with(cfg, rf, local_path)

    def test_file_marked_seen_after_successful_processing(self, tmp_path: Path) -> None:
        """After a successful poll, the file must be in the DB."""
        from watcher.sftp_watcher import SeenFilesDB, poll_once

        cfg = _make_config(tmp_path)
        db = SeenFilesDB(cfg.db_path)
        rf = _make_remote_file()
        local_path = cfg.staging_dir / rf.filename
        shutdown = threading.Event()

        with (
            patch("watcher.sftp_watcher.list_remote_files", return_value=[rf]),
            patch("watcher.sftp_watcher.download_file", return_value=local_path),
            patch("watcher.sftp_watcher.call_webhook"),
        ):
            poll_once(cfg, db, shutdown)

        assert db.is_seen(rf.filename, rf.mtime) is True

    def test_file_not_marked_seen_when_download_fails(self, tmp_path: Path) -> None:
        """If download raises, the file must NOT be marked seen (retry next poll)."""
        from watcher.sftp_watcher import SeenFilesDB, poll_once

        cfg = _make_config(tmp_path)
        db = SeenFilesDB(cfg.db_path)
        rf = _make_remote_file()
        shutdown = threading.Event()

        with (
            patch("watcher.sftp_watcher.list_remote_files", return_value=[rf]),
            patch("watcher.sftp_watcher.download_file", side_effect=OSError("network error")),
            patch("watcher.sftp_watcher.call_webhook"),
        ):
            poll_once(cfg, db, shutdown)  # must not raise

        # File must NOT be in the DB — next poll should retry it
        assert db.is_seen(rf.filename, rf.mtime) is False

    def test_file_not_marked_seen_when_webhook_fails(self, tmp_path: Path) -> None:
        """If the webhook raises, the file must NOT be marked seen."""
        from watcher.sftp_watcher import SeenFilesDB, poll_once

        cfg = _make_config(tmp_path, webhook_url="http://localhost/hook")
        db = SeenFilesDB(cfg.db_path)
        rf = _make_remote_file()
        local_path = cfg.staging_dir / rf.filename
        shutdown = threading.Event()

        with (
            patch("watcher.sftp_watcher.list_remote_files", return_value=[rf]),
            patch("watcher.sftp_watcher.download_file", return_value=local_path),
            patch(
                "watcher.sftp_watcher.call_webhook",
                side_effect=RuntimeError("webhook down"),
            ),
        ):
            poll_once(cfg, db, shutdown)  # must not raise

        assert db.is_seen(rf.filename, rf.mtime) is False

    def test_second_poll_skips_already_processed_file(self, tmp_path: Path) -> None:
        """Simulates two consecutive polls: second poll must skip the file entirely."""
        from watcher.sftp_watcher import SeenFilesDB, poll_once

        cfg = _make_config(tmp_path)
        db = SeenFilesDB(cfg.db_path)
        rf = _make_remote_file()
        local_path = cfg.staging_dir / rf.filename
        shutdown = threading.Event()

        with (
            patch("watcher.sftp_watcher.list_remote_files", return_value=[rf]),
            patch("watcher.sftp_watcher.download_file", return_value=local_path) as mock_dl,
            patch("watcher.sftp_watcher.call_webhook"),
        ):
            poll_once(cfg, db, shutdown)   # first poll — processes the file
            poll_once(cfg, db, shutdown)   # second poll — must skip

        # download_file must have been called exactly once across both polls
        assert mock_dl.call_count == 1, (
            f"download_file called {mock_dl.call_count} time(s); expected 1"
        )

    def test_multiple_files_processed_independently(self, tmp_path: Path) -> None:
        """Each file is tracked independently; one seen file does not block others."""
        from watcher.sftp_watcher import RemoteFile, SeenFilesDB, poll_once

        cfg = _make_config(tmp_path)
        db = SeenFilesDB(cfg.db_path)

        rf_old = _make_remote_file("old.xlsx", mtime=1_000_000_000)
        rf_new = _make_remote_file("new.xlsx", mtime=1_700_000_000)

        # Mark only the old file as already seen
        db.mark_seen(rf_old.filename, rf_old.mtime, cfg.staging_dir / rf_old.filename)

        local_new = cfg.staging_dir / rf_new.filename
        shutdown = threading.Event()

        with (
            patch("watcher.sftp_watcher.list_remote_files", return_value=[rf_old, rf_new]),
            patch("watcher.sftp_watcher.download_file", return_value=local_new) as mock_dl,
            patch("watcher.sftp_watcher.call_webhook"),
        ):
            poll_once(cfg, db, shutdown)

        # Only the new file should have been downloaded
        mock_dl.assert_called_once_with(cfg, rf_new)

    def test_shutdown_event_stops_mid_cycle(self, tmp_path: Path) -> None:
        """Setting the shutdown event mid-cycle must stop processing remaining files."""
        from watcher.sftp_watcher import RemoteFile, SeenFilesDB, poll_once

        cfg = _make_config(tmp_path)
        db = SeenFilesDB(cfg.db_path)
        shutdown = threading.Event()

        rf1 = _make_remote_file("a.xlsx", mtime=1_000_000_001)
        rf2 = _make_remote_file("b.xlsx", mtime=1_000_000_002)

        local_path = cfg.staging_dir / rf1.filename

        def _download_and_shutdown(c, rf):
            # Set shutdown after the first file is downloaded
            shutdown.set()
            return local_path

        with (
            patch("watcher.sftp_watcher.list_remote_files", return_value=[rf1, rf2]),
            patch("watcher.sftp_watcher.download_file", side_effect=_download_and_shutdown) as mock_dl,
            patch("watcher.sftp_watcher.call_webhook"),
        ):
            poll_once(cfg, db, shutdown)

        # Only the first file should have been attempted
        assert mock_dl.call_count == 1


# ── Config validation tests ───────────────────────────────────────────────────

class TestConfigValidation:
    """Tests for Config.from_env() startup validation."""

    def test_missing_host_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EnvironmentError raised when SFTP_HOST is absent."""
        from watcher.sftp_watcher import Config

        monkeypatch.delenv("SFTP_HOST", raising=False)
        monkeypatch.setenv("SFTP_USER", "u")
        monkeypatch.setenv("SFTP_PASSWORD", "p")

        with pytest.raises(EnvironmentError, match="SFTP_HOST"):
            Config.from_env()

    def test_missing_auth_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EnvironmentError raised when neither SFTP_PASSWORD nor SFTP_KEY_PATH is set."""
        from watcher.sftp_watcher import Config

        monkeypatch.setenv("SFTP_HOST", "sftp.test")
        monkeypatch.setenv("SFTP_USER", "u")
        monkeypatch.delenv("SFTP_PASSWORD", raising=False)
        monkeypatch.delenv("SFTP_KEY_PATH", raising=False)

        with pytest.raises(EnvironmentError, match="SFTP_PASSWORD"):
            Config.from_env()

    def test_invalid_port_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EnvironmentError raised when SFTP_PORT is not a valid integer."""
        from watcher.sftp_watcher import Config

        monkeypatch.setenv("SFTP_HOST", "sftp.test")
        monkeypatch.setenv("SFTP_USER", "u")
        monkeypatch.setenv("SFTP_PASSWORD", "p")
        monkeypatch.setenv("SFTP_PORT", "not_a_port")

        with pytest.raises(EnvironmentError, match="SFTP_PORT"):
            Config.from_env()

    def test_valid_minimal_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config.from_env() succeeds with only the required vars set."""
        from watcher.sftp_watcher import Config

        monkeypatch.setenv("SFTP_HOST", "sftp.test")
        monkeypatch.setenv("SFTP_USER", "u")
        monkeypatch.setenv("SFTP_PASSWORD", "p")
        # Clear optional vars so defaults are used
        for var in ("SFTP_PORT", "SFTP_KEY_PATH", "POLL_INTERVAL_SECONDS",
                    "WEBHOOK_URL", "WEBHOOK_TIMEOUT_SECONDS"):
            monkeypatch.delenv(var, raising=False)

        cfg = Config.from_env()
        assert cfg.host == "sftp.test"
        assert cfg.port == 22
        assert cfg.poll_interval == 60
        assert cfg.webhook_url == ""


# ── Webhook tests ─────────────────────────────────────────────────────────────

class TestCallWebhook:
    """Tests for the webhook delivery function."""

    def test_no_call_when_url_empty(self, tmp_path: Path) -> None:
        """call_webhook must be a no-op when webhook_url is empty."""
        from watcher.sftp_watcher import call_webhook

        cfg = _make_config(tmp_path, webhook_url="")
        rf = _make_remote_file()

        with patch("urllib.request.urlopen") as mock_urlopen:
            call_webhook(cfg, rf, tmp_path / rf.filename)

        mock_urlopen.assert_not_called()

    def test_payload_shape(self, tmp_path: Path) -> None:
        """Webhook POST body must contain all required fields."""
        from watcher.sftp_watcher import call_webhook

        cfg = _make_config(tmp_path, webhook_url="http://localhost/hook")
        rf = _make_remote_file()
        local_path = tmp_path / rf.filename

        captured_body: list[bytes] = []

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        def _fake_urlopen(req, timeout):
            captured_body.append(req.data)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            call_webhook(cfg, rf, local_path)

        assert len(captured_body) == 1
        payload = json.loads(captured_body[0])
        assert payload["event"] == "file_arrived"
        assert payload["filename"] == rf.filename
        assert payload["mtime"] == rf.mtime
        assert payload["size_bytes"] == rf.size
        assert "local_path" in payload
        assert "detected_at" in payload
