"""test_idempotency.py — watcher never re-processes a file with the same checksum."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from watcher.ftp_watcher import (
    _already_processed,
    _file_checksum,
    _mark_processed,
    _poll_once,
    _save_state,
)

_FAKE_CHECKSUM = "abc123def456" * 4  # 48-char fake hex digest
_FILENAME = "transactions.xlsx"


# ── Unit-level idempotency helpers ────────────────────────────────────────────

def test_already_processed_returns_false_for_new_file() -> None:
    """A filename not yet in state must not be considered processed."""
    assert _already_processed({}, _FILENAME, _FAKE_CHECKSUM) is False


def test_already_processed_returns_true_for_same_checksum() -> None:
    """Same filename + same checksum → skip."""
    state = {_FILENAME: {"checksum": _FAKE_CHECKSUM, "processed_at": "2024-01-01T00:00:00+00:00"}}
    assert _already_processed(state, _FILENAME, _FAKE_CHECKSUM) is True


def test_already_processed_returns_false_for_changed_checksum() -> None:
    """Same filename but different checksum → re-process."""
    state = {_FILENAME: {"checksum": "old_checksum", "processed_at": "2024-01-01T00:00:00+00:00"}}
    assert _already_processed(state, _FILENAME, _FAKE_CHECKSUM) is False


# ── Integration-level: poll_once calls pipeline exactly once ──────────────────

def test_pipeline_called_once_for_duplicate_file(tmp_path: Path) -> None:
    """Submitting the same file twice in one poll must trigger pipeline once.

    The second pass detects the checksum match and skips without calling
    ``_run_pipeline``.
    """
    # Create a real file so _file_checksum works
    fake_file = tmp_path / _FILENAME
    fake_file.write_bytes(b"fake xlsx content")

    state: dict = {}

    with (
        patch("watcher.ftp_watcher._list_ftp", return_value=[_FILENAME]),
        patch("watcher.ftp_watcher._download_ftp") as mock_download,
        patch("watcher.ftp_watcher._INCOMING_DIR", tmp_path),
        patch("watcher.ftp_watcher._run_with_retry", return_value=True) as mock_pipeline,
        patch("watcher.ftp_watcher._save_state"),
    ):
        # First poll — file is new
        _poll_once("ftp", state)
        assert mock_pipeline.call_count == 1

        # Second poll — same file, same checksum → must be skipped
        _poll_once("ftp", state)
        assert mock_pipeline.call_count == 1, (
            "Pipeline was called a second time for an already-processed file"
        )


def test_state_records_file_only_once(tmp_path: Path) -> None:
    """processed_files.json must contain exactly one entry after two polls."""
    fake_file = tmp_path / _FILENAME
    fake_file.write_bytes(b"fake xlsx content")

    state_file = tmp_path / "processed_files.json"
    state: dict = {}

    def fake_save(s: dict) -> None:
        state_file.write_text(json.dumps(s), encoding="utf-8")

    with (
        patch("watcher.ftp_watcher._list_ftp", return_value=[_FILENAME]),
        patch("watcher.ftp_watcher._download_ftp"),
        patch("watcher.ftp_watcher._INCOMING_DIR", tmp_path),
        patch("watcher.ftp_watcher._run_with_retry", return_value=True),
        patch("watcher.ftp_watcher._save_state", side_effect=fake_save),
    ):
        _poll_once("ftp", state)
        _poll_once("ftp", state)

    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    assert len(persisted) == 1
    assert _FILENAME in persisted
