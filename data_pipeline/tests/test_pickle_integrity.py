"""test_pickle_integrity.py — PickleIntegrityError raised on corrupted pickle."""

import uuid
from pathlib import Path

import pandas as pd
import pytest

from persistence.pickle_store import PickleIntegrityError, load, save


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Return a small 10-row DataFrame suitable for pickling.

    Returns:
        A :class:`pandas.DataFrame` with all 6 transaction columns.
    """
    return pd.DataFrame(
        {
            "txn_id": [str(uuid.uuid4()) for _ in range(10)],
            "account_id": ["ACC_001"] * 10,
            "txn_ts": pd.to_datetime(["2024-01-01"] * 10).tz_localize("UTC"),
            "amount": [float(i * 10) for i in range(1, 11)],
            "currency": ["USD"] * 10,
            "narration": ["payment"] * 10,
        }
    )


def test_clean_pickle_loads_successfully(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    """An uncorrupted pickle must load without raising any exception."""
    pkl_path = save(sample_df, version=1, artifacts_dir=tmp_path)
    loaded = load(pkl_path)
    assert len(loaded) == len(sample_df)


def test_corrupted_pickle_raises_integrity_error(
    sample_df: pd.DataFrame, tmp_path: Path
) -> None:
    """Flipping a byte in the .pkl file must raise PickleIntegrityError on load.

    The SHA-256 stored in .meta.json will no longer match the file on disk.
    """
    pkl_path = save(sample_df, version=1, artifacts_dir=tmp_path)

    # Flip the last byte of the pickle file
    raw = bytearray(pkl_path.read_bytes())
    raw[-1] ^= 0xFF
    pkl_path.write_bytes(bytes(raw))

    with pytest.raises(PickleIntegrityError, match="SHA-256 mismatch"):
        load(pkl_path)


def test_missing_meta_raises_file_not_found(
    sample_df: pd.DataFrame, tmp_path: Path
) -> None:
    """Deleting the .meta.json sidecar must raise FileNotFoundError."""
    pkl_path = save(sample_df, version=1, artifacts_dir=tmp_path)
    meta_path = pkl_path.with_suffix(".meta.json")
    meta_path.unlink()

    with pytest.raises(FileNotFoundError, match="Metadata file not found"):
        load(pkl_path)
