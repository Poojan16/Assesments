"""pickle_store.py — versioned pickle persistence with integrity verification."""

import hashlib
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


# ── Custom exception ──────────────────────────────────────────────────────────

class PickleIntegrityError(Exception):
    """Raised when a loaded pickle file fails SHA-256 or row-count verification."""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        path: Path to the file to hash.

    Returns:
        Lowercase hex string of the SHA-256 digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def _meta_path(pkl_path: Path) -> Path:
    """Return the sidecar metadata path for a given pickle path.

    Args:
        pkl_path: Path to the ``.pkl`` file.

    Returns:
        Path with the same stem but ``.meta.json`` extension.
    """
    return pkl_path.with_suffix(".meta.json")


def _versioned_stem(version: int) -> str:
    """Build the filename stem for a versioned artifact.

    Format: ``processed_df_v{version}_{YYYYMMDD_HHMMSS}``

    Args:
        version: Integer version number.

    Returns:
        Filename stem string.
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"processed_df_v{version}_{ts}"


# ── Public API ────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, version: int = 1, artifacts_dir: Path = _ARTIFACTS_DIR) -> Path:
    """Pickle *df* to a versioned file and write a sidecar ``.meta.json``.

    The metadata file contains:
    - ``row_count`` — number of rows in *df*
    - ``sha256``    — SHA-256 hex digest of the ``.pkl`` file
    - ``saved_at``  — ISO 8601 UTC timestamp

    Args:
        df: DataFrame to persist.
        version: Version number embedded in the filename.
        artifacts_dir: Directory to write artifacts into (created if absent).

    Returns:
        Path to the saved ``.pkl`` file.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    stem = _versioned_stem(version)
    pkl_path = artifacts_dir / f"{stem}.pkl"

    log.info("Saving DataFrame (%d rows) → %s", len(df), pkl_path)
    with pkl_path.open("wb") as fh:
        pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)

    checksum = _sha256(pkl_path)
    saved_at = datetime.now(tz=timezone.utc).isoformat()

    meta = {
        "row_count": len(df),
        "sha256": checksum,
        "saved_at": saved_at,
    }
    meta_path = _meta_path(pkl_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info("Metadata written → %s  (sha256=%s…)", meta_path.name, checksum[:12])
    return pkl_path


def load(pkl_path: str | Path) -> pd.DataFrame:
    """Load and verify a pickled DataFrame.

    Verification steps:
    1. Recompute SHA-256 of the ``.pkl`` file and compare to ``.meta.json``.
    2. Compare the loaded DataFrame's row count to the stored ``row_count``.

    Args:
        pkl_path: Path to the ``.pkl`` file to load.

    Returns:
        The verified :class:`pandas.DataFrame`.

    Raises:
        FileNotFoundError: If the ``.pkl`` or ``.meta.json`` file is missing.
        PickleIntegrityError: If the checksum or row count does not match.
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")

    meta_path = _meta_path(pkl_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    log.info("Verifying checksum for %s …", pkl_path.name)
    actual_checksum = _sha256(pkl_path)
    if actual_checksum != meta["sha256"]:
        raise PickleIntegrityError(
            f"SHA-256 mismatch for {pkl_path.name}: "
            f"expected {meta['sha256']}, got {actual_checksum}"
        )
    log.info("Checksum OK.")

    with pkl_path.open("rb") as fh:
        df: pd.DataFrame = pickle.load(fh)

    if len(df) != meta["row_count"]:
        raise PickleIntegrityError(
            f"Row count mismatch for {pkl_path.name}: "
            f"expected {meta['row_count']}, got {len(df)}"
        )
    log.info("Row count OK (%d rows). Load complete.", len(df))
    return df
