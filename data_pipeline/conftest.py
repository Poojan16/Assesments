"""conftest.py — shared pytest configuration and fixtures."""

import sys
import uuid
from pathlib import Path

import pandas as pd
import pytest

# Ensure data_pipeline/ root is importable from every test file
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture()
def minimal_df() -> pd.DataFrame:
    """Return a minimal valid 5-row transaction DataFrame.

    Returns:
        A :class:`pandas.DataFrame` with all 6 required columns,
        UTC-aware ``txn_ts``, and no nulls.
    """
    return pd.DataFrame(
        {
            "txn_id": [str(uuid.uuid4()) for _ in range(5)],
            "account_id": ["ACC_001"] * 5,
            "txn_ts": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ).tz_localize("UTC"),
            "amount": [100.0, 200.0, 300.0, 400.0, 500.0],
            "currency": ["USD"] * 5,
            "narration": ["test narration"] * 5,
        }
    )
