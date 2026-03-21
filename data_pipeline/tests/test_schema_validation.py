"""test_schema_validation.py — SchemaValidationError raised on missing columns."""

import uuid
from pathlib import Path

import pandas as pd
import pytest

from ingestion.pandas_ingestor import SchemaValidationError, _validate_schema


def _df_missing_column(drop: str) -> pd.DataFrame:
    """Build a valid 3-row DataFrame then drop *drop* column.

    Args:
        drop: Column name to remove.

    Returns:
        DataFrame with 5 required columns instead of 6.
    """
    df = pd.DataFrame(
        {
            "txn_id": [str(uuid.uuid4()) for _ in range(3)],
            "account_id": ["ACC_001", "ACC_002", "ACC_003"],
            "txn_ts": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "amount": [10.0, 20.0, 30.0],
            "currency": ["USD", "EUR", "GBP"],
            "narration": ["a", "b", "c"],
        }
    )
    return df.drop(columns=[drop])


@pytest.mark.parametrize("missing_col", [
    "txn_id", "account_id", "txn_ts", "amount", "currency", "narration"
])
def test_schema_validation_raises_for_missing_column(missing_col: str) -> None:
    """SchemaValidationError must be raised for each individually missing column.

    Args:
        missing_col: The column name to drop before validation.
    """
    df = _df_missing_column(missing_col)
    with pytest.raises(SchemaValidationError) as exc_info:
        _validate_schema(df)
    assert missing_col in str(exc_info.value)


def test_schema_validation_passes_for_complete_df() -> None:
    """No exception raised when all 6 required columns are present."""
    df = pd.DataFrame(columns=["txn_id", "account_id", "txn_ts", "amount", "currency", "narration"])
    _validate_schema(df)  # must not raise
