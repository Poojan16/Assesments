"""test_rolling_metric.py — rolling_7d_sum correctness on a known dataset."""

import uuid
from datetime import date, timedelta

import pandas as pd
import polars as pl
import pytest

from analytics.polars_analytics import run_analytics


def _build_synthetic_df() -> pd.DataFrame:
    """Build a 30-row DataFrame: 2 accounts × 15 consecutive days.

    Each row has:
    - amount = 100.0 (USD), so amount_usd = 100.0 after conversion
    - one row per (account, day) — no duplicates

    Returns:
        A :class:`pandas.DataFrame` with all 6 required columns and
        UTC-aware ``txn_ts``.
    """
    base_date = date(2024, 1, 1)
    rows = []
    for account in ["ACC_001", "ACC_002"]:
        for day_offset in range(15):
            day = base_date + timedelta(days=day_offset)
            rows.append(
                {
                    "txn_id": str(uuid.uuid4()),
                    "account_id": account,
                    "txn_ts": pd.Timestamp(day).tz_localize("UTC"),
                    "amount": 100.0,
                    "currency": "USD",
                    "narration": "test payment",
                }
            )
    return pd.DataFrame(rows)


def test_rolling_7d_sum_known_value() -> None:
    """rolling_7d_sum for ACC_001 on day 7 (index 6) must equal 700.0.

    Days 1–7 each contribute 100.0 USD → sum = 700.0.
    """
    df = _build_synthetic_df()
    result: pl.DataFrame = run_analytics(df)

    acc1 = (
        result.filter(pl.col("account_id") == "ACC_001")
        .sort("date")
    )

    # Day index 6 = 2024-01-07 (7th day, window is full)
    rolling_on_day7 = acc1["rolling_7d_sum"][6]
    assert rolling_on_day7 == pytest.approx(700.0, rel=1e-6), (
        f"Expected 700.0, got {rolling_on_day7}"
    )


def test_rolling_7d_sum_accounts_are_independent() -> None:
    """Rolling window must not bleed across account boundaries."""
    df = _build_synthetic_df()
    result: pl.DataFrame = run_analytics(df)

    for account in ["ACC_001", "ACC_002"]:
        acc_rows = result.filter(pl.col("account_id") == account).sort("date")
        # First row window = 1 day → rolling sum == daily_total
        assert acc_rows["rolling_7d_sum"][0] == pytest.approx(
            acc_rows["daily_total"][0], rel=1e-6
        )


def test_result_schema() -> None:
    """Output DataFrame must contain the expected columns."""
    df = _build_synthetic_df()
    result: pl.DataFrame = run_analytics(df)
    expected_cols = {"account_id", "date", "daily_total", "txn_count", "rolling_7d_sum"}
    assert expected_cols.issubset(set(result.columns))
