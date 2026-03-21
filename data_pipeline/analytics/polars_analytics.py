"""polars_analytics.py — lazy Polars analytics pipeline."""

import logging

import pandas as pd
import polars as pl

log = logging.getLogger(__name__)

# ── Currency → USD conversion rates ──────────────────────────────────────────
_FX: dict[str, float] = {
    "USD": 1.0,
    "EUR": 1.08,
    "INR": 0.012,
    "GBP": 1.27,
}


# ── Step builders (each returns a LazyFrame) ──────────────────────────────────

def _step_filter(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Remove rows where ``amount`` is zero or negative.

    Args:
        lf: Input LazyFrame.

    Returns:
        LazyFrame with only positive-amount rows.
    """
    return lf.filter(pl.col("amount") > 0)


def _step_with_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add ``date`` and ``amount_usd`` derived columns.

    - ``date``       — date portion of ``txn_ts`` (no time component).
    - ``amount_usd`` — ``amount`` converted to USD using :data:`_FX` rates;
      unrecognised currencies are treated as USD (rate 1.0).

    Args:
        lf: LazyFrame after filtering.

    Returns:
        LazyFrame with two additional columns.
    """
    fx_expr = (
        pl.when(pl.col("currency") == "EUR").then(pl.col("amount") * 1.08)
        .when(pl.col("currency") == "INR").then(pl.col("amount") * 0.012)
        .when(pl.col("currency") == "GBP").then(pl.col("amount") * 1.27)
        .otherwise(pl.col("amount"))  # USD or unknown → as-is
        .alias("amount_usd")
    )

    return lf.with_columns(
        pl.col("txn_ts").dt.date().alias("date"),
        fx_expr,
    )


def _step_groupby(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate to one row per (account_id, date).

    Computes:
    - ``daily_total`` — sum of ``amount_usd``
    - ``txn_count``   — number of transactions

    Args:
        lf: LazyFrame after column enrichment.

    Returns:
        Aggregated LazyFrame grouped by ``(account_id, date)``.
    """
    return lf.group_by(["account_id", "date"]).agg(
        pl.col("amount_usd").sum().alias("daily_total"),
        pl.len().alias("txn_count"),
    )


def _step_rolling(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Compute a 7-day rolling sum of ``daily_total`` per account.

    Sorts by ``(account_id, date)`` first, then uses
    :meth:`polars.Expr.rolling_sum` over a 7-row window
    (index-based, one row = one day after groupby).

    Args:
        lf: Aggregated LazyFrame.

    Returns:
        LazyFrame with an additional ``rolling_7d_sum`` column.
    """
    return (
        lf.sort(["account_id", "date"])
        .with_columns(
            pl.col("daily_total")
            .rolling_sum(window_size=7)
            .over("account_id")
            .alias("rolling_7d_sum")
        )
    )


# ── Public API ────────────────────────────────────────────────────────────────

def run_analytics(df: pd.DataFrame) -> pl.DataFrame:
    """Execute the full lazy analytics pipeline on a cleaned DataFrame.

    Pipeline (all lazy until the final collect):

    1. Convert ``pd.DataFrame`` → ``pl.LazyFrame``.
    2. Filter rows with ``amount <= 0``.
    3. Add ``date`` and ``amount_usd`` columns.
    4. Group by ``(account_id, date)`` → ``daily_total``, ``txn_count``.
    5. Sort and compute ``rolling_7d_sum`` per account over 7 days.
    6. ``.collect()`` once.

    Args:
        df: Cleaned :class:`pandas.DataFrame` from
            :func:`ingestion.pandas_ingestor.ingest`.

    Returns:
        A :class:`polars.DataFrame` with columns:
        ``account_id``, ``date``, ``daily_total``,
        ``txn_count``, ``rolling_7d_sum``.
    """
    log.info("Converting pandas DataFrame (%d rows) to Polars LazyFrame.", len(df))
    lf: pl.LazyFrame = pl.from_pandas(df).lazy()

    lf = _step_filter(lf)
    log.info("Step 1 — filter applied (amount > 0).")

    lf = _step_with_columns(lf)
    log.info("Step 2 — 'date' and 'amount_usd' columns added.")

    lf = _step_groupby(lf)
    log.info("Step 3 — grouped by (account_id, date).")

    lf = _step_rolling(lf)
    log.info("Step 4 — 7-day rolling sum computed.")

    log.info("Collecting lazy pipeline …")
    result: pl.DataFrame = lf.collect()
    log.info("Pipeline complete — %d rows × %d columns.", *result.shape)
    return result
