"""pandas_ingestor.py — Excel ingestion, validation, and cleaning."""

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

REQUIRED_COLUMNS: list[str] = [
    "txn_id", "account_id", "txn_ts", "amount", "currency", "narration"
]


# ── Custom exceptions ─────────────────────────────────────────────────────────

class SchemaValidationError(Exception):
    """Raised when the ingested file is missing one or more required columns."""


class DataTypeError(Exception):
    """Raised when a column cannot be coerced to its expected dtype."""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_utc_timestamp(series: pd.Series) -> pd.Series:
    """Parse a mixed-format timestamp series to UTC-aware datetimes.

    Handles ISO 8601 strings with tz offsets, plain datetime strings,
    and malformed values (coerced to NaT).

    Args:
        series: Raw string series from the Excel file.

    Returns:
        A :class:`pandas.Series` of ``datetime64[ns, UTC]``, with
        unparseable values as ``NaT``.
    """
    def _to_utc(val: object) -> "pd.Timestamp | pd.NaT":
        try:
            ts = pd.to_datetime(val, utc=False)
            if ts.tzinfo is None:
                return ts.tz_localize("UTC")
            return ts.tz_convert("UTC")
        except Exception:
            return pd.NaT

    parsed = series.map(_to_utc)
    nat_count = parsed.isna().sum()
    if nat_count:
        log.warning("txn_ts: %d value(s) could not be parsed → NaT", nat_count)
    return parsed


def _validate_schema(df: pd.DataFrame) -> None:
    """Assert that *df* contains every required column.

    Args:
        df: DataFrame to validate.

    Raises:
        SchemaValidationError: If one or more required columns are absent.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"Missing required column(s): {missing}"
        )
    log.info("Schema validation passed — all %d columns present.", len(REQUIRED_COLUMNS))


def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to their expected dtypes.

    - ``txn_id``    → ``string`` (pandas StringDtype)
    - ``txn_ts``    → UTC-aware datetime via :func:`_parse_utc_timestamp`
    - ``amount``    → ``float64``, errors coerced to ``NaN``

    Args:
        df: DataFrame after schema validation.

    Returns:
        DataFrame with corrected dtypes.

    Raises:
        DataTypeError: If ``amount`` cannot be cast even with coercion.
    """
    df = df.copy()

    # txn_id → string, drop duplicates
    df["txn_id"] = df["txn_id"].astype("string")
    before = len(df)
    df = df.drop_duplicates(subset=["txn_id"])
    dropped = before - len(df)
    if dropped:
        log.warning("txn_id: dropped %d duplicate row(s).", dropped)

    # txn_ts → UTC datetime
    df["txn_ts"] = _parse_utc_timestamp(df["txn_ts"].astype(str))

    # amount → float64
    try:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").astype("float64")
    except Exception as exc:
        raise DataTypeError(f"Failed to cast 'amount' to float64: {exc}") from exc

    log.info("Dtype coercion complete.")
    return df


def _fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values according to per-column rules.

    - ``amount``    → ``0.0``  (logs a warning with the null count)
    - ``currency``  → ``"UNKNOWN"``
    - ``narration`` → ``""``

    Args:
        df: DataFrame after dtype fixing.

    Returns:
        DataFrame with nulls filled.
    """
    df = df.copy()

    amount_nulls = df["amount"].isna().sum()
    if amount_nulls:
        log.warning("amount: filling %d null(s) with 0.0", amount_nulls)
    df["amount"] = df["amount"].fillna(0.0)

    currency_nulls = df["currency"].isna().sum()
    if currency_nulls:
        log.info("currency: filling %d null(s) with 'UNKNOWN'", currency_nulls)
    df["currency"] = df["currency"].fillna("UNKNOWN")

    narration_nulls = df["narration"].isna().sum()
    if narration_nulls:
        log.info("narration: filling %d null(s) with ''", narration_nulls)
    df["narration"] = df["narration"].fillna("")

    return df


# ── Public API ────────────────────────────────────────────────────────────────

def ingest(file_path: str | Path) -> pd.DataFrame:
    """Read, validate, and clean a transaction Excel file.

    Pipeline:
    1. Read ``.xlsx`` with the ``openpyxl`` engine.
    2. Validate all required columns are present.
    3. Fix dtypes (txn_id, txn_ts, amount).
    4. Fill missing values.

    Args:
        file_path: Path to the ``.xlsx`` file to ingest.

    Returns:
        A cleaned :class:`pandas.DataFrame` ready for downstream use.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        SchemaValidationError: If required columns are missing.
        DataTypeError: If dtype coercion fails.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    log.info("Reading file: %s", path)
    df = pd.read_excel(path, engine="openpyxl")
    log.info("Loaded %d rows × %d columns.", *df.shape)

    _validate_schema(df)
    df = _fix_dtypes(df)
    df = _fill_nulls(df)

    log.info("Ingestion complete — %d clean rows returned.", len(df))
    return df
