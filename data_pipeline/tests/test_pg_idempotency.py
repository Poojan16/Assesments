"""test_pg_idempotency.py — duplicate inserts leave row count unchanged.

Requires a live PostgreSQL instance with pgvector.
Skipped automatically when DB env vars are absent.
"""

import os
import uuid

import pandas as pd
import pytest

# Skip the entire module if DB credentials are not configured
pytestmark = pytest.mark.skipif(
    not all(
        os.environ.get(v)
        for v in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
    ),
    reason="PostgreSQL env vars not set — skipping pg idempotency test.",
)


@pytest.fixture(scope="module")
def pg_conn():
    """Open a psycopg2 connection for the test module, close on teardown.

    Yields:
        An open :class:`psycopg2.extensions.connection`.
    """
    from persistence.pg_store import get_connection, ensure_schema

    conn = get_connection()
    ensure_schema(conn)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def hundred_rows_df() -> pd.DataFrame:
    """Build a 100-row transaction DataFrame with unique txn_ids.

    Returns:
        A :class:`pandas.DataFrame` with all 6 required columns.
    """
    return pd.DataFrame(
        {
            "txn_id": [str(uuid.uuid4()) for _ in range(100)],
            "account_id": ["ACC_001"] * 100,
            "txn_ts": pd.to_datetime(["2024-03-01T00:00:00+00:00"] * 100, utc=True),
            "amount": [float(i) for i in range(1, 101)],
            "currency": ["USD"] * 100,
            "narration": [f"payment {i}" for i in range(1, 101)],
        }
    )


def _count_rows(conn, txn_ids: list[str]) -> int:
    """Return the number of rows in ``transactions`` matching *txn_ids*.

    Args:
        conn: Open psycopg2 connection.
        txn_ids: List of txn_id values to filter on.

    Returns:
        Integer row count.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM transactions WHERE txn_id = ANY(%s);",
            (txn_ids,),
        )
        return cur.fetchone()[0]


def test_double_insert_row_count_unchanged(pg_conn, hundred_rows_df: pd.DataFrame) -> None:
    """Inserting 100 rows twice must leave exactly 100 rows in the table.

    ON CONFLICT (txn_id) DO NOTHING guarantees idempotency.
    """
    from persistence.pg_store import insert_dataframe

    txn_ids = hundred_rows_df["txn_id"].tolist()

    # First insert
    insert_dataframe(hundred_rows_df, pg_conn)
    count_after_first = _count_rows(pg_conn, txn_ids)
    assert count_after_first == 100, (
        f"Expected 100 rows after first insert, got {count_after_first}"
    )

    # Second insert — identical rows
    insert_dataframe(hundred_rows_df, pg_conn)
    count_after_second = _count_rows(pg_conn, txn_ids)
    assert count_after_second == 100, (
        f"Expected 100 rows after second insert, got {count_after_second}"
    )


def test_double_insert_no_duplicate_txn_ids(pg_conn, hundred_rows_df: pd.DataFrame) -> None:
    """Each txn_id must appear exactly once after two inserts."""
    from persistence.pg_store import insert_dataframe

    insert_dataframe(hundred_rows_df, pg_conn)
    insert_dataframe(hundred_rows_df, pg_conn)

    txn_ids = hundred_rows_df["txn_id"].tolist()
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT txn_id, COUNT(*) FROM transactions "
            "WHERE txn_id = ANY(%s) GROUP BY txn_id HAVING COUNT(*) > 1;",
            (txn_ids,),
        )
        duplicates = cur.fetchall()

    assert duplicates == [], f"Duplicate txn_ids found: {duplicates}"
