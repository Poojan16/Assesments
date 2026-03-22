"""pg_store.py — PostgreSQL + pgvector persistence for transactions."""

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from psycopg2.extensions import connection as PgConnection

log = logging.getLogger(__name__)

_BATCH_SIZE = 500
_EMBEDDING_DIM = 128
_MODEL_NAME = "all-MiniLM-L6-v2"  # native output: 384 dims → truncated to 128

# Module-level model cache — loaded once, reused across calls
_model = None


# ── DB connection ─────────────────────────────────────────────────────────────

def get_connection() -> PgConnection:
    """Create and return a psycopg2 connection using env-var credentials.

    Reads: ``DB_HOST``, ``DB_PORT``, ``DB_NAME``, ``DB_USER``, ``DB_PASSWORD``.

    Returns:
        An open :class:`psycopg2.extensions.connection`.

    Raises:
        KeyError: If any required env var is missing.
        psycopg2.OperationalError: If the connection cannot be established.
    """
    conn = psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )
    log.info("Connected to PostgreSQL at %s:%s/%s",
             os.environ["DB_HOST"], os.environ["DB_PORT"], os.environ["DB_NAME"])
    return conn


# ── Schema bootstrap ──────────────────────────────────────────────────────────

def ensure_schema(conn: PgConnection) -> None:
    """Create the pgvector extension and ``transactions`` table if absent.

    Idempotent — safe to call on every startup.

    Args:
        conn: Open psycopg2 connection.
    """
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                txn_id      TEXT PRIMARY KEY,
                account_id  TEXT NOT NULL,
                txn_ts      TIMESTAMPTZ NOT NULL,
                amount      FLOAT,
                currency    TEXT,
                narration   TEXT,
                embedding   VECTOR(128)
            );
        """)
    conn.commit()
    log.info("Schema ready (extension: vector, table: transactions).")


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _get_model():
    """Return the cached :class:`SentenceTransformer` model, loading it once.

    Returns:
        Loaded ``all-MiniLM-L6-v2`` model instance.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading sentence-transformer model '%s' …", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
        log.info("Model loaded.")
    return _model


def _embed(texts: list[str]) -> np.ndarray:
    """Encode *texts* to 128-dimensional float32 vectors.

    The model produces 384-dim vectors; we truncate to the first 128
    dimensions and L2-normalise so cosine and L2 distances are equivalent.
    The same input text always produces the same vector.

    Args:
        texts: List of strings to encode. Empty strings are encoded as-is.

    Returns:
        Float32 array of shape ``(len(texts), 128)``.
    """
    model = _get_model()
    vecs: np.ndarray = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    truncated = vecs[:, :_EMBEDDING_DIM].astype(np.float32)
    # Re-normalise after truncation
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (truncated / norms).astype(np.float32)


# ── Insert ────────────────────────────────────────────────────────────────────

def insert_dataframe(df: pd.DataFrame, conn: PgConnection) -> None:
    """Insert *df* rows into ``transactions`` in batches of 500.

    Rows whose ``txn_id`` already exists are silently skipped
    (``ON CONFLICT DO NOTHING``).

    Embeddings are computed from the ``narration`` column before insert.

    Args:
        df: Cleaned :class:`pandas.DataFrame` with the standard 6 columns.
        conn: Open psycopg2 connection.
    """
    log.info("Computing embeddings for %d narrations …", len(df))
    narrations = df["narration"].fillna("").tolist()
    embeddings = _embed(narrations)

    sql = """
        INSERT INTO transactions
            (txn_id, account_id, txn_ts, amount, currency, narration, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (txn_id) DO NOTHING;
    """

    rows: list[tuple[Any, ...]] = [
        (
            str(row.txn_id),
            str(row.account_id),
            row.txn_ts.isoformat(),
            float(row.amount) if pd.notna(row.amount) else None,
            row.currency if pd.notna(row.currency) else None,
            row.narration if pd.notna(row.narration) else None,
            embeddings[i].tolist(),
        )
        for i, row in enumerate(df.itertuples(index=False))
        if pd.notna(row.txn_ts)
    ]

    skipped = len(df) - len(rows)
    if skipped:
        log.warning("Skipping %d row(s) with null txn_ts.", skipped)
    total = len(rows)
    inserted = 0
    with conn.cursor() as cur:
        for start in range(0, total, _BATCH_SIZE):
            batch = rows[start : start + _BATCH_SIZE]
            psycopg2.extras.execute_batch(cur, sql, batch)
            conn.commit()
            inserted += len(batch)
            log.info("Inserted batch %d–%d / %d", start + 1, inserted, total)

    log.info("Insert complete — %d rows processed.", total)


# ── Similarity search ─────────────────────────────────────────────────────────

def find_similar_narrations(
    query_text: str,
    conn: PgConnection,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Find the *top_k* transactions whose narration is closest to *query_text*.

    Uses the pgvector ``<->`` (L2 distance) operator for ANN search.

    Args:
        query_text: Free-text query to embed and search against.
        conn: Open psycopg2 connection.
        top_k: Number of nearest neighbours to return (default 5).

    Returns:
        List of dicts, each representing one matching transaction row
        with keys: ``txn_id``, ``account_id``, ``txn_ts``, ``amount``,
        ``currency``, ``narration``.
    """
    log.info("Embedding query: %r", query_text)
    query_vec = _embed([query_text])[0].tolist()

    sql = """
        SELECT txn_id, account_id, txn_ts, amount, currency, narration
        FROM transactions
        ORDER BY embedding <-> %s::vector
        LIMIT %s;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (query_vec, top_k))
        results = [dict(row) for row in cur.fetchall()]

    log.info("Similarity search returned %d result(s).", len(results))
    return results
