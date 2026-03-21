"""pipeline_runner.py — shared helper that executes the full pipeline for one file."""

import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Ensure data_pipeline/ root is importable when called from the dashboard sub-package
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def run_pipeline_for_file(filepath: str | Path) -> int:
    """Run ingest → analytics → pickle_save → pg_store for *filepath*.

    Args:
        filepath: Absolute or relative path to a ``.xlsx`` file.

    Returns:
        Number of rows successfully ingested.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        Any exception propagated from the pipeline steps.
    """
    from ingestion.pandas_ingestor import ingest
    from analytics.polars_analytics import run_analytics
    from persistence.pickle_store import save as pickle_save
    from persistence.pg_store import get_connection, ensure_schema, insert_dataframe

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    log.info("Pipeline start: %s", path.name)

    df = ingest(path)
    log.info("Ingestion complete — %d rows.", len(df))

    run_analytics(df)
    log.info("Analytics complete.")

    pkl_path = pickle_save(df, version=1)
    log.info("Pickle saved → %s", pkl_path)

    conn = get_connection()
    try:
        ensure_schema(conn)
        insert_dataframe(df, conn)
    finally:
        conn.close()

    log.info("Pipeline complete: %s", path.name)
    return len(df)
