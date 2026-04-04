"""graph.py — LangGraph anomaly-detection pipeline.

Answers: "Summarise anomalies in the ingested data
          (flag outliers, missing fields, unexpected values)."

Graph topology
--------------
START
  └─► load_data
        └─► validate_schema
              └─► compute_stats
                    └─► llm_analyze
                          └─► [route_by_validity]
                                ├─► format_output  (JSON valid)
                                └─► llm_analyze    (repair loop, max 1 retry)
                                      └─► format_output
                                            └─► END
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env", override=True)

from ai.prompts import ANOMALY_ANALYSIS_PROMPT, REPAIR_PROMPT  # noqa: E402
from ai.state import AnomalyState  # noqa: E402

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# Required columns as defined by pandas_ingestor
_REQUIRED_COLUMNS: list[str] = [
    "txn_id", "account_id", "txn_ts", "amount", "currency", "narration"
]
# Valid currency codes accepted by the pipeline
_VALID_CURRENCIES: frozenset[str] = frozenset({"USD", "EUR", "INR", "GBP", "UNKNOWN"})
# IQR multiplier for outlier detection
_IQR_FACTOR = 1.5
# Maximum number of LLM repair attempts before giving up
_MAX_REPAIR_ATTEMPTS = 1


# ── Pydantic output schema ────────────────────────────────────────────────────

class OutlierFinding(BaseModel):
    column: str
    description: str
    affected_rows: int = Field(ge=0)


class MissingFieldFinding(BaseModel):
    column: str
    null_count: int = Field(ge=0)
    null_pct: float = Field(ge=0.0, le=100.0)


class UnexpectedValueFinding(BaseModel):
    column: str
    description: str
    examples: list[str]


class AnomalyReport(BaseModel):
    """Structured anomaly report returned as the graph's final output."""

    summary: str
    outliers: list[OutlierFinding]
    missing_fields: list[MissingFieldFinding]
    unexpected_values: list[UnexpectedValueFinding]
    severity: str = Field(pattern=r"^(low|medium|high)$")
    recommendations: list[str]


# ── LLM factory ───────────────────────────────────────────────────────────────

def _build_llm() -> ChatOpenAI:
    """Instantiate ChatOpenAI(gpt-4o) from environment credentials.

    Reads OPENAI_API_KEY from the environment (loaded from .env).

    Returns:
        Configured ChatOpenAI instance.

    Raises:
        ValueError: If OPENAI_API_KEY is absent or empty.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to .env and restart."
        )
    return ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=api_key)


# ── Node 1 — load_data ────────────────────────────────────────────────────────

def load_data(state: AnomalyState) -> dict[str, Any]:
    """Load the source file into a pandas DataFrame.

    Supports .xlsx (via openpyxl), .csv, and .json files.
    The file path is read from state["source_path"].

    Args:
        state: Current graph state.  Must have ``source_path`` set.

    Returns:
        Partial state update with ``raw_data`` populated.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(state["source_path"])
    log.info("load_data: reading %s", path)

    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        df = pd.read_excel(path, engine="openpyxl")
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix!r}. Use .xlsx, .csv, or .json.")

    log.info("load_data: loaded %d rows × %d columns.", *df.shape)
    return {"raw_data": df}


# ── Node 2 — validate_schema ──────────────────────────────────────────────────

def validate_schema(state: AnomalyState) -> dict[str, Any]:
    """Check the DataFrame against the expected pipeline schema.

    Checks performed:
    - All six required columns are present.
    - ``amount`` column is numeric (or coercible).
    - ``txn_ts`` column is parseable as datetime.
    - ``txn_id`` has no duplicates.

    Args:
        state: Current graph state.  Must have ``raw_data`` set.

    Returns:
        Partial state update with ``validation_errors`` populated.
        An empty list means the data passed all checks.
    """
    df: pd.DataFrame = state["raw_data"]
    errors: list[str] = []

    # Missing columns
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    present = set(df.columns)

    # amount must be numeric
    if "amount" in present:
        non_numeric = pd.to_numeric(df["amount"], errors="coerce").isna().sum()
        original_nulls = df["amount"].isna().sum()
        coerce_failures = non_numeric - original_nulls
        if coerce_failures > 0:
            errors.append(
                f"'amount': {coerce_failures} value(s) cannot be cast to float."
            )

    # txn_ts must be parseable
    if "txn_ts" in present:
        unparseable = pd.to_datetime(df["txn_ts"], errors="coerce").isna().sum()
        original_nat = df["txn_ts"].isna().sum()
        bad_ts = unparseable - original_nat
        if bad_ts > 0:
            errors.append(
                f"'txn_ts': {bad_ts} value(s) cannot be parsed as datetime."
            )

    # Duplicate txn_ids
    if "txn_id" in present:
        dup_count = df["txn_id"].duplicated().sum()
        if dup_count > 0:
            errors.append(f"'txn_id': {dup_count} duplicate value(s) found.")

    if errors:
        log.warning("validate_schema: %d error(s) found.", len(errors))
    else:
        log.info("validate_schema: all checks passed.")

    return {"validation_errors": errors}


# ── Node 3 — compute_stats ────────────────────────────────────────────────────

def compute_stats(state: AnomalyState) -> dict[str, Any]:
    """Compute per-column statistics used by the LLM to identify anomalies.

    Statistics produced:
    - ``null_counts``       — {col: count} for every column
    - ``null_pcts``         — {col: pct} rounded to 2 dp
    - ``duplicate_txn_ids`` — count of duplicate txn_id values
    - ``numeric_outliers``  — IQR-based outlier counts for numeric columns
    - ``invalid_currencies``— set of currency values outside _VALID_CURRENCIES
    - ``amount_describe``   — pandas describe() output for the amount column
    - ``row_count``         — total rows

    Args:
        state: Current graph state.  Must have ``raw_data`` set.

    Returns:
        Partial state update with ``stats`` populated.
    """
    df: pd.DataFrame = state["raw_data"]
    n = len(df)
    stats: dict[str, Any] = {"row_count": n}

    # Null counts and percentages
    null_counts = df.isnull().sum().to_dict()
    stats["null_counts"] = {k: int(v) for k, v in null_counts.items()}
    stats["null_pcts"] = {
        k: round(v / n * 100, 2) if n > 0 else 0.0
        for k, v in null_counts.items()
    }

    # Duplicate txn_ids
    if "txn_id" in df.columns:
        stats["duplicate_txn_ids"] = int(df["txn_id"].duplicated().sum())

    # IQR-based outlier detection for every numeric column
    numeric_outliers: dict[str, int] = {}
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - _IQR_FACTOR * iqr, q3 + _IQR_FACTOR * iqr
        outlier_count = int(((series < lower) | (series > upper)).sum())
        if outlier_count > 0:
            numeric_outliers[col] = outlier_count
    stats["numeric_outliers"] = numeric_outliers

    # Invalid currency codes
    if "currency" in df.columns:
        all_currencies = set(df["currency"].dropna().unique())
        invalid = sorted(all_currencies - _VALID_CURRENCIES)
        stats["invalid_currencies"] = invalid
        stats["currency_value_counts"] = (
            df["currency"].value_counts().head(10).to_dict()
        )

    # Descriptive stats for amount
    if "amount" in df.columns:
        numeric_amount = pd.to_numeric(df["amount"], errors="coerce")
        desc = numeric_amount.describe()
        stats["amount_describe"] = {k: round(float(v), 4) for k, v in desc.items()}
        stats["negative_amounts"] = int((numeric_amount < 0).sum())
        stats["zero_amounts"] = int((numeric_amount == 0).sum())

    # Unparseable timestamps
    if "txn_ts" in df.columns:
        nat_count = pd.to_datetime(df["txn_ts"], errors="coerce").isna().sum()
        stats["unparseable_timestamps"] = int(nat_count)

    log.info("compute_stats: stats computed for %d rows.", n)
    return {"stats": stats}


# ── Node 4 — llm_analyze ──────────────────────────────────────────────────────

def llm_analyze(state: AnomalyState) -> dict[str, Any]:
    """Call the LLM with the computed statistics and return its raw response.

    Uses ANOMALY_ANALYSIS_PROMPT on the first call.
    Uses REPAIR_PROMPT when ``llm_response`` is already set (retry path),
    passing the previous bad response for correction.

    Args:
        state: Current graph state.  Must have ``stats`` and
               ``validation_errors`` set.

    Returns:
        Partial state update with ``llm_response`` populated.
    """
    df: pd.DataFrame = state["raw_data"]
    stats = state["stats"]
    errors = state["validation_errors"]

    llm = _build_llm()
    chain = StrOutputParser()

    # Retry path: previous response was not valid JSON — ask LLM to repair it
    if state.get("llm_response"):
        log.info("llm_analyze: repair attempt for malformed JSON response.")
        prompt_value = REPAIR_PROMPT.format_messages(
            bad_response=state["llm_response"]
        )
        raw: str = (llm | chain).invoke(prompt_value)
        return {"llm_response": raw}

    # First call: full analysis
    validation_text = (
        "\n".join(f"- {e}" for e in errors) if errors else "None — all checks passed."
    )
    stats_json = json.dumps(stats, indent=2, default=str)

    log.info("llm_analyze: invoking LLM (gpt-4o) for anomaly analysis.")
    prompt_value = ANOMALY_ANALYSIS_PROMPT.format_messages(
        row_count=stats.get("row_count", len(df)),
        column_list=", ".join(df.columns.tolist()),
        validation_errors=validation_text,
        stats_json=stats_json,
    )
    raw = (llm | chain).invoke(prompt_value)
    log.info("llm_analyze: received %d-char response.", len(raw))
    return {"llm_response": raw}


# ── Routing function ──────────────────────────────────────────────────────────

def route_by_validity(state: AnomalyState) -> str:
    """Decide whether the LLM response is valid JSON or needs a repair pass.

    Branching logic:
    - Attempt to parse ``state["llm_response"]`` as JSON and validate it
      against the AnomalyReport Pydantic model.
    - If parsing succeeds  → route to "format_output" (happy path).
    - If parsing fails AND this is the first attempt → route back to
      "llm_analyze" for one repair pass using REPAIR_PROMPT.
    - If parsing fails AND a repair was already attempted → route to
      "format_output" anyway (format_output handles the fallback gracefully
      rather than looping forever).

    Args:
        state: Current graph state.  Must have ``llm_response`` set.

    Returns:
        "format_output" or "llm_analyze".
    """
    raw = state.get("llm_response", "")

    # Strip accidental markdown fences that some models add despite instructions
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove ```json ... ``` wrapper
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        data = json.loads(cleaned)
        AnomalyReport(**data)          # validate against Pydantic schema
        log.info("route_by_validity: valid JSON — routing to format_output.")
        return "format_output"
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        # Check whether a repair was already attempted by inspecting whether
        # the response still looks like the original (heuristic: if it starts
        # with '{' it was already a repair attempt).
        already_repaired = cleaned.lstrip().startswith("{")
        if already_repaired:
            log.warning(
                "route_by_validity: repair attempt also invalid (%s) — "
                "routing to format_output with fallback handling.", exc
            )
            return "format_output"
        log.warning(
            "route_by_validity: invalid JSON (%s) — routing to llm_analyze for repair.", exc
        )
        return "llm_analyze"


# ── Node 5 — format_output ────────────────────────────────────────────────────

def format_output(state: AnomalyState) -> dict[str, Any]:
    """Parse the LLM response with Pydantic and serialise to a plain dict.

    If the response is still not valid JSON after the optional repair pass,
    a fallback AnomalyReport is constructed from the raw stats so the graph
    always returns a well-formed result.

    Args:
        state: Current graph state.  Must have ``llm_response`` and
               ``stats`` set.

    Returns:
        Partial state update with ``final_output`` populated.
    """
    raw = state.get("llm_response", "")
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        data = json.loads(cleaned)
        report = AnomalyReport(**data)
        log.info("format_output: Pydantic validation passed.")
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        log.error("format_output: falling back to stats-derived report (%s).", exc)
        report = _fallback_report(state)

    return {"final_output": report.model_dump()}


def _fallback_report(state: AnomalyState) -> AnomalyReport:
    """Build a minimal AnomalyReport directly from computed stats.

    Called only when the LLM response cannot be parsed even after a repair
    attempt.  Ensures the graph always returns a structured result.

    Args:
        state: Current graph state with ``stats`` and ``validation_errors``.

    Returns:
        AnomalyReport populated from raw statistics.
    """
    stats = state.get("stats", {})
    errors = state.get("validation_errors", [])

    missing_fields = [
        MissingFieldFinding(
            column=col,
            null_count=count,
            null_pct=stats.get("null_pcts", {}).get(col, 0.0),
        )
        for col, count in stats.get("null_counts", {}).items()
        if count > 0
    ]

    outliers = [
        OutlierFinding(
            column=col,
            description=f"IQR-based outlier detection flagged {count} row(s).",
            affected_rows=count,
        )
        for col, count in stats.get("numeric_outliers", {}).items()
    ]

    unexpected: list[UnexpectedValueFinding] = []
    if stats.get("invalid_currencies"):
        unexpected.append(
            UnexpectedValueFinding(
                column="currency",
                description="Currency codes outside the accepted set.",
                examples=stats["invalid_currencies"][:5],
            )
        )

    total_issues = len(missing_fields) + len(outliers) + len(unexpected) + len(errors)
    severity = "high" if total_issues > 10 else "medium" if total_issues > 3 else "low"

    return AnomalyReport(
        summary=(
            f"Automated fallback report: {total_issues} issue(s) detected across "
            f"{stats.get('row_count', '?')} rows."
        ),
        outliers=outliers,
        missing_fields=missing_fields,
        unexpected_values=unexpected,
        severity=severity,
        recommendations=[
            "Review validation errors and re-ingest after fixing source data.",
            "Investigate IQR outliers in numeric columns for data-entry errors.",
        ],
    )


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph() -> Any:
    """Assemble and compile the 5-node anomaly-detection StateGraph.

    Edges:
        START → load_data → validate_schema → compute_stats → llm_analyze
        llm_analyze → [route_by_validity] → format_output → END
                                          ↘ llm_analyze (repair, once)

    Returns:
        Compiled LangGraph runnable.
    """
    builder: StateGraph = StateGraph(AnomalyState)

    builder.add_node("load_data", load_data)
    builder.add_node("validate_schema", validate_schema)
    builder.add_node("compute_stats", compute_stats)
    builder.add_node("llm_analyze", llm_analyze)
    builder.add_node("format_output", format_output)

    # Linear edges
    builder.add_edge(START, "load_data")
    builder.add_edge("load_data", "validate_schema")
    builder.add_edge("validate_schema", "compute_stats")
    builder.add_edge("compute_stats", "llm_analyze")

    # Conditional edge: valid JSON → format_output, else → repair via llm_analyze
    builder.add_conditional_edges(
        "llm_analyze",
        route_by_validity,
        {"format_output": "format_output", "llm_analyze": "llm_analyze"},
    )

    builder.add_edge("format_output", END)

    return builder.compile()


# ── Public API ────────────────────────────────────────────────────────────────

def run_anomaly_pipeline(source_path: str | Path) -> dict[str, Any]:
    """Run the full anomaly-detection pipeline for *source_path*.

    Args:
        source_path: Absolute or relative path to a .xlsx, .csv, or .json file
                     written by the FTP/SFTP watcher.

    Returns:
        The ``final_output`` dict — a serialised AnomalyReport.
    """
    graph = build_graph()
    initial_state: AnomalyState = {
        "source_path": str(source_path),
        "raw_data": None,
        "validation_errors": [],
        "stats": {},
        "llm_response": "",
        "final_output": {},
    }
    final_state: AnomalyState = graph.invoke(initial_state)
    return final_state["final_output"]


# ── __main__ — minimal runnable example with synthetic data ───────────────────

if __name__ == "__main__":
    import tempfile
    import uuid
    import json as _json

    import numpy as np

    print("\n=== Anomaly Detection Pipeline — synthetic data demo ===\n")

    # Build a synthetic DataFrame that contains deliberate anomalies:
    #   - 3 null amounts
    #   - 2 null currencies
    #   - 1 invalid currency code ("USDX")
    #   - 2 duplicate txn_ids
    #   - 1 extreme outlier amount (999_999)
    #   - 2 unparseable timestamps
    rng = np.random.default_rng(42)
    n = 200

    txn_ids = [str(uuid.uuid4()) for _ in range(n)]
    txn_ids[10] = txn_ids[5]   # duplicate
    txn_ids[20] = txn_ids[15]  # duplicate

    amounts: list[Any] = rng.uniform(10, 5_000, n).round(2).tolist()
    amounts[30] = None          # null
    amounts[31] = None          # null
    amounts[32] = None          # null
    amounts[50] = 999_999.0     # extreme outlier

    currencies: list[Any] = rng.choice(["USD", "EUR", "INR", "GBP"], n).tolist()
    currencies[40] = None       # null
    currencies[41] = None       # null
    currencies[60] = "USDX"    # invalid

    timestamps: list[Any] = pd.date_range("2024-01-01", periods=n, freq="h").astype(str).tolist()
    timestamps[70] = "not-a-date"   # unparseable
    timestamps[71] = "NULL"         # unparseable

    synthetic_df = pd.DataFrame({
        "txn_id":     txn_ids,
        "account_id": rng.choice([f"ACC_{i:03d}" for i in range(1, 11)], n).tolist(),
        "txn_ts":     timestamps,
        "amount":     amounts,
        "currency":   currencies,
        "narration":  [f"Payment {i}" for i in range(n)],
    })

    # Write to a temp CSV so the graph's load_data node reads it from disk
    # (matching the real watcher workflow where files land on disk first)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
        synthetic_df.to_csv(tmp, index=False)
        tmp_path = tmp.name

    print(f"Synthetic data written to: {tmp_path}")
    print(f"Shape: {synthetic_df.shape}\n")

    result = run_anomaly_pipeline(tmp_path)

    print("=== Final Output (AnomalyReport) ===\n")
    print(_json.dumps(result, indent=2))
