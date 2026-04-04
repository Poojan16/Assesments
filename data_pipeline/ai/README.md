# Anomaly-Detection LangGraph Pipeline

Answers: **"Summarise anomalies in the ingested data (flag outliers, missing fields, unexpected values)."**

The pipeline is a 5-node `StateGraph` that reads a file written by the FTP/SFTP watcher, computes statistics with pandas, calls `gpt-4o`, validates the response with Pydantic, and returns structured JSON.

---

## Files

| File | Purpose |
|---|---|
| `ai/state.py` | `AnomalyState` TypedDict — the single state object passed between every node |
| `ai/prompts.py` | All `ChatPromptTemplate` objects — no inline f-string prompts anywhere |
| `ai/graph.py` | Graph definition, all 5 node functions, routing function, Pydantic schema, public API |

---

## Graph Topology

```
START
  └─► load_data
        └─► validate_schema
              └─► compute_stats
                    └─► llm_analyze
                          └─► [route_by_validity]
                                ├─► format_output  ← JSON valid
                                └─► llm_analyze    ← repair pass (once)
                                      └─► format_output
                                            └─► END
```

### Nodes

| # | Node | Responsibility |
|---|---|---|
| 1 | `load_data` | Read `.xlsx` / `.csv` / `.json` from `state["source_path"]` into a pandas DataFrame |
| 2 | `validate_schema` | Check required columns, numeric castability of `amount`, datetime parseability of `txn_ts`, duplicate `txn_id` |
| 3 | `compute_stats` | IQR outlier counts, null counts/pcts, invalid currencies, `amount` describe, unparseable timestamps |
| 4 | `llm_analyze` | Call `gpt-4o` via `ANOMALY_ANALYSIS_PROMPT`; on retry path use `REPAIR_PROMPT` |
| 5 | `format_output` | Parse LLM response with `AnomalyReport` Pydantic model; fallback to stats-derived report if still invalid |

### Conditional edge — `route_by_validity`

After `llm_analyze`, the router attempts `json.loads` + `AnomalyReport(**data)`:

- **Valid JSON** → `format_output`
- **Invalid JSON, first attempt** → back to `llm_analyze` (repair pass using `REPAIR_PROMPT`)
- **Invalid JSON, after repair** → `format_output` with stats-derived fallback (no infinite loop)

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | **Yes** | OpenAI API key — read from `.env`, never hardcoded |
| `LANGCHAIN_LLM_PROVIDER` | No | Reserved for future provider switching (default: `openai`) |

All other variables (`DB_*`, `FTP_*`, etc.) are used by the broader pipeline but not by this graph directly.

---

## Setup

**Docker (recommended)**

```bash
cp .env.example .env   # add OPENAI_API_KEY
docker compose up --build
docker compose exec web python ai/graph.py
```

**Non-Docker**

```bash
cp .env.example .env   # add OPENAI_API_KEY
./setup.sh             # creates venv, installs deps, runs migrations
source venv/bin/activate
python ai/graph.py
```

**Manual venv**

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY
python ai/graph.py
```

---

## Example Invocations

**Run the built-in synthetic demo (no live data needed)**

```bash
python ai/graph.py
```

**Run against a real watcher-produced file**

```python
from ai.graph import run_anomaly_pipeline

result = run_anomaly_pipeline("data/sample_transactions.xlsx")
print(result)
```

**Run against a CSV**

```python
result = run_anomaly_pipeline("tmp/incoming/transactions_2024_06_01.csv")
```

---

## Expected Output Shape

`run_anomaly_pipeline()` returns a plain `dict` that is the serialised `AnomalyReport` Pydantic model:

```json
{
  "summary": "17 anomalies detected across 200 rows: 3 null amounts, 2 null currencies, 1 invalid currency code, 2 duplicate txn_ids, and 1 extreme amount outlier.",
  "outliers": [
    {
      "column": "amount",
      "description": "1 value (999999.0) is 47× the IQR upper fence of ~7800.",
      "affected_rows": 1
    }
  ],
  "missing_fields": [
    { "column": "amount",   "null_count": 3, "null_pct": 1.5  },
    { "column": "currency", "null_count": 2, "null_pct": 1.0  }
  ],
  "unexpected_values": [
    {
      "column": "currency",
      "description": "Currency code 'USDX' is not in the accepted set {USD, EUR, INR, GBP, UNKNOWN}.",
      "examples": ["USDX"]
    },
    {
      "column": "txn_ts",
      "description": "2 timestamps cannot be parsed as datetime.",
      "examples": ["not-a-date", "NULL"]
    }
  ],
  "severity": "medium",
  "recommendations": [
    "Reject or quarantine rows with null amount before downstream aggregation.",
    "Add a currency allowlist check to the ingestion validator.",
    "Investigate txn_id=<uuid> for the 999999.0 amount — likely a data-entry error.",
    "Fix or drop the 2 unparseable timestamps before the Polars rolling window step."
  ]
}
```

---

## State Schema (`AnomalyState`)

```python
class AnomalyState(TypedDict):
    source_path:       str            # input: file path supplied by caller
    raw_data:          Any            # pandas DataFrame after load_data
    validation_errors: list[str]      # schema/dtype errors from validate_schema
    stats:             dict[str, Any] # aggregated stats from compute_stats
    llm_response:      str            # raw LLM string from llm_analyze
    final_output:      dict[str, Any] # serialised AnomalyReport from format_output
```

---

## Prompts (`ai/prompts.py`)

| Name | Template type | Used by |
|---|---|---|
| `ANOMALY_ANALYSIS_PROMPT` | `ChatPromptTemplate` | `llm_analyze` — first call |
| `REPAIR_PROMPT` | `ChatPromptTemplate` | `llm_analyze` — retry/repair call |

Both templates are module-level named constants. No f-string prompt construction exists anywhere in the codebase.
