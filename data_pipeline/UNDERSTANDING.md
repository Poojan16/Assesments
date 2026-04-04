# Financial Transaction Ingestion Service

> **Who is this for?** A newbie developer joining this project. Read this top to bottom before touching any code.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Quick Start](#2-quick-start)
3. [Tech Stack & Why](#3-tech-stack--why)
4. [Project Structure](#4-project-structure)
5. [System Flows](#5-system-flows)
6. [Data Model](#6-data-model)
7. [AI Layer](#7-ai-layer)
8. [Configuration & Environment Variables](#8-configuration--environment-variables)
9. [Running Tests](#9-running-tests)
10. [Ops Scripts](#10-ops-scripts)
11. [Key Decisions — Must Read Before Coding](#11-key-decisions--must-read-before-coding)
12. [Known Risks & Missing Pieces](#12-known-risks--missing-pieces)

---

## 1. What This System Does

This service **automatically ingests financial transaction Excel files from an FTP/SFTP server**, cleans and validates them, runs rolling analytics, stores results in PostgreSQL (with vector embeddings for semantic search), and exposes everything through a Django web dashboard and an AI-powered anomaly detection agent.

```
FTP/SFTP Server
    │
    ▼ (polls on interval)
Watcher (paramiko)
    │
    ▼ (download .xlsx)
Pandas Ingestor  ──► schema validation, dtype coercion, null filling
    │
    ▼
Polars Analytics ──► rolling 7-day sums per account
    │
    ├──► Pickle Store  ──► artifacts/ (local cache with SHA-256 integrity)
    │
    └──► PostgreSQL + pgvector  ──► transactions table + narration embeddings
                │
                ▼
          Django Dashboard  ──► list view, trigger endpoint, AI insights
                │
                ▼
          LangGraph AI Agent  ──► anomaly detection, conversational Q&A
```

**Primary users:** Data engineers and analysts who need automated, idempotent ingestion with AI-powered anomaly insights.

---

## 2. Quick Start

### Docker (recommended)

```bash
cp .env.example .env        # fill in DB, FTP, and OpenAI credentials
docker-compose up --build   # starts: PostgreSQL, Django dashboard, FTP watcher
```

Dashboard available at: `http://localhost:8000/dashboard/`

### Without Docker

```bash
bash setup.sh               # creates venv, installs deps, runs migrations, seeds data
bash start.sh               # starts Django + watcher as background processes
bash stop.sh                # graceful shutdown
```

### Windows

```bat
run_windows.bat
```

---

## 3. Tech Stack & Why

| Library | Version | Role | Why chosen | Alternative | Key difference |
|---|---|---|---|---|---|
| **pandas** | 2.2.1 | Excel read, schema validation, dtype coercion | Mature `.xlsx` / dtype ecosystem | polars alone | Polars has no openpyxl bridge; pandas handles messy Excel natively |
| **openpyxl** | 3.1.2 | `.xlsx` engine for `pd.read_excel` | Only maintained xlsx engine | xlrd | xlrd dropped `.xlsx` support in v2.0 — would break on all modern Excel files |
| **polars-lts-cpu** | 0.20.15 | Lazy analytics pipeline, rolling 7-day sums | Faster than pandas for aggregations; lazy query planner avoids full scans | dask | Dask is distributed/complex to operate; Polars is single-node but far simpler |
| **psycopg2-binary** | 2.9.9 | Raw PostgreSQL driver | Direct SQL control for `ON CONFLICT` idempotency | SQLAlchemy ORM | SQLAlchemy adds abstraction overhead; raw psycopg2 keeps insert logic explicit |
| **pgvector** | 0.2.4 | Python adapter for `VECTOR` type | Needed to pass float lists as `::vector` to PostgreSQL | asyncpg | asyncpg is async-only; psycopg2 is sync and matches the rest of the codebase |
| **Django** | 5.0.3 | Web dashboard + ORM for `ProcessedFile` tracking | Batteries-included: migrations, templates, admin | FastAPI | FastAPI has no built-in ORM or migrations; Django saves significant boilerplate |
| **paramiko** | 3.4.0 | SFTP client (key + password auth) | Pure-Python SSH; no system `libssh2` dependency | asyncssh | asyncssh is async; paramiko fits the sync polling loop |
| **python-dotenv** | 1.0.1 | Load `.env` into `os.environ` | Standard lightweight pattern | pydantic-settings | pydantic-settings adds type validation but is heavier; not needed here |
| **sentence-transformers** | 2.6.1 | Embed narration text → 128-dim vectors | Pre-trained models, no training needed, free | OpenAI embeddings API | OpenAI costs money per call; local model is free but slow on first load |
| **langchain** | 0.1.16 | PromptTemplate, memory, chains | Glue between LLM and graph nodes | llama-index | llama-index is RAG-focused; LangChain is more general-purpose for agent flows |
| **langgraph** | 0.0.38 | `StateGraph` for multi-node AI pipelines | Explicit node/edge topology, testable, supports conditional routing | Plain LangChain chains | Chains are linear; LangGraph supports the conditional repair loop in `graph.py` |
| **langchain-openai** | 0.1.3 | `ChatOpenAI` wrapper | Official OpenAI integration | anthropic SDK | Anthropic path exists in `_build_llm()` but requires a separate install |
| **faker** | 24.3.0 | Synthetic data generation | Realistic names/sentences for seeding | mimesis | Functionally equivalent; Faker has wider community and more locale support |
| **pytest** | 8.1.1 | Test runner | Standard; fixtures and parametrize are cleaner than unittest | unittest | unittest is more verbose; pytest is the ecosystem standard |

---

## 4. Project Structure

```
project-root/
│
├── .env / .env.example         # ALL secrets and feature flags — fill this first
├── pytest.ini                  # Sets project root so test imports resolve
├── conftest.py                 # Shared pytest fixture: minimal_df
├── generate_data.py            # Synthetic Excel data generator (used by Docker startup)
├── docker-compose.yml          # Orchestrates: db, web (Django), watcher
├── Dockerfile                  # Python 3.11-slim image
├── setup.sh / start.sh / stop.sh / ops.sh  # Lifecycle scripts (non-Docker)
├── run_windows.bat             # Windows equivalent of setup + start
├── SCRIPTS.md                  # Documentation for all bash scripts
│
├── ingestion/
│   └── pandas_ingestor.py      # Step 1: validates schema, coerces dtypes, fills nulls
│
├── analytics/
│   └── polars_analytics.py     # Step 2: lazy Polars pipeline → rolling 7-day sums
│
├── persistence/
│   ├── pg_store.py             # Step 3a: PostgreSQL writes + pgvector similarity search
│   └── pickle_store.py         # Step 3b: versioned pickle cache with SHA-256 integrity
│
├── watcher/
│   ├── ftp_watcher.py          # Poll loop: FTP + SFTP via paramiko, JSON idempotency state
│   ├── sftp_watcher.py         # Production SFTP-only watcher: SQLite state, webhook support
│   └── processed_files.json    # Auto-generated: tracks which files were already processed
│
├── ai/
│   ├── langchain_agent.py      # 3-node graph: data_loader → analyst → formatter (used by dashboard)
│   ├── graph.py                # 5-node anomaly graph with conditional repair loop
│   ├── prompts.py              # All prompt strings as module-level constants
│   └── state.py                # AnomalyState TypedDict — typed contract for graph.py
│
├── dashboard/
│   ├── manage.py               # Django CLI: migrations, run_pipeline management command
│   ├── root_urls.py            # Mounts everything under /dashboard/ prefix
│   └── dashboard/
│       ├── settings.py         # Django settings, DB config, logging
│       ├── urls.py             # 3 routes: list, trigger, insights
│       ├── views.py            # HTTP handlers for all 3 routes
│       ├── models.py           # ProcessedFile ORM model
│       ├── pipeline_runner.py  # Shared run_pipeline_for_file() — single source of truth
│       └── migrations/
│           └── 0001_initial.py # Creates ProcessedFile table
│
├── tests/
│   ├── test_schema_validation.py   # Parametrized: SchemaValidationError for each missing column
│   ├── test_rolling_metric.py      # Correctness of rolling_7d_sum on a known 30-row dataset
│   ├── test_idempotency.py         # Watcher skips duplicate checksums; JSON state has one entry
│   ├── test_pickle_integrity.py    # SHA-256 mismatch raises PickleIntegrityError
│   ├── test_pg_idempotency.py      # Double insert = exactly 100 rows (auto-skipped if no DB)
│   └── test_sftp_watcher.py        # SFTP watcher unit tests (content unverified)
│
└── artifacts/                  # Auto-generated: versioned pickle files accumulate here
```

### File roles in one line each

| File | Contains | Why it exists | Breaks if removed |
|---|---|---|---|
| `ingestion/pandas_ingestor.py` | Schema validation, dtype coercion, null filling | Core cleaning; raw Excel is messy | Dirty data hits DB |
| `analytics/polars_analytics.py` | 4-step lazy Polars pipeline | Rolling 7-day sums per account | No analytics computed |
| `persistence/pg_store.py` | `ensure_schema`, `insert_dataframe`, `find_similar_narrations` | PostgreSQL + pgvector persistence | No DB writes or similarity search |
| `persistence/pickle_store.py` | Versioned pickle save/load with SHA-256 + row-count verification | Local artifact cache with integrity check | No local artifact storage |
| `dashboard/dashboard/pipeline_runner.py` | `run_pipeline_for_file()` | Avoids duplicating pipeline logic between view and management command | View and command diverge |
| `watcher/ftp_watcher.py` | FTP/SFTP poll, download, retry, idempotency via JSON | Automated file pickup | No automated ingestion |
| `ai/langchain_agent.py` | 3-node LangGraph graph | Conversational insights via `/insights/` | AI endpoint breaks |
| `ai/graph.py` | 5-node anomaly graph with conditional repair loop | Structured anomaly detection | Anomaly detection unavailable |
| `ai/prompts.py` | `ANOMALY_ANALYSIS_PROMPT`, `REPAIR_PROMPT` | Centralised prompt management | `graph.py` nodes have no prompts |
| `ai/state.py` | `AnomalyState` TypedDict | Typed state contract for `graph.py` | Type errors at runtime |
| `generate_data.py` | Faker-based synthetic Excel generator | Seeding and testing without real FTP | No test data; Docker startup fails |
| `watcher/processed_files.json` | `{filename: {checksum, processed_at}}` | Watcher idempotency state | Watcher re-processes all files on restart |

---

## 5. System Flows

### Flow 1 — Automated File Ingestion (Watcher)

```
WATCHER_POLL_INTERVAL timer
  │
  ▼
ftp_watcher._poll_once()
  ├── _list_ftp() / _list_sftp()         # list remote .xlsx files
  ├── _download_ftp/sftp()               # download to tmp/incoming/
  ├── _file_checksum()                   # SHA-256 of local file
  └── _already_processed(state, chksum)  # check processed_files.json
        │
        ├── [already seen] → SKIP
        │
        └── [new file] → _run_with_retry(local_path)   # up to 3 attempts, exponential backoff
              │
              ├── ingest(path)                           # pandas: validate + clean
              ├── run_analytics(df)                      # polars: filter→enrich→groupby→rolling
              ├── pickle_save(df)                        # artifacts/*.pkl + *.meta.json
              └── get_connection() + ensure_schema()
                  + insert_dataframe(df)                 # pg_store: ON CONFLICT DO NOTHING
                        │
                        ▼
              _mark_processed() + _save_state()          # update processed_files.json
```

### Flow 2 — HTTP Pipeline Trigger

```
POST /dashboard/trigger/?file=sample.xlsx
  │
  ▼
views.trigger_pipeline()
  ├── ProcessedFile.objects.get_or_create(filename)     # Django ORM: track run
  └── pipeline_runner.run_pipeline_for_file(filepath)
        │
        ├── ingest → run_analytics → pickle_save → pg insert   (same chain as Flow 1)
        │
        └── record.status = SUCCESS / FAILED
              │
              ▼
        JsonResponse({"status": ..., "file": ...})
```

### Flow 3 — AI Insights Endpoint

```
GET /dashboard/insights/
  │
  ▼
views.insights()
  └── langchain_agent.run_agent(question)
        │
        ├── [Node 1] data_loader
        │     └── pg_store.get_connection()
        │           → SELECT top 10 accounts by SUM(amount)
        │
        ├── [Node 2] analyst
        │     └── _ANALYST_PROMPT + ConversationBufferMemory
        │           → ChatOpenAI(gpt-4o-mini)
        │
        └── [Node 3] formatter
              └── _FORMATTER_PROMPT → ChatOpenAI → markdown string
                    │
                    ▼
              JsonResponse({"insights": "<markdown>"})
```

### Flow 4 — Anomaly Detection (graph.py, standalone)

```
[Node 1] load_data
  → [Node 2] analyze_anomalies (LLM call with ANOMALY_ANALYSIS_PROMPT)
        → [Node 3] validate_output (Pydantic validation)
              │
              ├── [valid JSON] → [Node 4] enrich_results → [Node 5] format_report
              │
              └── [invalid JSON] → [Node R] repair (REPAIR_PROMPT → LLM → retry validate)
```

---

## 6. Data Model

### `transactions` table (PostgreSQL + pgvector)

| Column | Type | Notes |
|---|---|---|
| `txn_id` | TEXT | Primary key — idempotency via `ON CONFLICT DO NOTHING` |
| `account_id` | TEXT | Groups transactions for rolling analytics |
| `txn_ts` | TIMESTAMPTZ | UTC-normalised; rows with null `txn_ts` are skipped on insert |
| `amount` | FLOAT | Cleaned to `0.0` if null by ingestor |
| `currency` | TEXT | Filled to `"UNKNOWN"` if null |
| `narration` | TEXT | Source text for embedding |
| `embedding` | VECTOR(128) | Truncated + renormalised `all-MiniLM-L6-v2` output; used for `<->` similarity search |

### `dashboard_processedfile` table (Django ORM → `models.py`)

| Column | Notes |
|---|---|
| `filename` | Unique — the Excel filename |
| `processed_at` | Timestamp of last pipeline run |
| `row_count` | Rows successfully written |
| `status` | `pending` / `success` / `failed` |
| `error_message` | Set on failure |

> **No FK between these two tables.** `dashboard_processedfile` tracks pipeline runs; `transactions` holds the actual data. They are managed independently.

### `watcher/processed_files.json` (flat file)

```json
{
  "sample.xlsx": {
    "checksum": "sha256hex...",
    "processed_at": "2024-01-01T00:00:00"
  }
}
```

Watcher's idempotency state — keyed on **content checksum**, not filename.

### `watcher/sftp_seen.db` (SQLite — `sftp_watcher.py` only)

Table: `seen_files(filename, mtime)` — composite primary key. Keyed on server `mtime`, **not** content checksum. Different idempotency strategy from `ftp_watcher.py`.

---

## 7. AI Layer

There are **two independent LangGraph graphs**. They do not share state.

| | `langchain_agent.py` | `graph.py` |
|---|---|---|
| Nodes | 3 (data_loader → analyst → formatter) | 5 (load → analyze → validate → repair/enrich → format) |
| Purpose | Conversational Q&A insights | Structured anomaly detection with JSON output |
| Used by | `views.insights()` endpoint | Standalone CLI / direct call |
| Routing | Linear | Conditional (repair loop on invalid JSON) |
| Memory | `ConversationBufferMemory` (module-level, unbounded) | `AnomalyState` TypedDict (per-run) |

**Prompts** (`ai/prompts.py`): All prompt strings live here as module-level constants. Never hardcode prompts inside node functions.

**LLM switching** (`_build_llm()` in `langchain_agent.py`): Supports OpenAI (default) and Anthropic (requires separate install). Controlled via `.env`.

---

## 8. Configuration & Environment Variables

All variables live in `.env`. Copy `.env.example` and fill in values.

| Variable | Used by | Default | Risk if wrong |
|---|---|---|---|
| `DATABASE_URL` | `pg_store.py`, Django | — | Nothing connects to DB |
| `FTP_HOST / FTP_USER / FTP_PASS` | `ftp_watcher.py` | — | Watcher can't connect |
| `SFTP_HOST / SFTP_USER / SFTP_KEY_PATH` | `sftp_watcher.py` | — | SFTP watcher can't connect |
| `OPENAI_API_KEY` | `langchain_agent.py`, `graph.py` | — | AI endpoints fail |
| `WATCHER_POLL_INTERVAL` | `ftp_watcher.py` | `60` | Controls polling frequency |
| `DJANGO_SECRET_KEY` | `settings.py` | `dev-secret-key-change-in-production` | ⚠️ **Use in prod = security risk** |
| `DJANGO_DEBUG` | `settings.py` | `True` | ⚠️ **Set to `false` in production** |
| `DJANGO_ALLOWED_HOSTS` | `settings.py` | `*` | ⚠️ **Set to your domain in production** |

> **Critical:** `DJANGO_SECRET_KEY` has a hardcoded fallback in `settings.py`. If you forget to set it in `.env`, Django will silently use the insecure default in production.

---

## 9. Running Tests

```bash
pytest                          # run all tests
pytest tests/test_rolling_metric.py  # single file
pytest -v                       # verbose output
```

Tests that require a live PostgreSQL connection (`test_pg_idempotency.py`) auto-skip if no DB is available.

**Shared fixture** (`conftest.py`): `minimal_df` — a small valid DataFrame used by multiple tests.

**What is tested:**

| Test file | Covers |
|---|---|
| `test_schema_validation.py` | `SchemaValidationError` raised for each missing column (parametrized) |
| `test_rolling_metric.py` | Correctness of `rolling_7d_sum` on a known 30-row dataset |
| `test_idempotency.py` | Watcher skips duplicate checksums; `processed_files.json` has one entry |
| `test_pickle_integrity.py` | SHA-256 mismatch → `PickleIntegrityError`; missing meta → `FileNotFoundError` |
| `test_pg_idempotency.py` | Double insert leaves exactly 100 rows |

**What is NOT tested:** `ai/graph.py`, `ai/langchain_agent.py`, `views.py`, `pipeline_runner.py` — these require a live LLM key.

---

## 10. Ops Scripts

| Script | Purpose |
|---|---|
| `setup.sh` | Non-Docker bootstrap: venv, deps, migrations, seed data |
| `start.sh` | Start Django + watcher as background processes with PID files |
| `stop.sh` | Graceful shutdown: SIGTERM → SIGKILL with configurable timeout |
| `ops.sh` | Archive old `.xlsx` files; purge old archives (cron-safe) |
| `run_windows.bat` | Windows equivalent of setup + start |

See `SCRIPTS.md` for full documentation on each script.

---

## 11. Key Decisions — Must Read Before Coding

**1. Two watcher implementations exist — know which one is active.**
`ftp_watcher.py` handles both FTP and SFTP via paramiko, uses `processed_files.json` for state, and is what `docker-compose.yml` runs. `sftp_watcher.py` is more production-grade (SQLite state, webhook support, structured JSON logging) but is **not wired into Docker yet**.

**2. `pipeline_runner.py` is the single source of truth for the pipeline.**
Both `views.trigger_pipeline()` and the `run_pipeline` management command call `run_pipeline_for_file()`. Never duplicate this logic — always extend `pipeline_runner.py`.

**3. Idempotency is layered — both layers must hold.**
The watcher checks SHA-256 before running. `pg_store` uses `ON CONFLICT (txn_id) DO NOTHING`. If either layer breaks, you will get duplicate data.

**4. Polars never touches the original pandas DataFrame.**
`run_analytics()` converts `pandas → Polars` internally and returns a `pl.DataFrame`. The **pandas** `df` is what gets pickled and inserted into PostgreSQL.

**5. `_model` in `pg_store.py` is a module-level singleton.**
`SentenceTransformer` loads once per process. Running pg-related tests in parallel can cause contention — keep them serial.

**6. `DJANGO_SECRET_KEY` has a dangerous hardcoded fallback.**
`settings.py` defaults to `"dev-secret-key-change-in-production"`. Always explicitly set this in `.env` for any non-local environment.

**7. `DEBUG=True` and `ALLOWED_HOSTS=["*"]` are the defaults.**
Both are insecure for production. Set `DJANGO_DEBUG=false` and `DJANGO_ALLOWED_HOSTS=yourdomain.com` in `.env`.

**8. Two independent LangGraph graphs — `views.insights()` only calls one.**
`langchain_agent.py` (3-node, conversational) is what the dashboard uses. `graph.py` (5-node, structured anomaly detection) is standalone. They are not connected.

**9. FX rates in `polars_analytics.py` are duplicated.**
The `_FX` dict and the `_step_with_columns` `when-then` expression both define FX rates. Changing a rate requires updating **two places** — easy to miss.

**10. `rolling_sum` is index-based, not time-based.**
After `group_by((account_id, date))`, one row = one day — **only if there are no missing days**. Gaps in dates will silently produce wrong rolling sums. This is a known limitation.

---

## 12. Known Risks & Missing Pieces

| Issue | Location | Severity |
|---|---|---|
| `DJANGO_SECRET_KEY` hardcoded fallback | `settings.py` | 🔴 High — exposes session signing if used in prod |
| No authentication on any endpoint | `views.py` — all 3 views are public | 🔴 High — `/trigger/` can run the pipeline against arbitrary files |
| Pipeline runs synchronously inside HTTP request | `views.trigger_pipeline()` | 🟡 Medium — large files will time out; no Celery/task queue |
| `ConversationBufferMemory` is module-level and unbounded | `langchain_agent.py` | 🟡 Medium — memory grows forever in long CLI sessions |
| `sftp_watcher.py` not wired into `docker-compose.yml` | `docker-compose.yml` | 🟡 Medium — the better watcher is unreachable via Docker |
| No tests for `ai/graph.py`, `views.py`, `pipeline_runner.py` | `tests/` | 🟡 Medium — core paths untested without a live LLM key |
| `test_sftp_watcher.py` content unverified | `tests/` | 🟢 Low — may be empty or incomplete |
| FX rates duplicated in two places | `polars_analytics.py` | 🟢 Low — easy to introduce inconsistency |
| `sentence-transformers` model downloads from HuggingFace on first run | `pg_store.py` | 🟢 Low — breaks in air-gapped environments |
| Pickle artifacts accumulate indefinitely in `artifacts/` | `pickle_store.py` | 🟢 Low — disk fills over time; no retention policy |
| `rolling_sum` uses index-based window, not `period="7d"` | `polars_analytics.py` | 🟢 Low — silently wrong for sparse data |

---

## Contributing

1. Never duplicate pipeline logic — extend `pipeline_runner.py`.
2. Never hardcode prompts — add to `ai/prompts.py`.
3. All new secrets go in `.env.example` with a description.
4. Run `pytest` before pushing — all non-DB tests must pass without credentials.

---

*Document generated from codebase analysis. Keep in sync when adding new modules or changing data flows.*