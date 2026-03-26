   # data_pipeline

## 1. Project Overview

`data_pipeline` is a mini end-to-end data-ingestion service that watches an FTP/SFTP directory for Excel files, cleans and validates them with pandas, runs Polars analytics, persists results to PostgreSQL with pgvector embeddings, and surfaces insights through a Django dashboard and a LangGraph AI agent.

---

## 2. Project Structure

```
data_pipeline/
├── generate_data.py              # Synthetic Excel data generator (Faker)
├── ingestion/
│   ├── __init__.py
│   └── pandas_ingestor.py        # Schema validation, dtype fixing, null handling
├── analytics/
│   ├── __init__.py
│   └── polars_analytics.py       # Lazy Polars pipeline → rolling 7-day metrics
├── persistence/
│   ├── __init__.py
│   ├── pickle_store.py           # Versioned pickle save/load with SHA-256 integrity
│   └── pg_store.py               # PostgreSQL + pgvector insert and similarity search
├── watcher/
│   ├── __init__.py
│   ├── ftp_watcher.py            # FTP/SFTP poller with idempotency + retry
│   └── processed_files.json      # Auto-generated state file (gitignore this)
├── dashboard/
│   ├── manage.py
│   ├── root_urls.py
│   └── dashboard/
│       ├── __init__.py
│       ├── settings.py
│       ├── wsgi.py
│       ├── models.py             # ProcessedFile ORM model
│       ├── views.py              # List, trigger, and insights endpoints
│       ├── urls.py
│       ├── pipeline_runner.py    # Shared pipeline helper (view + management cmd)
│       ├── management/
│       │   └── commands/
│       │       └── run_pipeline.py
│       └── templates/
│           └── dashboard/
│               └── index.html    # Bootstrap 5 dashboard UI
├── ai/
│   ├── __init__.py
│   └── langchain_agent.py        # 3-node LangGraph agent with multi-turn memory
├── tests/
│   ├── __init__.py
│   ├── test_schema_validation.py
│   ├── test_rolling_metric.py
│   ├── test_idempotency.py
│   ├── test_pickle_integrity.py
│   └── test_pg_idempotency.py
├── data/                         # Output of generate_data.py (auto-created)
├── artifacts/                    # Versioned pickle files (auto-created)
├── tmp/incoming/                 # Watcher download staging area (auto-created)
├── venv/                         # Python virtual environment (local dev only)
├── Dockerfile                    # Container image for web + watcher services
├── docker-compose.yml            # Orchestrates db, web, and watcher containers
├── setup.sh                      # One-command bootstrap script (non-Docker)
├── stop.sh                       # Graceful shutdown script (non-Docker)
├── run_windows.bat               # Windows bootstrap script (non-Docker)
├── requirements.txt
├── .env.example
├── conftest.py                   # Shared pytest config + fixtures
├── pytest.ini
└── README.md
```

---

## 3. Prerequisites

| Requirement | Version / Notes |
|---|---|
| Docker | 24+ with Docker Compose v2 (`docker compose` command) |
| OpenAI API key | Required only if using the AI agent with `LANGCHAIN_LLM_PROVIDER=openai` |

No local Python or PostgreSQL installation needed — everything runs inside containers.
The `ankane/pgvector` image ships with the `vector` extension pre-installed.

---

## 4. Setup Instructions

**Step 1 — Get the code**
```bash
cd data_pipeline
```

**Step 2 — Configure environment variables**
```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```
Open `.env` and fill in your real credentials (DB password, OpenAI key, FTP details) before continuing.

**Step 3 — Build and start all services**
```bash
docker compose up --build
```

This single command:
1. Pulls `ankane/pgvector` (PostgreSQL 15 + pgvector extension)
2. Builds the `web` and `watcher` images from `Dockerfile`
3. Waits for the database to be healthy
4. Runs Django migrations (`manage.py migrate`)
5. Generates 5 000 rows of synthetic sample data (`data/sample_transactions.xlsx`)
6. Starts the Django development server on port **8000**
7. Starts the FTP/SFTP watcher in a separate container

**Step 4 — Stop everything**
```bash
docker compose down
```
Add `-v` to also delete the PostgreSQL volume: `docker compose down -v`

---

## 5. Environment Variables

All variables are read from `.env` (loaded via `python-dotenv`). Copy `.env.example` to `.env` and fill in real values.

| Variable | Description | Example Value |
|---|---|---|
| `DB_HOST` | PostgreSQL server hostname | `localhost` |
| `DB_PORT` | PostgreSQL port | `5432` |
| `DB_NAME` | Database name | `pipeline_db` |
| `DB_USER` | Database user | `pipeline_user` |
| `DB_PASSWORD` | Database password | `changeme` |
| `FTP_HOST` | FTP server hostname | `ftp.example.com` |
| `FTP_PORT` | FTP port | `21` |
| `FTP_USER` | FTP username | `ftp_user` |
| `FTP_PASSWORD` | FTP password | `changeme` |
| `FTP_REMOTE_DIR` | Remote directory to watch | `/incoming` |
| `SFTP_HOST` | SFTP server hostname | `sftp.example.com` |
| `SFTP_PORT` | SFTP port | `22` |
| `SFTP_USER` | SFTP username | `sftp_user` |
| `SFTP_KEY_PATH` | Absolute path to SSH private key | `~/.ssh/id_rsa` |
| `SFTP_REMOTE_DIR` | Remote SFTP directory to watch | `/incoming` |
| `WATCHER_POLL_INTERVAL` | Seconds between directory polls | `30` |
| `WATCHER_MODE` | Transport to use: `ftp` or `sftp` | `ftp` |
| `LANGCHAIN_LLM_PROVIDER` | LLM backend: `openai` or `anthropic` | `openai` |
| `OPENAI_API_KEY` | OpenAI API key (required for openai provider) | `sk-...` |

---

## 6. How to Run

**Start all services**
```bash
docker compose up --build
```

**Start in detached (background) mode**
```bash
docker compose up --build -d
```

**Stop all services**
```bash
docker compose down
```

**Generate synthetic test data**
```bash
docker compose exec web python generate_data.py --rows 5000
# Output: data/sample_transactions.xlsx
```

**Generate a full scenario suite (happy path + edge cases)**
```bash
docker compose exec web python generate_data.py --scenario suite --rows 5000 --output data/
# Outputs:
# data/sample_transactions_happy.xlsx
# data/sample_transactions_balanced.xlsx
# data/sample_transactions_noisy.xlsx
# data/sample_transactions_idempotency.xlsx
```

**Generate a custom stress file (single mode with tunable anomaly rates)**
```bash
docker compose exec web python generate_data.py \
  --scenario single \
  --rows 10000 \
  --output data/sample_transactions_stress.xlsx \
  --malformed-ts-rate 0.30 \
  --null-amount-rate 0.12 \
  --duplicate-txn-rate 0.08 \
  --invalid-currency-rate 0.10
```

**Run the pipeline manually for a specific file**
```bash
docker compose exec web python dashboard/manage.py run_pipeline --file data/sample_transactions.xlsx
```

**Run the watcher standalone**
```bash
docker compose exec watcher python watcher/ftp_watcher.py
# Polls every WATCHER_POLL_INTERVAL seconds; Ctrl-C to stop cleanly
```

**Run the AI agent (CLI, multi-turn)**
```bash
docker compose exec web python ai/langchain_agent.py --question "Which accounts spiked this week?"
# After the first answer, type follow-up questions interactively
# Ctrl-C to exit
```

**Dashboard UI**
```
GET http://localhost:8000/dashboard/
```

**AI insights via HTTP**
```
GET http://localhost:8000/dashboard/insights/
→ {"insights": "## Analysis\n..."}
```

**Trigger pipeline via HTTP**
```
POST http://localhost:8000/dashboard/trigger/?file=sample_transactions.xlsx
→ {"status": "success", "file": "sample_transactions.xlsx"}
```

**Run all tests**
```bash
docker compose exec web pytest tests/
# PostgreSQL tests auto-skip if DB env vars are not set
```

**Run a single test file**
```bash
docker compose exec web pytest tests/test_pickle_integrity.py -v
```

**View live logs**
```bash
docker compose logs -f web
docker compose logs -f watcher
```

---

## 7. Pickle Safety

### Why pickle is unsafe for untrusted input

Python's `pickle` module executes arbitrary bytecode during deserialisation. A maliciously crafted `.pkl` file can run any code on the host machine the moment `pickle.load()` is called — no special permissions required. This makes loading pickles from untrusted or external sources a critical security risk.

### How this project mitigates the risk

This project treats pickle as an **internal artifact format only** — files are never received from external parties. Three controls are in place:

1. **SHA-256 checksum** — a `.meta.json` sidecar is written at save time containing the hex digest of the `.pkl` file. On every load, the digest is recomputed and compared before `pickle.load()` is called. Any byte-level tampering raises `PickleIntegrityError` immediately.
2. **Row-count verification** — after loading, the DataFrame's row count is compared against the value stored in `.meta.json`. A mismatch (e.g. from a truncated write) raises `PickleIntegrityError`.
3. **Versioned, timestamped filenames** — artifacts are written to a controlled local directory (`artifacts/`) with names like `processed_df_v1_20240615_143022.pkl`, making accidental overwrite or injection difficult.

### Recommended alternatives for production

| Format | Library | Advantages |
|---|---|---|
| **Parquet** | `pandas.to_parquet` / `polars.write_parquet` | Columnar, compressed, language-agnostic, safe to deserialise |
| **Feather / Arrow IPC** | `pandas.to_feather` / `polars.write_ipc` | Fastest read/write, zero-copy memory mapping, no code execution on load |
| **CSV / JSON** | stdlib | Human-readable, universally safe, but large and slow for wide DataFrames |

For any pipeline that receives files from external systems, replace `pickle_store.py` with a Parquet-based store.

---

## 8. Design Decisions

**Fully lazy Polars pipeline**
All four analytics steps (filter, with_columns, group_by, rolling) are chained as `LazyFrame` operations and `.collect()` is called exactly once. This lets Polars optimise the full query plan — predicate pushdown, projection pruning — before touching memory.

**Idempotent inserts everywhere**
Both the FTP watcher (SHA-256 + `processed_files.json`) and the PostgreSQL layer (`ON CONFLICT (txn_id) DO NOTHING`) are independently idempotent. A file can be re-delivered or the process can crash mid-batch without producing duplicate data.

**Shared `pipeline_runner.py`**
The Django trigger view and the `run_pipeline` management command share a single `run_pipeline_for_file()` function. This avoids duplicating the ingest → analytics → pickle → pg chain and ensures both entry points behave identically.

**Module-level model cache in `pg_store`**
`SentenceTransformer` takes several seconds to load. Caching it at module level means it is loaded once per process regardless of how many files are processed in a session.

**LangGraph over a plain chain**
Using `StateGraph` makes each node independently testable and the data flow explicit. Adding a new node (e.g. a `validator` between `analyst` and `formatter`) requires only a new function and two `add_edge` calls.

**`ConversationBufferMemory` at module level**
The CLI memory object lives at module scope so it persists across `run_agent()` calls within the same process. The Django endpoint gets a fresh context per request (stateless HTTP), which is the correct behaviour for a web API.

---

## 9. Known Limitations + What I'd Improve Next

| Limitation | Improvement |
|---|---|
| Django trigger endpoint runs the pipeline **synchronously** in the request/response cycle — large files will time out | Replace with Celery + Redis: the view enqueues a task and returns `{"status": "queued"}` immediately |
| `ConversationBufferMemory` grows unbounded in long CLI sessions | Switch to `ConversationSummaryBufferMemory` with a token limit, or persist to a vector store for long-term memory |
| Pickle artifacts accumulate indefinitely in `artifacts/` | Add a retention policy: keep the last N versions per file, archive or delete older ones |
| FTP/SFTP connections are opened and closed on every poll cycle | Use a persistent connection with reconnect-on-failure to reduce handshake overhead on high-frequency polling |
| `sentence-transformers` model is downloaded from HuggingFace Hub on first run | Bundle the model in the repo or pre-download to a fixed path; set `TRANSFORMERS_OFFLINE=1` in production |
| No authentication on Django endpoints | Add Django's built-in session auth or token auth (DRF `TokenAuthentication`) before any public deployment |
| Tests for the AI agent require a live LLM API key | Add a `pytest` fixture that mocks `_build_llm()` with a `FakeLLM` so the graph topology can be tested offline |
| `rolling_sum` uses index-based windows (one row = one day) | Switch to a true time-based rolling window (`rolling` with `period="7d"`) to handle missing days correctly |
