# Django Process Monitor – Full Stack Blueprint

A production-ready scaffold for a Windows Agent (EXE) + Django REST backend + Bootstrap frontend to collect and visualize running processes from one or more Windows machines.

## Architecture Overview

- **Agent (Windows)**: Python script packaged with PyInstaller into a single EXE. Periodically enumerates processes via `psutil` and posts JSON to the backend with an API key header.
- **Backend (Django + DRF)**: Stores snapshots per machine in SQLite (dev) or Postgres (prod). Provides ingestion and read APIs.
- **Frontend (Bootstrap)**: Simple dashboard to select a machine, view the latest snapshot time, and navigate a collapsible parent→children process tree.

### Data Model
- `Machine(hostname)`
- `Snapshot(machine, created_at, process_count)`
- `ProcessRow(snapshot, pid, ppid, name, cpu_percent, memory_rss, memory_percent)`

### API Authentication
- Inbound agent requests use an API key header: `X-API-KEY: <key>`
- Key is configured in Django via `PMA_API_KEY` env var (or `settings.API_KEY`).

---

## Quickstart (Development)

### Prerequisites
- Linux/macOS/WSL (Windows for agent build) with Python 3.11
- Recommended: a dedicated virtual environment

### Clone and set up
```bash
cd /home/poojan16/Desktop/Project
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configure Django
- Optionally set an API key:
```bash
export PMA_API_KEY="dev-api-key"
```
- Development DB is SQLite at `PMA/db.sqlite3` by default.

### Run migrations and server
```bash
cd PMA
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```
Visit `http://localhost:8000` for the dashboard.

---

## Agent (Windows)

### Layout
- `agent/agent.py`: The Windows agent
- `agent/agent_config.json`: Local config for endpoint, api_key, interval

### Run locally (Windows)
```powershell
python -m pip install psutil requests
# Edit agent_config.json to point to your server and API key
python agent.py
```

### Build EXE (Windows)
```powershell
python -m pip install pyinstaller
pyinstaller --onefile --name ProcessAgent --add-data agent_config.json;. agent.py
# Result: dist/ProcessAgent.exe; keep agent_config.json next to the EXE
```

### Payload Format
```json
{
  "hostname": "MY-PC",
  "processes": [
    {
      "pid": 1234,
      "ppid": 456,
      "name": "explorer.exe",
      "cpu_percent": 1.2,
      "memory_rss": 34527232,
      "memory_percent": 0.9
    }
  ]
}
```

POST to `POST /api/ingest/` with header `X-API-KEY: <key>`.

---

## API Reference

### POST /api/ingest/
- Headers: `Content-Type: application/json`, `X-API-KEY: <key>`
- Body: payload as above
- Response: `{ "status": "ok", "snapshot_id": 1, "process_count": 327 }`

### GET /api/latest/<hostname>/
- Returns the latest snapshot for the given host
- Response shape:
```json
{
  "hostname": "MY-PC",
  "snapshot_time": "2025-08-21T10:12:34.567Z",
  "processes": [ { "pid": 1, "ppid": 0, "name": "System", ... } ]
}
```

### GET /api/machines/
- Returns list of hostnames: `{ "machines": ["HOST1", "HOST2"] }`

---

## Frontend
- Entry: `GET /` → template `main/templates/main/dashboard.html`
- Features:
  - Machine dropdown (populated server-side)
  - Latest snapshot time
  - Manual refresh button
  - Expandable parent→child process tree with toggles

---

## Deployment (Production)

### Settings
- `DEBUG = False`
- Set `ALLOWED_HOSTS` to your domain/IP
- Set `PMA_API_KEY` via environment variable
- Consider Postgres instead of SQLite (configure `DATABASES`)

### Example: Gunicorn + Nginx (Ubuntu)
```bash
# Install deps
pip install gunicorn

# Collect static (if needed)
python manage.py collectstatic --noinput

# Run gunicorn
exec gunicorn PMA.wsgi:application --bind 0.0.0.0:8000 --workers 3
```
Configure Nginx to reverse proxy to `127.0.0.1:8000` and serve TLS.

---

## Security Considerations
- Keep `PMA_API_KEY` secret; rotate periodically
- Limit ingestion endpoint IPs via firewall/reverse proxy if possible
- Use HTTPS in production; terminate TLS at Nginx/ALB
- Validate payload sizes; current code bulk-creates rows but does not enforce explicit limits

---

## Troubleshooting
- "ModuleNotFoundError: rest_framework": Ensure you are using the project venv
  - `source venv/bin/activate && pip install -r requirements.txt`
- VA-API `libva` warnings on Linux shells are unrelated; safe to ignore
- 403/401 from `/api/ingest/`: Verify `X-API-KEY` matches `PMA_API_KEY`
- No machines in dropdown: Visit dashboard after first successful agent post to create a `Machine`
- Slow dashboard with huge snapshots: consider server-side pagination or limit process fields

---

## Development Notes
- DRF is configured with JSON parser/renderer; Browsable API is enabled
- Database indexes on `(snapshot, pid)` and `(snapshot, ppid)` improve lookup
- The process tree is built client-side for simplicity

---

## License
MIT (adjust as needed) 