# Windows Process Agent

## Run locally

1. Install Python 3.11 and pip
2. Install dependencies:

```bash
pip install psutil requests
```

3. Configure `agent_config.json` (endpoint and api_key)
4. Run:

```bash
python agent.py
```

## Build EXE with PyInstaller

```bash
pip install pyinstaller
pyinstaller --onefile --name ProcessAgent --add-data agent_config.json;. agent.py
```

- Double-click `dist/ProcessAgent.exe` to run
- Put `agent_config.json` alongside the exe to change settings

The agent posts JSON to the backend with header `X-API-KEY`. 