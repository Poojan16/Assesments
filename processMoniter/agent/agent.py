import json
import os
import platform
import socket
import time
from pathlib import Path

import psutil
import requests


def load_config():
	config_path = Path(__file__).with_name("agent_config.json")
	if config_path.exists():
		with open(config_path, "r", encoding="utf-8") as f:
			return json.load(f)
	return {
		"endpoint": "http://localhost:8000/api/ingest/",
		"api_key": os.environ.get("PMA_API_KEY", "dev-api-key"),
		"interval_seconds": 30,
	}


def collect_processes():
	procs = []
	for p in psutil.process_iter(attrs=["pid", "ppid", "name", "memory_info"]):
		try:
			cpu = p.cpu_percent(interval=0.0)
			mem_info = p.info.get("memory_info")
			mem_rss = int(mem_info.rss) if mem_info else 0
			mem_percent = float(p.memory_percent())
			procs.append({
				"pid": int(p.info.get("pid") or p.pid),
				"ppid": int(p.info.get("ppid") or 0),
				"name": p.info.get("name") or "",
				"cpu_percent": float(cpu),
				"memory_rss": mem_rss,
				"memory_percent": mem_percent,
			})
		except (psutil.NoSuchProcess, psutil.AccessDenied):
			continue
	return procs


def main():
	cfg = load_config()
	endpoint = cfg.get("endpoint")
	api_key = cfg.get("api_key")
	interval = int(cfg.get("interval_seconds", 30))
	hostname = socket.gethostname()

	print(f"Agent starting on {platform.system()} host {hostname}. Posting to {endpoint}")
	while True:
		payload = {
			"hostname": hostname,
			"processes": collect_processes(),
		}
		try:
			resp = requests.post(endpoint, json=payload, headers={"X-API-KEY": api_key}, timeout=15)
			if resp.ok:
				data = resp.json()
				print(f"Posted {len(payload['processes'])} processes. Snapshot {data.get('snapshot_id')}.")
			else:
				print(f"Post failed: {resp.status_code} {resp.text}")
		except Exception as e:
			print(f"Error posting data: {e}")
		time.sleep(interval)


if __name__ == "__main__":
	main() 