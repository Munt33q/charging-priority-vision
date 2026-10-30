"""Watch the exact submitted Kaggle version, not the mutable kernel slug."""
import json
import os
import sys
import time
import requests
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_client import load_tokens
from kagglesdk import KaggleClient, KaggleEnv
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

ROOT = Path(__file__).resolve().parents[1]
USER = "mluyxp11"
slug = sys.argv[1] if len(sys.argv) > 1 else "charging-priority-train-retry"
version = sys.argv[2] if len(sys.argv) > 2 else None
saved = ROOT / "reports" / "kaggle_submission.json"
if version is None and saved.is_file():
    data = json.loads(saved.read_text())
    if data.get("slug") == slug:
        version = str(data["version"])
if not version:
    raise SystemExit("Specify the submitted version: python 11_watch_kaggle.py <slug> <version>")

token = load_tokens()[0]
sdk = KaggleClient(env=KaggleEnv.PROD, username=USER, api_token=token)
last_log = ""
for _ in range(210):
    raw = requests.get("https://api.kaggle.com/v1/kernels/status",
                       params={"userName": USER, "kernelSlug": slug},
                       headers={"Authorization": f"Bearer {token}"}, timeout=60)
    raw.raise_for_status()
    status = raw.json().get("status", "unknown").upper()
    oreq = ApiListKernelSessionOutputRequest()
    oreq.user_name, oreq.kernel_slug = USER, slug
    out = sdk.kernels.kernels_api_client.list_kernel_session_output(oreq)
    log = getattr(out, "log", "") or ""
    if log != last_log:
        print(f"[{datetime.now():%H:%M:%S}] {status} version={version} log_changed={bool(log)}", flush=True)
        last_log = log
    if status not in ("RUNNING", "QUEUED"):
        break
    time.sleep(120)
print("WATCH COMPLETE:", status)
