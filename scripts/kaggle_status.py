"""Inspect one exact Kaggle kernel version.

Usage: python kaggle_status.py <slug> [version]
The version defaults to reports/kaggle_submission.json when available.
"""
import json
import sys
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_client import load_tokens
from kagglesdk import KaggleClient as SdkClient, KaggleEnv
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

USER = "mluyxp11"
ROOT = Path(__file__).resolve().parents[1]
slug = sys.argv[1] if len(sys.argv) > 1 else "charging-priority-train"
submission = ROOT / "reports" / "kaggle_submission.json"
version = sys.argv[2] if len(sys.argv) > 2 else None
if version is None and submission.is_file():
    saved = json.loads(submission.read_text())
    if saved.get("slug") == slug:
        version = str(saved["version"])

token = load_tokens()[0]
raw = requests.get("https://api.kaggle.com/v1/kernels/status",
                   params={"userName": USER, "kernelSlug": slug},
                   headers={"Authorization": f"Bearer {token}"}, timeout=60)
raw.raise_for_status()
status_data = raw.json()
status = status_data.get("status", "unknown").upper()
failure = status_data.get("failureMessage", "") or ""
sdk = SdkClient(env=KaggleEnv.PROD, username=USER, api_token=token)
print(f"kernel={USER}/{slug} version={version or 'latest'}")
print("status:", status, "| failure:", failure)

oreq = ApiListKernelSessionOutputRequest()
oreq.user_name, oreq.kernel_slug = USER, slug
out = sdk.kernels.kernels_api_client.list_kernel_session_output(oreq)
files = getattr(out, "files", None) or []
print("output files:", [f.file_name for f in files][:20])
log = getattr(out, "log", None)
if isinstance(log, str) and log.strip():
    try:
        entries = json.loads(log)
        text = "".join(item.get("data", "") for item in entries)
    except (ValueError, TypeError):
        text = log
    report = ROOT / "reports" / f"{slug}_v{version or 'latest'}_log.txt"
    report.write_text(text, encoding="utf-8")
    print("--- latest log ---")
    print(text[-10000:].encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"))
    print("full log:", report)
elif status in ("RUNNING", "QUEUED"):
    print("No persisted output yet; RUNNING alone does not prove execution progress.")
