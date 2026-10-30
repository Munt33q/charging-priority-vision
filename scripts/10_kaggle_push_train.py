"""Push the training kernel to Kaggle (GPU P100) using the SDK.

Run AFTER the dataset upload completes. Kernel sources the dataset
mluyxp11/charging-priority-data and runs kaggle_train_kernel.py code inline.
"""
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_client import load_tokens

from kagglesdk import KaggleClient as SdkClient, KaggleEnv
from kagglesdk.kernels.types.kernels_api_service import (
    ApiSaveKernelRequest,
    KernelExecutionType,
)

USER = "mluyxp11"
KERNEL_SLUG = sys.argv[1] if len(sys.argv) > 1 else "charging-priority-train"
DATASET = f"{USER}/charging-priority-data"

code_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_train_kernel.py")
with open(code_path, "r", encoding="utf-8") as f:
    code = f.read()
# Override stale preparation scripts inside the previously uploaded bundle.
bundled = {}
for name in ("04_merge_split.py", "05_augment_weather.py", "paths.py", "weather_aug.py"):
    with open(os.path.join(os.path.dirname(code_path), name), encoding="utf-8") as f:
        bundled[name] = f.read()
code = "BUNDLED_SCRIPTS = " + repr(bundled) + "\n" + code
compile(code, "kaggle_submission.py", "exec")

token = load_tokens()[0]
sdk = SdkClient(env=KaggleEnv.PROD, username=USER, api_token=token)

req = ApiSaveKernelRequest()
req.slug = f"{USER}/{KERNEL_SLUG}"
req.new_title = KERNEL_SLUG.replace("-", " ")
req.text = code
req.language = "python"
req.kernel_type = "script"
req.is_private = True
req.enable_internet = True          # pip install ultralytics
req.dataset_data_sources = [DATASET, f"{USER}/charging-priority-metadata"]
req.machine_shape = "NvidiaTeslaP100"
req.enable_gpu = True

try:
    resp = sdk.kernels.kernels_api_client.save_kernel(req)
    print("PUSHED:", resp.ref, "| version:", resp.version_number, "| error:", resp.error)
    Path(os.path.dirname(os.path.abspath(__file__))).parents[0].joinpath("reports").mkdir(exist_ok=True)
    Path(os.path.dirname(os.path.abspath(__file__))).parents[0].joinpath("reports", "kaggle_submission.json").write_text(
        json.dumps({"user": USER, "slug": KERNEL_SLUG, "ref": resp.ref,
                    "version": resp.version_number, "dataset_sources": req.dataset_data_sources}, indent=2))
    if resp.error:
        raise RuntimeError(resp.error)
except Exception as e:
    r = getattr(e, "response", None)
    print("EXC:", str(e)[:300])
    if r is not None:
        print("body:", r.text[:600])
    raise
