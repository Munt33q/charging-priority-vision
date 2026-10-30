"""Try SaveKernel without slug; dump the raw 400 body for diagnosis."""
import sys

sys.path.insert(0, r"E:\RPHD\scripts")
from kaggle_client import load_tokens
from kagglesdk import KaggleClient as SdkClient, KaggleEnv
from kagglesdk.kernels.types.kernels_api_service import (
    ApiSaveKernelRequest,
    KernelExecutionType,
)

token = load_tokens()[0]
sdk = SdkClient(env=KaggleEnv.PROD, username="unknown", api_token=token)
req = ApiSaveKernelRequest()
req.new_title = "charging priority probe"
req.text = "print('probe ok')"
req.language = "python"
req.kernel_type = "script"
req.is_private = True
req.enable_gpu = False
req.enable_internet = False

try:
    resp = sdk.kernels.kernels_api_client.save_kernel(req)
    print("SAVED! ref =", resp.ref, "| error:", resp.error)
except Exception as e:
    r = getattr(e, "response", None)
    if r is not None:
        print("status:", r.status_code)
        print("body:", r.text[:800])
    else:
        print("EXC:", type(e).__name__, str(e)[:300])
