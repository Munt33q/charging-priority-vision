"""Check GPU quota + push a smoke-test kernel to verify GPU sessions work."""
import sys

sys.path.insert(0, r"E:\RPHD\scripts")
from kaggle_client import load_tokens
from kagglesdk import KaggleClient as SdkClient, KaggleEnv
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetAcceleratorQuotaStatisticsRequest,
    ApiSaveKernelRequest,
    ApiGetKernelSessionStatusRequest,
)

token = load_tokens()[0]
sdk = SdkClient(env=KaggleEnv.PROD, username="mluyxp11", api_token=token)

q = sdk.kernels.kernels_api_client.get_accelerator_quota_statistics(
    ApiGetAcceleratorQuotaStatisticsRequest())
g = q.gpu_quota
print("quota refresh:", q.quota_refresh_time)
if g:
    print("gpu quota: total =", getattr(g, "total", None), "| used =", getattr(g, "used", None))

# push a 30-second GPU smoke kernel
code = """
import torch, subprocess
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
print('torch:', torch.__version__)
x = torch.randn(1000, 1000, device='cuda')
print('matmul ok:', (x @ x).sum().item() != 0)
"""
req = ApiSaveKernelRequest()
req.slug = "mluyxp11/gpu-smoke-test"
req.new_title = "gpu smoke test"
req.text = code
req.language = "python"
req.kernel_type = "script"
req.is_private = True
req.enable_internet = False
req.machine_shape = "NvidiaTeslaP100"
try:
    resp = sdk.kernels.kernels_api_client.save_kernel(req)
    print("smoke kernel pushed:", resp.ref, "| version:", resp.version_number, "| error:", resp.error)
except Exception as e:
    r = getattr(e, "response", None)
    print("EXC:", str(e)[:200], "| body:", r.text[:400] if r is not None else "n/a")
