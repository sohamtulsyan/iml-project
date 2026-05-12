"""
shared/device.py
================
Single get_device() implementation shared by ALL neural models.
Priority: CUDA → MPS → CPU.
"""
import os
import torch

# Essential for stability on Apple Silicon MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


_DEVICE_CACHE = {}

def get_device(device_str: str = "auto") -> torch.device:
    """Return optimal torch.device (cached) and configure backend flags."""
    global _DEVICE_CACHE
    if device_str in _DEVICE_CACHE:
        return _DEVICE_CACHE[device_str]

    if device_str != "auto":
        dev = torch.device(device_str)
        _DEVICE_CACHE[device_str] = dev
        return dev

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Device] CUDA: {name}  ({vram:.1f} GB VRAM)")
        torch.backends.cudnn.benchmark     = True
        torch.backends.cudnn.deterministic = False
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[Device] Apple Silicon MPS")
    else:
        dev = torch.device("cpu")
        n = os.cpu_count()
        torch.set_num_threads(n)
        print(f"[Device] CPU — {n} threads")

    _DEVICE_CACHE["auto"] = dev
    return dev
