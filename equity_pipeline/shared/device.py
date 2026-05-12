"""
shared/device.py
================
Single get_device() implementation shared by ALL neural models.
Priority: CUDA → MPS → CPU.
"""
import os
import torch


def get_device(device: str = "auto") -> torch.device:
    """Return optimal torch.device and configure backend flags."""
    if device != "auto":
        return torch.device(device)

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
        # Do NOT set_num_threads for MPS; it can cause instability on some macOS versions
    else:
        dev = torch.device("cpu")
        n = os.cpu_count()
        torch.set_num_threads(n)
        print(f"[Device] CPU — {n} threads")

    return dev
