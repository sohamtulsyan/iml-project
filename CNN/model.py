"""
================================================================================
  CNN PIPELINE — MODEL ARCHITECTURE
================================================================================

  Multi-Scale Temporal CNN for cross-sectional equity return prediction.

  Input:  (batch, T, 7)   — time-first from SequenceBuilder
  Inside: transposed to (batch, 7, T)   — channel-first for Conv1d
  Output: (batch, 1)      — raw return score (regression)

  Architecture:
  ─────────────
  Input projection (1×1 conv) → 3 parallel causal branches (k=3, 6, 12)
  → concat → global avg + max pool → MLP head

  Key design choices:
  ───────────────────
  - CausalConv1d: left-only padding (no future leakage)
  - BatchNorm1d after every conv (stable on large monthly cross-sections)
  - Residual connections when shapes match (ResNet-style gradient flow)
  - Both AvgPool + MaxPool concatenated (mean signal + peak-spike capture)
  - Kaiming init for Conv1d; Xavier for Linear
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  Device detection  (shared logic; mirrors Transformer/model.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_device(device: str = "auto") -> torch.device:
    """CUDA → MPS → CPU, with appropriate runtime flags."""
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Device] CUDA GPU : {name}  ({vram:.1f} GB VRAM)")
        torch.backends.cudnn.benchmark     = True
        torch.backends.cudnn.deterministic = False
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[Device] Apple Silicon MPS backend")
        torch.set_num_threads(os.cpu_count())
    else:
        dev = torch.device("cpu")
        n  = os.cpu_count()
        torch.set_num_threads(n)
        torch.set_num_interop_threads(max(1, n // 2))
        print(f"[Device] CPU — {n} threads")

    return dev


# ─────────────────────────────────────────────────────────────────────────────
#  Causal Conv1d (left-only padding — no future leakage)
# ─────────────────────────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """
    1D convolution with causal (left-only) padding.

    Pads `kernel_size - 1` zeros on the LEFT of the time dimension so that
    the output at position t depends only on inputs at positions ≤ t.
    This is the CNN equivalent of the Transformer's upper-triangular mask.

    Input / output shape: (batch, channels, T)  — channel-first
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 bias: bool = False, **kwargs):
        super().__init__()
        self.left_pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0, bias=bias, **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, C, T)"""
        x = F.pad(x, (self.left_pad, 0))   # pad left only
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Single convolutional block:  CausalConv1d → BN → GELU (+ residual)
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    One causal conv layer + BatchNorm + GELU, with optional residual.
    Residual is applied only when in_channels == out_channels.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        self.causal_conv = CausalConv1d(in_ch, out_ch, kernel_size)
        self.bn           = nn.BatchNorm1d(out_ch)
        self.act          = nn.GELU()
        self.use_residual = (in_ch == out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn(self.causal_conv(x)))
        if self.use_residual:
            out = out + x
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Single multi-scale branch: stack of ConvBlocks with one kernel size
# ─────────────────────────────────────────────────────────────────────────────

class ConvBranch(nn.Module):
    """
    A sequential stack of n_blocks ConvBlocks, all with the same kernel_size.
    Captures factor patterns at one specific timescale.
    """

    def __init__(self, n_filters: int, kernel_size: int, n_blocks: int = 1):
        super().__init__()
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ConvBlock(n_filters, n_filters, kernel_size))
        self.branch = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-Scale Temporal CNN
# ─────────────────────────────────────────────────────────────────────────────

class TemporalCNN(nn.Module):
    """
    Multi-Scale 1D Temporal CNN for equity return prediction.

    Architecture:
    1. Input projection:  7 → n_filters via 1×1 causal conv + BN + GELU
    2. 3 parallel branches:  kernel sizes k_short, k_mid, k_long
    3. Concat branches → (batch, 3*n_filters, T)
    4. Global AvgPool1d + GlobalMaxPool1d → concat → (batch, 6*n_filters)
    5. MLP head: Linear(6*n_filters, 64) → GELU → Dropout → Linear(64, 1)
    """

    def __init__(
        self,
        n_features:    int         = 7,
        n_filters:     int         = 32,
        kernel_sizes:  tuple       = (3, 6, 12),
        n_conv_blocks: int         = 1,
        dropout:       float       = 0.1,
    ):
        super().__init__()
        assert len(kernel_sizes) == 3, "Exactly 3 branch kernel sizes required"

        self.n_features = n_features
        self.n_filters  = n_filters

        # ── Input projection: 7 → n_filters (1×1 conv across features) ────────
        self.input_proj = nn.Sequential(
            CausalConv1d(n_features, n_filters, kernel_size=1, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.GELU(),
        )

        # ── Three parallel branches ────────────────────────────────────────────
        self.branch_short  = ConvBranch(n_filters, kernel_sizes[0], n_conv_blocks)
        self.branch_medium = ConvBranch(n_filters, kernel_sizes[1], n_conv_blocks)
        self.branch_long   = ConvBranch(n_filters, kernel_sizes[2], n_conv_blocks)

        # ── MLP head ───────────────────────────────────────────────────────────
        # 3 branches × 2 pool types (avg + max) = 6 × n_filters input features
        head_in = 6 * n_filters
        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming uniform for Conv1d; Xavier uniform for Linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, T, 7)  — time-first from SequenceBuilder
        Returns: (batch, 1)
        """
        # ── Transpose to channel-first for Conv1d ──────────────────────────────
        x = x.transpose(1, 2)          # (batch, 7, T)

        # ── Input projection ───────────────────────────────────────────────────
        x = self.input_proj(x)         # (batch, n_filters, T)

        # ── Three branches in parallel ─────────────────────────────────────────
        b1 = self.branch_short(x)      # (batch, n_filters, T)
        b2 = self.branch_medium(x)     # (batch, n_filters, T)
        b3 = self.branch_long(x)       # (batch, n_filters, T)

        out = torch.cat([b1, b2, b3], dim=1)   # (batch, 3*n_filters, T)

        # ── Global pooling — avg + max — then concat ───────────────────────────
        avg = out.mean(dim=2)          # (batch, 3*n_filters)
        mx  = out.max(dim=2).values    # (batch, 3*n_filters)
        pooled = torch.cat([avg, mx], dim=1)   # (batch, 6*n_filters)

        return self.head(pooled)       # (batch, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience builder (accepts dict or CNNConfig)
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg, n_features: int = 7) -> TemporalCNN:
    """Instantiate TemporalCNN from a CNNConfig or plain dict."""
    def _get(k, default):
        if isinstance(cfg, dict):
            return cfg.get(k, default)
        return getattr(cfg, k, default)

    return TemporalCNN(
        n_features    = n_features,
        n_filters     = _get("n_filters",     32),
        kernel_sizes  = tuple(_get("kernel_sizes", (3, 6, 12))),
        n_conv_blocks = _get("n_conv_blocks", 1),
        dropout       = _get("dropout",       0.1),
    )
