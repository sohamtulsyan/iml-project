"""
================================================================================
  TRANSFORMER MODEL — ARCHITECTURE + DEVICE UTILITIES
================================================================================
"""

import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  Device detection
# ─────────────────────────────────────────────────────────────────────────────

def get_device(device: str = "auto") -> torch.device:
    """
    Auto-detect best device: CUDA > MPS > CPU.
    Logs device name and VRAM if CUDA is found.
    """
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Device] CUDA GPU : {name}  ({vram:.1f} GB VRAM)")
        # cuDNN optimizations for fixed input sizes
        torch.backends.cudnn.benchmark     = True
        torch.backends.cudnn.deterministic = False
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[Device] Apple Silicon MPS backend")
        # Use all CPU cores for ops that fall back to CPU on MPS
        torch.set_num_threads(os.cpu_count())
    else:
        dev = torch.device("cpu")
        n_threads = os.cpu_count()
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(max(1, n_threads // 2))
        print(f"[Device] CPU — {n_threads} threads")

    return dev


# ─────────────────────────────────────────────────────────────────────────────
#  Sinusoidal positional encoding
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encoding to the input embedding.
    Supports sequences up to max_len positions.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                         # (L, d)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)                                       # (1, L, d)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Transformer Encoder model
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoderModel(nn.Module):
    """
    Causal Transformer Encoder for cross-sectional equity return prediction.

    Input:  (batch_size, T, n_features)
    Output: (batch_size, 1)  — raw return score (regression)

    Architecture:
      Linear projection → Positional Encoding → TransformerEncoder (pre-LN)
      → Last-token pooling → Linear head
    """

    def __init__(
        self,
        n_features:     int   = 7,
        d_model:        int   = 64,
        nhead:          int   = 4,
        num_layers:     int   = 2,
        dim_feedforward: int  = 256,
        dropout:        float = 0.1,
        seq_len:        int   = 24,
    ):
        super().__init__()
        assert d_model % nhead == 0, \
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.seq_len  = seq_len
        self.d_model  = d_model

        # Input projection: raw features → d_model
        self.input_proj = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        # Causal mask (pre-built, registered as buffer so it moves with the model)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", causal_mask)

        # Encoder layers with Pre-LN (norm_first=True) for stable training
        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward = dim_feedforward,
            dropout        = dropout,
            activation     = "gelu",
            batch_first    = True,   # (batch, seq, feat) convention
            norm_first     = True,   # Pre-LN: more stable on small datasets
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers          = num_layers,
            norm                = nn.LayerNorm(d_model),
            enable_nested_tensor = False,  # required when norm_first=True
        )

        # Output regression head
        self.head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for Linear layers; zero bias."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, T, n_features)
        Returns: (batch, 1)
        """
        T = x.size(1)
        # Resize mask if T differs from seq_len (e.g. shorter history)
        mask = self.causal_mask[:T, :T]

        x = self.input_proj(x)        # (batch, T, d_model)
        x = self.pos_enc(x)           # (batch, T, d_model)  + positional info
        x = self.encoder(x, mask=mask)  # (batch, T, d_model)
        x = x[:, -1, :]               # last token — has attended to full history
        return self.head(x)           # (batch, 1)


def build_model(cfg, n_features: int = 7) -> TransformerEncoderModel:
    """Instantiate model from config dict or TransformerConfig."""
    return TransformerEncoderModel(
        n_features      = n_features,
        d_model         = cfg.d_model if hasattr(cfg, "d_model") else cfg["d_model"],
        nhead           = cfg.nhead if hasattr(cfg, "nhead") else cfg["nhead"],
        num_layers      = cfg.num_layers if hasattr(cfg, "num_layers") else cfg["num_layers"],
        dim_feedforward = cfg.dim_feedforward if hasattr(cfg, "dim_feedforward") else cfg["dim_feedforward"],
        dropout         = cfg.dropout if hasattr(cfg, "dropout") else cfg["dropout"],
        seq_len         = cfg.seq_len if hasattr(cfg, "seq_len") else cfg["seq_len"],
    )
