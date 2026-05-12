"""
================================================================================
  CNN PIPELINE — CONFIGURATION
================================================================================
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class CNNConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    data_path:          str   = "../project_database.csv"
    output_dir:         str   = "results"
    id_col:             str   = "co_code"
    date_col:           str   = "Month"
    target_col:         str   = "monthly_gross_return"
    features: tuple = (
        "BM_sep", "lag_mv", "OpProf", "Inv", "Momentum", "lag_ret", "mktcap"
    )
    log_transform_cols: tuple = ("mktcap",)

    # ── Preprocessing ─────────────────────────────────────────────────────────
    winsor_lower:  float = 0.01
    winsor_upper:  float = 0.99
    min_obs:       int   = 30

    # ── Sequence / Walk-forward ────────────────────────────────────────────────
    seq_len:       int  = 24
    train_window:  int  = 60
    val_months:    int  = 6

    # ── Default CNN architecture ──────────────────────────────────────────────
    n_filters:     int        = 32
    kernel_sizes:  tuple      = (3, 6, 12)   # short / medium / long branches
    n_conv_blocks: int        = 1            # stacked conv blocks per branch
    dropout:       float      = 0.1

    # ── Training ──────────────────────────────────────────────────────────────
    lr:              float = 3e-4
    weight_decay:    float = 1e-4
    batch_size:      int   = 512
    max_epochs:      int   = 100
    early_stop_patience: int = 10
    grad_clip:       float = 1.0
    device:          str  = "auto"

    # ── Hyperparameter search space ────────────────────────────────────────────
    search_space: dict = field(default_factory=lambda: {
        "n_filters":     [16, 32],
        "kernel_sizes":  [(3, 6, 12), (2, 4, 8)],
        "n_conv_blocks": [1, 2],
        "dropout":       [0.1, 0.2],
        "lr":            [1e-4, 3e-4],
        "batch_size":    [512],
        "seq_len":       [12, 24],
    })

    # ── Runtime ───────────────────────────────────────────────────────────────
    tune_hparams: bool = False
    verbose:      bool = False
    seed:         int  = 42
    n_workers:    int  = -1

    # ── Baselines (for reporting) ──────────────────────────────────────────────
    ridge_icir: float = 0.2504
    lgbm_icir:  float = 0.7347
    rf_icir:    float = 0.7248


DEFAULT_CONFIG = CNNConfig()
