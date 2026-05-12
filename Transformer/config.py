"""
================================================================================
  TRANSFORMER PIPELINE — CONFIGURATION
  All hyperparameters, paths, and defaults in one place.
================================================================================
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TransformerConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    data_path:     str  = "../project_database.csv"
    output_dir:    str  = "results"
    id_col:        str  = "co_code"
    date_col:      str  = "Month"
    target_col:    str  = "monthly_gross_return"
    features: tuple = (
        "BM_sep", "lag_mv", "OpProf", "Inv", "Momentum", "lag_ret", "mktcap"
    )
    log_transform_cols: tuple = ("mktcap",)

    # ── Preprocessing ─────────────────────────────────────────────────────────
    winsor_lower:  float = 0.01
    winsor_upper:  float = 0.99
    min_obs:       int   = 30    # min stocks per month to include fold

    # ── Sequence / Walk-forward ────────────────────────────────────────────────
    seq_len:       int  = 24     # T — months of history per token sequence
    train_window:  int  = 60     # months in each rolling training window
    val_months:    int  = 6      # last N months of training window used for val IC

    # ── Default model architecture ────────────────────────────────────────────
    d_model:        int   = 64
    nhead:          int   = 4
    num_layers:     int   = 2
    dim_feedforward: int  = 256
    dropout:        float = 0.1

    # ── Training ──────────────────────────────────────────────────────────────
    lr:              float = 3e-4
    weight_decay:    float = 1e-4
    batch_size:      int   = 512
    max_epochs:      int   = 100
    early_stop_patience: int = 10
    grad_clip:       float = 1.0
    device:          str  = "auto"   # auto | cuda | mps | cpu

    # ── Hyperparameter search space (tuned on first fold only) ────────────────
    search_space: dict = field(default_factory=lambda: {
        "d_model":         [32, 64],
        "nhead":           [2, 4],
        "num_layers":      [1, 2],
        "dim_feedforward": [128, 256],
        "dropout":         [0.1, 0.2],
        "lr":              [1e-4, 3e-4],
        "batch_size":      [512, 1024],
        "seq_len":         [12, 24],
    })

    # ── Runtime flags ─────────────────────────────────────────────────────────
    tune_hparams:  bool = False   # run hparam search on first fold
    verbose:       bool = False
    seed:          int  = 42
    n_workers:     int  = -1      # -1 = all CPU cores

    # ── Evaluation / baselines ────────────────────────────────────────────────
    ridge_ic:   float = 0.0344
    ridge_icir: float = 0.2504
    lgbm_icir:  float = 0.7347
    rf_icir:    float = 0.7248


# Singleton default config used when no CLI args override it
DEFAULT_CONFIG = TransformerConfig()
