"""
equity_pipeline/config.py
=========================
Single source of truth for ALL pipeline settings.
No magic numbers anywhere else — everything references cfg.*.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    # ── Paths ─────────────────────────────────────────────────────────────────
    data_path:   str = "project_database.csv"
    results_dir: str = "results"

    # ── Data identifiers ──────────────────────────────────────────────────────
    id_col:     str = "co_code"
    date_col:   str = "Month"
    target_col: str = "monthly_gross_return"   # raw; fwd_return constructed from this

    # Canonical feature set — lag_mv permanently excluded (VIF=204)
    features: tuple = (
        "BM_sep", "OpProf", "Inv", "Momentum", "lag_ret", "mktcap"
    )
    log_cols: tuple = ("mktcap",)   # log-transformed before winsorization

    # ── Preprocessing ─────────────────────────────────────────────────────────
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99

    # ── Walk-forward ──────────────────────────────────────────────────────────
    train_window: int = 60    # ALL models — including CART (fix from 1-month)
    val_months:   int = 6     # last N months of training window for validation IC
    min_obs:      int = 30    # min stocks per month to include fold

    # ── Sequence models (Transformer, CNN) ────────────────────────────────────
    seq_len: int = 24

    # ── Training (neural models) ──────────────────────────────────────────────
    max_epochs:           int   = 100
    early_stop_patience:  int   = 10
    batch_size:           int   = 512
    lr:                   float = 3e-4
    grad_clip:            float = 1.0
    weight_decay:         float = 1e-4

    # ── OOF ───────────────────────────────────────────────────────────────────
    n_oof_folds: int = 5

    # ── Runtime ───────────────────────────────────────────────────────────────
    seed:    int  = 42
    n_jobs:  int  = -1
    device:  str  = "auto"
    verbose: bool = False

    # ── Portfolio construction (backtest + RL) ────────────────────────────────
    long_pct:              float = 0.10   # top decile long
    short_pct:             float = 0.10   # bottom decile short
    rebalance_freq:        str   = "monthly"
    transaction_cost_bps:  float = 10.0   # one-way, basis points

    # ── Baselines (for reporting) ─────────────────────────────────────────────
    baselines: dict = field(default_factory=lambda: {
        "Ridge":        {"mean_ic": 0.0344, "icir": 0.2504},
        "CART":         {"mean_ic": 0.0304, "icir": 0.2561},
        "LightGBM":     {"mean_ic": 0.0551, "icir": 0.7347},
        "RandomForest": {"mean_ic": 0.0546, "icir": 0.7248},
    })


# Module-level singleton — import and use anywhere
DEFAULT_CONFIG = PipelineConfig()
