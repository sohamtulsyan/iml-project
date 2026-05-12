"""
================================================================================
  TRANSFORMER PIPELINE — MAIN ENTRY POINT
  Cross-Sectional Equity Return Prediction via Temporal Transformer Encoder

  Usage:
  ──────
    # Defaults (no hparam search, reduced grid):
    python run_transformer.py

    # With hyperparameter tuning on first fold:
    python run_transformer.py --tune_hparams

    # Full custom run:
    python run_transformer.py \\
      --data_path ../project_database.csv \\
      --output_dir results \\
      --seq_len 24 \\
      --train_window 60 \\
      --n_epochs 100 \\
      --seed 42 \\
      --tune_hparams \\
      --verbose

================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import argparse
import random
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# ── Ensure the Transformer package is importable ─────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config       import TransformerConfig
from model        import get_device
from data         import load_data
from walk_forward import walk_forward


# ─────────────────────────────────────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────────────────────
#  Dependency version check
# ─────────────────────────────────────────────────────────────────────────────

def check_dependencies():
    import importlib.metadata as meta
    from packaging.version import Version

    required = {
        "torch":        "2.0.0",
        "numpy":        "1.24.0",
        "pandas":       "2.0.0",
        "scikit-learn": "1.3.0",
        "scipy":        "1.10.0",
        "joblib":       "1.3.0",
        "pyarrow":      "12.0.0",
        "statsmodels":  "0.14.0",
    }
    ok = True
    for pkg, min_ver in required.items():
        try:
            installed = meta.version(pkg)
            if Version(installed) < Version(min_ver):
                print(f"  ⚠ {pkg} {installed} < {min_ver} (required)")
                ok = False
        except meta.PackageNotFoundError:
            print(f"  ✗ {pkg} not installed")
            ok = False
    return ok


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Temporal Transformer Encoder — Equity Return Prediction"
    )
    p.add_argument("--data_path",    default="../project_database.csv")
    p.add_argument("--output_dir",   default="results")
    p.add_argument("--seq_len",      type=int,   default=24)
    p.add_argument("--train_window", type=int,   default=60)
    p.add_argument("--val_months",   type=int,   default=6)
    p.add_argument("--n_epochs",     type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=512)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--d_model",      type=int,   default=64)
    p.add_argument("--nhead",        type=int,   default=4)
    p.add_argument("--num_layers",   type=int,   default=2)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--device",       default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--n_workers",    type=int,   default=-1)
    p.add_argument("--tune_hparams", action="store_true",
                   help="Run hyperparameter search on first fold")
    p.add_argument("--verbose",      action="store_true",
                   help="Print epoch-level training logs")
    p.add_argument("--skip_dep_check", action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 70)
    print("  TEMPORAL TRANSFORMER — EQUITY RETURN PREDICTION")
    print("=" * 70)
    print(f"  data_path    : {args.data_path}")
    print(f"  output_dir   : {args.output_dir}")
    print(f"  seq_len      : {args.seq_len}")
    print(f"  train_window : {args.train_window}")
    print(f"  tune_hparams : {args.tune_hparams}")
    print(f"  device       : {args.device}")
    print(f"  seed         : {args.seed}")

    # ── Seed ─────────────────────────────────────────────────────────────────
    set_seed(args.seed)

    # ── Dependency check ──────────────────────────────────────────────────────
    if not args.skip_dep_check:
        print("\n[Check] Verifying dependencies...")
        check_dependencies()

    # ── Device ───────────────────────────────────────────────────────────────
    print("\n[Device] Detecting hardware...")
    device = get_device(args.device)

    # ── Build config ──────────────────────────────────────────────────────────
    cfg = TransformerConfig(
        data_path       = args.data_path,
        output_dir      = args.output_dir,
        seq_len         = args.seq_len,
        train_window    = args.train_window,
        val_months      = args.val_months,
        max_epochs      = args.n_epochs,
        batch_size      = args.batch_size,
        lr              = args.lr,
        d_model         = args.d_model,
        nhead           = args.nhead,
        num_layers      = args.num_layers,
        dropout         = args.dropout,
        device          = args.device,
        seed            = args.seed,
        n_workers       = args.n_workers,
        tune_hparams    = args.tune_hparams,
        verbose         = args.verbose,
    )

    output_dir = Path(args.output_dir)

    # ── Load & preprocess data ────────────────────────────────────────────────
    print("\n[Data] Loading...")
    df = load_data(
        data_path  = args.data_path,
        id_col     = cfg.id_col,
        date_col   = cfg.date_col,
        features   = cfg.features,
        target_col = cfg.target_col,
    )

    # ── Walk-forward ──────────────────────────────────────────────────────────
    t_total = time.time()

    summary = walk_forward(
        df           = df,
        cfg          = cfg,
        device       = device,
        output_dir   = output_dir,
        tune_hparams = args.tune_hparams,
    )

    elapsed = (time.time() - t_total) / 3600
    print(f"\n[Done] Total wall time: {elapsed:.2f} hours")
    print(f"[Done] Results in: {output_dir.resolve()}/")


if __name__ == "__main__":
    main()
