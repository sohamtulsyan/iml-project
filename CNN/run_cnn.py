"""
================================================================================
  CNN PIPELINE — MAIN ENTRY POINT
================================================================================

  Usage:
  ──────
    # Default (reduced grid, no hparam search):
    python run_cnn.py

    # With hyperparameter tuning:
    python run_cnn.py --tune_hparams --verbose

    # Full custom:
    python run_cnn.py \\
      --data_path    ../project_database.csv \\
      --output_dir   results \\
      --seq_len      24 \\
      --n_filters    32 \\
      --n_epochs     100 \\
      --seed         42 \\
      --tune_hparams \\
      --verbose

================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, random, time, json, argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# ── Package path setup ────────────────────────────────────────────────────────
CNN_DIR         = Path(__file__).parent
TRANSFORMER_DIR = CNN_DIR.parent / "CNN"
sys.path.insert(0, str(CNN_DIR))
sys.path.insert(0, str(TRANSFORMER_DIR))

from config       import CNNConfig
from model        import get_device
from data     import load_data      # shared from Transformer
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
        "torch": "2.0.0", "numpy": "1.24.0", "pandas": "2.0.0",
        "scikit-learn": "1.3.0", "scipy": "1.10.0", "joblib": "1.3.0",
        "pyarrow": "12.0.0", "statsmodels": "0.14.0",
    }
    for pkg, min_ver in required.items():
        try:
            installed = meta.version(pkg)
            if Version(installed) < Version(min_ver):
                print(f"  ⚠ {pkg} {installed} < {min_ver}")
        except meta.PackageNotFoundError:
            print(f"  ✗ {pkg} not installed")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-Scale Temporal CNN — Equity Return Prediction"
    )
    p.add_argument("--data_path",    default="../project_database.csv")
    p.add_argument("--output_dir",   default="results")
    p.add_argument("--seq_len",      type=int,   default=24)
    p.add_argument("--train_window", type=int,   default=60)
    p.add_argument("--val_months",   type=int,   default=6)
    p.add_argument("--n_filters",    type=int,   default=32)
    p.add_argument("--n_epochs",     type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=512)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--device",       default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--n_workers",    type=int,   default=-1)
    p.add_argument("--tune_hparams", action="store_true")
    p.add_argument("--verbose",      action="store_true")
    p.add_argument("--skip_dep_check", action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 70)
    print("  MULTI-SCALE TEMPORAL CNN — EQUITY RETURN PREDICTION")
    print("=" * 70)
    print(f"  data_path    : {args.data_path}")
    print(f"  output_dir   : {args.output_dir}")
    print(f"  seq_len      : {args.seq_len}")
    print(f"  n_filters    : {args.n_filters}")
    print(f"  tune_hparams : {args.tune_hparams}")
    print(f"  device       : {args.device}")
    print(f"  seed         : {args.seed}")

    set_seed(args.seed)

    if not args.skip_dep_check:
        print("\n[Check] Verifying dependencies...")
        check_dependencies()

    print("\n[Device] Detecting hardware...")
    device = get_device(args.device)

    cfg = CNNConfig(
        data_path        = args.data_path,
        output_dir       = args.output_dir,
        seq_len          = args.seq_len,
        train_window     = args.train_window,
        val_months       = args.val_months,
        n_filters        = args.n_filters,
        max_epochs       = args.n_epochs,
        batch_size       = args.batch_size,
        lr               = args.lr,
        dropout          = args.dropout,
        device           = args.device,
        seed             = args.seed,
        n_workers        = args.n_workers,
        tune_hparams     = args.tune_hparams,
        verbose          = args.verbose,
    )

    output_dir = Path(args.output_dir)

    print("\n[Data] Loading...")
    df = load_data(
        data_path  = args.data_path,
        id_col     = cfg.id_col,
        date_col   = cfg.date_col,
        features   = cfg.features,
        target_col = cfg.target_col,
    )

    t_total = time.time()
    summary = walk_forward(
        df           = df,
        cfg          = cfg,
        device       = device,
        output_dir   = output_dir,
        tune_hparams = args.tune_hparams,
    )

    elapsed = (time.time() - t_total) / 3600
    print(f"\n[Done] Total wall time: {elapsed:.2f}h")
    print(f"[Done] Results in: {output_dir.resolve()}/")


if __name__ == "__main__":
    main()
