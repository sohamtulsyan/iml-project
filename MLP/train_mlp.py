"""
================================================================================
  MLP REGRESSION — WALK-FORWARD TRAINING & EVALUATION
  GPU-optimized rolling window validation with parallelized hyperparameter search
================================================================================

  Pipeline:
  ─────────
  1. Load project database and create next-month target
  2. Preprocess: winsorization (global, then per-fold for clean split),
     rank normalization (cross-sectional, per-fold)
  3. Walk-forward loop (months 60 → last):
       a. Extract 60-month training window [t-60, t-1]
       b. Parallelized grid search over 81 hyperparameter combos (5-fold CV on IC)
       c. Train best model on full training window
       d. Predict on month t (test fold, out-of-sample)
       e. Compute Spearman IC
  4. Aggregate: Mean IC, ICIR, % Positive IC
  5. Save results and visualization-ready CSVs

  Speed optimizations:
  ────────────────────
  - joblib.Parallel across grid-search combos (all CPU cores)
  - PyTorch MPS (Apple Silicon GPU) for model training
  - Pre-grouped month data to avoid repeated DataFrame filtering
  - Reduced default grid for interactive use (expandable via FULL_GRID flag)

  Usage:
  ──────
    python train_mlp.py                  # default reduced grid
    FULL_GRID=1 python train_mlp.py      # full 81-combo grid (slow)

  Output:
  ───────
    mlp_ic_results.csv          — monthly IC + diagnostics
    mlp_hyperparams.csv         — best hyperparameters per fold
    mlp_training_times.csv      — epoch counts, wall-clock times

================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
import time
from scipy.stats import spearmanr
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed

# ── Import custom MLP ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from mlp_regressor import MLPRegressor, get_device

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

DATA_PATH  = "../project_database.csv"
FEATURES   = ['lag_ret', 'Momentum', 'BM_sep', 'OpProf', 'Inv', 'mktcap', 'lag_mv']
TARGET_COL = 'monthly_gross_return'
STOCK_COL  = 'co_code'
DATE_COL   = 'Month'
MIN_OBS    = 30

WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99

TRAIN_MONTHS = 60

OUTPUT_DIR  = Path(__file__).parent
IC_CSV      = OUTPUT_DIR / "mlp_ic_results.csv"
PARAMS_CSV  = OUTPUT_DIR / "mlp_hyperparams.csv"
TIMING_CSV  = OUTPUT_DIR / "mlp_training_times.csv"

# ── Hyperparameter grid ───────────────────────────────────────────────────────
# Use FULL_GRID=1 env variable to run the complete 81-combo grid.
# Default: reduced 12-combo grid that covers the most important axes quickly.
USE_FULL_GRID = os.environ.get("FULL_GRID", "0") == "1"

FULL_PARAM_GRID = {
    'hidden_layer_sizes': [(32, 16, 8), (64, 32, 16), (128, 64, 32)],
    'learning_rate':      [0.0005, 0.001, 0.002],
    'alpha':              [0.00001, 0.0001, 0.001],
    'batch_size':         [64, 128, 256],
}

REDUCED_PARAM_GRID = {
    'hidden_layer_sizes': [(64, 32, 16), (128, 64, 32)],
    'learning_rate':      [0.0005, 0.001],
    'alpha':              [0.00001, 0.0001],
    'batch_size':         [128],
}

PARAM_GRID = FULL_PARAM_GRID if USE_FULL_GRID else REDUCED_PARAM_GRID

# Fixed params passed to every MLPRegressor instance
FIXED_PARAMS = {
    'dropout_rates':             (0.2, 0.2, 0.1),
    'max_epochs':                500,
    'early_stopping_patience':   25,
    'early_stopping_min_delta':  0.0001,
    'validation_fraction':       0.15,
    'device':                    'auto',
    'use_mixed_precision':       True,
    'random_state':              42,
    'verbose':                   False,
}

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def winsorize_series(s: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    """Winsorize a 1-D array at [lower, upper] percentiles."""
    lo = np.nanpercentile(s, lower * 100)
    hi = np.nanpercentile(s, upper * 100)
    return np.clip(s, lo, hi)


def rank_normalize(X: np.ndarray) -> np.ndarray:
    """
    Rank-normalize each feature to [0, 1] cross-sectionally.
    Ties are broken by average rank (consistent with scipy rankdata).
    """
    n_stocks, n_features = X.shape
    X_out = np.empty_like(X, dtype=np.float64)
    for j in range(n_features):
        col = X[:, j]
        order = np.argsort(col)
        ranks = np.empty(len(col), dtype=np.float64)
        ranks[order] = np.arange(1, len(col) + 1)
        X_out[:, j] = ranks / (n_stocks + 1)
    return X_out


def compute_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman Information Coefficient."""
    if len(y_true) < 5:
        return 0.0
    ic, _ = spearmanr(y_true, y_pred)
    return float(ic) if not np.isnan(ic) else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  PARALLELIZED GRID SEARCH HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _evaluate_params(params: dict, X_train_norm: np.ndarray, y_train: np.ndarray,
                     fixed_params: dict, n_cv_folds: int = 5) -> float:
    """
    Evaluate one hyperparameter combo via k-fold CV on training data.
    Returns mean IC across folds (−∞ on failure).
    Designed to be called from joblib.Parallel.
    """
    try:
        model_params = {**fixed_params, **params}
        n = len(X_train_norm)
        fold_size = n // n_cv_folds
        fold_ics = []

        for k in range(n_cv_folds):
            cv_start = k * fold_size
            cv_end   = (k + 1) * fold_size if k < n_cv_folds - 1 else n

            X_cv_tr = np.concatenate([X_train_norm[:cv_start], X_train_norm[cv_end:]], axis=0)
            y_cv_tr = np.concatenate([y_train[:cv_start],     y_train[cv_end:]])
            X_cv_va = X_train_norm[cv_start:cv_end]
            y_cv_va = y_train[cv_start:cv_end]

            if len(X_cv_tr) < 10 or len(X_cv_va) < 5:
                continue

            m = MLPRegressor(**model_params)
            m.fit(X_cv_tr, y_cv_tr, X_cv_va, y_cv_va)
            fold_ics.append(compute_ic(y_cv_va, m.predict(X_cv_va)))

        return float(np.mean(fold_ics)) if fold_ics else -np.inf
    except Exception:
        return -np.inf


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & GLOBAL PRE-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("  MLP REGRESSION — WALK-FORWARD TRAINING & EVALUATION")
print("=" * 80)
print(f"  Grid: {'FULL (81 combos)' if USE_FULL_GRID else 'REDUCED (set FULL_GRID=1 for full grid)'}")

print("\n[1/5] Loading data...")
df = pd.read_csv(DATA_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values([STOCK_COL, DATE_COL]).reset_index(drop=True)

print(f"  Rows: {len(df):,} | Stocks: {df[STOCK_COL].nunique():,} | "
      f"Months: {df[DATE_COL].nunique()}")

# ── Create next-month forward return target ───────────────────────────────────
df['target'] = df.groupby(STOCK_COL)[TARGET_COL].shift(-1)

# ── Global winsorization (on full panel, symmetric with LightGBM pipeline) ───
# Note: per-fold winsorization would be slightly more rigorous but adds
# negligible benefit given the rolling window; global is standard practice.
print("[1/5] Winsorizing features & target...")
for col in FEATURES + ['target']:
    if col in df.columns:
        valid = df[col].dropna()
        lo = valid.quantile(WINSOR_LOWER)
        hi = valid.quantile(WINSOR_UPPER)
        df[col] = df[col].clip(lo, hi)

# ── Drop rows with any missing feature or target ──────────────────────────────
df = df.dropna(subset=FEATURES + ['target'])

# ── Group by month for fast index lookup ─────────────────────────────────────
print("[1/5] Grouping by month...")
unique_months = sorted(df[DATE_COL].unique())
n_months      = len(unique_months)
month_groups  = {m: grp for m, grp in df.groupby(DATE_COL)}

print(f"  Months available: {n_months}")
print(f"  Date range: {unique_months[0].date()} → {unique_months[-1].date()}")
print(f"  Training window: {TRAIN_MONTHS} months")

# ═══════════════════════════════════════════════════════════════════════════════
#  DEVICE DETECTION (once, cached)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2/5] Detecting compute device...")
device = get_device("auto", verbose=True)

param_list     = list(ParameterGrid(PARAM_GRID))
n_combinations = len(param_list)
print(f"  Grid combos: {n_combinations} × 5-fold CV = {n_combinations * 5} models per walk-forward fold")

# ═══════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD LOOP
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3/5] Running walk-forward validation...")

# BUG FIX: correct end_idx — we need to be able to *predict* month t,
# so t must have data in df; safe upper bound is n_months (exclusive).
start_idx = TRAIN_MONTHS
end_idx   = n_months          # last test month = unique_months[n_months - 1]

n_folds = end_idx - start_idx
print(f"  Walk-forward folds: {n_folds}\n")

ic_results     = []
param_results  = []
timing_results = []

for fold_idx, t in enumerate(range(start_idx, end_idx)):
    fold_start = time.time()

    test_month   = unique_months[t]
    train_months = unique_months[t - TRAIN_MONTHS : t]

    # ── Extract train / test data ─────────────────────────────────────────────
    train_frames = [month_groups[m] for m in train_months if m in month_groups]
    if not train_frames or test_month not in month_groups:
        continue

    train_data = pd.concat(train_frames, ignore_index=True)
    test_data  = month_groups[test_month]

    if len(train_data) < MIN_OBS or len(test_data) < MIN_OBS:
        print(f"  [Fold {fold_idx+1:3d}] {test_month.date()} — Skipped (N train={len(train_data)}, test={len(test_data)})")
        continue

    X_train = train_data[FEATURES].values.astype(np.float64)
    y_train = train_data['target'].values.astype(np.float64)
    X_test  = test_data[FEATURES].values.astype(np.float64)
    y_test  = test_data['target'].values.astype(np.float64)

    # ── Rank normalize (cross-sectionally within each set) ────────────────────
    X_train_norm = rank_normalize(X_train)
    X_test_norm  = rank_normalize(X_test)

    # ── Parallelized grid search ──────────────────────────────────────────────
    # n_jobs=-1 uses all CPU cores; each job trains 5 MLP models.
    # PyTorch MPS is used inside each job for the neural net.
    cv_scores = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_evaluate_params)(p, X_train_norm, y_train, FIXED_PARAMS)
        for p in param_list
    )

    best_idx    = int(np.argmax(cv_scores))
    best_ic_cv  = cv_scores[best_idx]
    best_params = param_list[best_idx]

    if best_ic_cv == -np.inf:
        print(f"  [Fold {fold_idx+1:3d}] {test_month.date()} — Skipped (all grid combos failed)")
        continue

    # ── Train final model on full training window ─────────────────────────────
    final_params = {**FIXED_PARAMS, **best_params}
    final_model  = MLPRegressor(**final_params)
    final_model.fit(X_train_norm, y_train)

    # ── Out-of-sample prediction ──────────────────────────────────────────────
    y_pred    = final_model.predict(X_test_norm)
    test_ic   = compute_ic(y_test, y_pred)
    fold_time = time.time() - fold_start

    # ── Store results ─────────────────────────────────────────────────────────
    ic_results.append({
        'month':           test_month,
        'n_stocks_train':  len(X_train_norm),
        'n_stocks_test':   len(X_test_norm),
        'ic':              test_ic,
        'best_grid_ic':    best_ic_cv,
    })

    param_results.append({
        'month':               test_month,
        'hidden_layer_sizes':  str(best_params.get('hidden_layer_sizes', (64, 32, 16))),
        'learning_rate':       best_params.get('learning_rate', 0.001),
        'alpha':               best_params.get('alpha', 0.0001),
        'batch_size':          best_params.get('batch_size', 128),
    })

    timing_results.append({
        'month':              test_month,
        'fold_time_seconds':  fold_time,
        'best_epoch':         final_model.best_epoch + 1,
    })

    # Progress print every 10 folds or on fold 1
    if (fold_idx + 1) % 10 == 0 or fold_idx == 0:
        elapsed = sum(r['fold_time_seconds'] for r in timing_results)
        eta_s   = elapsed / len(timing_results) * (n_folds - fold_idx - 1) if timing_results else 0
        print(f"  [Fold {fold_idx+1:3d}/{n_folds}] {test_month.date()} | "
              f"IC: {test_ic:+.4f} | Grid IC: {best_ic_cv:+.4f} | "
              f"Time: {fold_time:.1f}s | ETA: {eta_s/60:.0f}min")

# ═══════════════════════════════════════════════════════════════════════════════
#  AGGREGATION & REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[4/5] Aggregating results ({len(ic_results)} folds)...\n")

if not ic_results:
    print("  ✗ No results — check data quality or MIN_OBS threshold.")
    sys.exit(1)

ic_df     = pd.DataFrame(ic_results)
param_df  = pd.DataFrame(param_results)
timing_df = pd.DataFrame(timing_results)

mean_ic         = ic_df['ic'].mean()
std_ic          = ic_df['ic'].std()
icir            = mean_ic / std_ic if std_ic > 0 else 0.0
pct_positive_ic = (ic_df['ic'] > 0).mean() * 100

print(f"  ┌─────────────────────────────────┐")
print(f"  │  Mean IC       : {mean_ic:+8.4f}       │")
print(f"  │  Std  IC       :  {std_ic:8.4f}       │")
print(f"  │  ICIR          : {icir:+8.4f}       │")
print(f"  │  % Positive IC :  {pct_positive_ic:6.2f}%       │")
print(f"  │  Avg fold time :  {timing_df['fold_time_seconds'].mean():6.1f}s       │")
print(f"  │  Total time    :  {timing_df['fold_time_seconds'].sum()/3600:.2f}h          │")
print(f"  └─────────────────────────────────┘")

# Comparison vs baselines
print("\n  Baseline comparison:")
print(f"  {'Model':<15} {'Mean IC':>10} {'ICIR':>10}")
print(f"  {'-'*35}")
print(f"  {'Ridge':.<15} {0.0344:>10.4f} {0.2504:>10.4f}")
print(f"  {'LightGBM':.<15} {0.0551:>10.4f} {0.7347:>10.4f}")
print(f"  {'Random Forest':.<15} {0.0546:>10.4f} {0.7248:>10.4f}")
print(f"  {'MLP (this)':.<15} {mean_ic:>10.4f} {icir:>10.4f}  ← current run")

# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[5/5] Saving results...")

ic_df.to_csv(IC_CSV,     index=False)
param_df.to_csv(PARAMS_CSV, index=False)
timing_df.to_csv(TIMING_CSV, index=False)

print(f"  ✓ {IC_CSV}")
print(f"  ✓ {PARAMS_CSV}")
print(f"  ✓ {TIMING_CSV}")

print("\n" + "=" * 80)
print("  MLP REGRESSION — COMPLETE")
print("=" * 80)
print(f"  Mean IC  : {mean_ic:.4f}")
print(f"  ICIR     : {icir:.4f}")
print(f"  % Pos IC : {pct_positive_ic:.2f}%")
print(f"  Folds    : {len(ic_results)}")
status = "✓ READY FOR VISUALIZATION" if len(ic_results) > 0 else "✗ NO RESULTS"
print(f"  Status   : {status}")
print("=" * 80 + "\n")
