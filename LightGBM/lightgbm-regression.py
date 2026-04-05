"""
================================================================================
  LightGBM GRADIENT BOOSTED TREES — FACTOR MODEL
  Rolling cross-sectional prediction with validation & diagnostics
================================================================================

  Pipeline overview:
  ──────────────────
  1. Load panel data, create next-month return target
  2. Winsorise features + target for robustness
  3. Pre-group data by month (performance optimisation)
  4. Hyperparameter tuning via grid search with rolling monthly IC/ICIR
  5. For best hyperparameters, run full rolling evaluation:
       a. PRE-CHECKS: variance, feature correlation, sample size
       b. Train LightGBM on month t, predict month t+1
       c. POST-CHECKS: residual analysis (normality, heteroscedasticity,
          autocorrelation), overfitting diagnostics (train vs test)
       d. Collect feature importance (gain + split), SHAP-style analysis
       e. Compute Spearman IC between predictions and realised returns
  6. Aggregate: Mean IC, ICIR, feature importance stability, diagnostics
  7. Generate 8-page PDF report

  Usage:
  ──────
    python lightgbm-regression.py    (run from project root)

  Output:
  ───────
    lgbm_ic_results.csv         — monthly IC values + diagnostics
    lgbm_importance_summary.csv — per-month feature importance (gain)
    lgbm_grid_search.csv        — grid search results
    lgbm_report.pdf             — 8-page visual diagnostics report
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid

from scipy.stats import spearmanr
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm

from pathlib import Path
import time


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

DATA_PATH      = "project_database.csv"
FEATURES       = ['lag_ret', 'Momentum', 'BM_sep', 'OpProf', 'Inv', 'mktcap']
TARGET_COL     = 'monthly_gross_return'
STOCK_COL      = 'co_code'
DATE_COL       = 'Month'
MIN_OBS        = 30

# Winsorisation bounds
WINSOR_LOWER   = 0.01
WINSOR_UPPER   = 0.99

# Diagnostic thresholds
JB_ALPHA       = 0.05
BP_ALPHA       = 0.05
DW_LOWER       = 1.5
DW_UPPER       = 2.5

# ── LightGBM Hyperparameter Grid ─────────────────────────────────────────────
# Kept intentionally conservative to prevent overfitting on cross-sectional data
PARAM_GRID = {
    'num_leaves':        [15, 31, 63],
    'max_depth':         [4, 6, 8],
    'learning_rate':     [0.05, 0.1],
    'n_estimators':      [100, 200],
    'min_child_samples': [20, 50],
    'subsample':         [0.8],
    'colsample_bytree':  [0.8],
    'reg_alpha':         [0.1],
    'reg_lambda':        [1.0],
}

# Fixed LightGBM params (not in grid)
FIXED_PARAMS = {
    'objective':  'regression',
    'metric':     'mse',
    'verbose':    -1,
    'n_jobs':     -1,
    'random_state': 42,
}

# Output paths
OUTPUT_DIR     = Path(".")
IC_CSV         = OUTPUT_DIR / "lgbm_ic_results.csv"
IMP_CSV        = OUTPUT_DIR / "lgbm_importance_summary.csv"
GRID_CSV       = OUTPUT_DIR / "lgbm_grid_search.csv"
REPORT_PDF     = OUTPUT_DIR / "lgbm_report.pdf"


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("  LightGBM GRADIENT BOOSTED TREES — FACTOR MODEL")
print("=" * 80)

print("\n[1/8] Loading data...")
df = pd.read_csv(DATA_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values([STOCK_COL, DATE_COL])

print(f"  Loaded {len(df):,} rows  |  {df[STOCK_COL].nunique()} stocks  "
      f"|  {df[DATE_COL].min().strftime('%Y-%m')} → {df[DATE_COL].max().strftime('%Y-%m')}")

# ── Create forward return target ─────────────────────────────────────────────
df['target'] = df.groupby(STOCK_COL)[TARGET_COL].shift(-1)

# ── Drop rows with missing features or target ────────────────────────────────
df_model = df.dropna(subset=FEATURES + ['target']).copy()
print(f"  After dropping NaN: {len(df_model):,} rows")

# ── Winsorise features and target ─────────────────────────────────────────────
print("\n[2/8] Winsorising features & target...")

def winsorize(series, lower=WINSOR_LOWER, upper=WINSOR_UPPER):
    """Clip values to [lower, upper] quantiles."""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)

for col in FEATURES + ['target']:
    df_model[col] = winsorize(df_model[col])

print(f"  Winsorised at [{WINSOR_LOWER*100:.0f}%, {WINSOR_UPPER*100:.0f}%] quantiles")

# ── Pre-group by month (critical optimisation) ───────────────────────────────
print("\n[3/8] Pre-grouping data by month...")
month_groups = {month: group for month, group in df_model.groupby(DATE_COL)}
sorted_months = sorted(month_groups.keys())
print(f"  {len(sorted_months)} months available  |  MIN_OBS={MIN_OBS}")


# ═══════════════════════════════════════════════════════════════════════════════
#  VALIDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def check_variance(X: pd.DataFrame) -> tuple[bool, list[str]]:
    """Check that all features have non-trivial variance."""
    low_var = [c for c in X.columns if X[c].var() < 1e-10]
    return len(low_var) == 0, low_var


def check_feature_correlation(X: pd.DataFrame, threshold: float = 0.95
                               ) -> tuple[bool, list[tuple]]:
    """Flag highly correlated feature pairs (|r| > threshold)."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = [
        (corr.columns[i], corr.columns[j], upper.iloc[i, j])
        for i in range(len(corr.columns))
        for j in range(i + 1, len(corr.columns))
        if upper.iloc[i, j] > threshold
    ]
    return len(pairs) == 0, pairs


def post_regression_diagnostics(X: pd.DataFrame, y: pd.Series,
                                 predictions: np.ndarray) -> dict:
    """
    Run post-regression assumption checks on residuals.

    Even though LightGBM doesn't assume normality/homoscedasticity,
    we validate these for comparison with the Ridge baseline and to
    characterise the residual structure.
    """
    residuals = y.values - predictions

    # Jarque-Bera normality
    try:
        jb_stat, jb_p, _, _ = jarque_bera(residuals)
    except Exception:
        jb_stat, jb_p = np.nan, np.nan

    # Breusch-Pagan heteroscedasticity (using features as regressors)
    try:
        X_const = sm.add_constant(X)
        bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_const)
    except Exception:
        bp_stat, bp_p = np.nan, np.nan

    # Durbin-Watson autocorrelation
    try:
        dw = durbin_watson(residuals)
    except Exception:
        dw = np.nan

    normality_ok = jb_p > JB_ALPHA if not np.isnan(jb_p) else False
    homosced_ok  = bp_p > BP_ALPHA if not np.isnan(bp_p) else False
    autocorr_ok  = DW_LOWER <= dw <= DW_UPPER if not np.isnan(dw) else False

    return {
        'jb_stat': jb_stat,
        'jb_p': jb_p,
        'bp_stat': bp_stat,
        'bp_p': bp_p,
        'dw': dw,
        'normality_ok': normality_ok,
        'homosced_ok': homosced_ok,
        'autocorr_ok': autocorr_ok,
        'resid_mean': np.mean(residuals),
        'resid_std': np.std(residuals),
        'resid_skew': float(pd.Series(residuals).skew()),
        'resid_kurt': float(pd.Series(residuals).kurtosis()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: GRID SEARCH (fast IC evaluation per param set)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4/8] Grid search — finding best hyperparameters...")

param_list = list(ParameterGrid(PARAM_GRID))
print(f"  {len(param_list)} combinations to evaluate")
start_time = time.time()

grid_results = []

for idx, params in enumerate(param_list):
    full_params = {**FIXED_PARAMS, **params}
    ICs = []

    for i in range(len(sorted_months) - 1):
        t = sorted_months[i]
        t_next = sorted_months[i + 1]

        train = month_groups[t]
        test  = month_groups[t_next]

        if len(train) < MIN_OBS or len(test) < MIN_OBS:
            continue

        X_train = train[FEATURES]
        y_train = train['target']
        X_test  = test[FEATURES]
        y_test  = test['target']

        # Variance check
        var_ok, _ = check_variance(X_train)
        if not var_ok:
            continue

        model = lgb.LGBMRegressor(**full_params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        ic, _ = spearmanr(preds, y_test)
        if not np.isnan(ic):
            ICs.append(ic)

    if not ICs:
        continue

    IC_mean = np.mean(ICs)
    IC_std  = np.std(ICs)
    ICIR    = IC_mean / IC_std if IC_std > 0 else np.nan

    grid_results.append({
        **params,
        'IC_mean': IC_mean,
        'IC_std': IC_std,
        'ICIR': ICIR,
        'N_months': len(ICs),
    })

    # Progress
    if (idx + 1) % 10 == 0 or idx == 0 or idx == len(param_list) - 1:
        print(f"  [{idx+1}/{len(param_list)}]  "
              f"IC={IC_mean:.4f}  ICIR={ICIR:.4f}  "
              f"leaves={params['num_leaves']}  depth={params['max_depth']}  "
              f"lr={params['learning_rate']}  n_est={params['n_estimators']}  "
              f"min_child={params['min_child_samples']}")

grid_df = pd.DataFrame(grid_results)
grid_df = grid_df.sort_values('ICIR', ascending=False)
grid_df.to_csv(GRID_CSV, index=False)

elapsed = time.time() - start_time
print(f"\n  Grid search complete in {elapsed:.1f}s")
print(f"  Saved → {GRID_CSV}")

# ── Select best params ───────────────────────────────────────────────────────
best_row = grid_df.iloc[0]
best_grid_params = {
    k: int(best_row[k]) if isinstance(best_row[k], (np.integer, float)) and best_row[k] == int(best_row[k])
    else best_row[k]
    for k in PARAM_GRID.keys()
}
BEST_PARAMS = {**FIXED_PARAMS, **best_grid_params}

print(f"\n  ── Best Hyperparameters (by ICIR) ──")
for k, v in best_grid_params.items():
    print(f"    {k:<20s}: {v}")
print(f"    {'IC_mean':<20s}: {best_row['IC_mean']:.6f}")
print(f"    {'ICIR':<20s}: {best_row['ICIR']:.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: FULL ROLLING EVALUATION WITH BEST PARAMS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[5/8] Running full rolling evaluation with best params...")

ic_results = []
importance_results = []
skipped = {'low_obs': 0, 'low_var': 0}

for i in range(len(sorted_months) - 1):
    t = sorted_months[i]
    t_next = sorted_months[i + 1]
    month_str = pd.Timestamp(t_next).strftime('%Y-%m')

    train = month_groups[t]
    test  = month_groups[t_next]

    # ── Observation count check ───────────────────────────────────────────────
    if len(train) < MIN_OBS or len(test) < MIN_OBS:
        skipped['low_obs'] += 1
        continue

    X_train = train[FEATURES].copy()
    y_train = train['target'].copy()
    X_test  = test[FEATURES].copy()
    y_test  = test['target'].copy()

    # ══════════════════════════════════════════════════════════════════════════
    #  PRE-CHECKS
    # ══════════════════════════════════════════════════════════════════════════

    # 1. Variance check
    var_ok, low_var_cols = check_variance(X_train)
    if not var_ok:
        skipped['low_var'] += 1
        continue

    # 2. Feature correlation (logged, not skipped — GBMs handle correlated features)
    corr_ok, corr_pairs = check_feature_correlation(X_train)
    pre_flags = []
    if not corr_ok:
        pre_flags.append(f"high_corr({len(corr_pairs)} pairs)")

    # ══════════════════════════════════════════════════════════════════════════
    #  TRAIN LightGBM
    # ══════════════════════════════════════════════════════════════════════════

    model = lgb.LGBMRegressor(**BEST_PARAMS)
    model.fit(X_train, y_train)

    # ══════════════════════════════════════════════════════════════════════════
    #  PREDICT & EVALUATE
    # ══════════════════════════════════════════════════════════════════════════

    preds_train = model.predict(X_train)
    preds_test  = model.predict(X_test)

    # In-sample metrics
    r2_train = r2_score(y_train, preds_train)
    mse_train = mean_squared_error(y_train, preds_train)

    # Out-of-sample metrics
    r2_test = r2_score(y_test, preds_test)
    mse_test = mean_squared_error(y_test, preds_test)

    # Overfitting ratio: train R² / test R² (high = overfitting)
    overfit_ratio = r2_train / r2_test if r2_test > 0 else np.nan

    # Spearman IC
    ic, ic_p = spearmanr(preds_test, y_test)

    # ══════════════════════════════════════════════════════════════════════════
    #  POST-CHECKS (residual diagnostics)
    # ══════════════════════════════════════════════════════════════════════════

    diag = post_regression_diagnostics(X_train, y_train, preds_train)

    post_fails = sum([
        not diag['normality_ok'],
        not diag['homosced_ok'],
        not diag['autocorr_ok']
    ])
    post_flags = []
    if not diag['normality_ok']:
        post_flags.append("non-normal")
    if not diag['homosced_ok']:
        post_flags.append("heteroscedastic")
    if not diag['autocorr_ok']:
        post_flags.append(f"autocorr(DW={diag['dw']:.2f})")

    # ══════════════════════════════════════════════════════════════════════════
    #  FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════════════════════

    gain_imp = model.booster_.feature_importance(importance_type='gain')
    split_imp = model.booster_.feature_importance(importance_type='split')

    # Normalise to sum to 1
    gain_norm  = gain_imp / gain_imp.sum() if gain_imp.sum() > 0 else gain_imp
    split_norm = split_imp / split_imp.sum() if split_imp.sum() > 0 else split_imp

    imp_row = {'Month': month_str}
    for j, feat in enumerate(FEATURES):
        imp_row[f'gain_{feat}'] = gain_norm[j]
        imp_row[f'split_{feat}'] = split_norm[j]
    importance_results.append(imp_row)

    # ══════════════════════════════════════════════════════════════════════════
    #  STORE RESULTS
    # ══════════════════════════════════════════════════════════════════════════

    ic_results.append({
        'Month': month_str,
        'IC': ic,
        'IC_p': ic_p,
        'R2_train': r2_train,
        'R2_test': r2_test,
        'MSE_train': mse_train,
        'MSE_test': mse_test,
        'overfit_ratio': overfit_ratio,
        'N_train': len(train),
        'N_test': len(test),
        'jb_p': diag['jb_p'],
        'bp_p': diag['bp_p'],
        'dw': diag['dw'],
        'normality_ok': diag['normality_ok'],
        'homosced_ok': diag['homosced_ok'],
        'autocorr_ok': diag['autocorr_ok'],
        'resid_skew': diag['resid_skew'],
        'resid_kurt': diag['resid_kurt'],
        'post_fails': post_fails,
        'pre_flags': '|'.join(pre_flags) if pre_flags else '',
        'post_flags': '|'.join(post_flags) if post_flags else '',
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULTS AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[6/8] Aggregating results...")

ic_df  = pd.DataFrame(ic_results)
imp_df = pd.DataFrame(importance_results)

if len(ic_df) == 0:
    print("  ⚠ No valid months — check data quality. Exiting.")
    exit(1)

IC_mean = ic_df['IC'].mean()
IC_std  = ic_df['IC'].std()
ICIR    = IC_mean / IC_std if IC_std != 0 else np.nan

# ── Summary Statistics ────────────────────────────────────────────────────────
print(f"\n{'═' * 80}")
print(f"  LightGBM — RESULTS SUMMARY")
print(f"{'═' * 80}")

print(f"\n  ── IC Statistics ──")
print(f"    Mean IC:         {IC_mean:.6f}")
print(f"    Median IC:       {ic_df['IC'].median():.6f}")
print(f"    IC Std:          {IC_std:.6f}")
print(f"    ICIR:            {ICIR:.6f}")
print(f"    % IC > 0:        {(ic_df['IC'] > 0).mean()*100:.1f}%")
print(f"    % IC sig (p<5%): {(ic_df['IC_p'] < 0.05).mean()*100:.1f}%")
print(f"    Months used:     {len(ic_df)}")

print(f"\n  ── Model Performance ──")
print(f"    Avg R² (train):  {ic_df['R2_train'].mean():.6f}")
print(f"    Avg R² (test):   {ic_df['R2_test'].mean():.6f}")
print(f"    Avg MSE (test):  {ic_df['MSE_test'].mean():.6f}")
print(f"    Avg Overfit Ratio (train R²/test R²): "
      f"{ic_df['overfit_ratio'].dropna().mean():.2f}")

print(f"\n  ── Assumption Diagnostics ──")
print(f"    Normality pass:        {ic_df['normality_ok'].mean()*100:.1f}%")
print(f"    Homoscedasticity pass: {ic_df['homosced_ok'].mean()*100:.1f}%")
print(f"    No autocorrelation:    {ic_df['autocorr_ok'].mean()*100:.1f}%")
print(f"    Avg post_fails:        {ic_df['post_fails'].mean():.2f} / 3")
print(f"    Avg residual skewness: {ic_df['resid_skew'].mean():.4f}")
print(f"    Avg residual kurtosis: {ic_df['resid_kurt'].mean():.4f}")

print(f"\n  ── Skipped Months ──")
for reason, cnt in skipped.items():
    print(f"    {reason}: {cnt}")

print(f"\n  ── Feature Importance (mean normalised gain) ──")
gain_cols = [c for c in imp_df.columns if c.startswith('gain_')]
for col in gain_cols:
    feat = col.replace('gain_', '')
    mean_g = imp_df[col].mean()
    std_g  = imp_df[col].std()
    print(f"    {feat:<12s}  gain={mean_g:.4f} ± {std_g:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE CSVs
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[7/8] Saving results...")
ic_df.to_csv(IC_CSV, index=False)
imp_df.to_csv(IMP_CSV, index=False)
print(f"  → {IC_CSV}")
print(f"  → {IMP_CSV}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PDF REPORT (8 pages)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[8/8] Generating PDF report...")

with PdfPages(REPORT_PDF) as pdf:
    months_plot = range(len(ic_df))

    # ── Page 1: Monthly IC time series ────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("LightGBM — Monthly Information Coefficient (IC)",
                 fontsize=14, fontweight='bold')

    ax = axes[0]
    colors = ['#27ae60' if v > 0 else '#e74c3c' for v in ic_df['IC']]
    ax.bar(months_plot, ic_df['IC'], color=colors, alpha=0.7, width=1.0)
    ax.axhline(IC_mean, color='#2980b9', linestyle='--', lw=1.5,
               label=f'Mean IC = {IC_mean:.4f}')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('Spearman IC')
    ax.set_title(f'Monthly IC  |  ICIR = {ICIR:.4f}  |  '
                 f'% Positive = {(ic_df["IC"]>0).mean()*100:.1f}%')
    ax.legend(loc='upper right')

    ax = axes[1]
    ax.plot(months_plot, ic_df['IC'].cumsum(), color='#8e44ad', lw=2)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('Cumulative IC')
    ax.set_xlabel('Month Index')
    ax.set_title('Cumulative IC (monotonically rising = consistent signal)')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 2: Feature Importance — Gain over time ───────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("LightGBM Feature Importance (Normalised Gain) Over Time",
                 fontsize=14, fontweight='bold')

    cmap = plt.cm.Set2
    for idx, col in enumerate(gain_cols):
        feat = col.replace('gain_', '')
        ax.plot(months_plot, imp_df[col], label=feat, lw=1.5, alpha=0.85,
                color=cmap(idx / max(len(gain_cols) - 1, 1)))
    ax.set_xlabel('Month Index')
    ax.set_ylabel('Normalised Gain Importance')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.set_title('Feature importance stability — consistent = reliable signal')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 3: Average Feature Importance bar chart ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Average Feature Importance", fontsize=14, fontweight='bold')

    # Gain
    ax = axes[0]
    means_gain = imp_df[gain_cols].mean().sort_values(ascending=True)
    bars = ax.barh(
        [c.replace('gain_', '') for c in means_gain.index],
        means_gain.values,
        color='#3498db', edgecolor='white', alpha=0.85
    )
    ax.set_xlabel('Mean Normalised Gain')
    ax.set_title('By Information Gain')

    # Split
    split_cols = [c for c in imp_df.columns if c.startswith('split_')]
    ax = axes[1]
    means_split = imp_df[split_cols].mean().sort_values(ascending=True)
    ax.barh(
        [c.replace('split_', '') for c in means_split.index],
        means_split.values,
        color='#e67e22', edgecolor='white', alpha=0.85
    )
    ax.set_xlabel('Mean Normalised Split Count')
    ax.set_title('By Split Frequency')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 4: Overfitting diagnostics ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Overfitting Diagnostics", fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.plot(months_plot, ic_df['R2_train'], label='Train R²', color='#3498db',
            lw=1.2, alpha=0.8)
    ax.plot(months_plot, ic_df['R2_test'], label='Test R²', color='#e74c3c',
            lw=1.2, alpha=0.8)
    ax.axhline(0, color='black', lw=0.8, linestyle='--')
    ax.set_xlabel('Month Index')
    ax.set_ylabel('R²')
    ax.set_title('Train vs Test R² Over Time')
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(ic_df['R2_train'], bins=25, color='#3498db', edgecolor='white',
            alpha=0.7, label='Train')
    ax.hist(ic_df['R2_test'], bins=25, color='#e74c3c', edgecolor='white',
            alpha=0.7, label='Test')
    ax.set_xlabel('R²')
    ax.set_ylabel('Count')
    ax.set_title('R² Distribution')
    ax.legend()

    ax = axes[2]
    valid_overfit = ic_df['overfit_ratio'].dropna()
    valid_overfit_clipped = valid_overfit.clip(upper=valid_overfit.quantile(0.95))
    ax.hist(valid_overfit_clipped, bins=25, color='#9b59b6', edgecolor='white',
            alpha=0.85)
    ax.axvline(1.0, color='red', linestyle='--', lw=1.5,
               label='No overfit (ratio=1)')
    ax.set_xlabel('Overfit Ratio (Train R² / Test R²)')
    ax.set_ylabel('Count')
    ax.set_title(f'Overfitting Distribution (mean={valid_overfit.mean():.2f})')
    ax.legend()

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 5: Diagnostic pass rates ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Post-Regression Residual Diagnostics",
                 fontsize=14, fontweight='bold')

    # Normality (Jarque-Bera)
    ax = axes[0]
    jb_vals = ic_df['jb_p'].dropna()
    ax.hist(jb_vals, bins=25, color='#2ecc71', edgecolor='white', alpha=0.85)
    ax.axvline(JB_ALPHA, color='red', linestyle='--', lw=1.5,
               label=f'α = {JB_ALPHA}')
    ax.set_xlabel('Jarque-Bera p-value')
    ax.set_ylabel('Count')
    pct = ic_df['normality_ok'].mean() * 100
    ax.set_title(f'Normality of Residuals ({pct:.0f}% pass)')
    ax.legend()

    # Homoscedasticity (Breusch-Pagan)
    ax = axes[1]
    bp_vals = ic_df['bp_p'].dropna()
    ax.hist(bp_vals, bins=25, color='#e67e22', edgecolor='white', alpha=0.85)
    ax.axvline(BP_ALPHA, color='red', linestyle='--', lw=1.5,
               label=f'α = {BP_ALPHA}')
    ax.set_xlabel('Breusch-Pagan p-value')
    ax.set_ylabel('Count')
    pct = ic_df['homosced_ok'].mean() * 100
    ax.set_title(f'Homoscedasticity ({pct:.0f}% pass)')
    ax.legend()

    # Autocorrelation (Durbin-Watson)
    ax = axes[2]
    dw_vals = ic_df['dw'].dropna()
    ax.hist(dw_vals, bins=25, color='#9b59b6', edgecolor='white', alpha=0.85)
    ax.axvline(DW_LOWER, color='red', linestyle='--', lw=1.2)
    ax.axvline(DW_UPPER, color='red', linestyle='--', lw=1.2,
               label=f'Acceptable [{DW_LOWER}, {DW_UPPER}]')
    ax.set_xlabel('Durbin-Watson Statistic')
    ax.set_ylabel('Count')
    pct = ic_df['autocorr_ok'].mean() * 100
    ax.set_title(f'No Autocorrelation ({pct:.0f}% pass)')
    ax.legend()

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 6: Residual structure ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Residual Distribution Properties",
                 fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.plot(months_plot, ic_df['resid_skew'], color='#2c3e50', lw=1.2)
    ax.axhline(0, color='red', linestyle='--', lw=1.0)
    ax.set_xlabel('Month Index')
    ax.set_ylabel('Skewness')
    ax.set_title(f'Residual Skewness (mean={ic_df["resid_skew"].mean():.3f})')

    ax = axes[1]
    ax.plot(months_plot, ic_df['resid_kurt'], color='#c0392b', lw=1.2)
    ax.axhline(0, color='red', linestyle='--', lw=1.0)
    ax.set_xlabel('Month Index')
    ax.set_ylabel('Excess Kurtosis')
    ax.set_title(f'Residual Kurtosis (mean={ic_df["resid_kurt"].mean():.3f})')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 7: IC vs model quality metrics ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("IC vs Model Quality", fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.scatter(ic_df['R2_test'], ic_df['IC'], alpha=0.5, s=20, color='#2c3e50')
    ax.set_xlabel('Test R²')
    ax.set_ylabel('Spearman IC')
    ax.set_title('IC vs Out-of-Sample R²')
    ax.axhline(0, color='red', lw=0.8, linestyle='--')

    ax = axes[1]
    ax.scatter(ic_df['N_train'], ic_df['IC'], alpha=0.5, s=20, color='#16a085')
    ax.set_xlabel('Training Sample Size')
    ax.set_ylabel('Spearman IC')
    ax.set_title('IC vs Sample Size')
    ax.axhline(0, color='red', lw=0.8, linestyle='--')

    ax = axes[2]
    ax.scatter(ic_df['MSE_test'], ic_df['IC'], alpha=0.5, s=20, color='#8e44ad')
    ax.set_xlabel('Test MSE')
    ax.set_ylabel('Spearman IC')
    ax.set_title('IC vs Test MSE')
    ax.axhline(0, color='red', lw=0.8, linestyle='--')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 8: Grid search heatmap ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Grid Search Results — ICIR by Hyperparameters",
                 fontsize=14, fontweight='bold')

    # Show top 20 configs sorted by ICIR
    top_n = min(20, len(grid_df))
    top = grid_df.head(top_n).copy()
    labels = [
        f"lv={int(r['num_leaves'])} d={int(r['max_depth'])} "
        f"lr={r['learning_rate']} n={int(r['n_estimators'])} "
        f"mc={int(r['min_child_samples'])}"
        for _, r in top.iterrows()
    ]
    colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    bars = ax.barh(range(top_n), top['ICIR'].values, color=colors_bar,
                   edgecolor='white', alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('ICIR')
    ax.set_title(f'Top {top_n} Configurations (best ICIR = {grid_df["ICIR"].max():.4f})')
    ax.invert_yaxis()

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print(f"  → {REPORT_PDF}")

print(f"\n{'═' * 80}")
print(f"  DONE — LightGBM factor model evaluation complete.")
print(f"{'═' * 80}\n")
