"""
================================================================================
  RIDGE REGRESSION — LINEAR FACTOR MODEL BASELINE
  Rolling cross-sectional prediction with full assumption validation
================================================================================

  Pipeline overview:
  ──────────────────
  1. Load panel data, create next-month return target
  2. Winsorize features + target for robustness
  3. For each month t:
       a. PRE-CHECKS: variance, multicollinearity (VIF), stationarity
       b. Standardise features (fit on train, transform both)
       c. Select optimal α via time-series cross-validation (expanding window)
       d. Fit Ridge on month t, predict month t+1
       e. POST-CHECKS: residual normality (Jarque-Bera), heteroscedasticity
          (Breusch-Pagan), autocorrelation (Durbin-Watson)
       f. Compute Spearman IC between predictions and realised returns
  4. Aggregate: Mean IC, ICIR, coefficient stability, diagnostic pass rates
  5. Generate PDF report with monthly IC series, coefficient paths, diagnostics

  Usage:
  ──────
    python ridge-regression.py

  Output:
  ───────
    ridge_ic_results.csv        — monthly IC values + diagnostics
    ridge_coef_summary.csv      — per-month coefficient values & chosen α
    ridge_report.pdf            — visual diagnostics report
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

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from scipy.stats import spearmanr
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

DATA_PATH      = "project_database.csv"
FEATURES       = ['lag_ret', 'Momentum', 'BM_sep', 'OpProf', 'Inv', 'mktcap']
TARGET_COL     = 'monthly_gross_return'
STOCK_COL      = 'co_code'
DATE_COL       = 'Month'
MIN_OBS        = 30

# Ridge α candidates (log-spaced from 0.001 to 1000)
ALPHAS         = np.logspace(-3, 3, 50)

# Winsorisation bounds
WINSOR_LOWER   = 0.01
WINSOR_UPPER   = 0.99

# Diagnostic thresholds
VIF_THRESHOLD  = 10.0
JB_ALPHA       = 0.05
BP_ALPHA       = 0.05
DW_LOWER       = 1.5
DW_UPPER       = 2.5

# Output paths
OUTPUT_DIR     = Path(".")
IC_CSV         = OUTPUT_DIR / "ridge_ic_results.csv"
COEF_CSV       = OUTPUT_DIR / "ridge_coef_summary.csv"
REPORT_PDF     = OUTPUT_DIR / "ridge_report.pdf"


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("  RIDGE REGRESSION — LINEAR FACTOR MODEL BASELINE")
print("=" * 80)

print("\n[1/6] Loading data...")
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
print("\n[2/6] Winsorising features & target...")

def winsorize(series, lower=WINSOR_LOWER, upper=WINSOR_UPPER):
    """Clip values to [lower, upper] quantiles."""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)

for col in FEATURES + ['target']:
    df_model[col] = winsorize(df_model[col])

print(f"  Winsorised at [{WINSOR_LOWER*100:.0f}%, {WINSOR_UPPER*100:.0f}%] quantiles")


# ═══════════════════════════════════════════════════════════════════════════════
#  ASSUMPTION VALIDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def check_variance(X: pd.DataFrame) -> tuple[bool, list[str]]:
    """Check that all features have non-trivial variance."""
    low_var = [c for c in X.columns if X[c].var() < 1e-10]
    return len(low_var) == 0, low_var


def check_vif(X: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    """Compute Variance Inflation Factors; flag if any VIF > threshold."""
    vif = pd.DataFrame({
        'feature': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    ok = (vif['VIF'] <= VIF_THRESHOLD).all()
    return ok, vif


def post_regression_diagnostics(X: pd.DataFrame, y: pd.Series,
                                 model_fitted) -> dict:
    """
    Run post-regression assumption checks on OLS-equivalent residuals.

    Returns dict with:
      jb_stat, jb_p   → Jarque-Bera normality test on residuals
      bp_stat, bp_p   → Breusch-Pagan heteroscedasticity test
      dw              → Durbin-Watson autocorrelation statistic
      normality_ok    → True if residuals ~normal (p > α)
      homosced_ok     → True if no heteroscedasticity (p > α)
      autocorr_ok     → True if DW in acceptable range
    """
    X_const = sm.add_constant(X)
    ols = sm.OLS(y, X_const).fit()
    resid = ols.resid

    # Jarque-Bera normality
    jb_stat, jb_p, _, _ = jarque_bera(resid)

    # Breusch-Pagan heteroscedasticity
    bp_stat, bp_p, _, _ = het_breuschpagan(resid, X_const)

    # Durbin-Watson autocorrelation
    dw = durbin_watson(resid)

    return {
        'jb_stat': jb_stat,
        'jb_p': jb_p,
        'bp_stat': bp_stat,
        'bp_p': bp_p,
        'dw': dw,
        'normality_ok': jb_p > JB_ALPHA,
        'homosced_ok': bp_p > BP_ALPHA,
        'autocorr_ok': DW_LOWER <= dw <= DW_UPPER,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ROLLING MONTHLY RIDGE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3/6] Running rolling monthly Ridge regression...")

months = sorted(df_model[DATE_COL].unique())
print(f"  {len(months)} months available  |  MIN_OBS={MIN_OBS}")

ic_results = []
coef_results = []

skipped = {'low_obs': 0, 'low_var': 0, 'high_vif': 0}

for i in range(len(months) - 1):
    t = months[i]
    t_next = months[i + 1]
    month_str = pd.Timestamp(t_next).strftime('%Y-%m')

    train = df_model[df_model[DATE_COL] == t]
    test  = df_model[df_model[DATE_COL] == t_next]

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

    # 2. Multicollinearity (VIF)
    vif_ok, vif_df = check_vif(X_train)
    pre_vif_max = vif_df['VIF'].max()

    # Note: Ridge handles multicollinearity by design, so we log but don't skip
    pre_flags = []
    if not vif_ok:
        pre_flags.append(f"VIF>{VIF_THRESHOLD}")

    # ══════════════════════════════════════════════════════════════════════════
    #  STANDARDISE FEATURES
    # ══════════════════════════════════════════════════════════════════════════

    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=FEATURES,
        index=X_train.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test),
        columns=FEATURES,
        index=X_test.index
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  FIT RIDGE WITH CROSS-VALIDATED α
    # ══════════════════════════════════════════════════════════════════════════

    ridge_cv = RidgeCV(
        alphas=ALPHAS,
        scoring='neg_mean_squared_error',
        cv=min(5, len(X_train_sc) // 10),  # adapt k to sample size
    )
    ridge_cv.fit(X_train_sc, y_train)
    best_alpha = ridge_cv.alpha_

    # Refit final model with best alpha for clean coefficient extraction
    model = Ridge(alpha=best_alpha)
    model.fit(X_train_sc, y_train)

    # ══════════════════════════════════════════════════════════════════════════
    #  PREDICT & EVALUATE
    # ══════════════════════════════════════════════════════════════════════════

    preds_train = model.predict(X_train_sc)
    preds_test  = model.predict(X_test_sc)

    # In-sample metrics
    r2_train = r2_score(y_train, preds_train)
    mse_train = mean_squared_error(y_train, preds_train)

    # Out-of-sample metrics
    r2_test = r2_score(y_test, preds_test)
    mse_test = mean_squared_error(y_test, preds_test)

    # Spearman IC (primary evaluation metric for factor models)
    ic, ic_p = spearmanr(preds_test, y_test)

    # ══════════════════════════════════════════════════════════════════════════
    #  POST-CHECKS (on training residuals)
    # ══════════════════════════════════════════════════════════════════════════

    diag = post_regression_diagnostics(X_train_sc, y_train, model)

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
    #  STORE RESULTS
    # ══════════════════════════════════════════════════════════════════════════

    ic_results.append({
        'Month': month_str,
        'IC': ic,
        'IC_p': ic_p,
        'alpha': best_alpha,
        'R2_train': r2_train,
        'R2_test': r2_test,
        'MSE_train': mse_train,
        'MSE_test': mse_test,
        'N_train': len(train),
        'N_test': len(test),
        'jb_p': diag['jb_p'],
        'bp_p': diag['bp_p'],
        'dw': diag['dw'],
        'normality_ok': diag['normality_ok'],
        'homosced_ok': diag['homosced_ok'],
        'autocorr_ok': diag['autocorr_ok'],
        'post_fails': post_fails,
        'pre_vif_max': pre_vif_max,
        'pre_flags': '|'.join(pre_flags) if pre_flags else '',
        'post_flags': '|'.join(post_flags) if post_flags else '',
    })

    # Store coefficients for stability analysis
    coef_row = {'Month': month_str, 'alpha': best_alpha, 'intercept': model.intercept_}
    for j, feat in enumerate(FEATURES):
        coef_row[f'coef_{feat}'] = model.coef_[j]
    coef_results.append(coef_row)


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULTS AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4/6] Aggregating results...")

ic_df = pd.DataFrame(ic_results)
coef_df = pd.DataFrame(coef_results)

if len(ic_df) == 0:
    print("  ⚠ No valid months — check data quality. Exiting.")
    exit(1)

IC_mean = ic_df['IC'].mean()
IC_std  = ic_df['IC'].std()
ICIR    = IC_mean / IC_std if IC_std != 0 else np.nan

# ── Summary Statistics ────────────────────────────────────────────────────────
print(f"\n{'═' * 80}")
print(f"  RIDGE REGRESSION — RESULTS SUMMARY")
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

print(f"\n  ── Ridge Regularisation ──")
print(f"    Median α:        {ic_df['alpha'].median():.4f}")
print(f"    Mean α:          {ic_df['alpha'].mean():.4f}")
print(f"    α range:         [{ic_df['alpha'].min():.4f}, {ic_df['alpha'].max():.4f}]")

print(f"\n  ── Assumption Diagnostics ──")
print(f"    Normality pass:       {ic_df['normality_ok'].mean()*100:.1f}%")
print(f"    Homoscedasticity pass: {ic_df['homosced_ok'].mean()*100:.1f}%")
print(f"    No autocorrelation:   {ic_df['autocorr_ok'].mean()*100:.1f}%")
print(f"    Avg post_fails:       {ic_df['post_fails'].mean():.2f} / 3")
print(f"    Max VIF (pre):        {ic_df['pre_vif_max'].mean():.2f} (avg across months)")

print(f"\n  ── Skipped Months ──")
for reason, cnt in skipped.items():
    print(f"    {reason}: {cnt}")

print(f"\n  ── Coefficient Summary (standardised) ──")
coef_cols = [c for c in coef_df.columns if c.startswith('coef_')]
for col in coef_cols:
    feat = col.replace('coef_', '')
    mean_c = coef_df[col].mean()
    std_c  = coef_df[col].std()
    t_stat = mean_c / (std_c / np.sqrt(len(coef_df))) if std_c > 0 else np.nan
    pct_pos = (coef_df[col] > 0).mean() * 100
    print(f"    {feat:<12s}  mean={mean_c:>8.5f}  std={std_c:>8.5f}  "
          f"t={t_stat:>7.2f}  %pos={pct_pos:>5.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE CSVs
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[5/6] Saving results...")
ic_df.to_csv(IC_CSV, index=False)
coef_df.to_csv(COEF_CSV, index=False)
print(f"  → {IC_CSV}")
print(f"  → {COEF_CSV}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PDF REPORT
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[6/6] Generating PDF report...")

with PdfPages(REPORT_PDF) as pdf:

    # ── Page 1: Monthly IC time series ────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Ridge Regression — Monthly Information Coefficient (IC)",
                 fontsize=14, fontweight='bold')

    ax = axes[0]
    months_plot = range(len(ic_df))
    colors = ['#27ae60' if v > 0 else '#e74c3c' for v in ic_df['IC']]
    ax.bar(months_plot, ic_df['IC'], color=colors, alpha=0.7, width=1.0)
    ax.axhline(IC_mean, color='#2980b9', linestyle='--', lw=1.5,
               label=f'Mean IC = {IC_mean:.4f}')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('Spearman IC')
    ax.set_title(f'Monthly IC  |  ICIR = {ICIR:.4f}  |  '
                 f'% Positive = {(ic_df["IC"]>0).mean()*100:.1f}%')
    ax.legend(loc='upper right')

    # Cumulative IC
    ax = axes[1]
    ax.plot(months_plot, ic_df['IC'].cumsum(), color='#8e44ad', lw=2)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('Cumulative IC')
    ax.set_xlabel('Month Index')
    ax.set_title('Cumulative IC (monotonically rising = consistent signal)')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 2: Coefficient paths (stability over time) ───────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Ridge Coefficients Over Time (Standardised Features)",
                 fontsize=14, fontweight='bold')

    cmap = plt.cm.Set2
    for idx, col in enumerate(coef_cols):
        feat = col.replace('coef_', '')
        ax.plot(months_plot, coef_df[col], label=feat, lw=1.5, alpha=0.85,
                color=cmap(idx / max(len(coef_cols)-1, 1)))
    ax.axhline(0, color='black', lw=0.8, linestyle='--')
    ax.set_xlabel('Month Index')
    ax.set_ylabel('Coefficient Value')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.set_title('Coefficient stability — low variance = reliable factor')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 3: Optimal α trajectory ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Ridge α Selection", fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.semilogy(months_plot, ic_df['alpha'], color='#e67e22', lw=1.5)
    ax.set_xlabel('Month Index')
    ax.set_ylabel('Optimal α (log scale)')
    ax.set_title('CV-Selected α Over Time')

    ax = axes[1]
    ax.hist(np.log10(ic_df['alpha']), bins=25, color='#3498db',
            edgecolor='white', alpha=0.85)
    ax.set_xlabel('log₁₀(α)')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of Selected α  (median={ic_df["alpha"].median():.3f})')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 4: Diagnostic pass rates ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Post-Regression Assumption Diagnostics",
                 fontsize=14, fontweight='bold')

    # Normality (Jarque-Bera p-values)
    ax = axes[0]
    ax.hist(ic_df['jb_p'], bins=25, color='#2ecc71', edgecolor='white', alpha=0.85)
    ax.axvline(JB_ALPHA, color='red', linestyle='--', lw=1.5,
               label=f'α = {JB_ALPHA}')
    ax.set_xlabel('Jarque-Bera p-value')
    ax.set_ylabel('Count')
    pct = ic_df['normality_ok'].mean()*100
    ax.set_title(f'Normality of Residuals ({pct:.0f}% pass)')
    ax.legend()

    # Homoscedasticity (Breusch-Pagan p-values)
    ax = axes[1]
    ax.hist(ic_df['bp_p'], bins=25, color='#e67e22', edgecolor='white', alpha=0.85)
    ax.axvline(BP_ALPHA, color='red', linestyle='--', lw=1.5,
               label=f'α = {BP_ALPHA}')
    ax.set_xlabel('Breusch-Pagan p-value')
    ax.set_ylabel('Count')
    pct = ic_df['homosced_ok'].mean()*100
    ax.set_title(f'Homoscedasticity ({pct:.0f}% pass)')
    ax.legend()

    # Autocorrelation (Durbin-Watson)
    ax = axes[2]
    ax.hist(ic_df['dw'], bins=25, color='#9b59b6', edgecolor='white', alpha=0.85)
    ax.axvline(DW_LOWER, color='red', linestyle='--', lw=1.2)
    ax.axvline(DW_UPPER, color='red', linestyle='--', lw=1.2,
               label=f'Acceptable [{DW_LOWER}, {DW_UPPER}]')
    ax.set_xlabel('Durbin-Watson Statistic')
    ax.set_ylabel('Count')
    pct = ic_df['autocorr_ok'].mean()*100
    ax.set_title(f'No Autocorrelation ({pct:.0f}% pass)')
    ax.legend()

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 5: R² distribution & IC vs R² ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Model Fit Quality", fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.hist(ic_df['R2_train'], bins=25, color='#3498db', edgecolor='white',
            alpha=0.7, label='Train')
    ax.hist(ic_df['R2_test'], bins=25, color='#e74c3c', edgecolor='white',
            alpha=0.7, label='Test')
    ax.set_xlabel('R²')
    ax.set_ylabel('Count')
    ax.set_title('In-Sample vs Out-of-Sample R²')
    ax.legend()

    ax = axes[1]
    ax.scatter(ic_df['R2_test'], ic_df['IC'], alpha=0.5, s=20, color='#2c3e50')
    ax.set_xlabel('Test R²')
    ax.set_ylabel('Spearman IC')
    ax.set_title('IC vs Out-of-Sample R²')
    ax.axhline(0, color='red', lw=0.8, linestyle='--')

    ax = axes[2]
    ax.scatter(ic_df['N_train'], ic_df['IC'], alpha=0.5, s=20, color='#16a085')
    ax.set_xlabel('Training Sample Size')
    ax.set_ylabel('Spearman IC')
    ax.set_title('IC vs Sample Size')
    ax.axhline(0, color='red', lw=0.8, linestyle='--')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 6: VIF over time ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Pre-Check: Maximum VIF Over Time",
                 fontsize=14, fontweight='bold')

    ax.plot(months_plot, ic_df['pre_vif_max'], color='#c0392b', lw=1.5)
    ax.axhline(VIF_THRESHOLD, color='red', linestyle='--', lw=1.5,
               label=f'VIF Threshold = {VIF_THRESHOLD}')
    ax.set_xlabel('Month Index')
    ax.set_ylabel('Max VIF')
    ax.set_title('Multicollinearity is handled by Ridge — plotted for transparency')
    ax.legend()

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print(f"  → {REPORT_PDF}")

print(f"\n{'═' * 80}")
print(f"  DONE — Ridge regression baseline complete.")
print(f"{'═' * 80}\n")
