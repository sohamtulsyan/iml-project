"""
================================================================================
  MLP VISUALIZATION — COMPARE AGAINST BASELINES
  Generates 6 high-resolution plots for comprehensive model evaluation
================================================================================

  Outputs:
  ────────
  1. IC Time Series (line plot) — MLP vs Ridge vs LightGBM vs Random Forest
  2. ICIR Comparison (bar chart) — Color-coded by performance tier
  3. IC Distribution (histogram + KDE) — Statistical comparison
  4. Rolling IC (60-month window) — Signal stability
  5. Hyperparameter Sensitivity (heatmap) — Learning rate vs L2 regularization
  6. Cumulative Long-Short Returns — Portfolio performance

  Usage:
  ──────
    python visualize_mlp_results.py

  Dependencies:
  ──────────────
    - matplotlib (static plots, 300 DPI PNG)
    - plotly (optional, for interactive HTML)
    - seaborn (optional, for enhanced styling)

================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, gaussian_kde
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

MLP_DIR = Path(__file__).parent
RIDGE_DIR = MLP_DIR.parent / "ridge-regression"
LGBM_DIR = MLP_DIR.parent / "LightGBM"
RF_DIR = MLP_DIR.parent / "Random Forest"

OUTPUT_DIR = MLP_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

# Result CSV paths
MLP_IC_CSV = MLP_DIR / "mlp_ic_results.csv"
MLP_PARAMS_CSV = MLP_DIR / "mlp_hyperparams.csv"

RIDGE_IC_CSV = RIDGE_DIR / "ridge_ic_results.csv"
LGBM_IC_CSV = LGBM_DIR / "lgbm_ic_results.csv"
RF_IC_CSV = RF_DIR / "rf_ic_results.csv"

# Figure settings
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
DPI = 300
FIG_SIZE = (14, 8)

# Color palette
colors = {
    'mlp': '#FF6B6B',
    'ridge': '#4ECDC4',
    'lgbm': '#45B7D1',
    'rf': '#FFA07A',
}

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("  MLP VISUALIZATION")
print("=" * 80)

print("\n[1/7] Loading results...")

# Load MLP results
mlp_ic = pd.read_csv(MLP_IC_CSV)
mlp_params = pd.read_csv(MLP_PARAMS_CSV)


def _normalize_ic_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase 'month' and 'ic' regardless of source."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]  # lowercase all columns
    # Ridge uses 'Month' → already lowercased to 'month'
    # Some CSVs may have 'IC' → normalize to 'ic'
    return df


mlp_ic     = _normalize_ic_df(mlp_ic)
mlp_params = _normalize_ic_df(mlp_params)

# Load baseline results
try:
    ridge_ic = _normalize_ic_df(pd.read_csv(RIDGE_IC_CSV))
    print("  ✓ Ridge results loaded")
except FileNotFoundError:
    ridge_ic = None
    print("  ✗ Ridge results not found")

try:
    lgbm_ic = _normalize_ic_df(pd.read_csv(LGBM_IC_CSV))
    print("  ✓ LightGBM results loaded")
except FileNotFoundError:
    lgbm_ic = None
    print("  ✗ LightGBM results not found")

try:
    rf_ic = _normalize_ic_df(pd.read_csv(RF_DIR / "rf_ic_results.csv"))
    print("  ✓ Random Forest results loaded")
except FileNotFoundError:
    rf_ic = None
    print("  ✗ Random Forest results not found")

# Ensure datetime
mlp_ic['month'] = pd.to_datetime(mlp_ic['month'])
if ridge_ic is not None:
    ridge_ic['month'] = pd.to_datetime(ridge_ic['month'])
if lgbm_ic is not None:
    lgbm_ic['month'] = pd.to_datetime(lgbm_ic['month'])
if rf_ic is not None:
    rf_ic['month'] = pd.to_datetime(rf_ic['month'])

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 1: IC TIME SERIES
# ═══════════════════════════════════════════════════════════════════════════════

print("[2/7] Generating IC Time Series plot...")

fig, ax = plt.subplots(figsize=FIG_SIZE)

ax.plot(mlp_ic['month'], mlp_ic['ic'], label='MLP', color=colors['mlp'], linewidth=2.5, alpha=0.9)
if ridge_ic is not None:
    ax.plot(ridge_ic['month'], ridge_ic['ic'], label='Ridge', color=colors['ridge'], linewidth=2, alpha=0.8)
if lgbm_ic is not None:
    ax.plot(lgbm_ic['month'], lgbm_ic['ic'], label='LightGBM', color=colors['lgbm'], linewidth=2, alpha=0.8)
if rf_ic is not None:
    ax.plot(rf_ic['month'], rf_ic['ic'], label='Random Forest', color=colors['rf'], linewidth=2, alpha=0.8)

ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Zero')
ax.axhline(y=mlp_ic['ic'].mean(), color=colors['mlp'], linestyle=':', linewidth=2, alpha=0.7, label=f'MLP Mean (IC={mlp_ic["ic"].mean():.4f})')

ax.set_xlabel('Month', fontsize=12, fontweight='bold')
ax.set_ylabel('Information Coefficient (IC)', fontsize=12, fontweight='bold')
ax.set_title('MLP vs Baselines: Information Coefficient Over Time', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_ic_timeseries.png", dpi=DPI, bbox_inches='tight')
plt.close()

print(f"  ✓ {OUTPUT_DIR / '01_ic_timeseries.png'}")

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 2: ICIR COMPARISON (BAR CHART)
# ═══════════════════════════════════════════════════════════════════════════════

print("[3/7] Generating ICIR Comparison plot...")

# Compute metrics
metrics = {}
metrics['MLP'] = {
    'ic': mlp_ic['ic'].mean(),
    'icir': mlp_ic['ic'].mean() / mlp_ic['ic'].std() if mlp_ic['ic'].std() > 0 else 0,
}

if ridge_ic is not None:
    metrics['Ridge'] = {
        'ic': ridge_ic['ic'].mean(),
        'icir': ridge_ic['ic'].mean() / ridge_ic['ic'].std() if ridge_ic['ic'].std() > 0 else 0,
    }

if lgbm_ic is not None:
    metrics['LightGBM'] = {
        'ic': lgbm_ic['ic'].mean(),
        'icir': lgbm_ic['ic'].mean() / lgbm_ic['ic'].std() if lgbm_ic['ic'].std() > 0 else 0,
    }

if rf_ic is not None:
    metrics['RF'] = {
        'ic': rf_ic['ic'].mean(),
        'icir': rf_ic['ic'].mean() / rf_ic['ic'].std() if rf_ic['ic'].std() > 0 else 0,
    }

models = list(metrics.keys())
icir_values = [metrics[m]['icir'] for m in models]

# Color bars by performance tier
def get_icir_color(icir):
    if icir >= 0.5:
        return '#2ECC71'  # Green (Good)
    elif icir >= 0.3:
        return '#F39C12'  # Yellow (Moderate)
    else:
        return '#E74C3C'  # Red (Weak)

bar_colors = [get_icir_color(icir) for icir in icir_values]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, icir_values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, icir in zip(bars, icir_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{icir:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add threshold lines
ax.axhline(y=0.5, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (≥0.5)')
ax.axhline(y=0.3, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Moderate (≥0.3)')

ax.set_ylabel('ICIR (Information Coefficient Information Ratio)', fontsize=12, fontweight='bold')
ax.set_title('Model Performance: ICIR Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_icir_comparison.png", dpi=DPI, bbox_inches='tight')
plt.close()

print(f"  ✓ {OUTPUT_DIR / '02_icir_comparison.png'}")

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 3: IC DISTRIBUTION (HISTOGRAM + KDE)
# ═══════════════════════════════════════════════════════════════════════════════

print("[4/7] Generating IC Distribution plot...")

fig, ax = plt.subplots(figsize=FIG_SIZE)

# MLP histogram and KDE
ax.hist(mlp_ic['ic'], bins=30, color=colors['mlp'], alpha=0.5, label='MLP', density=True)
kde_mlp = gaussian_kde(mlp_ic['ic'])
x_range = np.linspace(min(mlp_ic['ic'].min(), -0.1), max(mlp_ic['ic'].max(), 0.1), 200)
ax.plot(x_range, kde_mlp(x_range), color=colors['mlp'], linewidth=2.5)

# Baseline histograms
if ridge_ic is not None:
    ax.hist(ridge_ic['ic'], bins=30, color=colors['ridge'], alpha=0.3, label='Ridge', density=True)

if lgbm_ic is not None:
    ax.hist(lgbm_ic['ic'], bins=30, color=colors['lgbm'], alpha=0.3, label='LightGBM', density=True)

if rf_ic is not None:
    ax.hist(rf_ic['ic'], bins=30, color=colors['rf'], alpha=0.3, label='Random Forest', density=True)

ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
ax.set_xlabel('Information Coefficient (IC)', fontsize=12, fontweight='bold')
ax.set_ylabel('Density', fontsize=12, fontweight='bold')
ax.set_title('IC Distribution Comparison (Histogram + KDE)', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_ic_distribution.png", dpi=DPI, bbox_inches='tight')
plt.close()

print(f"  ✓ {OUTPUT_DIR / '03_ic_distribution.png'}")

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 4: ROLLING IC (60-MONTH WINDOW)
# ═══════════════════════════════════════════════════════════════════════════════

print("[5/7] Generating Rolling IC plot...")

fig, ax = plt.subplots(figsize=FIG_SIZE)

# Compute rolling means (60-month)
mlp_rolling = mlp_ic.sort_values('month').set_index('month')['ic'].rolling(60, min_periods=1).mean()

ax.plot(mlp_rolling.index, mlp_rolling, label='MLP (60-month)', color=colors['mlp'], linewidth=2.5)

if ridge_ic is not None:
    ridge_rolling = ridge_ic.sort_values('month').set_index('month')['ic'].rolling(60, min_periods=1).mean()
    ax.plot(ridge_rolling.index, ridge_rolling, label='Ridge (60-month)', color=colors['ridge'], linewidth=2)

if lgbm_ic is not None:
    lgbm_rolling = lgbm_ic.sort_values('month').set_index('month')['ic'].rolling(60, min_periods=1).mean()
    ax.plot(lgbm_rolling.index, lgbm_rolling, label='LightGBM (60-month)', color=colors['lgbm'], linewidth=2)

if rf_ic is not None:
    rf_rolling = rf_ic.sort_values('month').set_index('month')['ic'].rolling(60, min_periods=1).mean()
    ax.plot(rf_rolling.index, rf_rolling, label='RF (60-month)', color=colors['rf'], linewidth=2)

ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
ax.set_xlabel('Month', fontsize=12, fontweight='bold')
ax.set_ylabel('Rolling Mean IC', fontsize=12, fontweight='bold')
ax.set_title('Rolling Information Coefficient (60-Month Window)', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_rolling_ic.png", dpi=DPI, bbox_inches='tight')
plt.close()

print(f"  ✓ {OUTPUT_DIR / '04_rolling_ic.png'}")

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 5: HYPERPARAMETER SENSITIVITY (HEATMAP)
# ═══════════════════════════════════════════════════════════════════════════════

print("[6/7] Generating Hyperparameter Sensitivity heatmap...")

# Extract learning rate and alpha from params
mlp_params['learning_rate'] = pd.to_numeric(mlp_params['learning_rate'])
mlp_params['alpha'] = pd.to_numeric(mlp_params['alpha'])

# Join with IC results
param_ic = mlp_ic[['month', 'ic']].merge(mlp_params[['month', 'learning_rate', 'alpha']], on='month')

# Create pivot table for heatmap
pivot_data = param_ic.pivot_table(
    values='ic',
    index='alpha',
    columns='learning_rate',
    aggfunc='mean'
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlGn', center=0, ax=ax, cbar_kws={'label': 'Mean IC'})
ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('Alpha (L2 Regularization)', fontsize=12, fontweight='bold')
ax.set_title('Hyperparameter Sensitivity: Learning Rate vs Alpha', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_hyperparam_heatmap.png", dpi=DPI, bbox_inches='tight')
plt.close()

print(f"  ✓ {OUTPUT_DIR / '05_hyperparam_heatmap.png'}")

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 6: CUMULATIVE LONG-SHORT RETURNS
# ═══════════════════════════════════════════════════════════════════════════════

print("[7/7] Generating Cumulative Long-Short Returns plot...")

def compute_portfolio_returns(ic_series):
    """
    Compute cumulative long-short portfolio returns.
    Assume: rank predictions into deciles, long top 10%, short bottom 10%.
    Return premium ≈ IC × volatility × 10 (simplified).
    """
    monthly_excess_return = ic_series.values * 0.01  # Scale IC to approximate return
    cumulative = np.cumprod(1 + monthly_excess_return)
    return cumulative

fig, ax = plt.subplots(figsize=FIG_SIZE)

mlp_monthly = mlp_ic.sort_values('month')
mlp_cum = compute_portfolio_returns(mlp_monthly['ic'])
ax.plot(mlp_monthly['month'], mlp_cum, label='MLP', color=colors['mlp'], linewidth=2.5)

if ridge_ic is not None:
    ridge_monthly = ridge_ic.sort_values('month')
    ridge_cum = compute_portfolio_returns(ridge_monthly['ic'])
    ax.plot(ridge_monthly['month'], ridge_cum, label='Ridge', color=colors['ridge'], linewidth=2)

if lgbm_ic is not None:
    lgbm_monthly = lgbm_ic.sort_values('month')
    lgbm_cum = compute_portfolio_returns(lgbm_monthly['ic'])
    ax.plot(lgbm_monthly['month'], lgbm_cum, label='LightGBM', color=colors['lgbm'], linewidth=2)

if rf_ic is not None:
    rf_monthly = rf_ic.sort_values('month')
    rf_cum = compute_portfolio_returns(rf_monthly['ic'])
    ax.plot(rf_monthly['month'], rf_cum, label='Random Forest', color=colors['rf'], linewidth=2)

ax.axhline(y=1, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Breakeven')
ax.set_xlabel('Month', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Portfolio Value', fontsize=12, fontweight='bold')
ax.set_title('Cumulative Long-Short Portfolio Returns (IC-Based Signal)', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_cumulative_returns.png", dpi=DPI, bbox_inches='tight')
plt.close()

print(f"  ✓ {OUTPUT_DIR / '06_cumulative_returns.png'}")

# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  MLP VISUALIZATION — COMPLETE")
print("=" * 80)

print(f"\nResults saved to: {OUTPUT_DIR}/")
print(f"\nPerformance Summary:")
print(f"  • MLP Mean IC: {metrics['MLP']['ic']:.4f}")
print(f"  • MLP ICIR: {metrics['MLP']['icir']:.4f}")

if ridge_ic is not None:
    print(f"  • Ridge Mean IC: {metrics['Ridge']['ic']:.4f}")
    print(f"  • Ridge ICIR: {metrics['Ridge']['icir']:.4f}")

if lgbm_ic is not None:
    print(f"  • LightGBM Mean IC: {metrics['LightGBM']['ic']:.4f}")
    print(f"  • LightGBM ICIR: {metrics['LightGBM']['icir']:.4f}")

if rf_ic is not None:
    print(f"  • Random Forest Mean IC: {metrics['RF']['ic']:.4f}")
    print(f"  • Random Forest ICIR: {metrics['RF']['icir']:.4f}")

print("\nVisualization Files:")
print(f"  ✓ 01_ic_timeseries.png")
print(f"  ✓ 02_icir_comparison.png")
print(f"  ✓ 03_ic_distribution.png")
print(f"  ✓ 04_rolling_ic.png")
print(f"  ✓ 05_hyperparam_heatmap.png")
print(f"  ✓ 06_cumulative_returns.png")

print("\n" + "=" * 80 + "\n")
