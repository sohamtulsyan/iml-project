"""
================================================================================
  TRANSFORMER PIPELINE — VISUALIZATION
  Generates 6 plots consistent with MLP/LightGBM/RF visualization style.
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde

try:
    import seaborn as sns
    sns.set_palette("husl")
    plt.style.use("seaborn-v0_8-darkgrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

TRANSFORMER_DIR = Path(__file__).parent
RIDGE_DIR  = TRANSFORMER_DIR.parent / "ridge-regression"
LGBM_DIR   = TRANSFORMER_DIR.parent / "LightGBM"
RF_DIR     = TRANSFORMER_DIR.parent / "Random Forest"
MLP_DIR    = TRANSFORMER_DIR.parent / "MLP"

OUTPUT_DIR = TRANSFORMER_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

DPI      = 300
FIG_SIZE = (14, 8)

COLORS = {
    "transformer": "#9B59B6",
    "mlp":         "#FF6B6B",
    "ridge":       "#4ECDC4",
    "lgbm":        "#45B7D1",
    "rf":          "#FFA07A",
}


def _normalize_df(df: pd.DataFrame, ic_col: str = "IC", month_col: str = "Month") -> pd.DataFrame:
    """Lower-case all columns and ensure 'month' and 'ic' exist."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df


def _safe_load(path: Path, label: str):
    try:
        df = _normalize_df(pd.read_csv(path))
        df["month"] = pd.to_datetime(df["month"])
        print(f"  ✓ {label} loaded ({len(df)} months)")
        return df
    except FileNotFoundError:
        print(f"  ✗ {label} not found ({path})")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Load all IC series
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  TRANSFORMER VISUALIZATION")
print("=" * 70)
print("\n[1/7] Loading IC results...")

transformer_ic = _safe_load(TRANSFORMER_DIR / "results" / "transformer_ic_series.csv", "Transformer")
ridge_ic       = _safe_load(RIDGE_DIR / "ridge_ic_results.csv",              "Ridge")
lgbm_ic        = _safe_load(LGBM_DIR  / "lgbm_fixed_ic.csv",                "LightGBM")
mlp_ic         = _safe_load(MLP_DIR   / "mlp_ic_results.csv",                "MLP")
rf_ic          = _safe_load(RF_DIR    / "rf_ic_results.csv",                 "Random Forest")

if transformer_ic is None:
    print("  ✗ Transformer results not found — run run_transformer.py first.")
    raise SystemExit(1)


def _icir(df):
    mu, sigma = df["ic"].mean(), df["ic"].std()
    return (mu / sigma) if sigma > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 1: IC Time Series
# ─────────────────────────────────────────────────────────────────────────────

print("[2/7] IC Time Series...")

fig, ax = plt.subplots(figsize=FIG_SIZE)

ax.plot(transformer_ic["month"], transformer_ic["ic"],
        label="Transformer", color=COLORS["transformer"], lw=2.5, alpha=0.95)
for df_, label, col in [
    (ridge_ic, "Ridge",         COLORS["ridge"]),
    (lgbm_ic,  "LightGBM",     COLORS["lgbm"]),
    (mlp_ic,   "MLP",          COLORS["mlp"]),
    (rf_ic,    "Random Forest", COLORS["rf"]),
]:
    if df_ is not None:
        ax.plot(df_["month"], df_["ic"], label=label, color=col, lw=1.5, alpha=0.7)

ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.5)
ax.axhline(transformer_ic["ic"].mean(), color=COLORS["transformer"], lw=1.5, ls=":",
           label=f"Transformer mean ({transformer_ic['ic'].mean():.4f})")
ax.set_xlabel("Month", fontsize=12, fontweight="bold")
ax.set_ylabel("Spearman IC", fontsize=12, fontweight="bold")
ax.set_title("Transformer vs Baselines — Information Coefficient", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_ic_timeseries.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"  ✓ 01_ic_timeseries.png")

# ─────────────────────────────────────────────────────────────────────────────
#  Plot 2: ICIR Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

print("[3/7] ICIR Comparison...")

model_icirs = {"Transformer": _icir(transformer_ic)}
for df_, name in [(ridge_ic, "Ridge"), (lgbm_ic, "LightGBM"),
                  (mlp_ic, "MLP"), (rf_ic, "Random Forest")]:
    if df_ is not None:
        model_icirs[name] = _icir(df_)

def _icir_color(v):
    return "#2ECC71" if v >= 0.5 else ("#F39C12" if v >= 0.3 else "#E74C3C")

names  = list(model_icirs.keys())
values = [model_icirs[n] for n in names]
colors_bar = [_icir_color(v) for v in values]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(names, values, color=colors_bar, edgecolor="black", lw=1.2, alpha=0.85)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax.axhline(0.5, color="green",  ls="--", lw=1.8, alpha=0.6, label="Good ≥ 0.5")
ax.axhline(0.3, color="orange", ls="--", lw=1.8, alpha=0.6, label="Moderate ≥ 0.3")
ax.set_ylabel("ICIR", fontsize=12, fontweight="bold")
ax.set_title("Model Comparison — ICIR", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_icir_comparison.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"  ✓ 02_icir_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
#  Plot 3: IC Distribution (Histogram + KDE)
# ─────────────────────────────────────────────────────────────────────────────

print("[4/7] IC Distribution...")

fig, ax = plt.subplots(figsize=FIG_SIZE)

ic_vals_tr = transformer_ic["ic"].values
ax.hist(ic_vals_tr, bins=30, color=COLORS["transformer"], alpha=0.55,
        label="Transformer", density=True)
kde = gaussian_kde(ic_vals_tr)
x   = np.linspace(ic_vals_tr.min() - 0.05, ic_vals_tr.max() + 0.05, 300)
ax.plot(x, kde(x), color=COLORS["transformer"], lw=2.5)

for df_, label, col in [(ridge_ic, "Ridge", COLORS["ridge"]),
                         (lgbm_ic, "LightGBM", COLORS["lgbm"]),
                         (mlp_ic, "MLP", COLORS["mlp"])]:
    if df_ is not None:
        ax.hist(df_["ic"].values, bins=30, color=col, alpha=0.3,
                label=label, density=True)

ax.axvline(0, color="black", ls="--", lw=1.2, alpha=0.5)
ax.set_xlabel("Spearman IC", fontsize=12, fontweight="bold")
ax.set_ylabel("Density", fontsize=12, fontweight="bold")
ax.set_title("IC Distribution — Transformer vs Baselines", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_ic_distribution.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"  ✓ 03_ic_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
#  Plot 4: Rolling 12-month IC
# ─────────────────────────────────────────────────────────────────────────────

print("[5/7] Rolling IC...")

fig, ax = plt.subplots(figsize=FIG_SIZE)

def _rolling(df_):
    return df_.sort_values("month").set_index("month")["ic"].rolling(12, min_periods=1).mean()

tr_roll = _rolling(transformer_ic)
ax.plot(tr_roll.index, tr_roll, color=COLORS["transformer"], lw=2.5, label="Transformer (12m)")
for df_, label, col in [(ridge_ic, "Ridge", COLORS["ridge"]),
                         (lgbm_ic, "LightGBM", COLORS["lgbm"]),
                         (mlp_ic, "MLP", COLORS["mlp"])]:
    if df_ is not None:
        r = _rolling(df_)
        ax.plot(r.index, r, color=col, lw=1.5, alpha=0.75, label=f"{label} (12m)")

ax.axhline(0, color="black", ls="--", lw=1.2, alpha=0.5)
ax.set_xlabel("Month", fontsize=12, fontweight="bold")
ax.set_ylabel("Rolling Mean IC", fontsize=12, fontweight="bold")
ax.set_title("Rolling 12-Month Information Coefficient", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_rolling_ic.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"  ✓ 04_rolling_ic.png")

# ─────────────────────────────────────────────────────────────────────────────
#  Plot 5: Cumulative IC
# ─────────────────────────────────────────────────────────────────────────────

print("[6/7] Cumulative IC...")

fig, ax = plt.subplots(figsize=FIG_SIZE)

tr_sorted = transformer_ic.sort_values("month")
ax.plot(tr_sorted["month"], tr_sorted["ic"].cumsum(),
        color=COLORS["transformer"], lw=2.5, label="Transformer")
for df_, label, col in [(ridge_ic, "Ridge", COLORS["ridge"]),
                         (lgbm_ic, "LightGBM", COLORS["lgbm"]),
                         (mlp_ic, "MLP", COLORS["mlp"]),
                         (rf_ic, "Random Forest", COLORS["rf"])]:
    if df_ is not None:
        d = df_.sort_values("month")
        ax.plot(d["month"], d["ic"].cumsum(), color=col, lw=1.5, alpha=0.7, label=label)

ax.axhline(0, color="black", ls="--", lw=1.2, alpha=0.5)
ax.set_xlabel("Month", fontsize=12, fontweight="bold")
ax.set_ylabel("Cumulative IC", fontsize=12, fontweight="bold")
ax.set_title("Cumulative Information Coefficient", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_cumulative_ic.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"  ✓ 05_cumulative_ic.png")

# ─────────────────────────────────────────────────────────────────────────────
#  Plot 6: Summary comparison table
# ─────────────────────────────────────────────────────────────────────────────

print("[7/7] Summary table...")

rows = []
for df_, name in [(transformer_ic, "Transformer"),
                  (ridge_ic,       "Ridge"),
                  (lgbm_ic,        "LightGBM"),
                  (mlp_ic,         "MLP"),
                  (rf_ic,          "Random Forest")]:
    if df_ is not None:
        rows.append({
            "Model":     name,
            "Mean IC":   f"{df_['ic'].mean():.4f}",
            "IC Std":    f"{df_['ic'].std():.4f}",
            "ICIR":      f"{_icir(df_):.4f}",
            "% Pos IC":  f"{(df_['ic'] > 0).mean()*100:.1f}%",
            "N Months":  str(len(df_)),
        })

tbl_df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(12, max(3, len(rows) * 0.7 + 1.5)))
ax.axis("off")
tbl = ax.table(cellText=tbl_df.values, colLabels=tbl_df.columns,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.2, 1.8)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#4B2E83")
        cell.set_text_props(color="white", fontweight="bold")
    elif tbl_df.iloc[r - 1]["Model"] == "Transformer":
        cell.set_facecolor("#EDE0F5")
    elif r % 2 == 0:
        cell.set_facecolor("#F5F5F5")
    cell.set_edgecolor("white")

plt.title("Model Comparison Summary", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_summary_table.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"  ✓ 06_summary_table.png")

# ─────────────────────────────────────────────────────────────────────────────
#  Final summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  VISUALIZATION COMPLETE")
print("=" * 70)
print(f"  Output: {OUTPUT_DIR}/")
print()
print(tbl_df.to_string(index=False))
print("=" * 70 + "\n")
