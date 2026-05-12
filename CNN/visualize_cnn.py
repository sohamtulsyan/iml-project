"""
================================================================================
  CNN PIPELINE — VISUALIZATION
  6 plots consistent with MLP / Transformer / LightGBM / RF style.
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde

try:
    plt.style.use("seaborn-v0_8-darkgrid")
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────────────────────────

CNN_DIR   = Path(__file__).parent
BASE      = CNN_DIR.parent

RIDGE_DIR = BASE / "ridge-regression"
LGBM_DIR  = BASE / "LightGBM"
RF_DIR    = BASE / "Random Forest"
MLP_DIR   = BASE / "MLP"
TR_DIR    = BASE / "Transformer"

OUTPUT_DIR = CNN_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

DPI = 300

COLORS = {
    "cnn":         "#E67E22",
    "transformer": "#9B59B6",
    "mlp":         "#FF6B6B",
    "ridge":       "#4ECDC4",
    "lgbm":        "#45B7D1",
    "rf":          "#FFA07A",
}


def _load(path: Path, label: str):
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        df["month"] = pd.to_datetime(df["month"])
        print(f"  ✓ {label}")
        return df
    except FileNotFoundError:
        print(f"  ✗ {label} not found")
        return None


def _icir(df):
    mu, sd = df["ic"].mean(), df["ic"].std()
    return mu / sd if sd > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Load
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  CNN VISUALIZATION")
print("=" * 70)
print("\n[1/7] Loading results...")

cnn_ic = _load(CNN_DIR / "results" / "cnn_ic_series.csv",         "CNN")
tr_ic  = _load(TR_DIR  / "results" / "transformer_ic_series.csv", "Transformer")
rid_ic = _load(RIDGE_DIR / "ridge_ic_results.csv",                 "Ridge")
lgb_ic = _load(LGBM_DIR  / "lgbm_fixed_ic.csv",                   "LightGBM")
mlp_ic = _load(MLP_DIR   / "mlp_ic_results.csv",                  "MLP")
rf_ic  = _load(RF_DIR    / "rf_ic_results.csv",                   "Random Forest")

if cnn_ic is None:
    print("  ✗ CNN results not found — run run_cnn.py first.")
    raise SystemExit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  Plot 1: IC Time Series
# ─────────────────────────────────────────────────────────────────────────────

print("[2/7] IC Time Series...")
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(cnn_ic["month"], cnn_ic["ic"],
        label="CNN", color=COLORS["cnn"], lw=2.5, alpha=0.95)
for df_, lbl, col in [(tr_ic, "Transformer", COLORS["transformer"]),
                       (rid_ic, "Ridge", COLORS["ridge"]),
                       (lgb_ic, "LightGBM", COLORS["lgbm"]),
                       (mlp_ic, "MLP", COLORS["mlp"]),
                       (rf_ic, "Random Forest", COLORS["rf"])]:
    if df_ is not None:
        ax.plot(df_["month"], df_["ic"], label=lbl, color=col, lw=1.5, alpha=0.65)
ax.axhline(0, color="black", ls="--", lw=1.2, alpha=0.5)
ax.axhline(cnn_ic["ic"].mean(), color=COLORS["cnn"], ls=":",
           lw=1.5, label=f"CNN mean ({cnn_ic['ic'].mean():.4f})")
ax.set_xlabel("Month", fontsize=12, fontweight="bold")
ax.set_ylabel("Spearman IC", fontsize=12, fontweight="bold")
ax.set_title("CNN vs Baselines — Information Coefficient", fontsize=14, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_ic_timeseries.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  ✓ 01_ic_timeseries.png")

# ─────────────────────────────────────────────────────────────────────────────
#  Plot 2: ICIR Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

print("[3/7] ICIR Comparison...")
models_icir = {"CNN": _icir(cnn_ic)}
for df_, nm in [(tr_ic, "Transformer"), (rid_ic, "Ridge"),
                 (lgb_ic, "LightGBM"), (mlp_ic, "MLP"), (rf_ic, "Random Forest")]:
    if df_ is not None:
        models_icir[nm] = _icir(df_)

def _bar_color(v):
    return "#2ECC71" if v >= 0.5 else ("#F39C12" if v >= 0.3 else "#E74C3C")

names  = list(models_icir.keys())
vals   = [models_icir[n] for n in names]
colors = [_bar_color(v) for v in vals]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(names, vals, color=colors, edgecolor="black", lw=1.2, alpha=0.85)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
ax.axhline(0.5, color="green",  ls="--", lw=1.8, alpha=0.6, label="Good ≥ 0.5")
ax.axhline(0.3, color="orange", ls="--", lw=1.8, alpha=0.6, label="Moderate ≥ 0.3")
ax.set_ylabel("ICIR", fontsize=12, fontweight="bold")
ax.set_title("Model Comparison — ICIR", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_icir_comparison.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  ✓ 02_icir_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
#  Plot 3: IC Distribution
# ─────────────────────────────────────────────────────────────────────────────

print("[4/7] IC Distribution...")
fig, ax = plt.subplots(figsize=(12, 6))
ic_arr = cnn_ic["ic"].values
ax.hist(ic_arr, bins=30, color=COLORS["cnn"], alpha=0.55, density=True, label="CNN")
kde = gaussian_kde(ic_arr)
x   = np.linspace(ic_arr.min() - 0.05, ic_arr.max() + 0.05, 300)
ax.plot(x, kde(x), color=COLORS["cnn"], lw=2.5)
for df_, lbl, col in [(tr_ic, "Transformer", COLORS["transformer"]),
                       (rid_ic, "Ridge", COLORS["ridge"]),
                       (mlp_ic, "MLP", COLORS["mlp"])]:
    if df_ is not None:
        ax.hist(df_["ic"].values, bins=30, color=col, alpha=0.3, density=True, label=lbl)
ax.axvline(0, color="black", ls="--", lw=1.2, alpha=0.5)
ax.set_xlabel("Spearman IC", fontsize=12, fontweight="bold")
ax.set_ylabel("Density", fontsize=12, fontweight="bold")
ax.set_title("IC Distribution — CNN vs Baselines", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_ic_distribution.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  ✓ 03_ic_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
#  Plot 4: Rolling 12-month IC
# ─────────────────────────────────────────────────────────────────────────────

print("[5/7] Rolling IC...")
fig, ax = plt.subplots(figsize=(14, 7))
def _roll(df_):
    return df_.sort_values("month").set_index("month")["ic"].rolling(12, min_periods=1).mean()

r = _roll(cnn_ic)
ax.plot(r.index, r, color=COLORS["cnn"], lw=2.5, label="CNN (12m)")
for df_, lbl, col in [(tr_ic, "Transformer", COLORS["transformer"]),
                       (rid_ic, "Ridge", COLORS["ridge"]),
                       (lgb_ic, "LightGBM", COLORS["lgbm"])]:
    if df_ is not None:
        rr = _roll(df_)
        ax.plot(rr.index, rr, color=col, lw=1.5, alpha=0.7, label=f"{lbl} (12m)")
ax.axhline(0, color="black", ls="--", lw=1.2, alpha=0.5)
ax.set_xlabel("Month", fontsize=12, fontweight="bold")
ax.set_ylabel("Rolling Mean IC", fontsize=12, fontweight="bold")
ax.set_title("Rolling 12-Month IC", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_rolling_ic.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  ✓ 04_rolling_ic.png")

# ─────────────────────────────────────────────────────────────────────────────
#  Plot 5: Branch activation heatmap (CNN-specific interpretability)
# ─────────────────────────────────────────────────────────────────────────────

print("[6/7] Branch activations...")
acts_path = CNN_DIR / "results" / "cnn_filter_activations.csv"
if acts_path.exists():
    acts_df = pd.read_csv(acts_path)
    acts_df.columns = [c.lower() for c in acts_df.columns]
    acts_df["month"] = pd.to_datetime(acts_df["month"])

    branch_cols = [c for c in acts_df.columns if "branch" in c]
    if branch_cols:
        fig, ax = plt.subplots(figsize=(14, 5))
        labels = {"branch_short": "Short (k=3)", "branch_medium": "Medium (k=6)",
                  "branch_long": "Long (k=12)"}
        branch_colors = ["#E74C3C", "#F39C12", "#2ECC71"]
        for col, col_color in zip(branch_cols, branch_colors):
            if col in acts_df.columns:
                ax.plot(acts_df["month"], acts_df[col],
                        label=labels.get(col, col), color=col_color, lw=2, alpha=0.85)
        ax.set_xlabel("Month", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Activation Magnitude", fontsize=12, fontweight="bold")
        ax.set_title("CNN Branch Activation Magnitudes — Short / Medium / Long Scale",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "05_branch_activations.png", dpi=DPI, bbox_inches="tight")
        plt.close()
        print("  ✓ 05_branch_activations.png")
    else:
        print("  ✗ No branch_* columns in activations file")
else:
    print(f"  ✗ {acts_path} not found — skipping activation plot")

# ─────────────────────────────────────────────────────────────────────────────
#  Plot 6: Summary table
# ─────────────────────────────────────────────────────────────────────────────

print("[7/7] Summary table...")
rows = []
for df_, name in [(cnn_ic, "CNN"), (tr_ic, "Transformer"), (rid_ic, "Ridge"),
                   (lgb_ic, "LightGBM"), (mlp_ic, "MLP"), (rf_ic, "Random Forest")]:
    if df_ is not None:
        rows.append({
            "Model":    name,
            "Mean IC":  f"{df_['ic'].mean():.4f}",
            "IC Std":   f"{df_['ic'].std():.4f}",
            "ICIR":     f"{_icir(df_):.4f}",
            "% Pos IC": f"{(df_['ic'] > 0).mean()*100:.1f}%",
            "N Months": str(len(df_)),
        })

tbl = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(12, max(3, len(rows) * 0.7 + 1.5)))
ax.axis("off")
t = ax.table(cellText=tbl.values, colLabels=tbl.columns, loc="center", cellLoc="center")
t.auto_set_font_size(False)
t.set_fontsize(11)
t.scale(1.2, 1.8)
for (r, c), cell in t.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")
    elif r < len(rows) + 1 and tbl.iloc[r-1]["Model"] == "CNN":
        cell.set_facecolor("#FEF3E2")
    elif r % 2 == 0:
        cell.set_facecolor("#F5F5F5")
    cell.set_edgecolor("white")
plt.title("Model Comparison Summary", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_summary_table.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  ✓ 06_summary_table.png")

print("\n" + "=" * 70)
print("  CNN VISUALIZATION COMPLETE")
print("=" * 70)
print(f"  Output: {OUTPUT_DIR}/")
print()
print(tbl.to_string(index=False))
print("=" * 70 + "\n")
