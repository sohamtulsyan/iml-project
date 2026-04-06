"""
LightGBM Factor Investing Pipeline — FIXED VERSION
====================================================
Fixes applied vs original:
  1. Removed lag_mv (near-duplicate of mktcap → VIF = 204)
  2. Fixed hyperparameter tuning window (TUNE_WINDOW > TRAIN_WINDOW)
  3. Added leakage audit: verifies lag_ret is a true lag of the target
  4. Added log-transform of mktcap (highly skewed, skew=5.5)
  5. Added cross-sectional rank normalisation of features (standard
     in factor investing — removes time-varying scale differences)
  6. Added IC sanity gate: flags and investigates if mean IC > 0.15
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

from scipy.stats import spearmanr, jarque_bera
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from joblib import Parallel, delayed

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────
DATA_PATH    = "project_database.csv"
DATE_COL     = "Month"
TARGET_COL   = "monthly_gross_return"
ID_COL       = "co_code"
SIZE_COL     = "Size_Label"

# FIX 1: lag_mv removed — it is a near-duplicate of mktcap (VIF ~204).
# Keeping both adds no information and breaks the VIF check.
# FIX: Momentum temporarily removed for leakage test
FEATURES = [
    "BM_sep",
    "OpProf",
    "Inv",
    "lag_ret",
    "mktcap",       # log-transformed in code
]

# FIX 2: TUNE_WINDOW must be > TRAIN_WINDOW, otherwise the tuning loop
# range(TRAIN_WINDOW, len(tune_months)) is empty and best_params = {}.
TRAIN_WINDOW  = 60    # months of history used to train each fold
TUNE_WINDOW   = 84    # months used for hyperparameter tuning (must be > TRAIN_WINDOW)
RANDOM_STATE  = 42
OUTPUT_PREFIX = "lgbm_fixed"

# IC sanity threshold — anything above this triggers a leakage investigation
IC_SANITY_THRESHOLD = 0.15

# ─────────────────────────────────────────────────────────────
# 2. LOAD & BASIC CLEAN
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — Loading data")
print("=" * 65)

df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL])
df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

print(f"  Rows loaded      : {len(df):,}")
print(f"  Date range       : {df[DATE_COL].min()} → {df[DATE_COL].max()}")
print(f"  Unique companies : {df[ID_COL].nunique():,}")
print(f"  Unique months    : {df[DATE_COL].nunique():,}")

# ─────────────────────────────────────────────────────────────
# 2.5 FIX 2 — FEATURE LAG (PREVENT LOOK-AHEAD BIAS)
#     Shift features by 1 month per company. Predict t+1 returns using t information.
# ─────────────────────────────────────────────────────────────
print("\n  Shifting features by 1 month to prevent look-ahead bias...")
df[FEATURES] = df.groupby(ID_COL)[FEATURES].shift(1)

# ─────────────────────────────────────────────────────────────
# 3. LEAKAGE AUDIT
#    FIX 3: Verify lag_ret is a genuine lag of the target.
#    If corr(lag_ret_t, target_t) is very high (e.g. > 0.8),
#    lag_ret is likely the same-period return — remove it.
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — Leakage audit")
print("=" * 65)

# Check 1: correlation between lag_ret and the target in the SAME row
if "lag_ret" in df.columns:
    corr_same = df[["lag_ret", TARGET_COL]].dropna().corr().iloc[0, 1]
    print(f"  Corr(lag_ret[t], target[t])        = {corr_same:.4f}")

    # Check 2: is lag_ret[t] ≈ target[t-1] for the same stock?
    df_sorted = df.sort_values([ID_COL, DATE_COL])
    df_sorted["_prev_target"] = df_sorted.groupby(ID_COL)[TARGET_COL].shift(1)
    mask = df_sorted["lag_ret"].notna() & df_sorted["_prev_target"].notna()
    corr_lagged = df_sorted.loc[mask, ["lag_ret", "_prev_target"]].corr().iloc[0, 1]
    print(f"  Corr(lag_ret[t], target[t-1])      = {corr_lagged:.4f}")
    df.drop(columns=["_prev_target"], errors="ignore", inplace=True)

    if abs(corr_same) > 0.5:
        print("\n  *** LEAKAGE WARNING ***")
        print(f"  lag_ret[t] is highly correlated with target[t] ({corr_same:.4f}).")
        print("  This means lag_ret is not a true 1-month lag.")
        print("  ACTION: Dropping lag_ret from features to prevent leakage.\n")
        FEATURES = [f for f in FEATURES if f != "lag_ret"]
    elif abs(corr_lagged) > 0.5:
        print("  lag_ret looks correctly lagged (high corr with prior target).")
        print("  No leakage detected.\n")
    else:
        print("  lag_ret has low correlation with both t and t-1 target.")
        print("  Treating as potentially noisy but not leaking — keeping.\n")

# Check 3: Momentum leakage — similar check
if "Momentum" in df.columns:
    corr_mom = df[["Momentum", TARGET_COL]].dropna().corr().iloc[0, 1]
    print(f"  Corr(Momentum[t], target[t])       = {corr_mom:.4f}")
    if abs(corr_mom) > 0.5:
        print("  *** LEAKAGE WARNING: Momentum dropped ***")
        FEATURES = [f for f in FEATURES if f != "Momentum"]

print(f"\n  Features after leakage audit: {FEATURES}")

# ─────────────────────────────────────────────────────────────
# 4. FIX 4 — Log-transform mktcap
#    Raw mktcap: mean=38k, std=137k, skew=5.6 → log compresses this
# ─────────────────────────────────────────────────────────────
if "mktcap" in df.columns:
    df["mktcap"] = np.log1p(df["mktcap"].clip(lower=0))
    print(f"\n  mktcap log-transformed.")
    print(f"  Post-transform: mean={df['mktcap'].mean():.3f}, "
          f"std={df['mktcap'].std():.3f}, skew={df['mktcap'].skew():.3f}")

# ─────────────────────────────────────────────────────────────
# 5. MISSING DATA
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3 — Missing data")
print("=" * 65)

all_cols = FEATURES + [TARGET_COL]
miss = df[all_cols].isnull().mean() * 100
print(miss.round(2).to_string())

before = len(df)
df = df.dropna(subset=all_cols)
print(f"\n  Rows dropped : {before - len(df):,}  |  Remaining : {len(df):,}")

# ─────────────────────────────────────────────────────────────
# 6. WINSORISE at 1/99 pct
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4 — Winsorising at 1/99 pct")
print("=" * 65)

for col in all_cols:
    lo = df[col].quantile(0.01)
    hi = df[col].quantile(0.99)
    n  = ((df[col] < lo) | (df[col] > hi)).sum()
    df[col] = df[col].clip(lo, hi)
    print(f"  {col:<35} clipped {n:>5} rows")

# ─────────────────────────────────────────────────────────────
# 7. VIF CHECK
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5 — VIF (multicollinearity)")
print("=" * 65)

X_vif = pd.DataFrame(
    StandardScaler().fit_transform(df[FEATURES]),
    columns=FEATURES
)
vif_df = pd.DataFrame({
    "Feature": FEATURES,
    "VIF": [variance_inflation_factor(X_vif.values, i)
            for i in range(len(FEATURES))]
})
print(vif_df.round(3).to_string(index=False))

high = vif_df[vif_df["VIF"] > 10]
if not high.empty:
    print(f"\n  WARNING: {len(high)} feature(s) with VIF > 10")
    print(high.to_string(index=False))
else:
    print("\n  All VIF < 10 — multicollinearity acceptable.")

fig, ax = plt.subplots(figsize=(9, 4))
colors = ["#E24B4A" if v > 10 else "#378ADD" for v in vif_df["VIF"]]
ax.barh(vif_df["Feature"], vif_df["VIF"], color=colors)
ax.axvline(10, color="red", linestyle="--", label="VIF=10 threshold")
ax.set_title("VIF — Multicollinearity Check")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_vif.png", dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────
# 8. FEATURE DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6 — Feature distributions")
print("=" * 65)

dist = pd.DataFrame({
    "Feature" : FEATURES,
    "Mean"    : [df[f].mean() for f in FEATURES],
    "Std"     : [df[f].std()  for f in FEATURES],
    "Skew"    : [df[f].skew() for f in FEATURES],
    "Kurtosis": [df[f].kurt() for f in FEATURES],
})
print(dist.round(4).to_string(index=False))

n_feat = len(FEATURES)
ncols  = min(3, n_feat)
nrows  = int(np.ceil(n_feat / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = np.array(axes).flatten()
for i, feat in enumerate(FEATURES):
    axes[i].hist(df[feat], bins=40, color="#378ADD", edgecolor="white", alpha=0.8)
    axes[i].set_title(feat)
for j in range(i + 1, len(axes)):
    axes[j].axis("off")
plt.suptitle("Feature Distributions (post winsorisation & log-transform)", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_feature_distributions.png", dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────
# 9. FIX 5 — Cross-sectional rank normalisation helper
#    Standard in factor investing: within each month, convert
#    raw feature values to uniform [0,1] ranks across stocks.
#    This removes time-varying scale and makes features
#    comparable across different market regimes.
# ─────────────────────────────────────────────────────────────

def cs_rank(df_in: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Vectorized cross-sectional percentile ranks within each month."""
    df_out = df_in.copy()
    df_out[cols] = df_in.groupby(DATE_COL)[cols].rank(pct=True)
    return df_out

# ─────────────────────────────────────────────────────────────
# 10. HYPERPARAMETER TUNING — FIX 2 applied
#     TUNE_WINDOW (84) > TRAIN_WINDOW (60), so the loop
#     range(60, 84) = 24 valid folds to tune on.
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7 — Hyperparameter tuning")
print(f"         Using first {TUNE_WINDOW} months  |  "
      f"Train window = {TRAIN_WINDOW} months  |  "
      f"Tuning folds = {TUNE_WINDOW - TRAIN_WINDOW}")
print("=" * 65)

months      = sorted(df[DATE_COL].unique())
tune_months = months[:TUNE_WINDOW]

param_grid = {
    "n_estimators"     : [100, 200],
    "max_depth"        : [2, 3],
    "num_leaves"       : [3, 7],
    "min_child_samples": [100, 200],
    "learning_rate"    : [0.05],
}

n_configs   = len(list(ParameterGrid(param_grid)))
print(f"  Configs to evaluate : {n_configs}")
print(f"  Folds per config    : {TUNE_WINDOW - TRAIN_WINDOW}\n")

best_ic     = -np.inf
best_params = {}

def evaluate_config(params):
    fold_ics = []
    for t in range(TRAIN_WINDOW, len(tune_months)):
        tr_months = tune_months[t - TRAIN_WINDOW : t]
        te_month  = tune_months[t]

        tr = df[df[DATE_COL].isin(tr_months)].copy()
        te = df[df[DATE_COL] == te_month].copy()

        if len(te) < 10:
            continue

        # Using vectorized ranking
        tr = cs_rank(tr, FEATURES)
        te = cs_rank(te, FEATURES)

        # FIX 1: Remove StandardScaler (redundant after cross-sectional rank)
        Xtr = tr[FEATURES]
        Xte = te[FEATURES]

        m = lgb.LGBMRegressor(
            **params,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            reg_lambda       = 1.0,
            n_jobs           = 1,  # use 1 thread per model to avoid oversubscription
            random_state     = RANDOM_STATE,
            verbose          = -1,
        )
        m.fit(Xtr, tr[TARGET_COL].values)
        ic_val = spearmanr(m.predict(Xte), te[TARGET_COL].values).correlation
        if not np.isnan(ic_val):
            fold_ics.append(ic_val)
    
    return np.mean(fold_ics) if fold_ics else -np.inf

# Parallelize over hyperparameter configurations with progress feedback
grid = list(ParameterGrid(param_grid))
results_tuning = Parallel(n_jobs=-1, verbose=10)(delayed(evaluate_config)(p) for p in grid)

for i, mean_ic in enumerate(results_tuning):
    if mean_ic > best_ic:
        best_ic     = mean_ic
        best_params = grid[i]

if not best_params:
    print("  WARNING: No valid params found — using defaults.")
    best_params = {
        "n_estimators": 100, "max_depth": 3, "num_leaves": 7,
        "min_child_samples": 100, "learning_rate": 0.05
    }

print(f"  Best params  : {best_params}")
print(f"  Best mean IC : {best_ic:.4f}")

# ─────────────────────────────────────────────────────────────
# 11. WALK-FORWARD EVALUATION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 8 — Walk-forward evaluation")
print("=" * 65)

results          = []
feature_imp_rows = []
residual_records = []

def run_walk_forward_fold(t):
    tr_months = months[t - TRAIN_WINDOW : t]
    te_month  = months[t]

    tr = df[df[DATE_COL].isin(tr_months)].copy()
    te = df[df[DATE_COL] == te_month].copy()

    if len(te) < 10:
        return None

    # Cross-sectional rank normalisation
    tr = cs_rank(tr, FEATURES)
    te = cs_rank(te, FEATURES)

    # FIX 1: Remove StandardScaler
    Xtr  = tr[FEATURES]
    Xte  = te[FEATURES]
    ytr  = tr[TARGET_COL].values
    yte  = te[TARGET_COL].values

    model = lgb.LGBMRegressor(
        **best_params,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        reg_lambda       = 1.0,
        n_jobs           = 1,
        random_state     = RANDOM_STATE,
        verbose          = -1,
    )
    model.fit(Xtr, ytr)
    preds     = model.predict(Xte)
    residuals = yte - preds
    ic        = spearmanr(preds, yte).correlation

    # FIX 4 & 7: Noise and Shuffle Check
    rand_ic = spearmanr(np.random.randn(len(yte)), yte).correlation
    yte_shuffled = np.random.permutation(yte)
    ic_shuffle = spearmanr(preds, yte_shuffled).correlation

    return {
        "month"    : te_month,
        "ic"       : ic if not np.isnan(ic) else 0.0,
        "n_stocks" : len(yte),
        "fi"       : model.feature_importances_,
        "rand_ic"  : rand_ic,
        "shuffle_ic": ic_shuffle,
        "residual" : {
            "month": te_month, "residuals": residuals,
            "preds": preds,    "Xte": Xte,
        }
    }

# Parallelize walk-forward evaluation with progress feedback
print(f"  Running {len(months) - TRAIN_WINDOW} folds in parallel...")
wf_results = Parallel(n_jobs=-1, verbose=10)(delayed(run_walk_forward_fold)(t) for t in range(TRAIN_WINDOW, len(months)))

for res in wf_results:
    if res is not None:
        results.append({
            "month": res["month"], 
            "ic": res["ic"], 
            "n_stocks": res["n_stocks"],
            "rand_ic": res["rand_ic"],
            "shuffle_ic": res["shuffle_ic"]
        })
        feature_imp_rows.append(res["fi"])
        residual_records.append(res["residual"])

results_df = pd.DataFrame(results)
fi_df      = pd.DataFrame(feature_imp_rows, columns=FEATURES)

mean_ic  = results_df["ic"].mean()
mean_rand = results_df["rand_ic"].mean()
mean_shuf = results_df["shuffle_ic"].mean()

std_ic   = results_df["ic"].std()
icir     = mean_ic / std_ic if std_ic > 0 else np.nan
pct_pos  = (results_df["ic"] > 0).mean() * 100
n_months = len(results_df)

print(f"\n  Mean IC   : {mean_ic:.4f}")
print(f"  Mean Rand : {mean_rand:.4f} (expected ~0)")
print(f"  Mean Shuf : {mean_shuf:.4f} (expected ~0)")
print(f"  IC Std    : {std_ic:.4f}")
print(f"  ICIR      : {icir:.4f}")
print(f"  % Positive: {pct_pos:.1f}%")
print(f"  N months  : {n_months}")

# ─────────────────────────────────────────────────────────────
# 12. FIX 6 — IC SANITY GATE
#    If mean IC > threshold, surface the most likely cause
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 9 — IC Sanity check")
print("=" * 65)

if mean_ic > IC_SANITY_THRESHOLD:
    print(f"  *** IC SANITY ALERT: Mean IC = {mean_ic:.4f} > {IC_SANITY_THRESHOLD} ***")
    print("  Investigating most likely causes:\n")

    # Which feature has the highest mean Spearman correlation with target?
    for feat in FEATURES:
        r, p = spearmanr(df[feat].dropna(), df.loc[df[feat].notna(), TARGET_COL])
        print(f"  Spearman('{feat}', target) = {r:.4f}  p = {p:.4e}")

    print("\n  If any feature shows |r| > 0.5 with the target globally,")
    print("  that feature is likely leaking. Remove it and re-run.")
else:
    print(f"  IC = {mean_ic:.4f} is within plausible range. No leakage detected.")

# ─────────────────────────────────────────────────────────────
# 13. POST-ESTIMATION DIAGNOSTICS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 10 — Post-estimation diagnostics")
print("=" * 65)

jb_pvals, bp_pvals, dw_stats = [], [], []

for rec in residual_records:
    res = rec["residuals"]
    X   = rec["Xte"]

    _, jb_p = jarque_bera(res)
    jb_pvals.append(jb_p)

    try:
        _, bp_p, _, _ = het_breuschpagan(res, add_constant(X))
        bp_pvals.append(bp_p)
    except Exception:
        bp_pvals.append(np.nan)

    dw_stats.append(durbin_watson(res))

ALPHA   = 0.05
jb_pass = (np.array(jb_pvals) > ALPHA).mean() * 100
bp_pass = (np.array(bp_pvals) > ALPHA).mean() * 100
dw_pass = ((np.array(dw_stats) >= 1.5) & (np.array(dw_stats) <= 2.5)).mean() * 100

print(f"  Normality (JB)          — {jb_pass:.1f}% months pass")
print(f"  Homoscedasticity (BP)   — {bp_pass:.1f}% months pass")
print(f"  No autocorrelation (DW) — {dw_pass:.1f}% months pass")
print("\n  Note: Low JB pass-rate is expected for financial returns (fat tails).")
print("  LightGBM does not require normally distributed residuals.")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, vals, title, xlabel, colour in zip(
    axes,
    [jb_pvals, bp_pvals, dw_stats],
    [f"Normality JB ({jb_pass:.0f}% pass)",
     f"Homoscedasticity BP ({bp_pass:.0f}% pass)",
     f"No Autocorrelation DW ({dw_pass:.0f}% pass)"],
    ["JB p-value", "BP p-value", "DW statistic"],
    ["#4CAF50", "#FF9800", "#9C27B0"],
):
    ax.hist(vals, bins=30, color=colour, edgecolor="white", alpha=0.85)
    ax.axvline(ALPHA if "p-value" in xlabel else 1.5,
               color="red", linestyle="--", linewidth=1.2)
    if "DW" in xlabel:
        ax.axvline(2.5, color="red", linestyle="--", linewidth=1.2)
    ax.set_title(title); ax.set_xlabel(xlabel)
plt.suptitle("Post-Estimation Diagnostics", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_diagnostics.png", dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────
# 14. VISUALISATIONS — IC, Feature Importance
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 11 — Visualisations")
print("=" * 65)

# IC bar + cumulative
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9))
colors = ["#4CAF50" if v > 0 else "#F44336" for v in results_df["ic"]]
ax1.bar(range(len(results_df)), results_df["ic"], color=colors, width=0.8)
ax1.axhline(mean_ic, color="blue", linestyle="--",
            label=f"Mean IC = {mean_ic:.4f}")
ax1.axhline(0, color="black", linewidth=0.8)
ax1.set_title(f"Monthly IC  |  ICIR = {icir:.4f}  |  % Positive = {pct_pos:.1f}%")
ax1.set_ylabel("Spearman IC"); ax1.legend()
ax2.plot(results_df["ic"].cumsum().values, color="purple", linewidth=1.5)
ax2.set_title("Cumulative IC"); ax2.set_xlabel("Month Index"); ax2.set_ylabel("Cumulative IC")
plt.suptitle("LightGBM — Information Coefficient", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_ic.png", dpi=150)
plt.close()

# IC distribution
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(results_df["ic"], bins=35, color="#378ADD", edgecolor="white", alpha=0.85)
ax.axvline(mean_ic, color="red", linestyle="--", label=f"Mean = {mean_ic:.4f}")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("IC Distribution"); ax.set_xlabel("Spearman IC"); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_ic_distribution.png", dpi=150)
plt.close()

# Mean feature importance
mean_fi = fi_df.mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(mean_fi.index, mean_fi.values, color="#378ADD", edgecolor="white")
ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=10)
ax.set_title("Mean Feature Importance (across all walk-forward folds)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_feature_importance_mean.png", dpi=150)
plt.close()

# Feature importance over time
fig, ax = plt.subplots(figsize=(15, 6))
for feat in FEATURES:
    ax.plot(fi_df[feat].values, label=feat, alpha=0.75, linewidth=1.2)
ax.set_title("Feature Importance Over Time")
ax.set_xlabel("Month Index"); ax.set_ylabel("Importance")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_feature_importance_time.png", dpi=150)
plt.close()

print(f"  All plots saved with prefix: {OUTPUT_PREFIX}_*")

# ─────────────────────────────────────────────────────────────
# 16. COMPARISON TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 13 — Model comparison")
print("=" * 65)

comparison = pd.DataFrame({
    "Model"     : ["Ridge",  "CART",   "LightGBM (fixed)"],
    "Mean IC"   : [0.0344,   0.0304,   round(mean_ic, 4)],
    "IC Std"    : [0.1374,   0.1186,   round(std_ic,  4)],
    "ICIR"      : [0.2504,   0.2561,   round(icir,    4)],
    "% Positive": [61.6,     None,     round(pct_pos, 1)],
    "Months"    : [279,      279,      n_months],
})
print(comparison.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")
tbl = ax.table(cellText=comparison.values, colLabels=comparison.columns,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.3, 2.0)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1C7293"); cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#E6F1FB")
    cell.set_edgecolor("white")
plt.title("Model Comparison — Ridge vs CART vs LightGBM", fontsize=13, pad=20)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_comparison_table.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────
# 17. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PIPELINE COMPLETE")
print("=" * 65)
print(f"  Mean IC    : {mean_ic:.4f}")
print(f"  IC Std     : {std_ic:.4f}")
print(f"  ICIR       : {icir:.4f}")
print(f"  % Positive : {pct_pos:.1f}%")
print(f"  N months   : {n_months}")
print(f"  Best params: {best_params}")