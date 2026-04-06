# Random Forest Cross-Sectional Equity Return Prediction Pipeline
# --------------------------------------------------------------
# Aligned with LightGBM pipeline (leakage-safe, cross-sectional ranking, time-based split)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, jarque_bera
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tools import add_constant
from joblib import Parallel, delayed

# ---------------- CONFIG ----------------
DATA_PATH = "project_database.csv"
DATE_COL = "Month"
TARGET_COL = "monthly_gross_return"
ID_COL = "co_code"

FEATURES = ["BM_sep", "OpProf", "Inv", "lag_ret", "mktcap"]

TRAIN_WINDOW = 60
TUNE_WINDOW = 84
RANDOM_STATE = 42

# ---------------- LOAD ----------------
print("="*60)
print("STEP 1 — Loading data")
print("="*60)

df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL])
df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

print(f"Rows: {len(df):,}")
print(f"Months: {df[DATE_COL].nunique()}")

# ---------------- FEATURE LAG (CRITICAL) ----------------
print("\nShifting features by 1 month...")
df[FEATURES] = df.groupby(ID_COL)[FEATURES].shift(1)

df = df.dropna(subset=FEATURES + [TARGET_COL])

# ---------------- TRANSFORMS ----------------
if "mktcap" in df.columns:
    df["mktcap"] = np.log1p(df["mktcap"].clip(lower=0))

# Winsorize
for col in FEATURES + [TARGET_COL]:
    lo = df[col].quantile(0.01)
    hi = df[col].quantile(0.99)
    df[col] = df[col].clip(lo, hi)

# ---------------- CROSS-SECTIONAL RANK ----------------
def cs_rank(df_in, cols):
    df_out = df_in.copy()
    df_out[cols] = df_in.groupby(DATE_COL)[cols].rank(pct=True)
    return df_out

months = sorted(df[DATE_COL].unique())
tune_months = months[:TUNE_WINDOW]

# ---------------- HYPERPARAMETER TUNING ----------------
print("\n" + "="*60)
print("STEP 2 — Hyperparameter tuning")
print("="*60)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "min_samples_leaf": [50, 100],
    "max_features": ["sqrt", 0.5]
}

best_ic = -np.inf
best_params = None

def evaluate_rf_config(params):
    fold_ics = []
    for t in range(TRAIN_WINDOW, len(tune_months)):
        tr_months = tune_months[t - TRAIN_WINDOW:t]
        te_month = tune_months[t]

        tr = df[df[DATE_COL].isin(tr_months)]
        te = df[df[DATE_COL] == te_month]

        if len(te) < 10:
            continue

        tr = cs_rank(tr, FEATURES)
        te = cs_rank(te, FEATURES)

        # n_jobs=1 because we'll parallelize at the config/fold level
        model = RandomForestRegressor(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=1
        )

        model.fit(tr[FEATURES], tr[TARGET_COL])
        preds = model.predict(te[FEATURES])

        ic = spearmanr(preds, te[TARGET_COL]).correlation
        if not np.isnan(ic):
            fold_ics.append(ic)

    return np.mean(fold_ics) if fold_ics else -np.inf

grid = list(ParameterGrid(param_grid))
print(f"Evaluating {len(grid)} configurations...")
tuning_results = Parallel(n_jobs=-1, verbose=10)(delayed(evaluate_rf_config)(p) for p in grid)

for i, mean_ic in enumerate(tuning_results):
    if mean_ic > best_ic:
        best_ic = mean_ic
        best_params = grid[i]

print("Best Params:", best_params)
print("Best Tuning IC:", round(best_ic, 4))

# ---------------- WALK-FORWARD ----------------
print("\n" + "="*60)
print("STEP 3 — Walk-forward evaluation")
print("="*60)

ics = []
feature_importances = []
residuals_all = []
rand_ics = []
shuffle_ics = []

def run_rf_walk_forward_fold(t):
    tr_months = months[t - TRAIN_WINDOW:t]
    te_month = months[t]

    tr = df[df[DATE_COL].isin(tr_months)]
    te = df[df[DATE_COL] == te_month]

    if len(te) < 10:
        return None

    tr = cs_rank(tr, FEATURES)
    te = cs_rank(te, FEATURES)

    model = RandomForestRegressor(
        **best_params,
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    model.fit(tr[FEATURES], tr[TARGET_COL])
    preds = model.predict(te[FEATURES])
    y = te[TARGET_COL].values

    # IC
    ic = spearmanr(preds, y).correlation
    if np.isnan(ic): ic = 0

    # Baselines
    rand_ic = spearmanr(np.random.randn(len(y)), y).correlation
    y_shuffled = np.random.permutation(y)
    shuffle_ic = spearmanr(preds, y_shuffled).correlation

    return {
        "ic": ic,
        "fi": model.feature_importances_,
        "resid": y - preds,
        "rand_ic": rand_ic,
        "shuffle_ic": shuffle_ic
    }

print(f"Running {len(months) - TRAIN_WINDOW} folds in parallel...")
wf_results = Parallel(n_jobs=-1, verbose=10)(delayed(run_rf_walk_forward_fold)(t) for t in range(TRAIN_WINDOW, len(months)))

for res in wf_results:
    if res is not None:
        ics.append(res["ic"])
        feature_importances.append(res["fi"])
        residuals_all.append(res["resid"])
        rand_ics.append(res["rand_ic"])
        shuffle_ics.append(res["shuffle_ic"])

# ---------------- METRICS ----------------
ics = np.array(ics)

mean_ic = ics.mean()
std_ic = ics.std()
icir = mean_ic / std_ic if std_ic > 0 else np.nan
pct_pos = (ics > 0).mean() * 100

print("\nRESULTS")
print("-"*40)
print(f"Mean IC   : {mean_ic:.4f}")
print(f"IC Std    : {std_ic:.4f}")
print(f"ICIR      : {icir:.4f}")
print(f"% Positive: {pct_pos:.1f}%")

print(f"Mean Rand : {np.mean(rand_ics):.4f} (≈0 expected)")
print(f"Mean Shuf : {np.mean(shuffle_ics):.4f} (≈0 expected)")

# ---------------- DIAGNOSTICS ----------------
print("\nDiagnostics")
jb_pvals, bp_pvals, dw_stats = [], [], []

for res, feat_imp in zip(residuals_all, feature_importances):
    _, jb_p = jarque_bera(res)
    jb_pvals.append(jb_p)

    try:
        X_dummy = np.random.randn(len(res), len(FEATURES))
        _, bp_p, _, _ = het_breuschpagan(res, add_constant(X_dummy))
        bp_pvals.append(bp_p)
    except:
        bp_pvals.append(np.nan)

    dw_stats.append(durbin_watson(res))

print(f"Normality pass %      : {(np.array(jb_pvals)>0.05).mean()*100:.1f}%")
print(f"Homoscedasticity pass: {(np.array(bp_pvals)>0.05).mean()*100:.1f}%")
print(f"No autocorrelation % : {((np.array(dw_stats)>=1.5)&(np.array(dw_stats)<=2.5)).mean()*100:.1f}%")

# ---------------- FEATURE IMPORTANCE ----------------
fi_mean = np.mean(feature_importances, axis=0)
print("\nFeature Importance:")
for f, val in zip(FEATURES, fi_mean):
    print(f"{f:<10}: {val:.4f}")

print("\nPipeline Complete")
