import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from pathlib import Path
import time

# =========================
# CONFIG
# =========================
DATA_PATH = "project_database.csv"
FEATURES = ['lag_ret', 'Momentum', 'BM_sep', 'OpProf', 'Inv', 'mktcap']
MIN_OBS = 30

# Hyperparameter grid (Reduced for faster initial run, you can expand later)
PARAM_GRID = {
    'max_depth': [3, 5, 10],
    'min_samples_leaf': [5,10,15,25,50,100],
    'min_samples_split': [50]
}

# =========================
# LOAD DATA
# =========================
print("[1/5] Loading and Preparing Data...")
df = pd.read_csv(DATA_PATH)
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values(['co_code', 'Month'])

# Create next-month return target
df['target'] = df.groupby('co_code')['monthly_gross_return'].shift(-1)

# Drop missing rows
df_model = df.dropna(subset=FEATURES + ['target']).copy()

# Winsorization (robustness)
def winsorize(series, lower=0.01, upper=0.99):
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)

for col in FEATURES + ['target']:
    df_model[col] = winsorize(df_model[col])

# OPTIMIZATION: Pre-group data by Month to avoid repeated filtering
print("[2/5] Pre-grouping data by Month (Optimization)...")
month_groups = {month: group for month, group in df_model.groupby('Month')}
sorted_months = sorted(month_groups.keys())

# =========================
# VALIDATION CHECKS
# =========================
def validate_data(df_chunk):
    if len(df_chunk) < MIN_OBS:
        return False
    if df_chunk['target'].var() < 1e-10:
        return False
    for col in FEATURES:
        if df_chunk[col].var() < 1e-10:
            return False
    return True

# =========================
# TRAIN + IC FUNCTION
# =========================
def run_model(params):
    ICs = []
    
    # We iterate through months once for this specific param set
    for i in range(len(sorted_months) - 1):
        t = sorted_months[i]
        t_next = sorted_months[i+1]

        train = month_groups[t]
        test = month_groups[t_next]

        if not validate_data(train) or not validate_data(test):
            continue

        X_train = train[FEATURES]
        y_train = train['target']

        X_test = test[FEATURES]
        y_test = test['target']

        model = DecisionTreeRegressor(**params, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        
        # Spearman IC
        ic, _ = spearmanr(preds, y_test)
        if not np.isnan(ic):
            ICs.append(ic)

    if not ICs:
        return None

    IC_mean = np.mean(ICs)
    IC_std = np.std(ICs)
    ICIR = IC_mean / IC_std if IC_std > 0 else np.nan

    return {
        'params': params,
        'IC_mean': IC_mean,
        'IC_std': IC_std,
        'ICIR': ICIR,
        'ICs': ICs,
        'N_months': len(ICs)
    }

# =========================
# GRID SEARCH
# =========================
print(f"[3/5] Running Grid Search over {len(list(ParameterGrid(PARAM_GRID)))} combinations...")
results = []
start_time = time.time()

for params in ParameterGrid(PARAM_GRID):
    print(f"  Testing params: {params}", end=" ", flush=True)
    res = run_model(params)
    if res:
        results.append(res)
        print(f"-> Mean IC: {res['IC_mean']:.4f} | ICIR: {res['ICIR']:.4f}")
    else:
        print("-> SKIPPED (No valid data)")

if not results:
    print("Error: No valid models were trained. Check your filters and MIN_OBS.")
    exit(1)

# Convert to DataFrame
summary = pd.DataFrame([
    {**r['params'], 
     'IC_mean': r['IC_mean'], 
     'IC_std': r['IC_std'], 
     'ICIR': r['ICIR'],
     'N_months': r['N_months']}
    for r in results
])

# Find best model (by ICIR for consistency/risk-adjusted returns)
best_model = max(results, key=lambda x: x['ICIR'])

# =========================
# FINAL TRAIN & IMPORTANCE
# =========================
print(f"\n[4/5] Extracting Feature Importance from Best Model...")
# Use last available month for feature importance snapshot
last_month = sorted_months[-2] # Second to last for a valid test pair if needed
train_final = month_groups[last_month]
model_final = DecisionTreeRegressor(**best_model['params'], random_state=42)
model_final.fit(train_final[FEATURES], train_final['target'])

importance = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': model_final.feature_importances_
}).sort_values('Importance', ascending=False)

# =========================
# OUTPUT
# =========================
print("\n[5/5] Saving Results...")
summary.to_csv("tree_model_summary.csv", index=False)

best_ic_df = pd.DataFrame({
    'Month': [sorted_months[i+1] for i in range(len(best_model['ICs']))], # Approximate alignment
    'IC': best_model['ICs']
})
best_ic_df.to_csv("best_tree_ics.csv", index=False)

print("\n" + "="*30)
print("     BEST MODEL SUMMARY")
print("="*30)
print(f"Params:    {best_model['params']}")
print(f"Mean IC:   {best_model['IC_mean']:.4f}")
print(f"IC Std:    {best_model['IC_std']:.4f}")
print(f"ICIR:      {best_model['ICIR']:.4f}")
print(f"Months:    {best_model['N_months']}")
print("\n--- Feature Importance ---")
print(importance.to_string(index=False))

print(f"\nSaved:")
print("- tree_model_summary.csv")
print("- best_tree_ics.csv")
print(f"Total Time: {time.time() - start_time:.2f}s")