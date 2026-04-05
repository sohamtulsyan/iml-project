import pandas as pd

# Load data
df = pd.read_csv("regression_results_all_stocks.csv")

# --- 1. Basic counts ---
total_models = len(df)
ok_models = (df["status"] == "OK").sum()
skipped_models = (df["status"] == "SKIPPED").sum()

# Filter only successful regressions
df_ok = df[df["status"] == "OK"].copy()

# --- 2. Aggregate model performance ---
summary = {
    "Total Models": total_models,
    "OK Models": ok_models,
    "Skipped Models": skipped_models,
    
    "Avg R2": df_ok["R²"].mean(),
    "Median R2": df_ok["R²"].median(),
    "Avg Adj R2": df_ok["Adj R²"].mean(),
    
    "Avg F-stat": df_ok["F-stat"].mean(),
    "Significant Models (p<0.05)": (df_ok["F p-value"] < 0.05).mean()
}

# --- 3. Coefficient significance (factor-level) ---
factors = [
    "lag_ret", "Momentum", "BM_sep", 
    "OpProf", "Inv", "mktcap"
]

coef_summary = {}

for factor in factors:
    coef_col = f"coef_{factor}"
    pval_col = f"pval_{factor}"
    
    coef_summary[factor] = {
        "Avg Coef": df_ok[coef_col].mean(),
        "Significant (%)": (df_ok[pval_col] < 0.05).mean()
    }

# --- 4. Diagnostics ---
diagnostics = {
    "Avg Pre Fails": df_ok["pre_fails"].mean(),
    "Avg Post Fails": df_ok["post_fails"].mean(),
    "Avg Pre Warns": df_ok["pre_warns"].mean(),
    "Avg Post Warns": df_ok["post_warns"].mean(),
}

# --- 5. Print results ---
print("\n=== MODEL SUMMARY ===")
for k, v in summary.items():
    print(f"{k}: {v}")

print("\n=== FACTOR SUMMARY ===")
for factor, stats in coef_summary.items():
    print(f"\n{factor}")
    for k, v in stats.items():
        print(f"  {k}: {v}")

print("\n=== DIAGNOSTICS ===")
for k, v in diagnostics.items():
    print(f"{k}: {v}")