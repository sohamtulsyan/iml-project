import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from statsmodels.stats.stattools import durbin_watson, jarque_bera
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =========================
# CONFIG
# =========================
DATA_PATH = "project_database.csv"
FEATURES = ['lag_ret', 'Momentum', 'BM_sep', 'OpProf', 'Inv', 'mktcap']
MIN_OBS = 30

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values(['co_code', 'Month'])

# Create next-month return target
df['target'] = df.groupby('co_code')['monthly_gross_return'].shift(-1)

df[['co_code','Month','monthly_gross_return','target']].head(10)

# Drop missing rows
df_model = df.dropna(subset=FEATURES + ['target']).copy()

# =========================
# VALIDATION FUNCTIONS
# =========================

def check_variance(df):
    for col in FEATURES:
        if df[col].var() < 1e-10:
            return False, f"Low variance in {col}"
    return True, "OK"


def check_vif(X):
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
    if (vif_data['VIF'] > 10).any():
        return False, vif_data
    return True, vif_data


def regression_diagnostics(X, y):
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    residuals = model.resid

    # Normality
    jb_stat, jb_p, _, _ = jarque_bera(residuals)

    # Heteroscedasticity
    bp_test = het_breuschpagan(residuals, X_const)
    bp_p = bp_test[1]

    # Autocorrelation
    dw_stat = durbin_watson(residuals)

    return {
        'jb_p': jb_p,
        'bp_p': bp_p,
        'dw': dw_stat
    }

# =========================
# MAIN LOOP
# =========================
months = sorted(df_model['Month'].unique())
results = []

for i in range(len(months) - 1):
    t = months[i]
    t_next = months[i+1]

    train = df_model[df_model['Month'] == t]
    test = df_model[df_model['Month'] == t_next]

    if len(train) < MIN_OBS or len(test) < MIN_OBS:
        continue

    X_train = train[FEATURES]
    y_train = train['target']

    # ===== Pre checks =====
    var_ok, var_msg = check_variance(train)
    vif_ok, vif_data = check_vif(X_train)

    if not var_ok or not vif_ok:
        continue

    # ===== Train model =====
    model = LinearRegression()
    model.fit(X_train, y_train)

    # ===== Diagnostics =====
    diag = regression_diagnostics(X_train, y_train)

    # ===== Predict =====
    X_test = test[FEATURES]
    y_test = test['target']

    preds = model.predict(X_test)

    # ===== IC =====
    ic, _ = spearmanr(preds, y_test)

    results.append({
        'Month': t_next,
        'IC': ic,
        'N_train': len(train),
        'N_test': len(test),
        'jb_p': diag['jb_p'],
        'bp_p': diag['bp_p'],
        'dw': diag['dw']
    })

# =========================
# FINAL OUTPUT
# =========================
ic_df = pd.DataFrame(results)

IC_mean = ic_df['IC'].mean()
IC_std = ic_df['IC'].std()
ICIR = IC_mean / IC_std if IC_std != 0 else np.nan

print("Mean IC:", IC_mean)
print("IC Std:", IC_std)
print("ICIR:", ICIR)

IC_mean = ic_df['IC'].mean()
IC_std = ic_df['IC'].std()
ICIR = IC_mean / IC_std if IC_std != 0 else np.nan

print("===== IC SUMMARY =====")
print(f"Mean IC: {IC_mean:.4f}")
print(f"IC Std: {IC_std:.4f}")
print(f"ICIR: {ICIR:.4f}")

print("\n===== MONTHLY ICs =====")
print(ic_df.head())

# Save results
ic_df.to_csv("ic_results.csv", index=False)

print("\nSaved IC results to ic_results.csv")
