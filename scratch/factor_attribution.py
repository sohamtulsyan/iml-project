
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-muted')

def run_factor_analysis():
    results_dir = Path("results")
    pos_path = results_dir / "backtest" / "lightgbm_positions.parquet"
    db_path = Path("project_database.csv")
    
    if not pos_path.exists() or not db_path.exists():
        print("Missing data files.")
        return

    # 1. Load Data
    print("[1/4] Loading portfolio positions and market data...")
    pos = pd.read_parquet(pos_path)
    
    # Load only necessary columns from database to save memory
    db = pd.read_csv(db_path, usecols=['co_code', 'Month', 'monthly_gross_return', 'Size_Label', 'BM_Label', 'Mom_Label'])
    db['Month'] = pd.to_datetime(db['Month'])
    
    # 2. CAPM Alpha Calculation
    print("[2/4] Running CAPM Regression...")
    # Calculate Monthly Market Return (convert 1+r to r)
    mkt_ret = (db.groupby('Month')['monthly_gross_return'].mean() - 1).rename('mkt_ret')
    
    # Calculate Monthly Portfolio Return (net)
    # For a long-short portfolio, the (1+r) cancels out: sum(w*(1+r)) where sum(w)=0 is sum(w*r)
    pos_raw = pos.merge(db[['co_code', 'Month', 'monthly_gross_return']], on=['co_code', 'Month'], how='left')
    port_ret = pos_raw.groupby('Month').apply(lambda x: (x['weight'] * x['monthly_gross_return']).sum(), include_groups=False).rename('port_ret')
    
    reg_data = pd.concat([port_ret, mkt_ret], axis=1).dropna()
    
    # Regression: Port_Ret = Alpha + Beta * Mkt_Ret
    X = sm.add_constant(reg_data['mkt_ret'])
    model = sm.OLS(reg_data['port_ret'], X).fit()
    
    alpha = model.params['const']
    beta = model.params['mkt_ret']
    t_alpha = model.tvalues['const']
    
    print(f"  Alpha (Monthly) : {alpha:.4f} (t-stat: {t_alpha:.2f})")
    print(f"  Alpha (Annual)  : {(1+alpha)**12-1:.2%}")
    print(f"  Beta            : {beta:.4f}")

    # 3. Segment Analysis
    print("[3/4] Segmenting performance...")
    # Merge characteristics into positions
    pos_char = pos_raw.merge(db[['co_code', 'Month', 'Size_Label', 'BM_Label', 'Mom_Label']], on=['co_code', 'Month'], how='left')
    
    segments = ['Size_Label', 'BM_Label', 'Mom_Label']
    segment_results = {}
    
    for seg in segments:
        # Calculate mean return contribution per segment
        # We only care about the Long and Short legs
        ls_only = pos_char[pos_char['leg'].isin(['long', 'short'])]
        res = ls_only.groupby(seg).apply(lambda x: (x['weight'] * x['monthly_gross_return']).sum(), include_groups=False)
        segment_results[seg] = res

    # 4. Visualization
    print("[4/4] Generating attribution plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, seg in enumerate(segments):
        res = segment_results[seg].sort_values(ascending=False)
        sns.barplot(x=res.index, y=res.values, ax=axes[i], palette="viridis", hue=res.index, legend=False)
        axes[i].set_title(f"Performance by {seg.replace('_Label', '')}", fontsize=14)
        axes[i].set_ylabel("Avg. Monthly Contribution")
        axes[i].axhline(0, color='black', linewidth=0.8)

    plt.suptitle(f"Portfolio Alpha Attribution: Where is the return coming from?", fontsize=18, y=1.05)
    plt.tight_layout()
    plt.savefig(results_dir / "factor_attribution.png", dpi=300)
    
    # Save Alpha stats
    with open(results_dir / "capm_alpha_summary.json", "w") as f:
        import json
        json.dump({
            "monthly_alpha": float(alpha),
            "annualized_alpha": float((1+alpha)**12-1),
            "beta": float(beta),
            "t_stat_alpha": float(t_alpha),
            "p_value_alpha": float(model.pvalues['const'])
        }, f, indent=2)
    
    print(f"✓ Saved {results_dir / 'factor_attribution.png'}")

if __name__ == "__main__":
    run_factor_analysis()
