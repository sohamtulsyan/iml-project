
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-muted')
sns.set_palette("viridis")

results_dir = Path("results")
backtest_dir = results_dir / "backtest"
ens_dir = results_dir / "ensemble"

def generate_plots():
    # 1. Cumulative Returns Comparison
    models = ["lightgbm", "random_forest", "ridge", "simple_average", "meta_learner"]
    plt.figure(figsize=(12, 7))
    
    for model in models:
        # Check standard and ensemble paths
        p = backtest_dir / f"{model}_backtest_results.parquet"
        if not p.exists():
            p = ens_dir / model / "backtest" / f"{model}_backtest_results.parquet"
            
        if p.exists():
            df = pd.read_parquet(p)
            plt.plot(df["Month"], df["cumulative_net"], label=f"{model.upper()}")
            
    plt.title("Portfolio Equity Curves (Comparison)", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return (Net)", fontsize=12)
    plt.yscale('log') # Log scale for long-term growth
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(results_dir / "equity_curves_comparison.png", dpi=300)
    print(f"✓ Saved {results_dir / 'equity_curves_comparison.png'}")

    # 2. Buy/Sell Signal Dashboard (Latest Month)
    # We'll use LightGBM as the representative "Best Model"
    pos_path = backtest_dir / "lightgbm_positions.parquet"
    if pos_path.exists():
        df_pos = pd.read_parquet(pos_path)
        latest_month = df_pos["Month"].max()
        m_pos = df_pos[df_pos["Month"] == latest_month]
        
        longs = m_pos[m_pos["leg"] == "long"].head(10).copy()
        shorts = m_pos[m_pos["leg"] == "short"].head(10).copy()
        longs["co_code"] = longs["co_code"].astype(str)
        shorts["co_code"] = shorts["co_code"].astype(str)
        
        # Plotting the "Signal Conviction"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Longs
        sns.barplot(x="weight", y="co_code", data=longs, ax=ax1, palette="Greens_d", hue="co_code", legend=False)
        ax1.set_title(f"Top 10 LONG Positions ({latest_month.date()})", fontsize=14, color='green')
        ax1.set_xlabel("Portfolio Weight (%)")
        ax1.set_ylabel("Stock (co_code)")
        
        # Shorts
        sns.barplot(x="weight", y="co_code", data=shorts, ax=ax2, palette="Reds_r", hue="co_code", legend=False)
        ax2.set_title(f"Top 10 SHORT Positions ({latest_month.date()})", fontsize=14, color='red')
        ax2.set_xlabel("Portfolio Weight (%)")
        ax2.set_ylabel("")
        
        plt.suptitle("Model Conviction Dashboard: Highest & Lowest Predicted Returns", fontsize=18, y=1.05)
        plt.tight_layout()
        plt.savefig(results_dir / "buy_sell_dashboard.png", dpi=300)
        print(f"✓ Saved {results_dir / 'buy_sell_dashboard.png'}")

    # 3. IC Correlation Heatmap (How similar are the signals?)
    ic_data = {}
    for model in ["ridge", "lightgbm", "random_forest", "mlp", "transformer"]:
        p = results_dir / model / f"{model}_ic_series.csv"
        if p.exists():
            df = pd.read_csv(p)
            ic_data[model] = df.set_index("Month")["IC"]
            
    if ic_data:
        ic_df = pd.DataFrame(ic_data).dropna()
        plt.figure(figsize=(10, 8))
        sns.heatmap(ic_df.corr(), annot=True, cmap="coolwarm", center=0)
        plt.title("Model Signal Correlation Matrix", fontsize=16)
        plt.tight_layout()
        plt.savefig(results_dir / "signal_correlations.png", dpi=300)
        print(f"✓ Saved {results_dir / 'signal_correlations.png'}")

if __name__ == "__main__":
    generate_plots()
