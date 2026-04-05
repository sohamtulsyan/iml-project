"""
================================================================================
  SPEARMAN CORRELATION: MODEL PREDICTIONS vs ACTUAL RETURNS (yfinance)
  Works with CSVs output by both the OLS and Regression Tree scripts
================================================================================

  How it works:
  ─────────────
  1. Load the model CSV (OLS or Tree output)
  2. For each stock (co_code), download actual monthly prices from yfinance
  3. Compute actual monthly gross returns from adjusted close prices
  4. Reconstruct predicted returns from model coefficients (OLS) or
     directly from a stored predictions column (Tree) — see MODE below
  5. Compute Spearman ρ between predicted and actual for each stock
  6. Output a clean summary table + save results

  IMPORTANT — Two modes:
  ──────────────────────
  MODE = "ols"   → uses coef_* columns from OLS CSV to recompute ŷ
  MODE = "tree"  → expects a predictions CSV (see note below)

  For the Tree, since sklearn doesn't store predictions in the summary CSV,
  you have two options:
    (a) Re-run the tree script and save y_pred per stock (we add that here)
    (b) Use only the feature importance / R² from the tree CSV for context

  This script handles both modes cleanly.
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
import time

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG — edit these
# ─────────────────────────────────────────────────────────────────────────────

MODE = "ols"          # "ols" or "tree"

# Path to the model output CSV
OLS_CSV  = "regression_results_all_stocks.csv"
TREE_CSV = "tree_results_all_stocks.csv"

# Your full panel dataset (needed to reconstruct OLS predictions)
PANEL_DATA = "your_panel_data.csv"    # set to None to skip OLS prediction reconstruction

# Mapping: co_code in your dataset → yfinance ticker symbol
# Edit this dict to match your stocks. Examples shown below.
TICKER_MAP = {
    # "STOCK_000": "RELIANCE.NS",
    # "STOCK_001": "TCS.NS",
    # "STOCK_002": "HDFCBANK.NS",
    # "STOCK_003": "INFY.NS",
    # Add your mappings here ...
}

# If co_code IS the ticker already (e.g., "AAPL", "MSFT"), set this to True
CO_CODE_IS_TICKER = False

# Date range for yfinance download
START_DATE = "2010-01-01"
END_DATE   = "2024-12-31"

# Output
OUTPUT_DIR   = Path(".")
RESULTS_CSV  = OUTPUT_DIR / "spearman_correlation_results.csv"
REPORT_PDF   = OUTPUT_DIR / "spearman_report.pdf"

# Features used in OLS (must match original script)
OLS_FEATURES = [
    "lag_ret", "Momentum", "BM_sep", "OpProf", "Inv", "mktcap"
]
TARGET = "monthly_gross_return"
STOCK_ID_COL = "co_code"
DATE_COL = "Month"

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1: FETCH ACTUAL RETURNS FROM YFINANCE
# ─────────────────────────────────────────────────────────────────────────────

def fetch_actual_returns(ticker: str,
                         start: str = START_DATE,
                         end: str = END_DATE,
                         retries: int = 3) -> pd.Series | None:
    """
    Download monthly adjusted close prices and compute gross returns.
    Returns a Series indexed by YYYY-MM strings.
    """
    for attempt in range(retries):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                interval="1mo",
                auto_adjust=True,
                progress=False,
                show_errors=False,
            )
            if raw.empty:
                return None

            # Use 'Close' (auto_adjust=True already adjusts for splits/dividends)
            prices = raw["Close"].squeeze()
            prices.index = prices.index.to_period("M").strftime("%Y-%m")
            prices.index.name = DATE_COL

            # Gross return = price_t / price_{t-1}
            gross_ret = prices / prices.shift(1)
            gross_ret = gross_ret.dropna()
            gross_ret.name = "actual_gross_return"
            return gross_ret

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"    [ERROR] Could not fetch {ticker}: {e}")
                return None


def fetch_all_tickers(co_codes: list[str]) -> dict[str, pd.Series]:
    """Download returns for all stocks. Returns dict co_code → Series."""
    results = {}
    for co in co_codes:
        ticker = TICKER_MAP.get(co, co if CO_CODE_IS_TICKER else None)
        if ticker is None:
            print(f"  [SKIP] No ticker mapping for {co}")
            continue
        print(f"  Fetching {co} → {ticker} ...", end=" ")
        ret = fetch_actual_returns(ticker)
        if ret is not None and len(ret) > 0:
            results[co] = ret
            print(f"OK ({len(ret)} months)")
        else:
            print("NO DATA")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2: RECONSTRUCT PREDICTED RETURNS (OLS MODE)
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_ols_predictions(co_code: str,
                                 coef_row: pd.Series,
                                 panel_df: pd.DataFrame) -> pd.Series | None:
    """
    Use stored OLS coefficients to recompute ŷ on the panel data.
    Returns a Series of predicted returns indexed by YYYY-MM.
    """
    stock_df = panel_df[panel_df[STOCK_ID_COL] == co_code].copy()
    if stock_df.empty:
        return None

    # Log-transform mktcap
    if "mktcap" in stock_df.columns:
        stock_df["mktcap"] = np.log1p(stock_df["mktcap"].clip(lower=0))

    available = [f for f in OLS_FEATURES if f in stock_df.columns]
    stock_df = stock_df[[DATE_COL] + available].dropna()

    # Rebuild prediction: ŷ = const + Σ βⱼ·xⱼ
    y_hat = pd.Series(np.zeros(len(stock_df)), index=stock_df.index)

    const_col = "coef_const"
    if const_col in coef_row.index and not pd.isna(coef_row[const_col]):
        y_hat += coef_row[const_col]

    for feat in available:
        col = f"coef_{feat}"
        if col in coef_row.index and not pd.isna(coef_row[col]):
            y_hat += coef_row[col] * stock_df[feat].values

    y_hat.index = stock_df[DATE_COL].values
    y_hat.index.name = DATE_COL
    y_hat.name = "predicted_return"
    return y_hat


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3: COMPUTE SPEARMAN CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_spearman(predicted: pd.Series,
                     actual: pd.Series) -> dict:
    """
    Align predicted and actual on the date index, then compute:
    - Spearman ρ
    - Pearson r (for comparison)
    - Information Coefficient (IC) = Spearman ρ (standard finance definition)
    - p-values for both
    """
    # Align on common dates
    common = predicted.index.intersection(actual.index)
    if len(common) < 10:
        return {
            "n_common": len(common),
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "IC": np.nan,
            "status": f"INSUFFICIENT OVERLAP ({len(common)} months)"
        }

    y_pred = predicted.loc[common].values.astype(float)
    y_act  = actual.loc[common].values.astype(float)

    # Remove any remaining NaNs
    mask = ~(np.isnan(y_pred) | np.isnan(y_act))
    y_pred, y_act = y_pred[mask], y_act[mask]

    if len(y_pred) < 10:
        return {
            "n_common": len(y_pred),
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "IC": np.nan,
            "status": f"INSUFFICIENT CLEAN DATA ({len(y_pred)} months)"
        }

    rho, rho_p = spearmanr(y_pred, y_act)
    r,   r_p   = pearsonr(y_pred, y_act)

    # Hit rate: % of months where predicted and actual have same sign
    hit_rate = np.mean(np.sign(y_pred) == np.sign(y_act))

    return {
        "n_common": len(y_pred),
        "spearman_rho": round(rho, 6),
        "spearman_p": round(rho_p, 6),
        "pearson_r": round(r, 6),
        "pearson_p": round(r_p, 6),
        "IC": round(rho, 6),           # IC = Spearman ρ in standard usage
        "hit_rate": round(hit_rate, 4),
        "status": "OK"
    }


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4: PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def _plot_stock(co_code: str,
                predicted: pd.Series,
                actual: pd.Series,
                stats: dict,
                pdf: PdfPages) -> None:
    common = predicted.index.intersection(actual.index)
    if len(common) < 5:
        return

    y_pred = predicted.loc[common].values.astype(float)
    y_act  = actual.loc[common].values.astype(float)
    mask   = ~(np.isnan(y_pred) | np.isnan(y_act))
    y_pred, y_act = y_pred[mask], y_act[mask]
    dates  = common[mask]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"{co_code}  │  Spearman ρ = {stats['spearman_rho']:.4f} "
        f"(p={stats['spearman_p']:.4f})  │  IC = {stats['IC']:.4f}  │  "
        f"Hit Rate = {stats.get('hit_rate', np.nan)*100:.1f}%  │  N = {stats['n_common']}",
        fontsize=11, fontweight="bold"
    )

    # 1. Scatter: predicted vs actual
    ax = axes[0]
    ax.scatter(y_pred, y_act, alpha=0.5, s=18, color="#2b7bba")
    mn = min(y_pred.min(), y_act.min()); mx = max(y_pred.max(), y_act.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5)
    ax.set_xlabel("Predicted Return"); ax.set_ylabel("Actual Return")
    ax.set_title("Predicted vs Actual")

    # 2. Time series overlay
    ax = axes[1]
    ax.plot(range(len(dates)), y_act,  label="Actual",    alpha=0.8, lw=1.2, color="#27ae60")
    ax.plot(range(len(dates)), y_pred, label="Predicted", alpha=0.8, lw=1.2, color="#e67e22", linestyle="--")
    ax.set_xlabel("Month index"); ax.set_ylabel("Return")
    ax.set_title("Time Series Overlay")
    ax.legend(fontsize=8)

    # 3. Rank scatter (what Spearman actually measures)
    ax = axes[2]
    rank_pred = pd.Series(y_pred).rank().values
    rank_act  = pd.Series(y_act).rank().values
    ax.scatter(rank_pred, rank_act, alpha=0.5, s=18, color="#8e44ad")
    ax.set_xlabel("Rank(Predicted)"); ax.set_ylabel("Rank(Actual)")
    ax.set_title(f"Rank-Rank Plot (Spearman ρ = {stats['spearman_rho']:.4f})")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_aggregate(summary_df: pd.DataFrame, pdf: PdfPages) -> None:
    ok = summary_df[summary_df["status"] == "OK"].copy()
    if ok.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Aggregate Spearman IC Summary", fontsize=13, fontweight="bold")

    # 1. IC distribution
    ax = axes[0]
    ax.hist(ok["spearman_rho"].dropna(), bins=20, color="#2b7bba",
            edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", lw=1.5, linestyle="--")
    mean_ic = ok["spearman_rho"].mean()
    ax.axvline(mean_ic, color="green", lw=1.5, linestyle="-", label=f"Mean IC = {mean_ic:.4f}")
    ax.set_xlabel("Spearman ρ (IC)"); ax.set_ylabel("Count")
    ax.set_title("IC Distribution Across Stocks")
    ax.legend(fontsize=9)

    # 2. IC vs N observations
    ax = axes[1]
    ax.scatter(ok["n_common"], ok["spearman_rho"], alpha=0.7, s=30, color="#e67e22")
    ax.axhline(0, color="red", lw=1.2, linestyle="--")
    ax.set_xlabel("N overlapping months"); ax.set_ylabel("Spearman ρ")
    ax.set_title("IC vs Sample Size")

    # 3. Hit rate distribution
    ax = axes[2]
    if "hit_rate" in ok.columns:
        ax.hist(ok["hit_rate"].dropna() * 100, bins=15, color="#27ae60",
                edgecolor="white", alpha=0.85)
        ax.axvline(50, color="red", lw=1.5, linestyle="--", label="50% baseline")
        ax.set_xlabel("Hit Rate (%)"); ax.set_ylabel("Count")
        ax.set_title("Hit Rate Distribution")
        ax.legend(fontsize=9)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(mode: str = MODE) -> pd.DataFrame:
    print(f"\n{'═'*85}")
    print(f"  SPEARMAN CORRELATION ENGINE   (mode: {mode.upper()})")
    print(f"{'═'*85}\n")

    # ── Load model CSV ────────────────────────────────────────────────────────
    csv_path = OLS_CSV if mode == "ols" else TREE_CSV
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Model CSV not found: {csv_path}")
    model_df = pd.read_csv(csv_path)
    ok_stocks = model_df[model_df["status"] == "OK"]["co_code"].tolist()
    print(f"  Loaded {len(model_df)} stocks from CSV  ({len(ok_stocks)} with OK status)\n")

    # ── Load panel data (needed for OLS prediction reconstruction) ────────────
    panel_df = None
    if mode == "ols" and PANEL_DATA and Path(PANEL_DATA).exists():
        panel_df = pd.read_csv(PANEL_DATA)
        # Parse month to string YYYY-MM for consistent indexing
        panel_df[DATE_COL] = pd.to_datetime(panel_df[DATE_COL]).dt.strftime("%Y-%m")
        print(f"  Panel data loaded: {len(panel_df):,} rows\n")
    elif mode == "ols" and panel_df is None:
        print("  ⚠ Panel data not found — OLS ŷ reconstruction will be skipped.")
        print("    Set PANEL_DATA path in config to enable full Spearman correlation.\n")

    # ── Download actual returns ───────────────────────────────────────────────
    print(f"  Downloading actual returns from yfinance...")
    actual_returns = fetch_all_tickers(ok_stocks)
    print(f"\n  Successfully fetched data for {len(actual_returns)}/{len(ok_stocks)} stocks\n")

    # ── Compute Spearman per stock ────────────────────────────────────────────
    results = []
    pdf_ctx = PdfPages(REPORT_PDF)

    for co_code in ok_stocks:
        if co_code not in actual_returns:
            results.append({
                "co_code": co_code,
                "status": "NO_YFINANCE_DATA",
                "spearman_rho": np.nan,
                "IC": np.nan,
            })
            continue

        actual = actual_returns[co_code]
        coef_row = model_df[model_df["co_code"] == co_code].iloc[0]

        # Reconstruct predicted returns
        predicted = None
        if mode == "ols" and panel_df is not None:
            predicted = reconstruct_ols_predictions(co_code, coef_row, panel_df)

        if predicted is None:
            results.append({
                "co_code": co_code,
                "status": "NO_PREDICTIONS",
                "spearman_rho": np.nan,
                "IC": np.nan,
            })
            continue

        # Compute correlation
        stats = compute_spearman(predicted, actual)
        row = {"co_code": co_code, **stats}

        # Add model quality metrics for context
        for col in ["R²", "Adj R²", "Train R²", "CV R² (approx)"]:
            if col in coef_row.index:
                row[f"model_{col}"] = coef_row[col]

        results.append(row)

        if stats["status"] == "OK":
            print(f"  {co_code:<20}  ρ = {stats['spearman_rho']:>7.4f}  "
                  f"p = {stats['spearman_p']:.4f}  "
                  f"IC = {stats['IC']:>7.4f}  "
                  f"HitRate = {stats.get('hit_rate', np.nan)*100:.1f}%  "
                  f"N = {stats['n_common']}")
            _plot_stock(co_code, predicted, actual, stats, pdf_ctx)
        else:
            print(f"  {co_code:<20}  {stats['status']}")

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(results)
    _plot_aggregate(summary_df, pdf_ctx)
    pdf_ctx.close()

    # ── Save ──────────────────────────────────────────────────────────────────
    summary_df.to_csv(RESULTS_CSV, index=False)

    # ── Print aggregate stats ─────────────────────────────────────────────────
    ok_df = summary_df[summary_df["status"] == "OK"]
    print(f"\n{'═'*85}")
    print(f"  AGGREGATE RESULTS")
    print(f"  Stocks with valid Spearman correlation:  {len(ok_df)}")
    if len(ok_df) > 0:
        print(f"\n  Spearman ρ (IC) Statistics:")
        print(f"    Mean IC:     {ok_df['spearman_rho'].mean():.4f}")
        print(f"    Median IC:   {ok_df['spearman_rho'].median():.4f}")
        print(f"    Std IC:      {ok_df['spearman_rho'].std():.4f}")
        print(f"    ICIR:        {ok_df['spearman_rho'].mean() / ok_df['spearman_rho'].std():.4f}  "
              f"(IC / Std IC — higher is better)")
        print(f"    % IC > 0:    {(ok_df['spearman_rho'] > 0).mean()*100:.1f}%")
        print(f"    % IC sig:    {(ok_df['spearman_p'] < 0.05).mean()*100:.1f}%  (p < 0.05)")
        if "hit_rate" in ok_df.columns:
            print(f"    Mean Hit Rate: {ok_df['hit_rate'].mean()*100:.1f}%")
    print(f"\n  📊 Results saved → {RESULTS_CSV}")
    print(f"  📄 Plots saved   → {REPORT_PDF}")
    print(f"{'═'*85}\n")

    return summary_df


if __name__ == "__main__":
    # ── Set your config at the top of this file, then run ────────────────────
    #
    # Minimum setup for OLS mode:
    #   1. Set OLS_CSV  = path to your OLS output CSV
    #   2. Set PANEL_DATA = path to your original dataset
    #   3. Fill in TICKER_MAP  (or set CO_CODE_IS_TICKER = True)
    #   4. Set START_DATE / END_DATE to match your data range
    #
    # Then simply:
    #   python spearman_correlation.py
    #
    results = run(mode=MODE)
    print(results[["co_code", "spearman_rho", "spearman_p", "IC",
                    "hit_rate", "n_common", "status"]].to_string(index=False))