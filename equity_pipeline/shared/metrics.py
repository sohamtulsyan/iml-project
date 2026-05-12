"""
shared/metrics.py
=================
Single implementation of ALL evaluation and portfolio metrics.
No model reimplements any of these.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ─────────────────────────────────────────────────────────────────────────────
#  Statistical (IC) metrics
# ─────────────────────────────────────────────────────────────────────────────

def spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank IC. Returns 0.0 on insufficient data or NaN result."""
    if len(y_true) < 5:
        return 0.0
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho) if not np.isnan(rho) else 0.0


def compute_ic_series(ic_values: list[float]) -> dict:
    """
    Canonical summary dict written to every model's _summary.json.
    Returns: {mean_ic, ic_std, icir, pct_positive, n_months}
    """
    arr = np.asarray(ic_values, dtype=float)
    mu  = float(np.mean(arr))
    sd  = float(np.std(arr))
    return {
        "mean_ic":      round(mu, 6),
        "ic_std":       round(sd, 6),
        "icir":         round(mu / sd, 6) if sd > 0 else 0.0,
        "pct_positive": round(float(np.mean(arr > 0)) * 100, 2),
        "n_months":     int(len(arr)),
    }


def print_results_table(
    summary:    dict,
    model_name: str,
    baselines:  dict,
    extra:      dict | None = None,   # portfolio metrics from backtest if available
) -> None:
    """Print standardised results table — same format for every model."""
    width = 60
    print("\n" + "═" * width)
    print(f"  {model_name.upper()} — RESULTS")
    print("═" * width)
    print(f"  Mean IC       : {summary.get('mean_ic', 0):+.6f}")
    print(f"  IC Std        :  {summary.get('ic_std', 0):.6f}")
    print(f"  ICIR          : {summary.get('icir', 0):+.6f}")
    print(f"  % Positive IC :  {summary.get('pct_positive', 0):.2f}%")
    print(f"  Months        :  {summary.get('n_months', 0)}")
    if extra:
        print(f"  Sharpe (ann)  :  {extra.get('sharpe_ratio', '—')}")
        print(f"  Max Drawdown  :  {extra.get('max_drawdown', '—')}")
        print(f"  Ann. Return   :  {extra.get('annualized_return', '—')}")

    print("\n  Baseline Comparison:")
    print(f"  {'Model':<22} {'Mean IC':>9}  {'ICIR':>8}")
    print(f"  {'-' * 42}")
    for name, b in baselines.items():
        marker = " ←" if name.lower() == model_name.lower() else ""
        print(f"  {name:<22} {b['mean_ic']:>9.4f}  {b['icir']:>8.4f}{marker}")
    # Print current model if not in baselines
    if model_name not in baselines:
        print(f"  {model_name + ' (this)':<22} "
              f"{summary.get('mean_ic', 0):>9.4f}  {summary.get('icir', 0):>8.4f}  ←")
    print("═" * width + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Portfolio-level metrics (used by backtest + future RL reward)
# ─────────────────────────────────────────────────────────────────────────────

def annualized_return(returns: np.ndarray, periods_per_year: int = 12) -> float:
    """Compound annualized growth rate."""
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0.0
    total = np.prod(1 + r)
    return float(total ** (periods_per_year / len(r)) - 1)


def annualized_volatility(returns: np.ndarray, periods_per_year: int = 12) -> float:
    """Annualized standard deviation of returns."""
    r = np.asarray(returns, dtype=float)
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def annualized_sharpe(returns: np.ndarray, periods_per_year: int = 12) -> float:
    """Annualized Sharpe ratio (risk-free = 0)."""
    ann_ret = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    return float(ann_ret / ann_vol) if ann_vol > 0 else 0.0


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of cumulative return series."""
    r     = np.asarray(returns, dtype=float)
    cum   = np.cumprod(1 + r)
    peak  = np.maximum.accumulate(cum)
    dd    = (cum - peak) / peak
    return float(dd.min())


def calmar_ratio(returns: np.ndarray, periods_per_year: int = 12) -> float:
    """Annualized return / abs(max drawdown)."""
    ann_ret = annualized_return(returns, periods_per_year)
    mdd     = abs(max_drawdown(returns))
    return float(ann_ret / mdd) if mdd > 0 else 0.0


def sortino_ratio(returns: np.ndarray, periods_per_year: int = 12) -> float:
    """Sharpe using only downside deviation."""
    r         = np.asarray(returns, dtype=float)
    ann_ret   = annualized_return(r, periods_per_year)
    downside  = r[r < 0]
    down_vol  = float(np.std(downside, ddof=1) * np.sqrt(periods_per_year)) if len(downside) > 1 else 0.0
    return float(ann_ret / down_vol) if down_vol > 0 else 0.0


def long_short_return(
    pred_scores:       np.ndarray,
    realized_returns:  np.ndarray,
    top_pct:           float = 0.10,
    bottom_pct:        float = 0.10,
) -> float:
    """
    Long-short return for one month.
    R^LS_t = mean(top_pct returns) - mean(bottom_pct returns).
    Equal-weighted within each leg.
    """
    n     = len(pred_scores)
    ranks = np.argsort(np.argsort(-pred_scores))   # descending rank (0 = highest)
    n_top = max(1, int(np.ceil(n * top_pct)))
    n_bot = max(1, int(np.ceil(n * bottom_pct)))
    long_ret  = float(np.mean(realized_returns[ranks < n_top]))
    short_ret = float(np.mean(realized_returns[ranks >= n - n_bot]))
    return long_ret - short_ret


def compute_portfolio_metrics(
    ls_returns:       np.ndarray,
    periods_per_year: int = 12,
) -> dict:
    """
    Full portfolio summary dict — canonical format used by backtest engine
    AND RL reward logger.
    """
    r = np.asarray(ls_returns, dtype=float)
    return {
        "annualized_return":    round(annualized_return(r, periods_per_year),   6),
        "annualized_vol":       round(annualized_volatility(r, periods_per_year), 6),
        "sharpe_ratio":         round(annualized_sharpe(r, periods_per_year),   6),
        "max_drawdown":         round(max_drawdown(r),                           6),
        "calmar_ratio":         round(calmar_ratio(r, periods_per_year),         6),
        "sortino_ratio":        round(sortino_ratio(r, periods_per_year),        6),
        "pct_positive_months":  round(float(np.mean(r > 0)) * 100,              2),
        "total_return":         round(float(np.prod(1 + r) - 1),                6),
    }


def print_full_comparison(
    ic_summaries:       dict[str, dict],   # {model_name: ic_summary}
    portfolio_summaries: dict[str, dict] | None = None,   # {model_name: portfolio_summary}
) -> pd.DataFrame:
    """
    Print full comparison table — IC + portfolio metrics side by side.
    Returns DataFrame also saved to results/full_comparison.csv.
    """
    rows = []
    for name, ic in ic_summaries.items():
        port = (portfolio_summaries or {}).get(name, {})
        rows.append({
            "Model":      name,
            "Mean IC":    f"{ic.get('mean_ic', 0):+.4f}" if ic else "—",
            "ICIR":       f"{ic.get('icir', 0):+.4f}"   if ic else "—",
            "Sharpe":     f"{port.get('sharpe_ratio', '—'):.2f}" if port else "—",
            "MaxDD":      f"{port.get('max_drawdown', '—'):.1%}"  if port else "—",
            "Ann.Ret":    f"{port.get('annualized_return', '—'):.1%}" if port else "—",
            "Ann.Vol":    f"{port.get('annualized_vol', '—'):.1%}"    if port else "—",
        })

    df = pd.DataFrame(rows)
    width = 82
    print("\n" + "═" * width)
    print("  FULL PIPELINE RESULTS — ALL MODELS")
    print("═" * width)
    print(df.to_string(index=False))
    print(f"\n  Stat. threshold : Mean IC > 0.03, ICIR > 0.5 for reliable signal")
    print(f"  Econ. threshold : Sharpe > 0.8 to be considered implementable")
    print("═" * width + "\n")
    return df
