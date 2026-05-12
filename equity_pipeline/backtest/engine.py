"""
backtest/engine.py — BacktestEngine
Converts positions → P&L → performance tearsheet.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from ..shared.metrics import compute_portfolio_metrics


class BacktestEngine:
    """
    Input:  results/backtest/{name}_positions.parquet
    Output: results/backtest/{name}_backtest_results.parquet
            results/backtest/{name}_backtest_summary.json
    """

    def __init__(self, cfg):
        self.cfg      = cfg
        self.date_col = cfg.date_col
        self.id_col   = cfg.id_col

    def run(self, positions_path: Path, source_name: str) -> dict:
        print(f"\n[Backtest] Running {source_name}...")
        pos = pd.read_parquet(positions_path, engine="pyarrow")
        months = sorted(pos[self.date_col].unique())

        results = []
        prev_pos = None

        for month in months:
            m_pos = pos[pos[self.date_col] == month]
            gross = float((m_pos["weight"] * m_pos["fwd_return"]).sum())

            # Transaction costs
            tc = 0.0
            if prev_pos is not None:
                from .portfolio import PortfolioConstructor
                pc = PortfolioConstructor(self.cfg)
                tc = pc.apply_transaction_costs(m_pos, prev_pos)

            net = gross - tc

            long_leg  = m_pos[m_pos["leg"] == "long"]
            short_leg = m_pos[m_pos["leg"] == "short"]
            long_ret  = float((long_leg["weight"] * long_leg["fwd_return"]).sum()) if len(long_leg) else 0.0
            short_ret = float((short_leg["weight"] * short_leg["fwd_return"]).sum()) if len(short_leg) else 0.0

            results.append({
                self.date_col:       month,
                "gross_return":      gross,
                "transaction_cost":  tc,
                "net_return":        net,
                "long_leg_return":   long_ret,
                "short_leg_return":  short_ret,
                "n_long":            int((m_pos["leg"] == "long").sum()),
                "n_short":           int((m_pos["leg"] == "short").sum()),
            })
            prev_pos = m_pos

        df = pd.DataFrame(results)
        df["cumulative_net"] = (1 + df["net_return"]).cumprod()
        df["rolling_sharpe_12m"] = df["net_return"].rolling(12, min_periods=12).apply(
            lambda r: np.mean(r) / np.std(r, ddof=1) * np.sqrt(12) if np.std(r, ddof=1) > 0 else np.nan
        )

        net_arr = df["net_return"].values
        port    = compute_portfolio_metrics(net_arr)

        # Long/short attribution
        long_ann  = compute_portfolio_metrics(df["long_leg_return"].values)["annualized_return"]
        short_ann = compute_portfolio_metrics(df["short_leg_return"].values)["annualized_return"]
        tc_drag   = compute_portfolio_metrics(df["transaction_cost"].values)["annualized_return"]

        summary = {
            "source":              source_name,
            **port,
            "long_return":         round(long_ann, 6),
            "short_return":        round(short_ann, 6),
            "tc_drag_annualized":  round(tc_drag, 6),
            "n_months":            len(df),
            "long_pct_used":       self.cfg.long_pct,
            "short_pct_used":      self.cfg.short_pct,
            "tc_bps":              self.cfg.transaction_cost_bps,
        }

        # Print
        print(f"\n  Backtest — {source_name}")
        print(f"  Ann. Return : {port['annualized_return']:+.2%}")
        print(f"  Ann. Vol    :  {port['annualized_vol']:.2%}")
        print(f"  Sharpe      :  {port['sharpe_ratio']:.3f}")
        print(f"  Max DD      :  {port['max_drawdown']:.2%}")
        print(f"  Calmar      :  {port['calmar_ratio']:.3f}")

        # Save
        output_dir = Path(self.cfg.results_dir) / "backtest"
        output_dir.mkdir(parents=True, exist_ok=True)
        res_path = output_dir / f"{source_name}_backtest_results.parquet"
        df.to_parquet(res_path, engine="pyarrow", compression="snappy", index=False)
        sum_path = output_dir / f"{source_name}_backtest_summary.json"
        with open(sum_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  ✓ {res_path}")
        print(f"  ✓ {sum_path}")
        return summary
