"""
backtest/portfolio.py — PortfolioConstructor
Converts monthly pred_score arrays → long/short portfolio weights.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path


class PortfolioConstructor:
    """
    Input:  {model}_test_predictions.parquet  Schema: (co_code, Month, pred_score, fwd_return)
    Output: results/backtest/{name}_positions.parquet
            Schema: (co_code, Month, weight, fwd_return, leg)
    """

    def __init__(self, cfg):
        self.long_pct  = cfg.long_pct
        self.short_pct = cfg.short_pct
        self.tc_bps    = cfg.transaction_cost_bps
        self.id_col    = cfg.id_col
        self.date_col  = cfg.date_col

    def construct(self, predictions_path: Path) -> pd.DataFrame:
        df = pd.read_parquet(predictions_path, engine="pyarrow")
        records = []

        for month, grp in df.groupby(self.date_col):
            n      = len(grp)
            n_long = max(1, int(np.ceil(n * self.long_pct)))
            n_short= max(1, int(np.ceil(n * self.short_pct)))

            ranked = grp.sort_values("pred_score", ascending=False).reset_index(drop=True)

            for i, row in ranked.iterrows():
                if i < n_long:
                    weight, leg = 1.0 / n_long, "long"
                elif i >= n - n_short:
                    weight, leg = -1.0 / n_short, "short"
                else:
                    weight, leg = 0.0, "neutral"

                records.append({
                    self.id_col:   row[self.id_col],
                    self.date_col: month,
                    "weight":      weight,
                    "fwd_return":  row["fwd_return"],
                    "leg":         leg,
                })

        return pd.DataFrame(records)

    def apply_transaction_costs(
        self,
        positions_t:   pd.DataFrame,
        positions_tm1: pd.DataFrame,
    ) -> float:
        """Cost = sum(|w_t - w_{t-1}|) * tc_bps / 10000 (one-way)."""
        merged = positions_t[[self.id_col, "weight"]].merge(
            positions_tm1[[self.id_col, "weight"]].rename(columns={"weight": "weight_prev"}),
            on=self.id_col, how="outer",
        ).fillna(0)
        turnover = (merged["weight"] - merged["weight_prev"]).abs().sum()
        return float(turnover * self.tc_bps / 10_000)

    def run(self, predictions_path: Path, source_name: str,
            output_dir: Path) -> Path:
        print(f"[Portfolio] Constructing positions for {source_name}...")
        pos_df = self.construct(predictions_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{source_name}_positions.parquet"
        pos_df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        print(f"  ✓ {path}  ({len(pos_df):,} rows, "
              f"{pos_df[self.date_col].nunique()} months)")
        return path
