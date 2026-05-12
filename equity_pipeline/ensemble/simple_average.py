"""
ensemble/simple_average.py — SimpleAverageEnsemble
Inner join on (co_code, Month), compute mean pred_score, evaluate IC.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from ..shared.metrics  import spearman_ic, compute_ic_series, print_results_table
from ..shared.output   import (load_predictions, save_test_predictions,
                                save_ic_series, save_summary)


class SimpleAverageEnsemble:
    name = "simple_average"

    def __init__(self, model_names: list[str], results_dir: Path, cfg):
        self.model_names = model_names
        self.results_dir = Path(results_dir)
        self.cfg         = cfg

    def run(self) -> dict:
        print(f"\n[SimpleAverage] Models: {self.model_names}")

        frames = {}
        missing = []
        for mn in self.model_names:
            try:
                df = load_predictions(mn, "test", self.results_dir)
                frames[mn] = df.rename(columns={"pred_score": f"pred_{mn}"})
            except FileNotFoundError as e:
                missing.append(mn)
                print(f"  ✗ {e}")

        if missing:
            raise RuntimeError(
                f"Missing test predictions for: {missing}. Run the model(s) first."
            )

        # Inner join on (co_code, Month)
        id_col, date_col = self.cfg.id_col, self.cfg.date_col
        merged = frames[self.model_names[0]][[id_col, date_col, f"pred_{self.model_names[0]}", "fwd_return"]]
        for mn in self.model_names[1:]:
            merged = merged.merge(
                frames[mn][[id_col, date_col, f"pred_{mn}"]],
                on=[id_col, date_col], how="inner",
            )

        pred_cols = [f"pred_{mn}" for mn in self.model_names]
        merged["pred_score"] = merged[pred_cols].mean(axis=1)

        n_months = merged[date_col].nunique()
        print(f"[SimpleAverage] Month coverage: {n_months} months "
              f"({merged[date_col].min()} → {merged[date_col].max()})")

        # IC per month
        ic_series = []
        for month, grp in merged.groupby(date_col):
            ic = spearman_ic(grp["fwd_return"].values, grp["pred_score"].values)
            ic_series.append({"Month": month, "IC": ic})

        summary = compute_ic_series([r["IC"] for r in ic_series])
        print_results_table(summary, self.name, self.cfg.baselines)

        # Save
        ens_dir = self.results_dir / "ensemble"
        save_test_predictions(
            merged[[id_col, date_col, "pred_score", "fwd_return"]],
            self.name, ens_dir,
        )
        save_ic_series(ic_series, self.name, ens_dir)
        save_summary(summary, self.name, ens_dir)
        return summary
