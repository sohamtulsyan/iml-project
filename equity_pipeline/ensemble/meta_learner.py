"""
ensemble/meta_learner.py — MetaLearnerEnsemble
Stacked generalization: RidgeCV meta-learner trained on OOF predictions.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV

from ..shared.metrics import spearman_ic, compute_ic_series, print_results_table
from ..shared.output  import (load_predictions, save_test_predictions,
                               save_ic_series, save_summary)


class MetaLearnerEnsemble:
    name = "meta_learner"

    def __init__(self, model_names: list[str], results_dir: Path, cfg):
        self.model_names = model_names
        self.results_dir = Path(results_dir)
        self.cfg         = cfg

    def run(self) -> dict:
        id_col, date_col = self.cfg.id_col, self.cfg.date_col
        print(f"\n[MetaLearner] Models: {self.model_names}")

        # ── Validate all predictions exist before proceeding ─────────────────
        missing_oof  = [mn for mn in self.model_names
                        if not (self.results_dir / mn /
                                f"{mn}_oof_predictions.parquet").exists()]
        missing_test = [mn for mn in self.model_names
                        if not (self.results_dir / mn /
                                f"{mn}_test_predictions.parquet").exists()]
        if missing_oof or missing_test:
            msg = ""
            if missing_oof:  msg += f"Missing OOF predictions: {missing_oof}\n"
            if missing_test: msg += f"Missing test predictions: {missing_test}\n"
            raise RuntimeError(msg + "Run the model(s) first.")

        # ── TRAINING PHASE: align OOF frames ─────────────────────────────────
        oof_frames = {}
        for mn in self.model_names:
            df = load_predictions(mn, "oof", self.results_dir)
            df = df.rename(columns={"oof_pred": f"pred_{mn}"})
            oof_frames[mn] = df

        oof_merged = oof_frames[self.model_names[0]][
            [id_col, date_col, f"pred_{self.model_names[0]}", "target"]
        ]
        for mn in self.model_names[1:]:
            oof_merged = oof_merged.merge(
                oof_frames[mn][[id_col, date_col, f"pred_{mn}"]],
                on=[id_col, date_col], how="inner",
            )

        # ── TRAINING PHASE: align TEST frames ────────────────────────────────
        test_frames = {}
        for mn in self.model_names:
            df = load_predictions(mn, "test", self.results_dir)
            df = df.rename(columns={"pred_score": f"pred_{mn}"})
            test_frames[mn] = df

        test_merged = test_frames[self.model_names[0]][
            [id_col, date_col, f"pred_{self.model_names[0]}", "fwd_return"]
        ]
        for mn in self.model_names[1:]:
            test_merged = test_merged.merge(
                test_frames[mn][[id_col, date_col, f"pred_{mn}"]],
                on=[id_col, date_col], how="inner",
            )

        # ── OOF/Test month overlap check ──────────────────────────────────────
        oof_months  = set(oof_merged[date_col].unique())
        test_months = set(test_merged[date_col].unique())
        overlap     = oof_months & test_months
        assert not overlap, (
            f"[MetaLearner] OOF and test months overlap: {sorted(overlap)[:5]}... "
            "This indicates a look-ahead leak. Aborting."
        )

        # ── Fit meta-learner ──────────────────────────────────────────────────
        pred_cols = [f"pred_{mn}" for mn in self.model_names]
        X_oof     = oof_merged[pred_cols].values.astype(np.float32)
        y_oof     = oof_merged["target"].values.astype(np.float32)

        meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        meta.fit(X_oof, y_oof)

        weights = dict(zip(self.model_names, meta.coef_.tolist()))
        print(f"\n[MetaLearner] Learned weights (α={meta.alpha_:.4f}):")
        for nm, w in weights.items():
            print(f"  {nm:<20}: {w:+.4f}")

        # Save weights
        ens_dir = self.results_dir / "ensemble"
        ens_dir.mkdir(parents=True, exist_ok=True)
        with open(ens_dir / "meta_learner_weights.json", "w") as f:
            json.dump(weights, f, indent=2)

        # ── INFERENCE PHASE ───────────────────────────────────────────────────
        X_test = test_merged[pred_cols].values.astype(np.float32)
        test_merged["pred_score"] = meta.predict(X_test).astype(np.float32)

        n_months = test_merged[date_col].nunique()
        print(f"[MetaLearner] Test month coverage: {n_months} months "
              f"({test_merged[date_col].min()} → {test_merged[date_col].max()})")

        ic_series = []
        for month, grp in test_merged.groupby(date_col):
            ic = spearman_ic(grp["fwd_return"].values, grp["pred_score"].values)
            ic_series.append({"Month": month, "IC": ic})

        summary = compute_ic_series([r["IC"] for r in ic_series])
        print_results_table(summary, self.name, self.cfg.baselines)

        # ── Save ──────────────────────────────────────────────────────────────
        save_test_predictions(
            test_merged[[id_col, date_col, "pred_score", "fwd_return"]],
            self.name, ens_dir,
        )
        save_ic_series(ic_series, self.name, ens_dir)
        save_summary(summary, self.name, ens_dir)
        return summary
