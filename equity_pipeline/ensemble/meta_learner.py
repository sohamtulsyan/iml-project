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

        # ── Mode Selection: OOF vs Time-Split ────────────────────────────────
        missing_oof = [mn for mn in self.model_names
                       if not (self.results_dir / mn / f"{mn}_oof_predictions.parquet").exists()]
        
        use_time_split = len(missing_oof) > 0
        
        if use_time_split:
            print(f"[MetaLearner] WARNING: OOF predictions missing for {missing_oof}.")
            print(f"[MetaLearner] Falling back to Time-Split Stacking (Test-on-Test).")
            return self._run_time_split()
        else:
            return self._run_standard_oof()

    def _run_standard_oof(self) -> dict:
        id_col, date_col = self.cfg.id_col, self.cfg.date_col
        
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

        return self._fit_and_evaluate(oof_merged, test_merged, "target", "fwd_return")

    def _run_time_split(self) -> dict:
        id_col, date_col = self.cfg.id_col, self.cfg.date_col
        
        # Load all test predictions
        test_frames = {}
        for mn in self.model_names:
            df = load_predictions(mn, "test", self.results_dir)
            df = df.rename(columns={"pred_score": f"pred_{mn}"})
            test_frames[mn] = df

        merged = test_frames[self.model_names[0]][
            [id_col, date_col, f"pred_{self.model_names[0]}", "fwd_return"]
        ]
        for mn in self.model_names[1:]:
            merged = merged.merge(
                test_frames[mn][[id_col, date_col, f"pred_{mn}"]],
                on=[id_col, date_col], how="inner",
            )

        # Split by time (60/40)
        all_months = sorted(merged[date_col].unique())
        split_idx  = int(len(all_months) * 0.6)
        train_months = all_months[:split_idx]
        test_months  = all_months[split_idx:]
        
        print(f"[MetaLearner] Split Date: {test_months[0]}")
        print(f"[MetaLearner] Meta-Train: {len(train_months)} months | Meta-Test: {len(test_months)} months")

        train_df = merged[merged[date_col].isin(train_months)]
        test_df  = merged[merged[date_col].isin(test_months)]
        
        return self._fit_and_evaluate(train_df, test_df, "fwd_return", "fwd_return")

    def _fit_and_evaluate(self, train_df, test_df, train_target_col, test_target_col) -> dict:
        id_col, date_col = self.cfg.id_col, self.cfg.date_col
        pred_cols = [f"pred_{mn}" for mn in self.model_names]
        
        X_train = train_df[pred_cols].values.astype(np.float32)
        y_train = train_df[train_target_col].values.astype(np.float32)

        # Fit Meta-Learner (Ridge)
        meta = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])
        meta.fit(X_train, y_train)

        weights = dict(zip(self.model_names, meta.coef_.tolist()))
        print(f"\n[MetaLearner] Learned weights (α={meta.alpha_:.4f}):")
        for nm, w in weights.items():
            print(f"  {nm:<20}: {w:+.4f}")

        # Save weights
        ens_dir = self.results_dir / "ensemble"
        ens_dir.mkdir(parents=True, exist_ok=True)
        with open(ens_dir / "meta_learner_weights.json", "w") as f:
            json.dump(weights, f, indent=2)

        # Inference on Test Set
        X_test = test_df[pred_cols].values.astype(np.float32)
        test_df["pred_score"] = meta.predict(X_test).astype(np.float32)

        # IC Evaluation
        ic_series = []
        for month, grp in test_df.groupby(date_col):
            ic = spearman_ic(grp[test_target_col].values, grp["pred_score"].values)
            ic_series.append({"Month": month, "IC": ic})

        summary = compute_ic_series([r["IC"] for r in ic_series])
        print_results_table(summary, self.name, self.cfg.baselines)

        # Save results
        save_test_predictions(
            test_df[[id_col, date_col, "pred_score", test_target_col]],
            self.name, ens_dir,
        )
        save_ic_series(ic_series, self.name, ens_dir)
        save_summary(summary, self.name, ens_dir)
        return summary
