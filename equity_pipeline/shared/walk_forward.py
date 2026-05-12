"""
shared/walk_forward.py
======================
Generic walk-forward harness used by ALL models.
Optimized for sub-0.1s fold time with precomputed bounds.
"""
from __future__ import annotations

import time
import copy
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.base import clone as sklearn_clone

from .metrics import spearman_ic, compute_ic_series, print_results_table
from .output  import (save_test_predictions, save_oof_predictions,
                       save_ic_series, save_summary, save_feature_importance,
                       save_best_hparams)
from .sequences import build_sequences


class BaseModel(ABC):
    name: str = "base"
    uses_sequences: bool = False
    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val) -> None: ...
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def get_feature_importance(self) -> dict: return {}
    def set_feature_names(self, features): return self
    def clone(self) -> "BaseModel": return copy.deepcopy(self)


def run_walk_forward(
    model:         BaseModel,
    df:            pd.DataFrame,
    preprocessor_factory,
    cfg,
    output_dir:    str,
    tune_first_fold: bool = True,
    compute_oof:     bool = True,
) -> dict:
    date_col = cfg.date_col
    id_col   = cfg.id_col
    feat_list = list(cfg.features)
    
    all_months = sorted(df[date_col].unique())
    n_months   = len(all_months)
    start_idx  = cfg.train_window

    # Pre-group months for O(1) slicing
    month_groups = {m: g.reset_index(drop=True) for m, g in df.groupby(date_col)}
    
    # Pre-compute ALL winsorization bounds globally
    print(f"[WalkForward] Pre-computing winsorization bounds for {n_months} months...")
    grouped = df.groupby(date_col)[feat_list]
    global_lo = grouped.quantile(cfg.winsor_lower)
    global_hi = grouped.quantile(cfg.winsor_upper)
    precomputed_winsor = (global_lo, global_hi)

    # Initialize preprocessor with precomputed bounds
    preprocessor = preprocessor_factory(cfg, precomputed_winsor)
    
    ic_series = []
    all_preds = []
    all_oof   = []
    
    n_folds = n_months - start_idx
    print(f"\n[WalkForward] {model.name.upper()} | {n_folds} folds | window={cfg.train_window}m")

    for fold_idx, t in enumerate(range(start_idx, n_months)):
        test_month   = all_months[t]
        train_months = all_months[max(0, t - cfg.train_window) : t]
        val_months_  = train_months[-cfg.val_months:]
        tr_months_   = train_months[:-cfg.val_months]

        # Fast Slicing
        df_train = pd.concat([month_groups[m] for m in train_months], ignore_index=True)
        df_test  = month_groups[test_month]
        
        if len(df_test) < cfg.min_obs:
            continue

        t_pre_start = time.time()
        pp = sklearn_clone(preprocessor)
        
        # Training data (tr + val)
        df_tr = df_train[df_train[date_col].isin(tr_months_)]
        df_tr_proc = pp.fit_transform(df_tr)
        X_tr = df_tr_proc[feat_list].values
        y_tr = df_tr_proc["fwd_return"].values

        # Validation data
        df_val = df_train[df_train[date_col].isin(val_months_)]
        df_val_proc = pp.transform(df_val)
        X_val = df_val_proc[feat_list].values
        y_val = df_val_proc["fwd_return"].values

        # Sequence conversion if needed
        if model.uses_sequences:
            seq_len = getattr(cfg, "seq_len", 24)
            # Train sequences
            X_tr, y_tr, _, _ = build_sequences(df_tr_proc, id_col, date_col, feat_list, "fwd_return", seq_len)
            
            # Val sequences (need prior history from tr to fill the seq_len window for early val months)
            # We can just use the full df_train_proc and filter for val_months
            df_train_proc = pp.transform(df_train)
            X_val, y_val, _, _ = build_sequences(df_train_proc, id_col, date_col, feat_list, "fwd_return", seq_len, 
                                                 pred_months=val_months_)
            
            # Test sequences (need history from training window)
            recent_hist = df_train_proc[df_train_proc[date_col].isin(train_months[-seq_len:])]
            df_te_proc = pp.transform(df_test)
            X_te, y_te, ids_te, _ = build_sequences(
                pd.concat([recent_hist, df_te_proc], ignore_index=True),
                id_col, date_col, feat_list, "fwd_return", seq_len,
                pred_months=[test_month]
            )
        else:
            # Test data (already transformed above for sequence models)
            df_te_proc = pp.transform(df_test)
            X_te = df_te_proc[feat_list].values
            y_te = df_te_proc["fwd_return"].values
            ids_te = df_te_proc[id_col].values

        # Hyperparameter Tuning (Optional: Only on first fold)
        if fold_idx == 0 and tune_first_fold:
            print(f"  [{model.name}] Tuning hyperparameters on first fold...")
            if hasattr(model, "tune_hyperparameters"):
                best_params = model.tune_hyperparameters(X_tr, y_tr, X_val, y_val, None)
                print(f"  [{model.name}] Best params: {best_params}")
                save_best_hparams(best_params, model.name, output_dir)

        t_pre = time.time() - t_pre_start

        if len(X_tr) < cfg.min_obs or len(X_te) < 5:
            continue

        t_train_start = time.time()
        print(f"  [{model.name}] calling fit...")
        model.fit(X_tr, y_tr, X_val, y_val)
        t_train = time.time() - t_train_start
        
        preds = model.predict(X_te)
        test_ic = spearman_ic(y_te, preds)
        
        if compute_oof:
            if model.uses_sequences:
                seq_len = getattr(cfg, "seq_len", 24)
                # If df_train_proc wasn't already created in the sequence block above
                if 'df_train_proc' not in locals():
                    df_train_proc = pp.transform(df_train)
                X_oof, y_oof, ids_oof, m_oof = build_sequences(df_train_proc, id_col, date_col, feat_list, "fwd_return", seq_len)
                if len(X_oof) > 0:
                    oof_preds = model.predict(X_oof)
                    all_oof.append(pd.DataFrame({
                        id_col: ids_oof,
                        date_col: m_oof,
                        "pred_score": oof_preds,
                        "fwd_return": y_oof
                    }))
            else:
                df_train_proc = pp.transform(df_train)
                oof_preds = model.predict(df_train_proc[feat_list].values)
                all_oof.append(pd.DataFrame({
                    id_col: df_train_proc[id_col],
                    date_col: df_train_proc[date_col],
                    "pred_score": oof_preds,
                    "fwd_return": df_train_proc["fwd_return"]
                }))

        ic_series.append({"Month": test_month, "IC": test_ic})
        all_preds.append(pd.DataFrame({
            id_col: ids_te, date_col: test_month,
            "pred_score": preds, "fwd_return": y_te
        }))
        
        if (fold_idx + 1) % 50 == 0 or (fold_idx + 1) == n_folds:
            mean_ic = np.mean([r["IC"] for r in ic_series])
            print(f"Fold {fold_idx+1:3d}/{n_folds} | pre={t_pre:.3f}s | train={t_train:.3f}s | meanIC={mean_ic:+.4f}")

    summary = compute_ic_series([r["IC"] for r in ic_series])
    print_results_table(summary, model.name, cfg.baselines)
    if not ic_series:
        print(f"\n[WalkForward] Warning: No folds were processed for {model.name}. Check your data and min_obs settings.")
        return {
            "mean_ic": np.nan, "ic_std": np.nan, "icir": 0.0,
            "pct_positive": np.nan, "n_months": 0
        }

    save_ic_series(ic_series, model.name, output_dir)
    save_summary(summary, model.name, output_dir)
    save_test_predictions(pd.concat(all_preds, ignore_index=True), model.name, output_dir)
    if all_oof:
        save_oof_predictions(pd.concat(all_oof, ignore_index=True).drop_duplicates(subset=[id_col, date_col]), model.name, output_dir)
    
    return summary
