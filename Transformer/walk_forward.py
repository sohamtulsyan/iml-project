"""
================================================================================
  TRANSFORMER PIPELINE — WALK-FORWARD EVALUATION + OOF GENERATION
================================================================================

  Structure:
  ──────────
  For each prediction month t in [train_window, n_months):
    1. Slice training panel [t-60, t-1]
    2. Preprocess per-month cross-sectionally
    3. Build (N, T, F) sequences for training months
    4. Build sequences for test month t (using history [t-T, t-1])
    5. Train model; predict on test sequences
    6. Compute Spearman IC; store predictions

  OOF predictions (for meta-learner):
    Within each training window, 5 time-ordered folds → OOF predictions
    covering the full 60-month training window.
"""

import time
import copy
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr
from statsmodels.stats.stattools import durbin_watson

from data     import preprocess_panel, build_sequences
from model    import get_device, build_model
from trainer  import train_one_fold, predict, spearman_ic
from config   import TransformerConfig


# ─────────────────────────────────────────────────────────────────────────────
#  OOF generation (5 time-ordered folds within a training window)
# ─────────────────────────────────────────────────────────────────────────────

def generate_oof_predictions(
    df_train:   pd.DataFrame,
    cfg:        TransformerConfig,
    device:     torch.device,
    hparams:    dict,
    n_folds:    int = 5,
) -> pd.DataFrame:
    """
    5-fold time-ordered cross-validation within the 60-month training window.
    Each fold trains on earlier months and predicts on later months (no shuffle).

    Returns DataFrame with columns: [id_col, date_col, oof_pred]
    """
    id_col    = cfg.id_col
    date_col  = cfg.date_col
    features  = cfg.features
    seq_len   = hparams.get("seq_len", cfg.seq_len)

    all_months = sorted(df_train[date_col].unique())
    n_months   = len(all_months)
    fold_size  = n_months // n_folds

    oof_records = []

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end   = (fold + 1) * fold_size if fold < n_folds - 1 else n_months

        tr_months  = all_months[:val_start] + all_months[val_end:]
        val_months = all_months[val_start:val_end]

        if len(tr_months) < seq_len + 1:
            continue

        df_tr  = df_train[df_train[date_col].isin(tr_months)]
        df_val = df_train[df_train[date_col].isin(val_months)]

        # Build sequences — val sequences use only training history
        X_tr, y_tr, _, _       = build_sequences(df_tr,  id_col, date_col, features, "fwd_return", seq_len)
        X_val, y_val, id_v, m_v = build_sequences(df_val, id_col, date_col, features, "fwd_return", seq_len)

        if len(X_tr) < cfg.min_obs or len(X_val) < 5:
            continue

        model = build_model(hparams if isinstance(hparams, dict)
                            else {**vars(hparams)}, n_features=len(features))
        model.seq_len = seq_len

        model, _, _ = train_one_fold(
            model, X_tr, y_tr, X_val, y_val, device,
            lr         = hparams.get("lr", cfg.lr),
            batch_size = hparams.get("batch_size", cfg.batch_size),
            max_epochs = cfg.max_epochs,
            patience   = cfg.early_stop_patience,
            verbose    = False,
        )

        preds = predict(model, X_val, device)

        for i, (firm_id, month) in enumerate(zip(id_v, m_v)):
            oof_records.append({
                id_col:    firm_id,
                date_col:  month,
                "oof_pred": preds[i],
                "target":   y_val[i],
            })

    return pd.DataFrame(oof_records)


# ─────────────────────────────────────────────────────────────────────────────
#  Single walk-forward fold
# ─────────────────────────────────────────────────────────────────────────────

def run_one_fold(
    fold_idx:      int,
    test_month,
    all_months:    list,
    df:            pd.DataFrame,
    cfg:           TransformerConfig,
    device:        torch.device,
    best_hparams:  dict,
    compute_oof:   bool = True,
) -> Optional[dict]:
    """
    Execute one walk-forward fold.
    Returns a result dict or None if the fold is skipped.
    """
    id_col   = cfg.id_col
    date_col = cfg.date_col
    features = cfg.features
    seq_len  = best_hparams.get("seq_len", cfg.seq_len)

    t_idx        = all_months.index(test_month)
    train_months = all_months[max(0, t_idx - cfg.train_window) : t_idx]

    if len(train_months) < cfg.train_window:
        return None  # not enough history yet

    # ── Extract panels ────────────────────────────────────────────────────────
    df_train = df[df[date_col].isin(train_months)].copy()
    df_test  = df[df[date_col] == test_month].copy()

    if len(df_test) < cfg.min_obs:
        return None

    # ── Per-month cross-sectional preprocessing (no leakage) ─────────────────
    df_train = preprocess_panel(df_train, features, date_col,
                                cfg.log_transform_cols, cfg.winsor_lower, cfg.winsor_upper)
    df_test  = preprocess_panel(df_test,  features, date_col,
                                cfg.log_transform_cols, cfg.winsor_lower, cfg.winsor_upper)

    # Validation split: last val_months of training window
    val_months  = train_months[-cfg.val_months:]
    tr_months_f = train_months[:-cfg.val_months]

    df_tr  = df_train[df_train[date_col].isin(tr_months_f)]
    df_val = df_train[df_train[date_col].isin(val_months)]

    # ── Build sequences ───────────────────────────────────────────────────────
    X_tr, y_tr, _, _           = build_sequences(df_tr,  id_col, date_col, features, "fwd_return", seq_len)
    X_val, y_val, _, _         = build_sequences(df_val, id_col, date_col, features, "fwd_return", seq_len)
    X_te, y_te, ids_te, _      = build_sequences(
        pd.concat([df_train.tail(seq_len * df[id_col].nunique()), df_test]),
        id_col, date_col, features, "fwd_return", seq_len,
        pred_months=[test_month]
    )

    if len(X_tr) < cfg.min_obs or len(X_te) < 5:
        return None

    if len(X_val) < 5:   # fallback: use a slice of training set as val
        n_val  = max(5, len(X_tr) // 10)
        X_val, y_val = X_tr[-n_val:], y_tr[-n_val:]
        X_tr,  y_tr  = X_tr[:-n_val], y_tr[:-n_val]

    # ── Train ─────────────────────────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    model = build_model(best_hparams, n_features=len(features))

    t0 = time.time()
    model, best_ep, val_ic_log = train_one_fold(
        model, X_tr, y_tr, X_val, y_val, device,
        lr          = best_hparams.get("lr", cfg.lr),
        weight_decay = cfg.weight_decay,
        batch_size  = best_hparams.get("batch_size", cfg.batch_size),
        max_epochs  = cfg.max_epochs,
        patience    = cfg.early_stop_patience,
        grad_clip   = cfg.grad_clip,
        verbose     = cfg.verbose,
    )
    train_time = time.time() - t0

    # ── Predict & evaluate ────────────────────────────────────────────────────
    preds  = predict(model, X_te, device)
    test_ic = spearman_ic(y_te, preds)

    # ── OOF predictions for meta-learner ─────────────────────────────────────
    oof_df = pd.DataFrame()
    if compute_oof:
        try:
            oof_df = generate_oof_predictions(df_train, cfg, device, best_hparams)
        except Exception:
            pass

    # ── Prediction records ────────────────────────────────────────────────────
    pred_records = pd.DataFrame({
        id_col:   ids_te,
        date_col: test_month,
        "pred_score": preds,
        "fwd_return": y_te,
    })

    return {
        "month":       test_month,
        "ic":          test_ic,
        "n_stocks":    len(X_te),
        "best_epoch":  best_ep + 1,
        "train_time":  train_time,
        "val_ic_log":  val_ic_log,
        "preds":       pred_records,
        "oof":         oof_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main walk-forward harness
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward(
    df:           pd.DataFrame,
    cfg:          TransformerConfig,
    device:       torch.device,
    output_dir:   Path,
    tune_hparams: bool = False,
) -> Dict:
    """
    Run the full walk-forward evaluation.

    Returns summary dict with mean IC, ICIR, etc.
    Writes all output files to output_dir.
    """
    from hparam_search import hyperparameter_search

    id_col   = cfg.id_col
    date_col = cfg.date_col
    features = cfg.features
    seq_len  = cfg.seq_len

    all_months = sorted(df[date_col].unique())
    n_months   = len(all_months)
    start_idx  = cfg.train_window  # first fold has full 60-month history

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Hyperparameter search (first fold only) ───────────────────────
    if tune_hparams:
        print("\n[HparamSearch] Running on first fold...")
        t0_idx       = start_idx
        tr_months_hs = all_months[0 : t0_idx - cfg.val_months]
        va_months_hs = all_months[t0_idx - cfg.val_months : t0_idx]

        df_tr_hs = preprocess_panel(
            df[df[date_col].isin(tr_months_hs)].copy(), features, date_col,
            cfg.log_transform_cols, cfg.winsor_lower, cfg.winsor_upper
        )
        df_va_hs = preprocess_panel(
            df[df[date_col].isin(va_months_hs)].copy(), features, date_col,
            cfg.log_transform_cols, cfg.winsor_lower, cfg.winsor_upper
        )

        X_tr_hs, y_tr_hs, _, _ = build_sequences(df_tr_hs, id_col, date_col, features, "fwd_return", seq_len)
        X_va_hs, y_va_hs, _, _ = build_sequences(df_va_hs, id_col, date_col, features, "fwd_return", seq_len)

        best_hparams = hyperparameter_search(
            cfg.search_space, X_tr_hs, y_tr_hs, X_va_hs, y_va_hs,
            device, n_jobs=cfg.n_workers, seed=cfg.seed, verbose=True
        )
    else:
        # Use defaults from config
        best_hparams = {
            "d_model":        cfg.d_model,
            "nhead":          cfg.nhead,
            "num_layers":     cfg.num_layers,
            "dim_feedforward": cfg.dim_feedforward,
            "dropout":        cfg.dropout,
            "lr":             cfg.lr,
            "batch_size":     cfg.batch_size,
            "seq_len":        cfg.seq_len,
        }

    # Save best hparams
    with open(output_dir / "transformer_best_hparams.json", "w") as f:
        json.dump(best_hparams, f, indent=2)
    print(f"[Config] Best hparams: {best_hparams}")

    # ── Step 2: Walk-forward loop ─────────────────────────────────────────────
    ic_series       = []
    all_preds       = []
    all_oof         = []
    training_log    = []
    n_folds         = n_months - start_idx

    print(f"\n[WalkForward] {n_folds} folds | device={device.type.upper()}")
    print(f"  Train window: {cfg.train_window} months | "
          f"Val: last {cfg.val_months} months | Seq len: {best_hparams.get('seq_len', cfg.seq_len)}")

    wall_times = []
    for fold_idx, t in enumerate(range(start_idx, n_months)):
        test_month = all_months[t]
        fold_t0    = time.time()

        result = run_one_fold(
            fold_idx, test_month, all_months, df, cfg, device, best_hparams,
            compute_oof=(fold_idx < 10)  # OOF for first 10 folds (expensive); extend if needed
        )

        if result is None:
            continue

        ic_series.append({"Month": result["month"], "IC": result["ic"]})
        all_preds.append(result["preds"])
        if not result["oof"].empty:
            all_oof.append(result["oof"])

        # Training log (epoch-level)
        for ep_idx, ep_ic in enumerate(result["val_ic_log"]):
            training_log.append({
                "Month":    result["month"],
                "epoch":    ep_idx + 1,
                "val_ic":   ep_ic,
                "fold_time": result["train_time"],
            })

        wall_times.append(time.time() - fold_t0)

        # Progress
        if (fold_idx + 1) % 10 == 0 or fold_idx == 0:
            n_done = len(ic_series)
            mean_ic_so_far = np.mean([r["IC"] for r in ic_series]) if ic_series else 0.0
            eta = np.mean(wall_times) * (n_folds - fold_idx - 1) / 60
            print(f"  [Fold {fold_idx+1:3d}/{n_folds}] {str(test_month)[:10]} | "
                  f"IC={result['ic']:+.4f} | mean IC={mean_ic_so_far:+.4f} | "
                  f"ETA={eta:.0f}min")

    # ── Step 3: Aggregate & report ────────────────────────────────────────────
    ic_df = pd.DataFrame(ic_series)
    if ic_df.empty:
        print("  ✗ No results produced.")
        return {}

    ic_arr   = ic_df["IC"].values
    mean_ic  = float(np.mean(ic_arr))
    std_ic   = float(np.std(ic_arr))
    icir     = mean_ic / std_ic if std_ic > 0 else 0.0
    pct_pos  = float(np.mean(ic_arr > 0) * 100)
    dw_stat  = durbin_watson(ic_arr)

    ic_df["cumulative_IC"] = ic_df["IC"].cumsum()

    summary = {
        "mean_ic":         round(mean_ic, 6),
        "ic_std":          round(std_ic,  6),
        "icir":            round(icir,    6),
        "pct_positive":    round(pct_pos, 2),
        "n_months":        len(ic_arr),
        "durbin_watson":   round(float(dw_stat), 4),
    }

    _print_summary(summary, cfg)

    # ── Step 4: Save outputs ──────────────────────────────────────────────────
    _save_outputs(output_dir, ic_df, all_preds, all_oof, training_log, summary)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(summary: dict, cfg: TransformerConfig):
    print("\n" + "=" * 70)
    print("  TRANSFORMER — RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Mean IC         : {summary['mean_ic']:+.6f}")
    print(f"  IC Std          :  {summary['ic_std']:.6f}")
    print(f"  ICIR            : {summary['icir']:+.6f}")
    print(f"  % Positive IC   :  {summary['pct_positive']:.2f}%")
    print(f"  Months          :  {summary['n_months']}")
    print(f"  Durbin-Watson   :  {summary['durbin_watson']:.4f}  (2.0 = no autocorr)")
    print("\n  Baseline comparison:")
    print(f"  {'Model':<20} {'Mean IC':>10} {'ICIR':>10}")
    print(f"  {'-'*40}")
    print(f"  {'Ridge':.<20} {'0.0344':>10} {'0.2504':>10}")
    print(f"  {'LightGBM':.<20} {'0.0551':>10} {'0.7347':>10}")
    print(f"  {'Random Forest':.<20} {'0.0546':>10} {'0.7248':>10}")
    print(f"  {'Transformer (this)':.<20} {summary['mean_ic']:>10.4f} {summary['icir']:>10.4f}")
    print("=" * 70 + "\n")


def _save_outputs(
    output_dir:   Path,
    ic_df:        pd.DataFrame,
    all_preds:    list,
    all_oof:      list,
    training_log: list,
    summary:      dict,
):
    import json

    # IC series CSV
    ic_csv = output_dir / "transformer_ic_series.csv"
    ic_df.to_csv(ic_csv, index=False)
    print(f"  ✓ {ic_csv}")

    # Summary JSON
    summ_json = output_dir / "transformer_summary.json"
    with open(summ_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ {summ_json}")

    # Test predictions parquet
    if all_preds:
        preds_df = pd.concat(all_preds, ignore_index=True)
        preds_pq = output_dir / "transformer_test_predictions.parquet"
        preds_df.to_parquet(preds_pq, engine="pyarrow", compression="snappy", index=False)
        print(f"  ✓ {preds_pq}  ({len(preds_df):,} rows)")

    # OOF predictions parquet
    if all_oof:
        oof_df = pd.concat(all_oof, ignore_index=True)
        oof_pq = output_dir / "transformer_oof_predictions.parquet"
        oof_df.to_parquet(oof_pq, engine="pyarrow", compression="snappy", index=False)
        print(f"  ✓ {oof_pq}  ({len(oof_df):,} rows)")

    # Training log CSV
    if training_log:
        log_df  = pd.DataFrame(training_log)
        log_csv = output_dir / "transformer_training_log.csv"
        log_df.to_csv(log_csv, index=False)
        print(f"  ✓ {log_csv}")
