"""
================================================================================
  CNN PIPELINE — WALK-FORWARD HARNESS + OOF GENERATOR
================================================================================

  Shares data loading / preprocessing with the Transformer pipeline.
  Only the model build step and output filenames differ.
"""

import sys
import json
import time
import copy
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from statsmodels.stats.stattools import durbin_watson

# ── Shared from Transformer ───────────────────────────────────────────────────
_TR_DIR = Path(__file__).parent.parent / "Transformer"
sys.path.insert(0, str(_TR_DIR))
from data    import preprocess_panel, build_sequences
from trainer import spearman_ic

# ── CNN-specific ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from model   import build_model, get_device
from trainer import train_one_fold, predict
from config  import CNNConfig


# ─────────────────────────────────────────────────────────────────────────────
#  OOF generator (5 time-ordered folds within a training window)
# ─────────────────────────────────────────────────────────────────────────────

def generate_oof_predictions(
    df_train: pd.DataFrame,
    cfg:      CNNConfig,
    device:   torch.device,
    hparams:  dict,
    n_folds:  int = 5,
) -> pd.DataFrame:
    """
    5-fold temporally ordered CV inside the 60-month training window.
    Returns DataFrame: [id_col, date_col, oof_pred, target]
    """
    id_col   = cfg.id_col
    date_col = cfg.date_col
    features = cfg.features
    seq_len  = hparams.get("seq_len", cfg.seq_len)

    all_months = sorted(df_train[date_col].unique())
    n_months   = len(all_months)
    fold_size  = n_months // n_folds
    oof_rows   = []

    for fold in range(n_folds):
        va_start = fold * fold_size
        va_end   = (fold + 1) * fold_size if fold < n_folds - 1 else n_months
        tr_months = all_months[:va_start] + all_months[va_end:]
        va_months = all_months[va_start:va_end]

        if len(tr_months) < seq_len + 1:
            continue

        df_tr = df_train[df_train[date_col].isin(tr_months)]
        df_va = df_train[df_train[date_col].isin(va_months)]

        X_tr, y_tr, _, _         = build_sequences(df_tr, id_col, date_col, features, "fwd_return", seq_len)
        X_va, y_va, id_v, m_v    = build_sequences(df_va, id_col, date_col, features, "fwd_return", seq_len)

        if len(X_tr) < cfg.min_obs or len(X_va) < 5:
            continue

        m = build_model(hparams, n_features=len(features))
        m, _, _ = train_one_fold(
            m, X_tr, y_tr, X_va, y_va, device,
            lr         = hparams.get("lr", cfg.lr),
            batch_size = hparams.get("batch_size", cfg.batch_size),
            max_epochs = cfg.max_epochs,
            patience   = cfg.early_stop_patience,
        )
        preds = predict(m, X_va, device)

        for i in range(len(preds)):
            oof_rows.append({
                id_col:     id_v[i],
                date_col:   m_v[i],
                "oof_pred": preds[i],
                "target":   y_va[i],
            })

    return pd.DataFrame(oof_rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Activation analysis (per-branch average activation magnitude)
# ─────────────────────────────────────────────────────────────────────────────

def compute_branch_activations(
    model:   torch.nn.Module,
    X_batch: np.ndarray,
    device:  torch.device,
) -> dict:
    """
    Forward pass with hooks to capture per-branch mean activation magnitude.
    Returns dict: {"branch_short": float, "branch_medium": float, "branch_long": float}
    """
    activations = {}
    hooks = []

    for branch_name in ("branch_short", "branch_medium", "branch_long"):
        branch = getattr(model, branch_name, None)
        if branch is None:
            continue

        def make_hook(name):
            def hook(module, inp, out):
                activations[name] = float(out.detach().abs().mean().cpu())
            return hook

        h = branch.register_forward_hook(make_hook(branch_name))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        Xb = torch.from_numpy(X_batch[:min(256, len(X_batch))]).float().to(device)
        model(Xb)

    for h in hooks:
        h.remove()

    return activations


# ─────────────────────────────────────────────────────────────────────────────
#  Single walk-forward fold
# ─────────────────────────────────────────────────────────────────────────────

def run_one_fold(
    fold_idx:     int,
    test_month,
    all_months:   list,
    df:           pd.DataFrame,
    cfg:          CNNConfig,
    device:       torch.device,
    best_hparams: dict,
    compute_oof:  bool = True,
) -> dict | None:
    """Execute one CNN walk-forward fold. Returns result dict or None if skipped."""
    id_col   = cfg.id_col
    date_col = cfg.date_col
    features = cfg.features
    seq_len  = best_hparams.get("seq_len", cfg.seq_len)

    t_idx        = all_months.index(test_month)
    train_months = all_months[max(0, t_idx - cfg.train_window) : t_idx]

    if len(train_months) < cfg.train_window:
        return None

    df_train = df[df[date_col].isin(train_months)].copy()
    df_test  = df[df[date_col] == test_month].copy()

    if len(df_test) < cfg.min_obs:
        return None

    # Per-month cross-sectional preprocessing (no leakage)
    df_train = preprocess_panel(df_train, features, date_col,
                                cfg.log_transform_cols, cfg.winsor_lower, cfg.winsor_upper)
    df_test  = preprocess_panel(df_test,  features, date_col,
                                cfg.log_transform_cols, cfg.winsor_lower, cfg.winsor_upper)

    # Train / val split
    val_months_  = train_months[-cfg.val_months:]
    tr_months_f  = train_months[:-cfg.val_months]

    df_tr  = df_train[df_train[date_col].isin(tr_months_f)]
    df_val = df_train[df_train[date_col].isin(val_months_)]

    # Build sequences
    X_tr, y_tr, _, _  = build_sequences(df_tr, id_col, date_col, features, "fwd_return", seq_len)
    X_val, y_val, _, _ = build_sequences(df_val, id_col, date_col, features, "fwd_return", seq_len)

    # Test sequences: use recent training history + test month
    recent_hist = df_train[df_train[date_col].isin(train_months[-(seq_len):])].copy()
    X_te, y_te, ids_te, _ = build_sequences(
        pd.concat([recent_hist, df_test], ignore_index=True),
        id_col, date_col, features, "fwd_return", seq_len,
        pred_months=[test_month],
    )

    if len(X_tr) < cfg.min_obs or len(X_te) < 5:
        return None

    if len(X_val) < 5:
        n_val = max(5, len(X_tr) // 10)
        X_val, y_val = X_tr[-n_val:], y_tr[-n_val:]
        X_tr,  y_tr  = X_tr[:-n_val], y_tr[:-n_val]

    # Train
    torch.manual_seed(cfg.seed)
    model = build_model(best_hparams, n_features=len(features))

    t0 = time.time()
    model, best_ep, val_ic_log = train_one_fold(
        model, X_tr, y_tr, X_val, y_val, device,
        lr           = best_hparams.get("lr", cfg.lr),
        weight_decay = cfg.weight_decay,
        batch_size   = best_hparams.get("batch_size", cfg.batch_size),
        max_epochs   = cfg.max_epochs,
        patience     = cfg.early_stop_patience,
        grad_clip    = cfg.grad_clip,
        verbose      = cfg.verbose,
    )
    train_time = time.time() - t0

    # Predict + IC
    preds   = predict(model, X_te, device)
    test_ic = spearman_ic(y_te, preds)

    # Branch activation magnitudes (interpretability)
    branch_acts = compute_branch_activations(model, X_te, device)

    # OOF
    oof_df = pd.DataFrame()
    if compute_oof:
        try:
            oof_df = generate_oof_predictions(df_train, cfg, device, best_hparams)
        except Exception:
            pass

    pred_records = pd.DataFrame({
        id_col:        ids_te,
        date_col:      test_month,
        "pred_score":  preds,
        "fwd_return":  y_te,
    })

    return {
        "month":       test_month,
        "ic":          test_ic,
        "n_stocks":    len(X_te),
        "best_epoch":  best_ep + 1,
        "train_time":  train_time,
        "val_ic_log":  val_ic_log,
        "branch_acts": branch_acts,
        "preds":       pred_records,
        "oof":         oof_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main walk-forward harness
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward(
    df:           pd.DataFrame,
    cfg:          CNNConfig,
    device:       torch.device,
    output_dir:   Path,
    tune_hparams: bool = False,
) -> dict:
    """Full walk-forward evaluation. Writes all output files."""
    from hparam_search import hyperparameter_search

    id_col   = cfg.id_col
    date_col = cfg.date_col
    features = cfg.features
    seq_len  = cfg.seq_len

    all_months = sorted(df[date_col].unique())
    n_months   = len(all_months)
    start_idx  = cfg.train_window

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Hyperparameter search (first fold) ─────────────────────────────────────
    if tune_hparams:
        print("\n[HparamSearch] Running on first fold...")
        tr_m = all_months[: start_idx - cfg.val_months]
        va_m = all_months[start_idx - cfg.val_months : start_idx]

        df_tr_hs = preprocess_panel(df[df[date_col].isin(tr_m)].copy(), features,
                                    date_col, cfg.log_transform_cols,
                                    cfg.winsor_lower, cfg.winsor_upper)
        df_va_hs = preprocess_panel(df[df[date_col].isin(va_m)].copy(), features,
                                    date_col, cfg.log_transform_cols,
                                    cfg.winsor_lower, cfg.winsor_upper)

        X_tr_hs, y_tr_hs, _, _ = build_sequences(df_tr_hs, id_col, date_col, features, "fwd_return", seq_len)
        X_va_hs, y_va_hs, _, _ = build_sequences(df_va_hs, id_col, date_col, features, "fwd_return", seq_len)

        best_hparams = hyperparameter_search(
            cfg.search_space, X_tr_hs, y_tr_hs, X_va_hs, y_va_hs,
            device, n_jobs=cfg.n_workers, seed=cfg.seed, verbose=True,
        )
    else:
        best_hparams = {
            "n_filters":     cfg.n_filters,
            "kernel_sizes":  list(cfg.kernel_sizes),
            "n_conv_blocks": cfg.n_conv_blocks,
            "dropout":       cfg.dropout,
            "lr":            cfg.lr,
            "batch_size":    cfg.batch_size,
            "seq_len":       cfg.seq_len,
        }

    with open(output_dir / "cnn_best_hparams.json", "w") as f:
        hparams_serializable = {k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in best_hparams.items()}
        json.dump(hparams_serializable, f, indent=2)
    print(f"[Config] Best hparams: {best_hparams}")

    # ── Walk-forward loop ──────────────────────────────────────────────────────
    n_folds      = n_months - start_idx
    ic_series    = []
    all_preds    = []
    all_oof      = []
    training_log = []
    all_acts     = []
    wall_times   = []

    print(f"\n[WalkForward] CNN | {n_folds} folds | device={device.type.upper()}")
    print(f"  Filters: {best_hparams.get('n_filters', cfg.n_filters)} | "
          f"Kernels: {best_hparams.get('kernel_sizes', cfg.kernel_sizes)} | "
          f"Seq len: {best_hparams.get('seq_len', cfg.seq_len)}")

    for fold_idx, t in enumerate(range(start_idx, n_months)):
        test_month = all_months[t]
        fold_t0    = time.time()

        result = run_one_fold(
            fold_idx, test_month, all_months, df, cfg, device, best_hparams,
            compute_oof=(fold_idx < 10),
        )

        if result is None:
            continue

        ic_series.append({"Month": result["month"], "IC": result["ic"]})
        all_preds.append(result["preds"])
        if not result["oof"].empty:
            all_oof.append(result["oof"])
        all_acts.append({"Month": result["month"], **result["branch_acts"]})

        for ep_idx, ep_ic in enumerate(result["val_ic_log"]):
            training_log.append({
                "Month":     result["month"],
                "epoch":     ep_idx + 1,
                "val_ic":    ep_ic,
                "fold_time": result["train_time"],
            })

        wall_times.append(time.time() - fold_t0)

        if (fold_idx + 1) % 10 == 0 or fold_idx == 0:
            mean_so_far = np.mean([r["IC"] for r in ic_series]) if ic_series else 0.0
            eta = np.mean(wall_times) * (n_folds - fold_idx - 1) / 60
            print(f"  [Fold {fold_idx+1:3d}/{n_folds}] {str(test_month)[:10]} | "
                  f"IC={result['ic']:+.4f} | mean IC={mean_so_far:+.4f} | "
                  f"ETA={eta:.0f}min")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    if not ic_series:
        print("  ✗ No results.")
        return {}

    ic_df   = pd.DataFrame(ic_series)
    ic_arr  = ic_df["IC"].values
    mean_ic = float(np.mean(ic_arr))
    std_ic  = float(np.std(ic_arr))
    icir    = mean_ic / std_ic if std_ic > 0 else 0.0
    pct_pos = float(np.mean(ic_arr > 0) * 100)
    dw      = float(durbin_watson(ic_arr))

    ic_df["cumulative_IC"] = ic_df["IC"].cumsum()

    summary = {
        "mean_ic":       round(mean_ic, 6),
        "ic_std":        round(std_ic,  6),
        "icir":          round(icir,    6),
        "pct_positive":  round(pct_pos, 2),
        "n_months":      len(ic_arr),
        "durbin_watson": round(dw,      4),
    }

    _print_summary(summary)
    _save_outputs(output_dir, ic_df, all_preds, all_oof, training_log, all_acts, summary)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(s: dict):
    print("\n" + "=" * 70)
    print("  CNN — RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Mean IC         : {s['mean_ic']:+.6f}")
    print(f"  IC Std          :  {s['ic_std']:.6f}")
    print(f"  ICIR            : {s['icir']:+.6f}")
    print(f"  % Positive IC   :  {s['pct_positive']:.2f}%")
    print(f"  Months          :  {s['n_months']}")
    print(f"  Durbin-Watson   :  {s['durbin_watson']:.4f}  (2.0 = no autocorr)")
    print("\n  Baseline comparison:")
    print(f"  {'Model':<20} {'ICIR':>10}")
    print(f"  {'-'*30}")
    print(f"  {'Ridge':.<20} {'0.2504':>10}")
    print(f"  {'LightGBM':.<20} {'0.7347':>10}")
    print(f"  {'Random Forest':.<20} {'0.7248':>10}")
    print(f"  {'CNN (this)':.<20} {s['icir']:>10.4f}")
    print("=" * 70 + "\n")


def _save_outputs(output_dir, ic_df, all_preds, all_oof, training_log, all_acts, summary):
    # IC series
    ic_csv = output_dir / "cnn_ic_series.csv"
    ic_df.to_csv(ic_csv, index=False)
    print(f"  ✓ {ic_csv}")

    # Summary JSON
    with open(output_dir / "cnn_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ {output_dir / 'cnn_summary.json'}")

    # Test predictions parquet
    if all_preds:
        preds_df = pd.concat(all_preds, ignore_index=True)
        pq = output_dir / "cnn_test_predictions.parquet"
        preds_df.to_parquet(pq, engine="pyarrow", compression="snappy", index=False)
        print(f"  ✓ {pq}  ({len(preds_df):,} rows)")

    # OOF parquet
    if all_oof:
        oof_df = pd.concat(all_oof, ignore_index=True)
        pq = output_dir / "cnn_oof_predictions.parquet"
        oof_df.to_parquet(pq, engine="pyarrow", compression="snappy", index=False)
        print(f"  ✓ {pq}  ({len(oof_df):,} rows)")

    # Training log
    if training_log:
        pd.DataFrame(training_log).to_csv(output_dir / "cnn_training_log.csv", index=False)
        print(f"  ✓ {output_dir / 'cnn_training_log.csv'}")

    # Branch activation magnitudes
    if all_acts:
        pd.DataFrame(all_acts).to_csv(output_dir / "cnn_filter_activations.csv", index=False)
        print(f"  ✓ {output_dir / 'cnn_filter_activations.csv'}")
