"""
================================================================================
  TRANSFORMER PIPELINE — HYPERPARAMETER SEARCH
================================================================================

  Strategy:
  ─────────
  - Run on first walk-forward fold only (first 24 validation months)
  - Criterion: ICIR on the 6-month validation split inside the training window
  - Parallelized across CPU cores via joblib (GPU path: sequential, max batches)
  - Invalid combos (nhead doesn't divide d_model) are skipped automatically
  - Returns best config dict frozen for all subsequent folds
"""

import time
import copy
import itertools
import numpy as np
from typing import Dict, Any, List

from joblib import Parallel, delayed


def _is_valid_combo(hparams: dict) -> bool:
    """Check that nhead divides d_model."""
    return hparams["d_model"] % hparams["nhead"] == 0


def _evaluate_combo(
    hparams:    dict,
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    device_str: str,
    seed:       int = 42,
) -> float:
    """
    Train one config and return validation IC.
    Designed to be called from joblib.Parallel.
    device_str is a string so it survives pickling.
    """
    import torch
    from model   import build_model, TransformerEncoderModel
    from trainer import train_one_fold, predict, spearman_ic
    from config  import TransformerConfig

    device = torch.device(device_str)

    # Build a minimal config-like object
    cfg = TransformerConfig(
        d_model        = hparams["d_model"],
        nhead          = hparams["nhead"],
        num_layers     = hparams["num_layers"],
        dim_feedforward = hparams["dim_feedforward"],
        dropout        = hparams["dropout"],
        seq_len        = hparams.get("seq_len", 24),
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(cfg, n_features=X_train.shape[2])

    try:
        model, best_ep, val_ics = train_one_fold(
            model       = model,
            X_train     = X_train,
            y_train     = y_train,
            X_val       = X_val,
            y_val       = y_val,
            device      = device,
            lr          = hparams.get("lr", 3e-4),
            batch_size  = hparams.get("batch_size", 512),
            max_epochs  = 30,    # reduced for search
            patience    = 5,
            verbose     = False,
        )
        preds  = predict(model, X_val, device)
        val_ic = spearman_ic(y_val, preds)
        return val_ic
    except Exception as e:
        return -np.inf


def hyperparameter_search(
    search_space: dict,
    X_train:      np.ndarray,
    y_train:      np.ndarray,
    X_val:        np.ndarray,
    y_val:        np.ndarray,
    device,
    n_jobs:       int  = -1,
    seed:         int  = 42,
    verbose:      bool = True,
) -> Dict[str, Any]:
    """
    Grid search over search_space, parallelized via joblib.

    Parameters
    ----------
    search_space : dict  {param_name: [values]}
    Returns
    -------
    best_hparams : dict  winning hyperparameter configuration
    """
    # Enumerate all valid combos
    keys   = list(search_space.keys())
    combos = [
        dict(zip(keys, vals))
        for vals in itertools.product(*search_space.values())
        if _is_valid_combo(dict(zip(keys, vals)))
    ]

    n_combos = len(combos)
    if verbose:
        print(f"[HparamSearch] {n_combos} valid combos "
              f"(of {len(list(itertools.product(*search_space.values())))} total)")

    device_str = str(device)

    # On CUDA: run sequentially (CUDA context can't be forked)
    # On MPS/CPU: parallelize across cores
    if device.type == "cuda":
        scores = [
            _evaluate_combo(c, X_train, y_train, X_val, y_val, device_str, seed)
            for c in combos
        ]
    else:
        scores = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_evaluate_combo)(c, X_train, y_train, X_val, y_val, device_str, seed)
            for c in combos
        )

    best_idx = int(np.argmax(scores))
    best     = combos[best_idx]

    if verbose:
        print(f"[HparamSearch] Best combo: {best}  →  val IC = {scores[best_idx]:.4f}")

    return best
