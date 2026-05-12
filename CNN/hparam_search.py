"""
================================================================================
  CNN PIPELINE — HYPERPARAMETER SEARCH
================================================================================

  Same strategy as Transformer: tune on first fold only, ICIR criterion.
  Parallelized via joblib (threads on MPS/CPU; sequential on CUDA).
  Invalid combos (non-integer kernel sizes, etc.) auto-filtered.
"""

import sys
import itertools
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed

# ── Shared spearman_ic from Transformer ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "Transformer"))
from trainer import spearman_ic


def _evaluate_combo(
    hparams:    dict,
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    device_str: str,
    seed:       int = 42,
) -> float:
    """Train one CNN config and return val IC. Runs in joblib worker."""
    import torch
    import numpy as np
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from model   import build_model
    from trainer import train_one_fold, predict

    device = torch.device(device_str)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(hparams, n_features=X_train.shape[2])

    try:
        model, _, _ = train_one_fold(
            model, X_train, y_train, X_val, y_val, device,
            lr         = hparams.get("lr", 3e-4),
            batch_size = hparams.get("batch_size", 512),
            max_epochs = 20,    # reduced for search speed
            patience   = 5,
            verbose    = False,
        )
        preds = predict(model, X_val, device)
        return spearman_ic(y_val, preds)
    except Exception:
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
) -> dict:
    """Grid search → return best hyperparameter dict."""
    keys   = list(search_space.keys())
    combos = [dict(zip(keys, vals))
              for vals in itertools.product(*search_space.values())]

    if verbose:
        print(f"[HparamSearch] {len(combos)} combos to evaluate")

    device_str = str(device)

    # CUDA: sequential (can't fork CUDA context)
    # MPS / CPU: parallel threads
    if device.type == "cuda":
        scores = [_evaluate_combo(c, X_train, y_train, X_val, y_val, device_str, seed)
                  for c in combos]
    else:
        scores = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_evaluate_combo)(c, X_train, y_train, X_val, y_val, device_str, seed)
            for c in combos
        )

    best_idx = int(np.argmax(scores))
    best     = combos[best_idx]

    if verbose:
        print(f"[HparamSearch] Best: {best}  val_IC={scores[best_idx]:.4f}")

    return best
