"""
================================================================================
  TRANSFORMER PIPELINE — TRAINING LOOP
================================================================================

  Features:
  ─────────
  - AdamW optimizer + CosineAnnealingLR scheduler
  - CUDA: AMP (autocast + GradScaler), torch.compile if PyTorch >= 2.0
  - MPS / CPU: float32 throughout (no AMP)
  - Gradient clipping (max_norm=1.0)
  - Early stopping on validation Spearman IC with best-weight restoration
  - Cross-sectional z-score of targets inside each batch (per PRD)
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr
from typing import Optional, Tuple


def _zscore_targets(y: torch.Tensor) -> torch.Tensor:
    """Z-score a (N,) or (N, 1) target tensor in-place within a batch."""
    mu    = y.mean()
    sigma = y.std().clamp(min=1e-8)
    return (y - mu) / sigma


def spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation — the primary evaluation metric."""
    if len(y_true) < 5:
        return 0.0
    r, _ = spearmanr(y_true, y_pred)
    return float(r) if not np.isnan(r) else 0.0


def train_one_fold(
    model:         nn.Module,
    X_train:       np.ndarray,   # (N_train, T, F)
    y_train:       np.ndarray,   # (N_train,)
    X_val:         np.ndarray,   # (N_val, T, F)
    y_val:         np.ndarray,   # (N_val,)
    device:        torch.device,
    lr:            float = 3e-4,
    weight_decay:  float = 1e-4,
    batch_size:    int   = 512,
    max_epochs:    int   = 100,
    patience:      int   = 10,
    grad_clip:     float = 1.0,
    verbose:       bool  = False,
) -> Tuple[nn.Module, int, list]:
    """
    Train model for one walk-forward fold.

    Returns
    -------
    model        : best model (weights restored to best val IC epoch)
    best_epoch   : epoch index of best val IC
    val_ic_log   : list of val IC per epoch
    """
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr * 0.01
    )
    criterion = nn.MSELoss()

    # ── DataLoader ─────────────────────────────────────────────────────────────
    X_t = torch.from_numpy(X_train).float()
    y_t = torch.from_numpy(y_train).float()
    pin = (device.type == "cuda")
    num_workers = 0  # 0 is safest; avoids fork issues on MPS/macOS

    loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size  = batch_size,
        shuffle     = True,
        drop_last   = False,
        pin_memory  = pin,
        num_workers = num_workers,
    )

    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_np = y_val  # keep as numpy for spearmanr

    # ── AMP setup (CUDA only) ─────────────────────────────────────────────────
    use_amp = (device.type == "cuda")
    if use_amp:
        amp_scaler = torch.amp.GradScaler("cuda")

    # ── torch.compile (CUDA + PyTorch >= 2.0) ────────────────────────────────
    _torch_major = int(torch.__version__.split(".")[0])
    if device.type == "cuda" and _torch_major >= 2:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            if verbose:
                print("[Trainer] torch.compile enabled")
        except Exception:
            pass  # compile is optional; continue without it

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_ic    = -np.inf
    patience_count = 0
    best_epoch     = 0
    best_state     = None
    val_ic_log     = []

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            # Cross-sectional z-score of targets (inside each batch)
            y_batch_z = _zscore_targets(y_batch).unsqueeze(1)  # (N, 1)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    pred = model(X_batch)          # (N, 1)
                    loss = criterion(pred, y_batch_z)
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                pred = model(X_batch)
                loss = criterion(pred, y_batch_z)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)

        scheduler.step()
        epoch_loss /= len(X_t)

        # ── Validation IC ────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).cpu().numpy().flatten()
        val_ic = spearman_ic(y_val_np, val_pred)
        val_ic_log.append(val_ic)

        if val_ic > best_val_ic:
            best_val_ic    = val_ic
            best_epoch     = epoch
            patience_count = 0
            best_state     = copy.deepcopy(model.state_dict())
        else:
            patience_count += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  [Ep {epoch+1:3d}] loss={epoch_loss:.5f} | "
                  f"val_IC={val_ic:+.4f} | best={best_val_ic:+.4f} | "
                  f"pat={patience_count}/{patience}")

        if patience_count >= patience:
            if verbose:
                print(f"  [Early stop] epoch {epoch+1}, best epoch {best_epoch+1}")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_epoch, val_ic_log


def predict(
    model:  nn.Module,
    X:      np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Run inference in batches (avoids OOM on large cross-sections).
    Returns flat (N,) float32 array.
    """
    model.eval()
    results = []
    n = len(X)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            X_batch = torch.from_numpy(X[i:i+batch_size]).float().to(device)
            out = model(X_batch).cpu().numpy().flatten()
            results.append(out)

    return np.concatenate(results).astype(np.float32)
