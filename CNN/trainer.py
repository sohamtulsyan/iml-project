"""
================================================================================
  CNN PIPELINE — TRAINING LOOP
================================================================================

  Imports shared utilities from the Transformer pipeline (same project).
  Only the model type changes; all training logic is identical:
    - AdamW + CosineAnnealingLR
    - CUDA: AMP (autocast + GradScaler), torch.compile
    - MPS/CPU: float32 throughout
    - Gradient clipping, early stopping, best-weight restoration
    - Batch-level z-score of targets
"""

import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from scipy.stats import spearmanr

# ── Shared utility from Transformer pipeline ──────────────────────────────────
_TRANSFORMER_DIR = Path(__file__).parent.parent / "Transformer"
sys.path.insert(0, str(_TRANSFORMER_DIR))
from trainer import spearman_ic   # reuse the shared helper


def _zscore(y: torch.Tensor) -> torch.Tensor:
    mu = y.mean()
    sd = y.std().clamp(min=1e-8)
    return (y - mu) / sd


def train_one_fold(
    model,
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    X_val:       np.ndarray,
    y_val:       np.ndarray,
    device:      torch.device,
    lr:          float = 3e-4,
    weight_decay: float = 1e-4,
    batch_size:  int   = 512,
    max_epochs:  int   = 100,
    patience:    int   = 10,
    grad_clip:   float = 1.0,
    verbose:     bool  = False,
):
    """
    Train TemporalCNN for one walk-forward fold.
    Interface identical to Transformer's train_one_fold.

    Returns (model_best, best_epoch, val_ic_log)
    """
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr * 0.01
    )
    criterion = nn.MSELoss()

    X_t = torch.from_numpy(X_train).float()
    y_t = torch.from_numpy(y_train).float()
    pin = (device.type == "cuda")

    loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size  = batch_size,
        shuffle     = True,
        drop_last   = False,
        pin_memory  = pin,
        num_workers = 0,
    )

    X_val_t  = torch.from_numpy(X_val).float().to(device)
    y_val_np = y_val

    # AMP — CUDA only (MPS unsupported)
    use_amp = (device.type == "cuda")
    if use_amp:
        amp_scaler = torch.amp.GradScaler("cuda")

    # torch.compile — CUDA + PyTorch >= 2.0
    if device.type == "cuda" and int(torch.__version__.split(".")[0]) >= 2:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass

    best_val_ic    = -np.inf
    patience_count = 0
    best_epoch     = 0
    best_state     = None
    val_ic_log     = []

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0

        for X_b, y_b in loader:
            X_b = X_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            y_b_z = _zscore(y_b).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    pred = model(X_b)
                    loss = criterion(pred, y_b_z)
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                pred = model(X_b)
                loss = criterion(pred, y_b_z)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            epoch_loss += loss.item() * X_b.size(0)

        scheduler.step()
        epoch_loss /= len(X_t)

        # Validation IC
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
                print(f"  [Early stop] ep {epoch+1}, best ep {best_epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_epoch, val_ic_log


def predict(model, X: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    """Batched inference — returns flat (N,) float32."""
    model.eval()
    results = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            Xb = torch.from_numpy(X[i:i+batch_size]).float().to(device)
            results.append(model(Xb).cpu().numpy().flatten())
    return np.concatenate(results).astype(np.float32)
