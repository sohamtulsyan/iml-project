"""
models/mlp.py — MLPModel
PyTorch MLP; uses_sequences=False (flat N×F input).
Architecture: 64→32→16→1 with ReLU + Dropout (per PRD).
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ..shared.walk_forward import BaseModel
from ..shared.device import get_device
from ..shared.metrics import spearman_ic


class _MLP(nn.Module):
    def __init__(self, n_features: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),         nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16),         nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(16, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class MLPModel(BaseModel):
    name           = "mlp"
    uses_sequences = False

    def __init__(
        self,
        device_str:  str   = "auto",
        lr:          float = 3e-4,
        batch_size:  int   = 512,
        max_epochs:  int   = 100,
        patience:    int   = 10,
        grad_clip:   float = 1.0,
        dropout:     float = 0.2,
        weight_decay:float = 1e-4,
        seed:        int   = 42,
    ):
        self.device_str   = device_str
        self.lr           = lr
        self.batch_size   = batch_size
        self.max_epochs   = max_epochs
        self.patience     = patience
        self.grad_clip    = grad_clip
        self.dropout      = dropout
        self.weight_decay = weight_decay
        self.seed         = seed
        self._model       = None
        self._device      = None

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        torch.manual_seed(self.seed)
        self._device = get_device(self.device_str)
        n_feat = X_train.shape[1]
        net    = _MLP(n_feat, self.dropout).to(self._device)

        optimizer = optim.AdamW(net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.lr * 0.01
        )
        criterion = nn.MSELoss()
        use_amp   = (self._device.type == "cuda")
        if use_amp:
            scaler = torch.amp.GradScaler("cuda")

        pin = (self._device.type == "cuda")
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y_train).float(),
            ),
            batch_size=self.batch_size, shuffle=True,
            pin_memory=pin, num_workers=0,
        )
        # Move to device lazily or carefully
        X_val_t = torch.from_numpy(X_val).float()
        if self._device.type == "cuda":
            X_val_t = X_val_t.to(self._device)
        # For MPS, we'll move it in the eval loop to avoid upfront memory spikes

        best_ic, best_state, patience_count = -np.inf, None, 0

        for epoch in range(self.max_epochs):
            net.train()
            for Xb, yb in loader:
                Xb = Xb.to(self._device, non_blocking=True)
                yb = yb.to(self._device, non_blocking=True)
                # z-score targets per batch
                yb = (yb - yb.mean()) / yb.std().clamp(min=1e-8)
                yb = yb.unsqueeze(1)
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        loss = criterion(net(Xb), yb)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss = criterion(net(Xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
                    optimizer.step()
            scheduler.step()

            net.eval()
            with torch.no_grad():
                X_val_batch = X_val_t.to(self._device)
                val_pred = net(X_val_batch).cpu().numpy().flatten()
            val_ic = spearman_ic(y_val, val_pred)

            if val_ic > best_ic:
                best_ic    = val_ic
                best_state = copy.deepcopy(net.state_dict())
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= self.patience:
                break

        if best_state:
            net.load_state_dict(best_state)
        self._model = net

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._model.eval()
        results = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                Xb = torch.from_numpy(X[i:i+self.batch_size]).float().to(self._device)
                results.append(self._model(Xb).cpu().numpy().flatten())
        return np.concatenate(results).astype(np.float32)
