"""
models/cnn.py — CNNModel
Multi-Scale Temporal CNN ported from CNN/ directory.
CausalConv1d (left-only padding), 3 branches (k=3,6,12),
global avg+max pool, uses_sequences=True.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ..shared.walk_forward import BaseModel
from ..shared.device import get_device
from ..shared.metrics import spearman_ic


class _CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, bias=False):
        super().__init__()
        self.left_pad = kernel_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_pad, 0)))


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.conv = _CausalConv1d(in_ch, out_ch, kernel_size)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.GELU()
        self.skip = (in_ch == out_ch)

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        return out + x if self.skip else out


class _TemporalCNN(nn.Module):
    def __init__(self, n_features=6, n_filters=32,
                 kernel_sizes=(3, 6, 12), n_blocks=1, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            _CausalConv1d(n_features, n_filters, 1),
            nn.BatchNorm1d(n_filters), nn.GELU(),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(*[_ConvBlock(n_filters, n_filters, k) for _ in range(n_blocks)])
            for k in kernel_sizes
        ])
        self.head = nn.Sequential(
            nn.Linear(6 * n_filters, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):                  # x: (B, T, F)
        x = x.transpose(1, 2)             # (B, F, T)
        x = self.proj(x)                  # (B, n_filters, T)
        outs = [b(x) for b in self.branches]
        cat  = torch.cat(outs, dim=1)     # (B, 3*n_filters, T)
        pooled = torch.cat([cat.mean(dim=2), cat.max(dim=2).values], dim=1)
        return self.head(pooled)           # (B, 1)


class CNNModel(BaseModel):
    name           = "cnn"
    uses_sequences = True

    def __init__(
        self,
        device_str:  str   = "auto",
        n_filters:   int   = 32,
        kernel_sizes: tuple = (3, 6, 12),
        n_conv_blocks:int   = 1,
        dropout:     float = 0.1,
        lr:          float = 3e-4,
        batch_size:  int   = 256,
        max_epochs:  int   = 100,
        patience:    int   = 10,
        grad_clip:   float = 1.0,
        weight_decay:float = 1e-4,
        seed:        int   = 42,
    ):
        self.device_str   = device_str
        self.n_filters    = n_filters
        self.kernel_sizes = kernel_sizes
        self.n_conv_blocks= n_conv_blocks
        self.dropout      = dropout
        self.lr           = 1e-4
        self.batch_size   = 256
        self.max_epochs   = max_epochs
        self.patience     = patience
        self.grad_clip    = grad_clip
        self.weight_decay = weight_decay
        self.seed         = seed
        self._model       = None
        self._device      = get_device(self.device_str)

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        print(f"  [{self.name}] fit: starting (device={self._device})")
        torch.manual_seed(self.seed)
        if self._device.type == "mps":
            print(f"  [{self.name}] fit: clearing mps cache")
            torch.mps.empty_cache()
            torch.mps.synchronize()

        n_feat = X_train.shape[2]
        print(f"  [{self.name}] fit: building net (n_feat={n_feat})")
        net = _TemporalCNN(n_feat, self.n_filters,
                           self.kernel_sizes, self.n_conv_blocks,
                           self.dropout).to(self._device)
        print(f"  [{self.name}] fit: net built and moved")

        use_amp = (self._device.type == "cuda")
        if use_amp:
            scaler = torch.amp.GradScaler("cuda")

        optimizer = optim.AdamW(net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        
        # OneCycleLR is perfect for "super-convergence" in very few epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr,
            epochs=self.max_epochs, steps_per_epoch=len(loader),
            pct_start=0.3, div_factor=10, final_div_factor=100
        )
        criterion = nn.MSELoss()
        pin = (self._device.type == "cuda")

        print(f"  [{self.name}] fit: creating dataloader")
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y_train).float(),
            ),
            batch_size=self.batch_size, shuffle=True,
            pin_memory=pin, num_workers=0,
        )
        print(f"  [{self.name}] fit: dataloader created")

        # Move to device lazily or carefully
        X_val_t = torch.from_numpy(X_val).float()
        if self._device.type == "cuda":
            X_val_t = X_val_t.to(self._device)
        # For MPS, we'll move it in the eval loop to avoid upfront memory spikes

        best_ic, best_state, patience_count = -np.inf, None, 0

        for epoch in range(self.max_epochs):
            net.train()
            for Xb, yb in loader:
                is_mps = (self._device.type == "mps")
                Xb = Xb.to(self._device, non_blocking=not is_mps)
                yb = yb.to(self._device, non_blocking=not is_mps)
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
            # Normalize validation targets to match training scale for IC calculation
            y_val_norm = (y_val - y_val.mean()) / y_val.std().clip(min=1e-8)
            val_ic = spearman_ic(y_val_norm, val_pred)

            if val_ic > best_ic:
                best_ic = val_ic; best_state = copy.deepcopy(net.state_dict())
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
