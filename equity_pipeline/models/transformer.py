"""
models/transformer.py — TransformerModel
Ported from Transformer/ directory (architecture unchanged).
FIX: OOF generated for ALL folds (no fold_idx < 10 guard).
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


# ── Sinusoidal positional encoding ────────────────────────────────────────────
class _SinPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class _TransformerNet(nn.Module):
    def __init__(self, n_features, seq_len=24, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = _SinPE(d_model, max_len=seq_len + 10, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        x = self.pos_enc(self.input_proj(x))
        T = x.size(1)
        x = self.encoder(x, mask=self.causal_mask[:T, :T])
        return self.head(x[:, -1, :])   # use last time step


class TransformerModel(BaseModel):
    name           = "transformer"
    uses_sequences = True

    def __init__(
        self,
        device_str:      str   = "auto",
        d_model:         int   = 64,
        nhead:           int   = 4,
        num_layers:      int   = 2,
        dim_feedforward: int   = 256,
        dropout:         float = 0.1,
        lr:              float = 3e-4,
        batch_size:      int   = 512,
        max_epochs:      int   = 100,
        patience:        int   = 10,
        grad_clip:       float = 1.0,
        weight_decay:    float = 1e-4,
        seed:            int   = 42,
    ):
        self.device_str      = device_str
        self.d_model         = d_model
        self.nhead           = nhead
        self.num_layers      = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout         = dropout
        self.lr              = lr
        self.batch_size      = batch_size
        self.max_epochs      = max_epochs
        self.patience        = patience
        self.grad_clip       = grad_clip
        self.weight_decay    = weight_decay
        self.seed            = seed
        self._model          = None
        self._device         = get_device(self.device_str)

    def _build_net(self, n_features, seq_len):
        return _TransformerNet(
            n_features=n_features, seq_len=seq_len,
            d_model=self.d_model, nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        torch.manual_seed(self.seed)
        if self._device.type == "mps":
            torch.mps.empty_cache()

        n_feat, seq_len = X_train.shape[2], X_train.shape[1]
        net = self._build_net(n_feat, seq_len).to(self._device)

        use_amp = (self._device.type == "cuda")
        if use_amp:
            scaler = torch.amp.GradScaler("cuda")
        if self._device.type == "cuda" and int(torch.__version__.split(".")[0]) >= 2:
            try:
                net = torch.compile(net, mode="reduce-overhead")
            except Exception:
                pass

        optimizer = optim.AdamW(net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.lr * 0.01
        )
        criterion = nn.MSELoss()
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
                is_mps = (self._device.type == "mps")
                Xb = Xb.to(self._device, non_blocking=not is_mps)
                yb = yb.to(self._device, non_blocking=not is_mps)
                yb = ((yb - yb.mean()) / yb.std().clamp(min=1e-8)).unsqueeze(1)
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
