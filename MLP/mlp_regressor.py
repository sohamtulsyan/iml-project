"""
================================================================================
  MLP (MULTI-LAYER PERCEPTRON) — GPU-ACCELERATED NEURAL NETWORK REGRESSOR
  High-performance PyTorch implementation with Metal Performance Shaders (MPS)
================================================================================

  Architecture:
  ─────────────
  - Input: 7 features (BM_sep, lag_mv, OpProf, Inv, Momentum, lag_ret, mktcap)
  - Hidden: 3 layers (64→32→16) with ReLU + Dropout
  - Output: Single continuous value (forward return prediction)
  - GPU: Apple Silicon (MPS) or falls back to CPU

  Key optimizations:
  ──────────────────
  1. GPU acceleration via PyTorch (Metal Performance Shaders on macOS)
  2. Batch processing with configurable batch size
  3. Mixed precision training (CUDA only; MPS uses float32)
  4. Learning rate scheduling with early stopping
  5. Best-weights restoration on early stopping

  Usage:
  ──────
    from mlp_regressor import MLPRegressor
    model = MLPRegressor(device='auto')  # Auto-detect GPU
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

================================================================================
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
#  GPU DEVICE MANAGEMENT (CACHED - CALLED ONCE)
# ═══════════════════════════════════════════════════════════════════════════════

_DEVICE_CACHE = {}  # Cache to avoid repeated device detection

def get_device(device: str = "auto", verbose: bool = True) -> torch.device:
    """
    Auto-detect and return optimal device (MPS > CUDA > CPU).
    Results are cached to avoid repeated detection and printing.
    
    Parameters
    ----------
    device : str
        "auto" (default) — auto-detect once
        "mps" — force Metal Performance Shaders (Apple Silicon)
        "cuda" — force CUDA (NVIDIA)
        "cpu" — force CPU
    verbose : bool
        Whether to print device info (default True)
    
    Returns
    -------
    torch.device
        Selected device
    """
    cache_key = device
    if cache_key in _DEVICE_CACHE:
        return _DEVICE_CACHE[cache_key]
    
    if device == "auto":
        if torch.backends.mps.is_available():
            selected_device = "mps"
        elif torch.cuda.is_available():
            selected_device = "cuda"
        else:
            selected_device = "cpu"
    else:
        selected_device = device
    
    torch_device = torch.device(selected_device)
    
    if verbose:
        device_name = {
            "mps": "Apple Silicon GPU (Metal Performance Shaders)",
            "cuda": "NVIDIA GPU (CUDA)",
            "cpu": "CPU"
        }.get(selected_device, selected_device)
        print(f"[GPU] Using device: {device_name}")
    
    _DEVICE_CACHE[cache_key] = torch_device
    return torch_device


# ═══════════════════════════════════════════════════════════════════════════════
#  PYTORCH MLP ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

class MLPNet(nn.Module):
    """High-performance MLP with configurable architecture and dropout."""
    
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dims: tuple = (64, 32, 16),
        dropout_rates: tuple = (0.2, 0.2, 0.1),
        use_batch_norm: bool = False,
    ):
        """
        Initialize MLP network.
        
        Parameters
        ----------
        input_dim : int
            Number of input features (default 7)
        hidden_dims : tuple
            Hidden layer dimensions (default (64, 32, 16))
        dropout_rates : tuple
            Dropout rates for each hidden layer (default (0.2, 0.2, 0.1))
        use_batch_norm : bool
            Whether to use batch normalization (default False)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rates[i]))
            
            prev_dim = hidden_dim
        
        # Output layer (linear activation for regression)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """He initialization for ReLU networks."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  SKLEARN-COMPATIBLE WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class MLPRegressor:
    """
    GPU-accelerated MLP regressor with sklearn-compatible interface.
    
    Supports:
    - Automatic device detection (MPS > CUDA > CPU)
    - Mixed precision training (CUDA only; MPS uses float32 natively)
    - Early stopping with best-weight restoration
    - Learning rate scheduling (ReduceLROnPlateau)
    - Batch processing for memory efficiency
    """
    
    def __init__(
        self,
        hidden_layer_sizes: tuple = (64, 32, 16),
        dropout_rates: tuple = (0.2, 0.2, 0.1),
        learning_rate: float = 0.001,
        alpha: float = 0.0001,  # L2 regularization (weight decay)
        batch_size: int = 128,
        max_epochs: int = 500,
        early_stopping_patience: int = 25,
        early_stopping_min_delta: float = 0.0001,
        validation_fraction: float = 0.15,
        device: str = "auto",
        use_mixed_precision: bool = True,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rates = dropout_rates
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.validation_fraction = validation_fraction
        self.device = get_device(device, verbose=False)
        # BUG FIX: Mixed precision (GradScaler) is CUDA-only; MPS uses float32.
        self.use_mixed_precision = use_mixed_precision and self.device.type == "cuda"
        self.random_state = random_state
        self.verbose = verbose
        
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        self.model = None
        self.optimizer = None
        self.best_loss = float('inf')
        self.best_epoch = 0
        
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> 'MLPRegressor':
        """
        Fit the MLP model.
        
        Parameters
        ----------
        X : np.ndarray  shape (N, 7)
        y : np.ndarray  shape (N,)
        X_val : np.ndarray, optional  — if None, splits validation from X
        y_val : np.ndarray, optional
        
        Returns
        -------
        self
        """
        # Validation split
        if X_val is None:
            n_val = max(1, int(len(X) * self.validation_fraction))
            rng = np.random.default_rng(self.random_state)
            idx = rng.permutation(len(X))
            val_idx, train_idx = idx[:n_val], idx[n_val:]
            X_val, y_val = X[val_idx], y[val_idx]
            X, y = X[train_idx], y[train_idx]
        
        # To tensors
        X_t     = torch.from_numpy(X.astype(np.float32)).to(self.device)
        y_t     = torch.from_numpy(y.astype(np.float32)).view(-1, 1).to(self.device)
        X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(self.device)
        y_val_t = torch.from_numpy(y_val.astype(np.float32)).view(-1, 1).to(self.device)
        
        # Build model
        self.model = MLPNet(
            input_dim=X.shape[1],
            hidden_dims=self.hidden_layer_sizes,
            dropout_rates=self.dropout_rates,
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.alpha,
        )
        
        # BUG FIX: removed deprecated `verbose` kwarg from ReduceLROnPlateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )
        
        criterion = nn.MSELoss()
        
        # BUG FIX: GradScaler is CUDA-only; use torch.amp.GradScaler for CUDA
        if self.use_mixed_precision:
            amp_scaler = torch.amp.GradScaler("cuda")
        
        train_loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        
        best_val_loss  = float('inf')
        patience_counter = 0
        # BUG FIX: save best weights for restoration on early stopping
        best_state_dict = None
        
        if self.verbose:
            print(f"\n[Training] Epochs: {self.max_epochs}, Batch: {self.batch_size}, "
                  f"Device: {self.device.type.upper()}")
        
        for epoch in range(self.max_epochs):
            # ── Train ──────────────────────────────────────────────────────────
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad(set_to_none=True)   # slightly faster than zero_grad()
                
                if self.use_mixed_precision:
                    with torch.amp.autocast("cuda"):
                        y_pred = self.model(X_batch)
                        loss   = criterion(y_pred, y_batch)
                    amp_scaler.scale(loss).backward()
                    amp_scaler.step(self.optimizer)
                    amp_scaler.update()
                else:
                    y_pred = self.model(X_batch)
                    loss   = criterion(y_pred, y_batch)
                    loss.backward()
                    self.optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(X_t)
            
            # ── Validate ───────────────────────────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
            
            scheduler.step(val_loss)
            
            # ── Early stopping & best-weight tracking ─────────────────────────
            if val_loss < best_val_loss - self.early_stopping_min_delta:
                best_val_loss  = val_loss
                patience_counter = 0
                self.best_epoch  = epoch
                # BUG FIX: deep-copy weights at best validation loss
                best_state_dict = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
            
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"  [Ep {epoch+1:3d}] Train: {train_loss:.6f} | "
                      f"Val: {val_loss:.6f} | Pat: {patience_counter}/{self.early_stopping_patience}")
            
            if patience_counter >= self.early_stopping_patience:
                if self.verbose:
                    print(f"\n[Early Stop] epoch {epoch+1}, best epoch {self.best_epoch+1}")
                break
        
        # BUG FIX: restore best weights
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray  shape (N, 7)
        
        Returns
        -------
        np.ndarray  shape (N,)
        """
        if self.model is None:
            raise ValueError("Model must be fit before calling predict()")
        
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_t)
        
        return y_pred.cpu().numpy().flatten()


if __name__ == "__main__":
    print("Testing MLP Regressor...")
    
    device = get_device("auto")
    
    X_train = np.random.randn(1000, 7).astype(np.float32)
    y_train = np.random.randn(1000).astype(np.float32)
    X_test  = np.random.randn(100, 7).astype(np.float32)
    
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        batch_size=128,
        max_epochs=50,
        early_stopping_patience=10,
        verbose=True,
    )
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    print(f"✓ Test passed! Predictions shape: {preds.shape}")
