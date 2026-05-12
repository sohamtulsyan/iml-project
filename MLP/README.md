# MLP (Multi-Layer Perceptron) — GPU-Accelerated Neural Network for Equity Return Prediction

## Overview

This directory contains an optimized PyTorch-based MLP (Multi-Layer Perceptron) implementation for cross-sectional equity return prediction. The model is designed for **speed** with full GPU acceleration (Apple Silicon MPS, NVIDIA CUDA, or CPU fallback).

### Key Features
- ✅ **GPU-Accelerated**: Metal Performance Shaders (MPS) on Apple Silicon, CUDA on NVIDIA, CPU fallback
- ✅ **Mixed Precision Training**: FP16 for 2x speedup on GPU
- ✅ **Automatic Device Detection**: Seamlessly switches between MPS > CUDA > CPU
- ✅ **Batch Processing**: Configurable batch sizes for memory efficiency
- ✅ **Early Stopping**: Stops training when validation loss plateaus
- ✅ **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning rates
- ✅ **Grid Search**: Hyperparameter tuning per walk-forward fold using 5-fold CV

---

## Architecture

```
Input (7 features)
    ↓
Dense(64) + ReLU + Dropout(0.2)
    ↓
Dense(32) + ReLU + Dropout(0.2)
    ↓
Dense(16) + ReLU + Dropout(0.1)
    ↓
Dense(1) — Linear activation (regression)
    ↓
Output (continuous return prediction)
```

**Rationale:**
- **ReLU**: Simple, fast, standard for regression; no need for GELU (that's for NLP/transformers)
- **3 Hidden Layers (64→32→16)**: Balance between model capacity and overfitting risk
- **Dropout (0.2, 0.2, 0.1)**: Cross-sectional data is noisy; reduces spurious monthly patterns
- **No Batch Normalization**: Dropout is sufficient for this task

---

## Installation

### 1. Install Dependencies

```bash
# From project root
cd /Users/sohamtulsyan/Documents/Coursework/IML/Project
pip install -r requirements.txt
```

### 2. Verify GPU Support

```python
import torch

# Check if GPU is available
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"MPS Available: {torch.backends.mps.is_available()}")

# Devices
if torch.backends.mps.is_available():
    print("✓ Apple Silicon GPU (Metal Performance Shaders) available")
if torch.cuda.is_available():
    print("✓ NVIDIA GPU (CUDA) available")
```

---

## Usage

### Step 1: Train MLP with Walk-Forward Validation

```bash
cd MLP/
python train_mlp.py
```

**Output:**
- `mlp_ic_results.csv` — Monthly IC values + diagnostics
- `mlp_hyperparams.csv` — Best hyperparameters per fold
- `mlp_training_times.csv` — Training time & epoch counts

**Expected Runtime:**
- ~2-5 minutes per fold on GPU (Apple Silicon MPS or NVIDIA CUDA)
- ~5-15 minutes per fold on CPU
- Total (219 folds): 5-30 hours on GPU, 20-70 hours on CPU

### Step 2: Visualize Results

```bash
python visualize_mlp_results.py
```

**Output (in `MLP/visualizations/`):**
1. `01_ic_timeseries.png` — IC vs Ridge, LightGBM, Random Forest
2. `02_icir_comparison.png` — ICIR bar chart (color-coded by performance)
3. `03_ic_distribution.png` — Histogram + KDE comparison
4. `04_rolling_ic.png` — 60-month rolling IC (signal stability)
5. `05_hyperparam_heatmap.png` — Learning rate vs L2 regularization sensitivity
6. `06_cumulative_returns.png` — Long-short portfolio returns

---

## Configuration

Edit `config.py` to customize:

```python
# Device selection (auto-detect by default)
DEVICE = "auto"  # or "mps", "cuda", "cpu"

# Architecture
HIDDEN_LAYER_SIZES = (64, 32, 16)
DROPOUT_RATES = (0.2, 0.2, 0.1)

# Training
LEARNING_RATE = 0.001
ALPHA = 0.0001  # L2 regularization
BATCH_SIZE = 128
MAX_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 25

# Hyperparameter grid search
PARAM_GRID = {
    'hidden_layer_sizes': [(32, 16, 8), (64, 32, 16), (128, 64, 32)],
    'learning_rate': [0.0005, 0.001, 0.002],
    'alpha': [0.00001, 0.0001, 0.001],
    'batch_size': [64, 128, 256],
}
# Total: 81 combinations × 5-fold CV = 405 model trainings per fold
```

---

## Performance Targets

### Comparison Baselines
| Model | Mean IC | ICIR | % Positive IC |
|-------|---------|------|---------------|
| Ridge (baseline) | 0.0344 | 0.2504 | - |
| LightGBM | 0.0551 | 0.7347 | - |
| Random Forest | 0.0546 | 0.7248 | - |
| **MLP (target)** | **≥0.0400** | **≥0.60** | **≥60%** |

### Success Criteria
- ✅ **Acceptable**: ICIR ≥ 0.25 (beats Ridge baseline)
- ✅ **Good**: ICIR ≥ 0.50
- ✅ **Excellent**: ICIR ≥ 0.60 (approaches tree-based models)

---

## Model Specification

### Input Layer
- **Features**: 7 (BM_sep, lag_mv, OpProf, Inv, Momentum, lag_ret, mktcap)
- **Preprocessing**: Cross-sectional z-score normalization + rank normalization ([0, 1])

### Hyperparameter Grid
| Parameter | Range | Default |
|-----------|-------|---------|
| Hidden Layer Sizes | (32,16,8) / (64,32,16) / (128,64,32) | (64, 32, 16) |
| Learning Rate | 0.0005, 0.001, 0.002 | 0.001 |
| L2 Regularization (alpha) | 0.00001, 0.0001, 0.001 | 0.0001 |
| Batch Size | 64, 128, 256 | 128 |
| Dropout | (0.2, 0.2, 0.1) | - |

### Training Configuration
- **Optimizer**: Adam (β₁=0.9, β₂=0.999, ε=1e-8)
- **Loss**: Mean Squared Error (MSE)
- **Learning Rate Schedule**: ReduceLROnPlateau (factor=0.5, patience=10, min_lr=1e-6)
- **Early Stopping**: Monitor validation loss, patience=25 epochs
- **Batch Size**: 128 (optimal for GPU memory)
- **Max Epochs**: 500
- **Validation Split**: 15% of training data (monthly stratification)

### Walk-Forward Validation
- **Training Window**: 60 months (5 years) rolling
- **Test Window**: 1 month (out-of-sample prediction)
- **Grid Search**: 5-fold cross-validation per fold (on IC metric)
- **Total Folds**: 219 (May 2001 → March 2025)

---

## Optimization Details

### GPU Acceleration

#### Apple Silicon (M1/M2/M3/M4)
- **Framework**: PyTorch with Metal Performance Shaders (MPS)
- **Auto-Detection**: `torch.backends.mps.is_available()`
- **Mixed Precision**: FP16 (2x speedup, minimal precision loss)
- **Expected Speed**: 2-5 min/fold (vs 5-15 min on CPU)

#### NVIDIA GPU (CUDA)
- **Framework**: PyTorch with CUDA
- **Auto-Detection**: `torch.cuda.is_available()`
- **Mixed Precision**: FP16 with `torch.cuda.amp.GradScaler`
- **Expected Speed**: 2-5 min/fold

#### CPU Fallback
- **Framework**: PyTorch on CPU
- **Mixed Precision**: Disabled (no VRAM available)
- **Expected Speed**: 5-15 min/fold (slower but functional)

### Optimization Techniques

1. **Batch Processing**: Configurable batch sizes (default 128) for gradient noise reduction
2. **Early Stopping**: Stops after 25 epochs of no validation improvement (~30-50% time savings)
3. **Learning Rate Scheduling**: Adaptive learning rates reduce oscillations
4. **Vectorized Operations**: All computations use PyTorch/NumPy for efficiency
5. **Memory-Efficient Validation**: Validation loop uses `torch.no_grad()` context

---

## Example Usage

### Train & Evaluate

```python
from mlp_regressor import MLPRegressor
import numpy as np

# Generate sample data (7 features, N stocks)
X_train = np.random.randn(2000, 7)
y_train = np.random.randn(2000)
X_test = np.random.randn(500, 7)

# Create model (auto-detect GPU)
model = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
    learning_rate=0.001,
    alpha=0.0001,
    batch_size=128,
    early_stopping_patience=25,
    device='auto',
    use_mixed_precision=True,
    verbose=True,
)

# Fit on training data
model.fit(X_train, y_train)

# Predict on test data
predictions = model.predict(X_test)

print(f"Predictions shape: {predictions.shape}")
print(f"Predictions mean: {predictions.mean():.4f}")
```

---

## Output Interpretation

### mlp_ic_results.csv
```
month,n_stocks_train,n_stocks_test,ic,best_grid_ic
2005-06-01,2145,2087,0.0523,0.0487
2005-07-01,2156,2102,0.0389,0.0401
...
```

- `ic`: Out-of-sample Information Coefficient (test month prediction)
- `best_grid_ic`: Best CV IC from grid search (validation estimate)
- `n_stocks_train/test`: Number of stocks in each fold

### mlp_hyperparams.csv
```
month,hidden_layer_sizes,learning_rate,alpha,batch_size
2005-06-01,(64, 32, 16),0.001,0.0001,128
2005-07-01,(128, 64, 32),0.0005,0.00001,256
...
```

- Best hyperparameters selected during grid search for each fold

### mlp_training_times.csv
```
month,fold_time_seconds,best_epoch
2005-06-01,342.5,187
2005-07-01,289.3,156
...
```

- `fold_time_seconds`: Wall-clock training time per fold
- `best_epoch`: Epoch where early stopping triggered

---

## Troubleshooting

### Issue: CUDA/MPS Not Detected

```python
import torch
print(torch.cuda.is_available())        # Check NVIDIA GPU
print(torch.backends.mps.is_available()) # Check Apple Silicon GPU
```

**Solution**: Update PyTorch to latest version:
```bash
pip install --upgrade torch torchvision torchaudio
```

### Issue: Out of Memory (OOM)

**Solution**: Reduce batch size in `config.py`:
```python
BATCH_SIZE = 64  # or 32 for smaller GPU VRAM
```

Or reduce hidden layer sizes:
```python
HIDDEN_LAYER_SIZES = (32, 16, 8)  # smaller network
```

### Issue: Training Too Slow

**Solution**: Ensure GPU is being used:
```python
from mlp_regressor import get_device
device = get_device('auto')
print(device)  # Should show 'mps' or 'cuda' if GPU available
```

---

## Files

- `mlp_regressor.py` — Core PyTorch MLP implementation (sklearn-compatible)
- `train_mlp.py` — Walk-forward training script with grid search
- `visualize_mlp_results.py` — Generate 6 high-resolution comparison plots
- `config.py` — Configuration file (defaults, grid search params)
- `README.md` — This file

---

## Future Enhancements

1. **LSTM/Transformer**: Capture temporal dependencies across months
2. **Attention Mechanisms**: Learn feature importance dynamically
3. **Ensemble Methods**: Combine MLP with tree-based models
4. **Calibration**: Post-hoc probability calibration for risk management
5. **Explainability**: SHAP values for feature importance

---

## References

- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- Information Coefficient (IC): Spearman rank correlation between predictions and realized returns
- Early Stopping: Common regularization technique to prevent overfitting
- Learning Rate Scheduling: Adaptive learning rate adjustment for stable convergence

---

## License

Educational use only. Part of IML Project (Cross-Sectional Equity Return Prediction).

---

## Support

For issues or questions:
1. Check GPU availability: `python -c "import torch; print(torch.backends.mps.is_available())"`
2. Review `config.py` for parameter tuning
3. Check output CSVs for diagnostic information
4. Run `visualize_mlp_results.py` for performance comparison

---

**Last Updated**: May 2026
