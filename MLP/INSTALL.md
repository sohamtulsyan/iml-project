# Installation & Setup Guide

## Quick Start (3 steps)

### 1. Install PyTorch with GPU Support

```bash
# macOS (Apple Silicon M1/M2/M3/M4 - includes MPS support)
pip install torch torchvision torchaudio

# Linux/Windows with NVIDIA GPU (CUDA support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only (slower)
pip install torch torchvision torchaudio
```

### 2. Install Project Dependencies

```bash
cd /Users/sohamtulsyan/Documents/Coursework/IML/Project
pip install -r requirements.txt
```

### 3. Verify GPU Availability

```bash
cd MLP/
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Detailed Installation

### macOS (Apple Silicon M1/M2/M3/M4)

**Best for:** Maximum speed with Metal Performance Shaders (MPS)

```bash
# 1. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install PyTorch with MPS support
pip install torch torchvision torchaudio

# 4. Install project dependencies
pip install -r requirements.txt

# 5. Verify installation
python << 'EOF'
import torch
import numpy as np
import pandas as pd

print("✓ Core imports successful")
print(f"  • PyTorch: {torch.__version__}")
print(f"  • NumPy: {np.__version__}")
print(f"  • Pandas: {pd.__version__}")

if torch.backends.mps.is_available():
    print(f"  • GPU (MPS): Available ✓")
    device = torch.device("mps")
    x = torch.randn(100, 100, device=device)
    y = torch.matmul(x, x)
    print(f"    - MPS test: PASSED")
else:
    print(f"  • GPU (MPS): NOT available")

EOF
```

### Linux (NVIDIA GPU - CUDA 11.8)

**Best for:** NVIDIA GPUs (RTX 3000+, A100, etc.)

```bash
# 1. Install CUDA toolkit (if not already installed)
# See: https://developer.nvidia.com/cuda-downloads

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install project dependencies
pip install -r requirements.txt

# 5. Verify installation
python << 'EOF'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  • GPU: {torch.cuda.get_device_name(0)}")
    print(f"  • VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test CUDA
    x = torch.randn(100, 100, device='cuda')
    y = torch.matmul(x, x)
    print(f"  • CUDA test: PASSED ✓")

EOF
```

### Windows (CPU or NVIDIA GPU)

**Best for:** Windows machines with or without NVIDIA GPU

```batch
REM 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

REM 2. Install PyTorch (choose one)
REM For NVIDIA GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM For CPU only:
pip install torch torchvision torchaudio

REM 3. Install project dependencies
pip install -r requirements.txt

REM 4. Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```bash
# Reinstall PyTorch
pip install --upgrade torch torchvision torchaudio

# Or use specific index URL
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
```

### Issue: MPS/CUDA Not Detected

```python
import torch

# Check what's available
print(f"MPS Available: {torch.backends.mps.is_available()}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Force CPU if needed
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size in `config.py`
```python
BATCH_SIZE = 64  # or 32
```

Or reduce hidden layers:
```python
HIDDEN_LAYER_SIZES = (32, 16, 8)  # smaller network
```

### Issue: Very Slow Training (using CPU)

**Solution:** Ensure PyTorch is using GPU:
```python
from mlp_regressor import get_device
device = get_device('auto')
print(device)  # Should show 'mps' or 'cuda', not 'cpu'
```

---

## Environment Variables (Optional)

### Control GPU Usage

```bash
# Disable GPU (force CPU)
export CUDA_VISIBLE_DEVICES=""

# Use specific GPU (for multi-GPU setups)
export CUDA_VISIBLE_DEVICES=0

# MacOS: Force CPU for debugging
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Memory Management

```bash
# Reduce PyTorch memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# For MPS (MacOS)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

---

## Performance Tuning

### For Maximum Speed

**GPU (Apple Silicon MPS):**
```python
# config.py
DEVICE = "mps"
USE_MIXED_PRECISION = True  # FP16 for 2x speedup
BATCH_SIZE = 256            # Larger batch = faster
```

**GPU (NVIDIA CUDA):**
```python
# config.py
DEVICE = "cuda"
USE_MIXED_PRECISION = True  # FP16 for 2x speedup
BATCH_SIZE = 256            # Larger batch = faster
```

### For Memory Efficiency

```python
# config.py
BATCH_SIZE = 64             # Smaller batch = less memory
HIDDEN_LAYER_SIZES = (32, 16, 8)  # Smaller network
EARLY_STOPPING_PATIENCE = 20       # Faster convergence
```

---

## Verify Installation

```bash
cd MLP/

# Run minimal test
python << 'EOF'
from mlp_regressor import MLPRegressor, get_device
import numpy as np

# Check device
device = get_device('auto')
print(f"✓ Device: {device}")

# Create dummy data
X = np.random.randn(100, 7)
y = np.random.randn(100)

# Train tiny model
model = MLPRegressor(
    hidden_layer_sizes=(16, 8),
    batch_size=32,
    max_epochs=5,
    verbose=False
)
model.fit(X, y)

# Predict
pred = model.predict(X)
print(f"✓ Model training/inference: OK")
print(f"✓ Predictions shape: {pred.shape}")
print("\nInstallation verified! Ready to train MLP.")

EOF
```

---

## Next Steps

Once installation is verified, run the full pipeline:

```bash
# From MLP directory
python run_mlp_pipeline.py

# Or run individually:
python train_mlp.py           # Training
python visualize_mlp_results.py  # Visualization
```

---

## System Requirements

### Minimum
- Python 3.9+
- 4 GB RAM
- CPU: Modern processor (Intel i5/i7 or equivalent)
- GPU: Optional (CPU fallback supported)

### Recommended
- Python 3.10+
- 16 GB RAM
- GPU: Apple Silicon M1+ or NVIDIA RTX 3060+
- SSD: 20 GB free space

### Optimal
- Python 3.11+
- 32 GB RAM
- GPU: Apple Silicon M2/M3+ or NVIDIA A100
- SSD: 50 GB free space

---

## Conda Installation (Alternative)

```bash
# Create conda environment
conda create -n mlp python=3.11

# Activate environment
conda activate mlp

# Install PyTorch (macOS)
conda install pytorch torchvision torchaudio -c pytorch

# Install dependencies
pip install -r requirements.txt
```

---

## Docker Deployment (Optional)

```dockerfile
FROM continuumio/miniconda3:latest

WORKDIR /app

# Install PyTorch
RUN conda install pytorch torchvision torchaudio -c pytorch

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy project
COPY MLP/ .

# Run training
CMD ["python", "train_mlp.py"]
```

Build and run:
```bash
docker build -t mlp-trainer .
docker run --rm mlp-trainer
```

---

## Support & Debugging

### Check Versions

```bash
python << 'EOF'
import sys
import torch
import numpy as np
import pandas as pd
import sklearn

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
EOF
```

### GPU Diagnostics

```bash
# For NVIDIA GPU
nvidia-smi

# For macOS (MPS)
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

### Common Issues Log

See README.md → Troubleshooting section for detailed solutions.

---

**Last Updated**: May 2026
