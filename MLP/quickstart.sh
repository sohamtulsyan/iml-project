#!/bin/bash
# Quick-start installation and training script for MLP module

set -e  # Exit on error

echo "═══════════════════════════════════════════════════════════════════════════"
echo "  MLP QUICK START"
echo "═══════════════════════════════════════════════════════════════════════════"

# Step 1: Check Python version
echo ""
echo "[1/4] Checking Python version..."
python --version

# Step 2: Install/upgrade PyTorch with GPU support
echo ""
echo "[2/4] Installing PyTorch with GPU support..."
echo "    • This will install PyTorch with Metal Performance Shaders (MPS) support"
echo "    • If you have NVIDIA GPU, CUDA will be auto-detected"

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "    • macOS detected — installing PyTorch with MPS support"
    pip install torch torchvision torchaudio
else
    echo "    • Linux/other OS detected — standard PyTorch installation"
    pip install torch torchvision torchaudio
fi

# Step 3: Verify GPU availability
echo ""
echo "[3/4] Verifying GPU availability..."
python << 'EOF'
import torch
print(f"  • CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    - GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"    - GPU Count: {torch.cuda.device_count()}")

print(f"  • MPS Available: {torch.backends.mps.is_available()}")

if torch.backends.mps.is_available():
    print(f"    ✓ Apple Silicon GPU (Metal Performance Shaders) ready")
elif torch.cuda.is_available():
    print(f"    ✓ NVIDIA GPU (CUDA) ready")
else:
    print(f"    ⚠ No GPU detected — will use CPU (slower)")
EOF

# Step 4: Summary
echo ""
echo "[4/4] Installation complete!"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  NEXT STEPS"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "1. Train MLP with walk-forward validation:"
echo "   cd MLP/"
echo "   python train_mlp.py"
echo ""
echo "2. Generate visualization plots:"
echo "   python visualize_mlp_results.py"
echo ""
echo "3. Results will be saved to:"
echo "   • mlp_ic_results.csv"
echo "   • mlp_hyperparams.csv"
echo "   • mlp_training_times.csv"
echo "   • visualizations/*.png"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
