"""
MLP Module — GPU-Accelerated Neural Network for Equity Return Prediction

Main exports:
    - MLPRegressor: sklearn-compatible MLP regressor with GPU support
    - get_device: Auto-detect and return optimal device (MPS > CUDA > CPU)
"""

from .mlp_regressor import MLPRegressor, MLPNet, get_device

__all__ = ["MLPRegressor", "MLPNet", "get_device"]
__version__ = "1.0.0"
