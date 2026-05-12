"""
MLP Configuration
=================

This module provides default hyperparameters and configuration for the MLP model.
"""

import torch

# ═══════════════════════════════════════════════════════════════════════════════
#  DEVICE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Auto-detect device (MPS > CUDA > CPU)
DEVICE = "auto"

# Force specific device (override auto-detection)
# DEVICE = "mps"    # Apple Silicon GPU
# DEVICE = "cuda"   # NVIDIA GPU
# DEVICE = "cpu"    # CPU only

# ═══════════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_DIM = 7  # Number of input features

HIDDEN_LAYER_SIZES = (64, 32, 16)  # Hidden layer dimensions

DROPOUT_RATES = (0.2, 0.2, 0.1)  # Dropout rates for each hidden layer

OUTPUT_DIM = 1  # Single output (regression)

# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING HYPERPARAMETERS (DEFAULTS)
# ═══════════════════════════════════════════════════════════════════════════════

LEARNING_RATE = 0.001  # Initial learning rate for Adam

ALPHA = 0.0001  # L2 regularization coefficient (weight decay)

BATCH_SIZE = 128  # Batch size (tune for GPU memory)

MAX_EPOCHS = 500  # Maximum number of training epochs

EARLY_STOPPING_PATIENCE = 25  # Patience for early stopping (epochs)

EARLY_STOPPING_MIN_DELTA = 0.0001  # Minimum loss improvement to count as convergence

VALIDATION_FRACTION = 0.15  # Fraction of training data for validation

USE_MIXED_PRECISION = True  # Use FP16 for 2x speedup (GPU only)

RANDOM_STATE = 42

# ═══════════════════════════════════════════════════════════════════════════════
#  GRID SEARCH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PARAM_GRID = {
    'hidden_layer_sizes': [
        (32, 16, 8),
        (64, 32, 16),
        (128, 64, 32),
    ],
    'learning_rate': [0.0005, 0.001, 0.002],
    'alpha': [0.00001, 0.0001, 0.001],
    'batch_size': [64, 128, 256],
}

# Total grid combinations: 3 × 3 × 3 × 3 = 81
# × 5-fold CV = 405 model trainings per walk-forward fold

# ═══════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD VALIDATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

TRAIN_WINDOW_MONTHS = 60  # Training window size (months)

TEST_WINDOW_MONTHS = 1  # Test window size (out-of-sample prediction)

MIN_OBS_PER_FOLD = 30  # Minimum observations per fold to proceed

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

WINSOR_LOWER = 0.01  # Lower percentile for winsorization

WINSOR_UPPER = 0.99  # Upper percentile for winsorization

# Features used in model (7 total)
FEATURES = [
    'lag_ret',       # Lagged return
    'Momentum',      # Momentum factor
    'BM_sep',        # Book-to-market
    'OpProf',        # Operating profitability
    'Inv',           # Investment factor
    'mktcap',        # Market capitalization
    'lag_mv',        # Lagged market value
]

# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════

# Performance thresholds
GOOD_ICIR = 0.5        # ICIR ≥ 0.5 is "Good"
MODERATE_ICIR = 0.3    # ICIR ≥ 0.3 is "Moderate"
# ICIR < 0.3 is "Weak"

# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT & LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

VERBOSE = False  # Print training progress per epoch

SAVE_BEST_WEIGHTS = True  # Save best model weights

SAVE_RESULTS_CSV = True  # Save results to CSV

# ═══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE TARGETS (vs BASELINES)
# ═══════════════════════════════════════════════════════════════════════════════

RIDGE_BASELINE_IC = 0.0344
RIDGE_BASELINE_ICIR = 0.2504

LGBM_TARGET_ICIR = 0.7347
RF_TARGET_ICIR = 0.7248

ACCEPTABLE_ICIR = 0.25  # Minimum acceptable ICIR

SUCCESS_ICIR = 0.60  # ICIR ≥ 0.60 is "Success" (approaching tree models)
