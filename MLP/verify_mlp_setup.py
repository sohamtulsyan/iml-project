#!/usr/bin/env python
"""
================================================================================
  MLP VERIFICATION SCRIPT
  Comprehensive system checks before training
================================================================================

This script verifies:
  ✓ Python version (≥ 3.9)
  ✓ PyTorch installation & GPU availability
  ✓ Required packages (numpy, pandas, scipy, scikit-learn)
  ✓ GPU acceleration (MPS, CUDA, or CPU)
  ✓ Data availability
  ✓ Model basic functionality
  ✓ Configuration file integrity

Usage:
    python verify_mlp_setup.py

Exit codes:
    0 = All checks passed ✓
    1 = Some checks failed ✗
    2 = Critical error (cannot proceed)

================================================================================
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
#  COLOR OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.ENDC}\n")

def print_check(name: str, status: bool, detail: str = ""):
    icon = f"{Colors.GREEN}✓{Colors.ENDC}" if status else f"{Colors.RED}✗{Colors.ENDC}"
    detail_str = f" — {detail}" if detail else ""
    print(f"  {icon} {name}{detail_str}")

def print_warning(text: str):
    print(f"  {Colors.YELLOW}⚠{Colors.ENDC} {text}")

def print_error(text: str):
    print(f"  {Colors.RED}✗{Colors.ENDC} {text}")

def print_success(text: str):
    print(f"  {Colors.GREEN}✓{Colors.ENDC} {text}")

# ═══════════════════════════════════════════════════════════════════════════════
#  VERIFICATION CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def check_python_version():
    """Check Python version ≥ 3.9"""
    print_header("1. Python Version")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor >= 9:
        print_check("Python Version", True, version_str)
        return True
    else:
        print_check("Python Version", False, f"{version_str} (required ≥ 3.9)")
        return False

def check_pytorch():
    """Check PyTorch installation and GPU availability"""
    print_header("2. PyTorch Installation")
    
    try:
        import torch
        print_check("PyTorch Import", True, f"v{torch.__version__}")
    except ImportError:
        print_check("PyTorch Import", False)
        print_error("PyTorch not installed. Run: pip install torch")
        return False
    
    # Check GPU availability
    mps_available = torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    
    print_check("Apple Silicon GPU (MPS)", mps_available)
    if mps_available:
        print_success("Metal Performance Shaders (MPS) available — 2-5x speedup!")
    
    print_check("NVIDIA GPU (CUDA)", cuda_available)
    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
            print_success(f"GPU: {device_name}")
        except:
            pass
    
    if not mps_available and not cuda_available:
        print_warning("No GPU detected. Will use CPU (slower but functional).")
    
    return True

def check_dependencies():
    """Check required packages"""
    print_header("3. Required Packages")
    
    packages = {
        'numpy': 'np',
        'pandas': 'pd',
        'scipy': 'scipy',
        'sklearn': 'sklearn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'statsmodels': 'statsmodels',
    }
    
    all_ok = True
    for package_name, import_name in packages.items():
        try:
            __import__(import_name)
            module = sys.modules[import_name]
            version = getattr(module, '__version__', 'unknown')
            print_check(package_name, True, version)
        except ImportError:
            print_check(package_name, False)
            all_ok = False
    
    return all_ok

def check_project_structure():
    """Check project files exist"""
    print_header("4. Project Structure")
    
    mlp_dir = Path(__file__).parent
    required_files = {
        'mlp_regressor.py': 'Core MLP implementation',
        'train_mlp.py': 'Training script',
        'visualize_mlp_results.py': 'Visualization script',
        'config.py': 'Configuration',
        '../project_database.csv': 'Project database',
    }
    
    all_ok = True
    for filename, description in required_files.items():
        filepath = mlp_dir / filename
        exists = filepath.exists()
        print_check(description, exists, filepath.name)
        if not exists:
            all_ok = False
    
    return all_ok

def check_data_availability():
    """Check if project database exists and has data"""
    print_header("5. Data Availability")
    
    try:
        import pandas as pd
        data_path = Path(__file__).parent.parent / 'project_database.csv'
        
        if not data_path.exists():
            print_check("Data File", False, "project_database.csv not found")
            return False
        
        print_check("Data File", True, data_path.name)
        
        # Try reading first few rows
        df = pd.read_csv(data_path, nrows=10)
        n_rows = len(open(data_path).readlines()) - 1  # Rough estimate
        print_check("Data Loadable", True, f"~{n_rows:,} rows, {len(df.columns)} columns")
        
        # Check required columns
        required_cols = ['lag_ret', 'Momentum', 'BM_sep', 'OpProf', 'Inv', 'mktcap', 'lag_mv', 'monthly_gross_return']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            print_error(f"Missing columns: {missing}")
            return False
        
        print_check("Required Columns", True, f"All {len(required_cols)} present")
        return True
        
    except Exception as e:
        print_error(f"Error checking data: {e}")
        return False

def check_mlp_module():
    """Test MLP module import and basic functionality"""
    print_header("6. MLP Module Functionality")
    
    try:
        from mlp_regressor import MLPRegressor, get_device, MLPNet
        print_check("MLPRegressor Import", True)
    except ImportError as e:
        print_check("MLPRegressor Import", False)
        print_error(f"Import error: {e}")
        return False
    
    try:
        import numpy as np
        
        # Test device detection
        device = get_device('auto')
        print_check("Device Auto-Detection", True, f"{device}")
        
        # Test minimal model
        X_tiny = np.random.randn(50, 7)
        y_tiny = np.random.randn(50)
        
        model = MLPRegressor(
            hidden_layer_sizes=(16, 8),
            batch_size=16,
            max_epochs=3,
            early_stopping_patience=2,
            verbose=False,
        )
        
        model.fit(X_tiny, y_tiny)
        pred = model.predict(X_tiny)
        
        print_check("Model Training", True, f"Tiny model (3 epochs)")
        print_check("Model Prediction", True, f"{len(pred)} predictions")
        
        return True
        
    except Exception as e:
        print_check("Model Testing", False)
        print_error(f"Model test error: {e}")
        return False

def check_config():
    """Verify configuration file"""
    print_header("7. Configuration File")
    
    try:
        import config
        
        # Check key settings
        print_check("Config Import", True)
        print_check("Device Setting", True, config.DEVICE)
        print_check("Hidden Layers", True, str(config.HIDDEN_LAYER_SIZES))
        print_check("Learning Rate", True, str(config.LEARNING_RATE))
        print_check("Batch Size", True, str(config.BATCH_SIZE))
        print_check("Grid Search Enabled", True, f"{len(list(__import__('itertools').product(*config.PARAM_GRID.values())))} combinations")
        
        return True
        
    except Exception as e:
        print_check("Configuration", False)
        print_error(f"Config error: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{Colors.BOLD}MLP SETUP VERIFICATION{Colors.ENDC}")
    print(f"Python {sys.version}")
    print(f"Location: {Path(__file__).parent}")
    
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch", check_pytorch),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Data Availability", check_data_availability),
        ("MLP Module", check_mlp_module),
        ("Configuration", check_config),
    ]
    
    results = []
    for check_name, check_fn in checks:
        try:
            result = check_fn()
            results.append(result)
        except Exception as e:
            print_error(f"Unexpected error in {check_name}: {e}")
            results.append(False)
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    total = len(results)
    passed = sum(results)
    failed = total - passed
    
    if failed == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ ALL CHECKS PASSED{Colors.ENDC}")
        print(f"\nYou are ready to train the MLP model!\n")
        print(f"Next steps:")
        print(f"  1. python train_mlp.py              # Train MLP (7-10 hours on GPU)")
        print(f"  2. python visualize_mlp_results.py  # Generate plots")
        print(f"  3. Check visualizations/            # View comparison plots\n")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ {failed} CHECK(S) FAILED{Colors.ENDC}")
        print(f"\nPassed: {passed}/{total}")
        print(f"Failed: {failed}/{total}\n")
        
        if failed == total:
            print("Critical errors detected. Cannot proceed.")
            return 2
        else:
            print("Some checks failed. Review above and fix issues.")
            return 1

if __name__ == '__main__':
    sys.exit(main())
