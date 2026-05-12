"""
smoke_test.py — Quick import + sanity check. Run before full pipeline.
Usage:  python smoke_test.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

print("Running smoke tests...")

errors = []

# ── 1. Imports ────────────────────────────────────────────────────────────────
def _test_imports():
    try:
        from equity_pipeline.config import PipelineConfig, DEFAULT_CONFIG
        from equity_pipeline.shared.loader import load_data, build_target, lag_features
        from equity_pipeline.shared.metrics import (
            spearman_ic, compute_ic_series, long_short_return,
            compute_portfolio_metrics
        )
        from equity_pipeline.shared.preprocessing import (
            LogTransformer, CrossSectionalWinsorizer,
            CrossSectionalRankNormalizer, CrossSectionalZScorer,
            make_tree_preprocessor, make_linear_preprocessor,
        )
        from equity_pipeline.shared.walk_forward import BaseModel, run_walk_forward
        from equity_pipeline.shared.output import (
            save_summary, save_test_predictions, save_oof_predictions
        )
        from equity_pipeline.models.ridge       import RidgeModel
        from equity_pipeline.models.cart        import CARTModel
        from equity_pipeline.models.lightgbm_model import LightGBMModel
        from equity_pipeline.models.random_forest  import RandomForestModel
        from equity_pipeline.models.mlp        import MLPModel
        from equity_pipeline.models.transformer import TransformerModel
        from equity_pipeline.models.cnn        import CNNModel
        from equity_pipeline.ensemble.simple_average import SimpleAverageEnsemble
        from equity_pipeline.ensemble.meta_learner   import MetaLearnerEnsemble
        from equity_pipeline.backtest.portfolio import PortfolioConstructor
        from equity_pipeline.backtest.engine    import BacktestEngine
        from equity_pipeline.layers.rl_agent   import RLAgent
        print("  ✓ All imports OK")
    except ImportError as e:
        errors.append(f"Import error: {e}")
        print(f"  ✗ Import error: {e}")

# ── 2. Metrics correctness ────────────────────────────────────────────────────
def _test_metrics():
    import numpy as np
    from equity_pipeline.shared.metrics import (
        spearman_ic, compute_portfolio_metrics, long_short_return
    )
    # Perfect correlation → IC = 1
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ic = spearman_ic(y, y)
    assert abs(ic - 1.0) < 1e-9, f"Expected IC=1, got {ic}"

    # Anti-correlated → IC = -1
    ic_neg = spearman_ic(y, -y)
    assert abs(ic_neg + 1.0) < 1e-9, f"Expected IC=-1, got {ic_neg}"

    # Long-short return
    pred  = np.array([0.1, 0.05, 0.0, -0.05, -0.1])
    ret   = np.array([0.02, 0.01, 0.0, -0.01, -0.02])
    ls    = long_short_return(pred, ret, top_pct=0.2, bottom_pct=0.2)
    assert ls > 0, "Expected positive L/S return for aligned pred/ret"

    port = compute_portfolio_metrics(np.array([0.01, -0.005, 0.02, 0.015, -0.001]))
    assert "sharpe_ratio" in port and "max_drawdown" in port
    print("  ✓ Metrics OK")

# ── 3. Preprocessing no-leakage check ────────────────────────────────────────
def _test_preprocessing():
    import pandas as pd, numpy as np
    from equity_pipeline.config import PipelineConfig
    from equity_pipeline.shared.preprocessing import CrossSectionalWinsorizer
    cfg = PipelineConfig()
    rng = pd.date_range("2010-01-01", periods=24, freq="ME")
    df  = pd.DataFrame({
        "Month":  np.repeat(rng, 50),
        "BM_sep": np.random.randn(24 * 50),
        "mktcap": np.abs(np.random.randn(24 * 50)) * 1000,
    })
    all_months = sorted(df["Month"].unique())
    train_months = all_months[:18]   # first 18 months
    test_months  = all_months[18:]   # last 6 months

    df_train = df[df["Month"].isin(train_months)]
    df_test  = df[df["Month"].isin(test_months)]

    w = CrossSectionalWinsorizer()
    w.fit(df_train)
    out_train = w.transform(df_train)
    out_test  = w.transform(df_test)

    assert len(out_train) == len(df_train), "Train row count should be preserved"
    assert len(out_test)  == len(df_test),  "Test row count should be preserved"

    # Test months must NOT be in fitted bounds (no look-ahead)
    for tm in test_months:
        assert tm not in w.bounds_, f"Test month {tm} should not be in fitted bounds"

    print("  ✓ Preprocessing no-leakage OK")

# ── 4. Causal padding check (CNN) ─────────────────────────────────────────────
def _test_causal_conv():
    import torch
    import torch.nn as nn
    from equity_pipeline.models.cnn import _CausalConv1d, _TemporalCNN
    conv = _CausalConv1d(1, 1, kernel_size=3, bias=False)
    nn.init.constant_(conv.conv.weight, 1.0)
    x      = torch.zeros(1, 1, 5)
    x[0, 0, 2] = 1.0          # pulse at t=2
    out    = conv(x)[0, 0]    # shape (5,)
    # t=0: sees only t=0 → should be 0
    # t=1: sees t=0,1 → should be 0
    # t=2: sees t=0,1,2 → should be 1
    # t=3: sees t=1,2,3 → should be 1 (or more with running sum)
    assert out[0].item() == 0.0, "CausalConv: t=0 should see no future"
    assert out[2].item() == 1.0, "CausalConv: pulse should appear at t=2"
    assert out[1].item() == 0.0, "CausalConv: t=1 should not see t=2"
    print("  ✓ Causal padding OK")

# ── 5. BaseModel interface check ──────────────────────────────────────────────
def _test_base_model():
    from equity_pipeline.models.ridge import RidgeModel
    m = RidgeModel()
    assert hasattr(m, "fit") and hasattr(m, "predict") and hasattr(m, "clone")
    m2 = m.clone()
    assert m2 is not m, "clone() must return a different object"
    print("  ✓ BaseModel interface OK")

# ── Run all ───────────────────────────────────────────────────────────────────
for test in [_test_imports, _test_metrics, _test_preprocessing,
             _test_causal_conv, _test_base_model]:
    try:
        test()
    except Exception as e:
        errors.append(f"{test.__name__}: {e}")
        print(f"  ✗ {test.__name__}: {e}")

print()
if errors:
    print(f"FAILED — {len(errors)} error(s):")
    for e in errors:
        print(f"  • {e}")
    sys.exit(1)
else:
    print(f"ALL SMOKE TESTS PASSED ✓")
