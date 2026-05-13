
import pandas as pd
import numpy as np
from pathlib import Path
from equity_pipeline.backtest.portfolio import PortfolioConstructor
from equity_pipeline.config import PipelineConfig

cfg = PipelineConfig()
pc = PortfolioConstructor(cfg)
pred_path = Path("results/ensemble/meta_learner/meta_learner/meta_learner_test_predictions.parquet")

print("--- Inspecting Meta-Learner Predictions ---")
df = pd.read_parquet(pred_path)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Dtypes:\n{df.dtypes}")

try:
    print("\n--- Testing Portfolio Construction ---")
    pos_df = pc.construct(pred_path)
    print(f"Success! Created {len(pos_df)} positions.")
except Exception as e:
    print(f"FAILED construction: {e}")
    import traceback
    traceback.print_exc()
