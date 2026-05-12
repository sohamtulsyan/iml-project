"""
shared/output.py
================
Standardised file I/O for ALL models. No model writes files directly.
All parquet files use engine="pyarrow", compression="snappy".
"""
from __future__ import annotations

import json
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _model_dir(model_name: str, results_dir: Path) -> Path:
    d = results_dir / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pq_write(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    print(f"  ✓ {path}  ({len(df):,} rows)")


# ─────────────────────────────────────────────────────────────────────────────
#  Writers
# ─────────────────────────────────────────────────────────────────────────────

def save_test_predictions(df: pd.DataFrame, model_name: str, results_dir: Path) -> Path:
    """
    Schema: co_code, Month, pred_score, fwd_return
    Output: results/{model_name}/{model_name}_test_predictions.parquet
    """
    path = _model_dir(model_name, results_dir) / f"{model_name}_test_predictions.parquet"
    _pq_write(df, path)
    return path


def save_oof_predictions(df: pd.DataFrame, model_name: str, results_dir: Path) -> Path:
    """
    Schema: co_code, Month, oof_pred, target
    Output: results/{model_name}/{model_name}_oof_predictions.parquet
    """
    path = _model_dir(model_name, results_dir) / f"{model_name}_oof_predictions.parquet"
    _pq_write(df, path)
    return path


def save_ic_series(ic_series: list[dict], model_name: str, results_dir: Path) -> Path:
    """
    Schema: Month, IC, cumulative_IC
    """
    df   = pd.DataFrame(ic_series)
    df["cumulative_IC"] = df["IC"].cumsum()
    path = _model_dir(model_name, results_dir) / f"{model_name}_ic_series.csv"
    df.to_csv(path, index=False)
    print(f"  ✓ {path}")
    return path


def save_summary(summary: dict, model_name: str, results_dir: Path) -> Path:
    """
    Schema: {mean_ic, ic_std, icir, pct_positive, n_months, ...}
    """
    path = _model_dir(model_name, results_dir) / f"{model_name}_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  ✓ {path}")
    return path


def save_feature_importance(importance: dict, model_name: str, results_dir: Path) -> Path:
    """
    Schema: feature, importance
    Output: results/{model_name}/{model_name}_feature_importance.csv
    """
    if not importance:
        return None
    df   = pd.DataFrame(list(importance.items()), columns=["feature", "importance"])
    df   = df.sort_values("importance", ascending=False).reset_index(drop=True)
    path = _model_dir(model_name, results_dir) / f"{model_name}_feature_importance.csv"
    df.to_csv(path, index=False)
    print(f"  ✓ {path}")
    return path


def save_best_hparams(hparams: dict, model_name: str, results_dir: Path) -> Path:
    path = _model_dir(model_name, results_dir) / f"{model_name}_best_hparams.json"
    with open(path, "w") as f:
        # Make tuples serializable
        clean = {k: (list(v) if isinstance(v, tuple) else v) for k, v in hparams.items()}
        json.dump(clean, f, indent=2)
    print(f"  ✓ {path}")
    return path


def save_training_log(rows: list[dict], model_name: str, results_dir: Path) -> Path:
    if not rows:
        return None
    path = _model_dir(model_name, results_dir) / f"{model_name}_training_log.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  ✓ {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_predictions(
    model_name:  str,
    pred_type:   str,          # "test" or "oof"
    results_dir: Path,
) -> pd.DataFrame:
    """Load saved predictions. pred_type: 'test' or 'oof'."""
    fname = f"{model_name}_{pred_type}_predictions.parquet"
    path  = results_dir / model_name / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {pred_type} predictions for {model_name}: {path}\n"
            f"Run the model first."
        )
    return pd.read_parquet(path, engine="pyarrow")


def load_summary(model_name: str, results_dir: Path) -> dict | None:
    path = results_dir / model_name / f"{model_name}_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
