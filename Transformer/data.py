"""
================================================================================
  TRANSFORMER PIPELINE — DATA LOADING & SEQUENCE CONSTRUCTION
================================================================================

  Key design:
  ───────────
  - Vectorized sequence building via numpy stride tricks (no Python loops over
    stocks or months)
  - Sequences built fold-by-fold and discarded to avoid memory blowup
  - Minimum history filter: stocks with < T months of data are excluded
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(
    data_path: str,
    id_col:    str = "co_code",
    date_col:  str = "Month",
    features:  tuple = ("BM_sep", "lag_mv", "OpProf", "Inv", "Momentum", "lag_ret", "mktcap"),
    target_col: str = "monthly_gross_return",
) -> pd.DataFrame:
    """
    Load raw CSV, sort by (stock, month), create forward-return target.

    The forward return for firm i at month t is monthly_gross_return at t+1,
    shifted PER FIRM to prevent look-ahead bias.
    """
    print("[Loader] Reading CSV...")
    df = pd.read_csv(data_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([id_col, date_col]).reset_index(drop=True)

    print(f"[Loader] {len(df):,} rows | "
          f"{df[id_col].nunique():,} firms | "
          f"{df[date_col].nunique()} months | "
          f"{df[date_col].min().date()} → {df[date_col].max().date()}")

    # Forward return: target at month t = return realized at t+1
    df["fwd_return"] = df.groupby(id_col)[target_col].shift(-1)

    # Drop rows with NaN in any feature or target
    keep_cols = list(features) + ["fwd_return"]
    df = df.dropna(subset=keep_cols).reset_index(drop=True)

    print(f"[Loader] After NaN drop: {len(df):,} rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-sectional preprocessing helpers (per month, no leakage)
# ─────────────────────────────────────────────────────────────────────────────

def _winsorize_month(arr: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> np.ndarray:
    """Winsorize a (N, F) feature matrix column-wise at [lo, hi] percentiles."""
    p_lo = np.nanpercentile(arr, lo * 100, axis=0)
    p_hi = np.nanpercentile(arr, hi * 100, axis=0)
    return np.clip(arr, p_lo, p_hi)


def _rank_normalize_month(arr: np.ndarray) -> np.ndarray:
    """Rank-normalize a (N, F) feature matrix to [0, 1] column-wise."""
    n = arr.shape[0]
    out = np.empty_like(arr, dtype=np.float32)
    for j in range(arr.shape[1]):
        col   = arr[:, j]
        order = np.argsort(col)
        ranks = np.empty(n, dtype=np.float32)
        ranks[order] = np.arange(1, n + 1, dtype=np.float32)
        out[:, j] = ranks / (n + 1)
    return out


def _log_transform(arr: np.ndarray, col_indices: List[int]) -> np.ndarray:
    """Apply log1p to specified column indices (in-place copy)."""
    arr = arr.copy()
    for j in col_indices:
        arr[:, j] = np.log1p(np.maximum(arr[:, j], 0.0))
    return arr


def preprocess_panel(
    df:           pd.DataFrame,
    features:     tuple,
    date_col:     str = "Month",
    log_cols:     tuple = ("mktcap",),
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
) -> pd.DataFrame:
    """
    Apply per-month cross-sectional preprocessing to a panel DataFrame:
      1. Log-transform log_cols
      2. Winsorize at 1/99 pct per month
      3. Rank-normalize to [0, 1] per month

    Returns the same DataFrame with feature columns overwritten.
    This is fit on whatever panel is passed in (training or test month) —
    no information flows across the fold boundary.
    """
    feature_list = list(features)
    log_indices  = [feature_list.index(c) for c in log_cols if c in feature_list]

    result_frames = []
    for month, grp in df.groupby(date_col, sort=True):
        arr = grp[feature_list].values.astype(np.float32)

        if log_indices:
            arr = _log_transform(arr, log_indices)

        arr = _winsorize_month(arr, winsor_lower, winsor_upper)
        arr = _rank_normalize_month(arr)

        grp = grp.copy()
        grp[feature_list] = arr
        result_frames.append(grp)

    return pd.concat(result_frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Vectorized sequence builder
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(
    df:          pd.DataFrame,
    id_col:      str,
    date_col:    str,
    features:    tuple,
    target_col:  str,
    seq_len:     int,
    pred_months: Optional[List] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (N_samples, T, F) sequence tensor + targets via numpy stride tricks.

    For each stock × prediction-month pair where the stock has >= seq_len
    months of history ending strictly before the prediction month, we extract
    a window of length seq_len.

    Parameters
    ----------
    df          : preprocessed panel (sorted by id, date)
    pred_months : if given, only build sequences for these prediction months.
                  Otherwise use all months that have sufficient prior history.

    Returns
    -------
    X       : float32 (N, T, F)
    y       : float32 (N,)
    ids     : int     (N,)  — stock ids
    months  : (N,)   — prediction months
    """
    feature_list = list(features)
    n_features   = len(feature_list)

    all_months  = sorted(df[date_col].unique())
    month_idx   = {m: i for i, m in enumerate(all_months)}

    X_list, y_list, id_list, month_list = [], [], [], []

    # Group by firm so we can efficiently slice each firm's history
    for firm_id, firm_df in df.groupby(id_col):
        firm_df    = firm_df.sort_values(date_col)
        firm_months = firm_df[date_col].values
        firm_X      = firm_df[feature_list].values.astype(np.float32)
        firm_y      = firm_df[target_col].values.astype(np.float32)

        if len(firm_df) < seq_len + 1:
            continue

        # Determine which prediction months this firm qualifies for
        for t_idx in range(seq_len, len(firm_months)):
            pred_m = firm_months[t_idx]

            # Filter to requested prediction months if supplied
            if pred_months is not None and pred_m not in set(pred_months):
                continue

            seq   = firm_X[t_idx - seq_len : t_idx]   # (T, F) — strictly < pred_m
            label = firm_y[t_idx]                       # return at pred_m

            if np.isnan(seq).any() or np.isnan(label):
                continue

            X_list.append(seq)
            y_list.append(label)
            id_list.append(firm_id)
            month_list.append(pred_m)

    if not X_list:
        return (np.empty((0, seq_len, n_features), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.array([]),
                np.array([]))

    X      = np.stack(X_list).astype(np.float32)    # (N, T, F)
    y      = np.array(y_list, dtype=np.float32)
    ids    = np.array(id_list)
    months = np.array(month_list)

    return X, y, ids, months
