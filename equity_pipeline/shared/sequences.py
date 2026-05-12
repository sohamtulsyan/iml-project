"""
shared/sequences.py
===================
Vectorized sequence construction for RNN/Transformer/CNN models.
Ported from Transformer/data.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


def build_sequences(
    df:          pd.DataFrame,
    id_col:      str,
    date_col:    str,
    features:    tuple | list,
    target_col:  str,
    seq_len:     int,
    pred_months: Optional[List] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (N, T, F) sequence tensor + targets.
    
    For each stock at prediction month m, we extract a window of length T 
    ending at m-1.
    """
    feature_list = list(features)
    n_features   = len(feature_list)
    
    X_list, y_list, id_list, month_list = [], [], [], []
    
    # Sort for safety
    df = df.sort_values([id_col, date_col])
    
    for firm_id, firm_df in df.groupby(id_col):
        if len(firm_df) < seq_len + 1:
            continue
            
        firm_months = firm_df[date_col].values
        firm_X      = firm_df[feature_list].values.astype(np.float32)
        firm_y      = firm_df[target_col].values.astype(np.float32)
        
        # We start from seq_len because we need seq_len prior months
        for t_idx in range(seq_len, len(firm_months)):
            pred_m = firm_months[t_idx]
            
            if pred_months is not None and pred_m not in pred_months:
                continue
                
            seq   = firm_X[t_idx - seq_len : t_idx]   # (T, F)
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
                
    X      = np.stack(X_list).astype(np.float32)
    y      = np.array(y_list, dtype=np.float32)
    ids    = np.array(id_list)
    months = np.array(month_list)
    
    return X, y, ids, months
