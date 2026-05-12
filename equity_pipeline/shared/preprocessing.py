"""
shared/preprocessing.py
=======================
High-performance vectorized transformers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CrossSectionalWinsorizer(BaseEstimator, TransformerMixin):
    """Numpy-vectorized winsorization. Minimal copies."""
    def __init__(self, lower: float = 0.01, upper: float = 0.99, date_col: str = "Month", precomputed_bounds: tuple[pd.DataFrame, pd.DataFrame] | None = None):
        self.lower = lower
        self.upper = upper
        self.date_col = date_col
        self.precomputed_bounds = precomputed_bounds
        
        # Internal state
        self.bounds_low_ = precomputed_bounds[0] if precomputed_bounds else None
        self.bounds_high_ = precomputed_bounds[1] if precomputed_bounds else None
        self.last_month_ = None
        self.feature_cols = None

    def fit(self, X: pd.DataFrame, y=None):
        if self.feature_cols is None:
            if self.bounds_low_ is not None:
                self.feature_cols = list(self.bounds_low_.columns)
            else:
                exclude = {self.date_col, "co_code", "gvkey", "permno", "ticker", "fwd_return", "Year", "Corrected_Year"}
                self.feature_cols = [c for c in X.select_dtypes(include=[np.number]).columns if c not in exclude]

        if not self.feature_cols:
            return self

        if self.bounds_low_ is None:
            grouped = X.groupby(self.date_col)[self.feature_cols]
            self.bounds_low_  = grouped.quantile(self.lower)
            self.bounds_high_ = grouped.quantile(self.upper)
        
        if not self.bounds_low_.empty:
            self.last_month_ = self.bounds_low_.index[-1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.bounds_low_ is None: return X
        
        X = X.copy()
        months = X[self.date_col].values
        lo_df = self.bounds_low_.reindex(months)
        hi_df = self.bounds_high_.reindex(months)
        
        if lo_df.isna().any().any():
            last_lo, last_hi = self.bounds_low_.loc[self.last_month_], self.bounds_high_.loc[self.last_month_]
            lo_df, hi_df = lo_df.fillna(last_lo), hi_df.fillna(last_hi)

        X[self.feature_cols] = np.clip(X[self.feature_cols].values, lo_df.values, hi_df.values)
        return X


class CrossSectionalRankNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, date_col: str = "Month"):
        self.date_col = date_col
    def fit(self, X, y=None): return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        exclude = {self.date_col, "co_code", "gvkey", "permno", "ticker", "fwd_return", "Year", "Corrected_Year"}
        feat_cols = [c for c in X.select_dtypes(include=[np.number]).columns if c not in exclude]
        if feat_cols:
            X[feat_cols] = X.groupby(self.date_col)[feat_cols].rank(pct=True)
        return X


class CrossSectionalZScorer(BaseEstimator, TransformerMixin):
    def __init__(self, date_col: str = "Month"):
        self.date_col = date_col
    def fit(self, X, y=None): return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        exclude = {self.date_col, "co_code", "gvkey", "permno", "ticker", "fwd_return", "Year", "Corrected_Year"}
        feat_cols = [c for c in X.select_dtypes(include=[np.number]).columns if c not in exclude]
        if feat_cols:
            grouped = X.groupby(self.date_col)[feat_cols]
            mu = grouped.transform("mean")
            sd = grouped.transform("std").fillna(1).replace(0, 1)
            X[feat_cols] = (X[feat_cols].values - mu.values) / sd.values
        return X


class SequentialPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, steps: list):
        self.steps = steps
    def fit(self, X, y=None):
        Xt = X
        for _, t in self.steps: Xt = t.fit_transform(Xt, y)
        return self
    def transform(self, X):
        Xt = X
        for _, t in self.steps: Xt = t.transform(Xt)
        return Xt
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_sequence_preprocessor(cfg, precomputed_winsor=None) -> SequentialPreprocessor:
    return SequentialPreprocessor([
        ("winsor", CrossSectionalWinsorizer(lower=cfg.winsor_lower, upper=cfg.winsor_upper, date_col=cfg.date_col, precomputed_bounds=precomputed_winsor)),
        ("rank",   CrossSectionalRankNormalizer(date_col=cfg.date_col)),
    ])

def make_tree_preprocessor(cfg, precomputed_winsor=None) -> SequentialPreprocessor:
    return SequentialPreprocessor([
        ("winsor", CrossSectionalWinsorizer(lower=cfg.winsor_lower, upper=cfg.winsor_upper, date_col=cfg.date_col, precomputed_bounds=precomputed_winsor)),
        ("rank",   CrossSectionalRankNormalizer(date_col=cfg.date_col)),
    ])

def make_linear_preprocessor(cfg, precomputed_winsor=None) -> SequentialPreprocessor:
    return SequentialPreprocessor([
        ("winsor", CrossSectionalWinsorizer(lower=cfg.winsor_lower, upper=cfg.winsor_upper, date_col=cfg.date_col, precomputed_bounds=precomputed_winsor)),
        ("zscore", CrossSectionalZScorer(date_col=cfg.date_col)),
    ])
