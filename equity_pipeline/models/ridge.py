"""
models/ridge.py — RidgeModel
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import RidgeCV
from ..shared.walk_forward import BaseModel


class RidgeModel(BaseModel):
    name           = "ridge"
    uses_sequences = False

    def __init__(self, alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)):
        self.alphas    = alphas
        self._model    = None
        self._features = None

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        import random, torch
        from ..config import DEFAULT_CONFIG
        self._model = RidgeCV(alphas=self.alphas, cv=5)
        # Combine train + val for Ridge (RidgeCV does its own CV internally)
        X_all = np.concatenate([X_train, X_val], axis=0)
        y_all = np.concatenate([y_train, y_val], axis=0)
        self._model.fit(X_all, y_all)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X).astype(np.float32)

    def get_feature_importance(self) -> dict:
        if self._model is None or self._features is None:
            return {}
        coef = np.abs(self._model.coef_)
        return dict(zip(self._features, coef.tolist()))

    def set_feature_names(self, features):
        self._features = list(features)
        return self
