"""
models/cart.py — CARTModel
FIX: training window is now 60 months (not 1 month as in the old script).
"""
from __future__ import annotations
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ParameterGrid
from ..shared.walk_forward import BaseModel
from ..shared.metrics import spearman_ic


class CARTModel(BaseModel):
    name           = "cart"
    uses_sequences = False

    _DEFAULT_PARAMS = {
        "max_depth":         5,
        "min_samples_leaf": 25,
        "random_state":     42,
    }

    def __init__(self, **params):
        self._params   = {**self._DEFAULT_PARAMS, **params}
        self._model    = None
        self._features = None

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        self._model = DecisionTreeRegressor(**self._params)
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X).astype(np.float32)

    def get_feature_importance(self) -> dict:
        if self._model is None or self._features is None:
            return {}
        return dict(zip(self._features, self._model.feature_importances_.tolist()))

    def set_feature_names(self, features):
        self._features = list(features)
        return self

    def tune_hyperparameters(self, X_train, y_train, X_val, y_val,
                              search_space: dict) -> dict:
        space = search_space or {
            "max_depth":        [3, 5, 10],
            "min_samples_leaf": [5, 10, 25, 50, 100],
        }
        best_ic, best = -np.inf, {}
        for params in ParameterGrid(space):
            m = DecisionTreeRegressor(**params, random_state=self._params["random_state"])
            m.fit(X_train, y_train)
            ic = spearman_ic(y_val, m.predict(X_val))
            if ic > best_ic:
                best_ic, best = ic, params
        if best:
            self._params.update(best)
        return {**self._params}
