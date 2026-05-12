"""
models/random_forest.py — RandomForestModel
Ported from Random Forest/rf_pipeline.py.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from ..shared.walk_forward import BaseModel
from ..shared.metrics import spearman_ic


_SEARCH_SPACE = {
    "n_estimators":    [100, 200],
    "max_depth":       [3, 5],
    "min_samples_leaf":[50, 100],
    "max_features":    ["sqrt", 0.5],
}


class RandomForestModel(BaseModel):
    name           = "random_forest"
    uses_sequences = False

    def __init__(self, seed: int = 42, **hparams):
        self._seed    = seed
        self._hparams = {
            "n_estimators": 100, "max_depth": 5,
            "min_samples_leaf": 50, "max_features": "sqrt",
            **hparams,
        }
        self._model    = None
        self._features = None

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        self._model = RandomForestRegressor(
            **self._hparams,
            n_jobs      = -1,
            random_state= self._seed,
        )
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X).astype(np.float32)

    def get_feature_importance(self) -> dict:
        if self._model is None or self._features is None:
            return {}
        return dict(zip(self._features,
                        self._model.feature_importances_.astype(float).tolist()))

    def set_feature_names(self, features):
        self._features = list(features)
        return self

    def tune_hyperparameters(self, X_train, y_train, X_val, y_val,
                              search_space: dict) -> dict:
        space = search_space or _SEARCH_SPACE
        grid  = list(ParameterGrid(space))

        def _eval(params):
            m = RandomForestRegressor(**params, n_jobs=1, random_state=self._seed)
            m.fit(X_train, y_train)
            return spearman_ic(y_val, m.predict(X_val))

        scores = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_eval)(p) for p in grid
        )
        best = grid[int(np.argmax(scores))]
        self._hparams.update(best)
        return {**self._hparams}
