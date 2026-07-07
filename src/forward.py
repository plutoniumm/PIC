"""Forward models: data-driven black-box surrogate and a grey-box physics scaffold."""

from __future__ import annotations
import numpy as np


class BlackBoxForward:
    """Per-output gradient-boosted surrogate on [v, v^2] features."""

    def __init__(
        self, max_iter: int = 300, learning_rate: float = 0.08, random_state: int = 0
    ):
        self.kw = dict(
            max_iter=max_iter, learning_rate=learning_rate, random_state=random_state
        )
        self.models = None

    @staticmethod
    def _feat(X):
        X = np.atleast_2d(np.asarray(X, float))
        return np.hstack([X, X**2])

    def fit(self, X, Y):
        from sklearn.ensemble import HistGradientBoostingRegressor

        F = self._feat(X)
        Y = np.atleast_2d(np.asarray(Y, float))
        self.models = []
        for k in range(Y.shape[1]):
            g = HistGradientBoostingRegressor(**self.kw)
            g.fit(F, Y[:, k])
            self.models.append(g)
        return self

    def predict(self, X):
        if self.models is None:
            raise RuntimeError("call fit() first")
        F = self._feat(X)
        return np.column_stack([m.predict(F) for m in self.models])

    def __call__(self, v):
        return self.predict(v)[0]


class GreyBoxForward:
    """Physics forward model (phi = phi2*v^2 + phi0 per heater, composed through the
    src.mzi mesh topology). Stub: needs topology + per-heater characterization."""

    def __init__(self, topology=None, params=None):
        self.topology = topology
        self.params = params

    def fit(self, *args, **kw):
        raise NotImplementedError(
            "grey-box fit needs topology + (homodyne) characterization data"
        )

    def predict(self, X):
        raise NotImplementedError(
            "grey-box predict pending topology + characterization"
        )
