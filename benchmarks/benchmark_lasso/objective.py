import numpy as np
from numpy.linalg import norm

from benchopt import BaseObjective


class Objective(BaseObjective):
    """L1 regularized linear regression."""

    name = "Lasso Regression"
    min_benchopt_version = "1.7"
    parameters = {
        "fit_intercept": [False],
        "reg": [0.01],
    }

    def __init__(self, reg=0.01, fit_intercept=False):
        self.reg = reg
        self.fit_intercept = fit_intercept

    def set_data(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        self.n_features = self.X.shape[1]
        self.lmbd = self.reg * self._get_lambda_max()

    def get_one_result(self):
        n_features = self.n_features + int(self.fit_intercept)
        return dict(beta=np.zeros(n_features, dtype=float))

    def evaluate_result(self, beta):
        beta = np.asarray(beta, dtype=np.float64)
        if self.fit_intercept:
            coef = beta[: self.n_features]
            intercept = beta[self.n_features :]
        else:
            coef = beta
            intercept = 0.0

        diff = self.y - self.X @ coef
        if self.fit_intercept:
            diff -= intercept

        p_obj = 0.5 * diff.dot(diff) + self.lmbd * abs(coef).sum()
        scaling = max(1, norm(self.X.T @ diff, ord=np.inf) / self.lmbd)
        d_obj = (norm(self.y) ** 2 / 2.0) - (norm(self.y - diff / scaling) ** 2 / 2.0)
        return dict(
            value=p_obj,
            support_size=(coef != 0).sum(),
            duality_gap=p_obj - d_obj,
        )

    def _get_lambda_max(self):
        if self.fit_intercept:
            return abs(self.X.T @ (self.y - self.y.mean())).max()
        return abs(self.X.T @ self.y).max()

    def get_objective(self):
        return dict(
            X=self.X,
            y=self.y,
            lmbd=self.lmbd,
            fit_intercept=self.fit_intercept,
        )
