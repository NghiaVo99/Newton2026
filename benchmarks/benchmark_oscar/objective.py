import numpy as np
from numpy.linalg import norm

from benchopt import BaseObjective


class Objective(BaseObjective):
    """OSCAR objective via equivalent SLOPE weighted sorted-L1 form."""

    name = "OSCAR Regression"
    min_benchopt_version = "1.7"
    parameters = {
        "w1": [1e-3],
        "w2": [1e-4],
        "fit_intercept": [False],
    }

    def __init__(self, w1=1e-3, w2=1e-4, fit_intercept=False):
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.fit_intercept = bool(fit_intercept)

    def set_data(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        self.n_samples, self.n_features = self.X.shape

        if self.w1 < 0 or self.w2 < 0:
            raise ValueError("w1 and w2 must be nonnegative.")

        # Equivalent SLOPE weights for OSCAR: alpha_i = w1 + w2 * (p - 1 - i)
        idx = np.arange(self.n_features, dtype=float)
        self.alphas = self.w1 + self.w2 * (self.n_features - 1 - idx)

    def get_one_result(self):
        return dict(beta=np.zeros(self.n_features + 1, dtype=float))

    def evaluate_result(self, beta):
        beta = np.asarray(beta, dtype=float).reshape(-1)
        intercept = beta[0]
        coefs = beta[1:]

        diff = self.y - self.X @ coefs - intercept
        p_obj = 1.0 / (2 * self.n_samples) * diff.dot(diff)
        p_obj += float(np.sum(self.alphas * np.sort(np.abs(coefs))[::-1]))

        theta = diff.copy()
        theta /= max(1.0, self._dual_norm_slope(theta, self.alphas))
        d_obj = (norm(self.y) ** 2 - norm(self.y - theta * self.n_samples) ** 2) / (
            2 * self.n_samples
        )

        gap = p_obj - d_obj
        return dict(
            value=p_obj,
            duality_gap=gap,
            rel_duality_gap=gap / (1e-10 + abs(p_obj)),
            support_size=int((coefs != 0).sum()),
        )

    def _dual_norm_slope(self, theta, alphas):
        Xtheta = np.sort(np.abs(self.X.T @ theta))[::-1]
        taus = 1.0 / np.cumsum(alphas)
        return float(np.max(np.cumsum(Xtheta) * taus))

    def get_objective(self):
        return dict(
            X=self.X,
            y=self.y,
            alphas=self.alphas,
            fit_intercept=self.fit_intercept,
        )
