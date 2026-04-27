from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse


class Solver(BaseSolver):
    name = "ista"
    sampling_strategy = "callback"

    references = [
        "I. Daubechies, M. Defrise and C. De Mol, "
        '"An iterative thresholding algorithm for linear inverse problems '
        'with a sparsity constraint", Comm. Pure Appl. Math., '
        "vol. 57, pp. 1413-1457, no. 11, Wiley Online Library (2004)"
    ]

    def skip(self, X, y, lmbd, fit_intercept):
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"
        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

    def run(self, callback):
        L = self.compute_lipschitz_constant()
        n_features = self.X.shape[1]
        self.w = w = np.zeros(n_features)

        while callback():
            w -= self.X.T @ (self.X @ w - self.y) / L
            w = self.st(w, self.lmbd / L)
            self.w = w

    def st(self, w, mu):
        w -= np.clip(w, -mu, mu)
        return w

    def get_result(self):
        return dict(beta=self.w)

    def compute_lipschitz_constant(self):
        if not sparse.issparse(self.X):
            return np.linalg.norm(self.X, ord=2) ** 2
        return sparse.linalg.svds(self.X, k=1)[1][0] ** 2
