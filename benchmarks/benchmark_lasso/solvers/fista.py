from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse


class Solver(BaseSolver):
    name = "fista"
    sampling_strategy = "callback"

    references = [
        "A. Beck and M. Teboulle, "
        '"A fast iterative shrinkage-thresholding algorithm for linear inverse '
        'problems", SIAM J. Imaging Sci., vol. 2, no. 1, pp. 183-202 (2009)'
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
        z = np.zeros(n_features)
        t_new = 1.0

        while callback():
            t_old = t_new
            t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
            w_old = w.copy()
            z -= self.X.T @ (self.X @ z - self.y) / L
            w = self.st(z, self.lmbd / L)
            z = w + (t_old - 1.0) / t_new * (w - w_old)
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
