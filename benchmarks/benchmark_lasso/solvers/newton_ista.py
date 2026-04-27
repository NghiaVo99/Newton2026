from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    from benchmark_utils.lasso_utils import run_newton_ista


class Solver(BaseSolver):
    name = "newton_ista"
    sampling_strategy = "iteration"
    support_sparse = False

    def skip(self, X, y, lmbd, fit_intercept):
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"
        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X = X
        self.y = y
        self.lmbd = lmbd
        self.beta = np.zeros(X.shape[1], dtype=float)

    def run(self, n_iter):
        if int(n_iter) <= 0:
            return
        self.beta = run_newton_ista(self.X, self.y, self.lmbd, n_iter)

    def get_result(self):
        return dict(beta=self.beta)
