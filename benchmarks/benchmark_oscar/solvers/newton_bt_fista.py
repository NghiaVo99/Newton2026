from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    from benchmark_utils.oscar_utils import _make_cached_subproblem_solver
    from benchmark_utils.oscar_utils import run_newton_bt_fista
    from benchmark_utils.oscar_utils import warm_up_fast_prox


class Solver(BaseSolver):
    name = "newton_bt_fista"
    sampling_strategy = "iteration"
    support_sparse = False

    def skip(self, X, y, alphas, fit_intercept):
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"
        return False, None

    def set_objective(self, X, y, alphas, fit_intercept):
        self.X = X
        self.y = y
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.subproblem_solver = _make_cached_subproblem_solver(X)
        warm_up_fast_prox(X.shape[1])
        self.beta = np.zeros(X.shape[1] + 1, dtype=float)

    def run(self, n_iter):
        if int(n_iter) <= 0:
            return
        self.beta = run_newton_bt_fista(
            self.X,
            self.y,
            self.alphas,
            n_iter,
            fit_intercept=self.fit_intercept,
            subproblem_solver=self.subproblem_solver,
        )

    @staticmethod
    def get_next(previous):
        if previous <= 0:
            return 1
        return 2 * previous

    def get_result(self):
        return dict(beta=self.beta)
