import pathlib
import sys

from benchopt import BaseSolver
from benchopt import safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

BENCHMARK_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

with safe_import_context() as import_ctx:
    import numpy as np

    from benchmark_utils.shared import get_l2norm
    from benchmark_utils.shared import grad_huber
    from benchmark_utils.tv1d_utils import has_tv_prox_available
    from benchmark_utils.tv1d_utils import make_tv_prox

    if not has_tv_prox_available():
        raise ImportError("Primal PGD analysis requires TV prox support")


class Solver(BaseSolver):
    """Proximal gradient descent for analysis formulation."""

    name = "Primal PGD analysis"
    install_cmd = "conda"
    requirements = []
    stopping_criterion = SufficientProgressCriterion(patience=3, strategy="callback")

    parameters = {"alpha": [1.0], "use_acceleration": [False, True]}

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A = A
        self.y = y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit
        self.prox = make_tv_prox(A.shape[1])

    def run(self, callback):
        p = self.A.shape[1]
        stepsize = self.alpha / get_l2norm(self.A) ** 2
        self.u = self.c * np.ones(p)
        u_acc = self.u.copy()
        u_old = self.u.copy()

        t_new = 1
        while callback():
            if self.use_acceleration:
                t_old = t_new
                t_new = (1 + np.sqrt(1 + 4 * t_old**2)) / 2
                u_old[:] = self.u
                self.u[:] = u_acc
            self.u = self.prox(
                self.u - stepsize * self.grad(self.A, self.u),
                self.reg * stepsize,
            )
            if self.use_acceleration:
                u_acc[:] = self.u + (t_old - 1.0) / t_new * (self.u - u_old)

    def get_result(self):
        return dict(u=self.u)

    def grad(self, A, u):
        R = self.y - A @ u
        if self.data_fit == "quad":
            return -A.T @ R
        return -A.T @ grad_huber(R, self.delta)
