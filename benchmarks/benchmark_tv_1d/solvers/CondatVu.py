import pathlib
import sys

from benchopt import BaseSolver
from benchopt import safe_import_context
from benchopt.stopping_criterion import SufficientDescentCriterion

BENCHMARK_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

with safe_import_context() as import_ctx:
    import numpy as np

    from benchmark_utils.shared import get_l2norm
    from benchmark_utils.shared import grad_huber


class Solver(BaseSolver):
    """Primal-dual splitting method for analysis formulation."""

    name = "CondatVu analysis"
    stopping_criterion = SufficientDescentCriterion(patience=3, strategy="callback")
    parameters = {"ratio": [1.0], "eta": [1.0]}

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A = A
        self.y = y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        _, p = self.A.shape
        LD = 2.0
        LA = get_l2norm(self.A)
        sigma = 1.0 / (self.ratio * LD)
        tau = 1 / (LA**2 / 2 + sigma * LD**2)
        eta = self.eta
        self.u = self.c * np.ones(p)
        v = np.zeros(p - 1)

        while callback():
            u_tmp = (
                self.u
                - tau * self.grad(self.A, self.u)
                - tau * (-np.diff(v, append=0, prepend=0))
            )
            v_tmp = np.clip(
                v + sigma * np.diff(2 * u_tmp - self.u), -self.reg, self.reg
            )
            self.u = eta * u_tmp + (1 - eta) * self.u
            v = eta * v_tmp + (1 - eta) * v

    def get_result(self):
        return dict(u=self.u)

    def grad(self, A, u):
        R = self.y - A @ u
        if self.data_fit == "quad":
            return -A.T @ R
        return -A.T @ grad_huber(R, self.delta)
