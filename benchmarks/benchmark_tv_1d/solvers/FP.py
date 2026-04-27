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

    from benchmark_utils.shared import grad_huber
    from benchmark_utils.shared import prox_z


class Solver(BaseSolver):
    """Fixed point with block updates for synthesis formulation."""

    name = "FP synthesis"
    stopping_criterion = SufficientDescentCriterion(patience=3, strategy="callback")
    parameters = {"alpha": [1.9], "use_acceleration": [False, True]}

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A = A
        self.y = y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        n, p = self.A.shape
        L = np.tri(p)
        AL = self.A @ L
        stepsize = self.alpha / (n * np.max((AL**2).sum(axis=1)))
        self.z = np.zeros(p)
        self.z[0] = self.c
        z_old = self.z.copy()
        z_acc = self.z.copy()

        t_new = 1
        while callback():
            if self.use_acceleration:
                t_old = t_new
                t_new = (1 + np.sqrt(1 + 4 * t_old**2)) / 2
                z_old[:] = self.z
                self.z[:] = z_acc
            mu = self.z - stepsize * (n * self.grad(AL, self.z) * AL.T).T
            nu = np.mean(mu, axis=0)
            self.z = prox_z(nu, stepsize * self.reg)
            if self.use_acceleration:
                z_acc[:] = self.z + (t_old - 1.0) / t_new * (self.z - z_old)

    def get_result(self):
        return dict(u=np.cumsum(self.z))

    def grad(self, A, u):
        R = self.y - A @ u
        if self.data_fit == "quad":
            return -R
        return -grad_huber(R, self.delta)
