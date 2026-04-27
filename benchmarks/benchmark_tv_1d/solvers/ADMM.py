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
    from scipy.optimize import minimize
    from scipy.sparse import spdiags
    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse.linalg import cg

    from benchmark_utils.shared import grad_huber
    from benchmark_utils.shared import huber
    from benchmark_utils.shared import st


def loss(y, A, u, delta, z, mu, gamma):
    R = y - A @ u
    return huber(R, delta) + gamma / 2 * np.linalg.norm(
        np.diff(u) - z + mu / gamma, ord=2
    ) ** 2


def jac_loss(y, A, u, delta, z, mu, gamma):
    R = y - A @ u
    return -A.T @ grad_huber(R, delta) - gamma * np.diff(
        np.diff(u) - z + mu / gamma, append=0, prepend=0
    )


class Solver(BaseSolver):
    """Alternating direction method for analysis formulation."""

    name = "ADMM analysis"
    stopping_criterion = SufficientDescentCriterion(patience=3, strategy="callback")
    parameters = {"gamma": [25.0], "update_pen": [False]}

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A = A
        self.y = y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        p = self.A.shape[1]
        self.u = u = self.c * np.ones(p)
        z = np.zeros(p - 1)
        mu = np.zeros(p - 1)
        gamma = self.gamma
        tol_cg = 1e-12
        Aty = self.A.T @ self.y
        if isinstance(self.A, np.ndarray):
            data = np.array([-np.ones(p), np.ones(p)])
            diags = np.array([0, 1])
            D = spdiags(data, diags, p - 1, p)
            AtA_gDtD_inv = np.linalg.pinv(self.A.T @ self.A + gamma * D.T @ D)
        else:
            AtA_gDtD = LinearOperator(
                shape=(p, p),
                matvec=lambda x: self.A.T @ (self.A @ x)
                - gamma * np.diff(np.diff(x), append=0, prepend=0),
            )

        while callback():
            z_old = z
            if self.data_fit == "quad":
                u_tmp = (
                    Aty
                    + np.diff(mu, append=0, prepend=0)
                    - gamma * np.diff(z, append=0, prepend=0)
                )
                if isinstance(self.A, np.ndarray):
                    u = np.ravel(AtA_gDtD_inv @ u_tmp)
                else:
                    u, _ = cg(AtA_gDtD, u_tmp, x0=u, tol=tol_cg)
            else:
                u = minimize(
                    lambda u_: loss(self.y, self.A, u_, self.delta, z, mu, gamma),
                    x0=u,
                    jac=lambda u_: jac_loss(
                        self.y, self.A, u_, self.delta, z, mu, gamma
                    ),
                    method="BFGS",
                    tol=1e-15,
                ).x

            z = st(np.diff(u) + mu / gamma, self.reg / gamma)
            mu += gamma * (np.diff(u) - z)

            if self.update_pen:
                r = np.linalg.norm(np.diff(u) - z, ord=2)
                s = np.linalg.norm(
                    gamma * np.diff(z - z_old, append=0, prepend=0), ord=2
                )
                if r > 10 * s:
                    gamma *= 2
                if s > 10 * r:
                    gamma /= 2
            self.u = u

    def get_result(self):
        return dict(u=self.u)
