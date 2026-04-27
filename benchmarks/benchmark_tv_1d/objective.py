import pathlib
import sys

from benchopt import BaseObjective
from benchopt import safe_import_context

BENCHMARK_DIR = pathlib.Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import optimize

    from benchmark_utils.shared import grad_huber
    from benchmark_utils.shared import huber


class Objective(BaseObjective):
    name = "TV1D"
    min_benchopt_version = "1.5"

    parameters = {
        "reg": [0.5],
        "delta": [0.9],
        "data_fit": ["quad", "huber"],
    }

    def set_data(self, A, y, x):
        self.A = A
        self.y = np.asarray(y, dtype=float).reshape(-1)
        self.x = np.asarray(x, dtype=float).reshape(-1)
        S = self.A @ np.ones(self.A.shape[1])
        self.c = self.get_c(S, self.delta)
        self.reg_scaled = self.reg * self.get_reg_max(self.c)

    def evaluate_result(self, u):
        u = np.asarray(u, dtype=float).reshape(-1)
        R = self.y - self.A @ u
        reg_tv = abs(np.diff(u)).sum()
        if self.data_fit == "quad":
            loss = 0.5 * R @ R
        elif self.data_fit == "huber":
            loss = huber(R, self.delta)
        else:
            raise ValueError(f"Unknown data_fit={self.data_fit!r}.")
        norm_x = np.linalg.norm(u - self.x)

        return dict(value=loss + self.reg_scaled * reg_tv, norm_x=norm_x)

    def get_one_result(self):
        return dict(u=np.zeros(self.A.shape[1]))

    def get_objective(self):
        return dict(
            A=self.A,
            reg=self.reg_scaled,
            y=self.y,
            c=self.c,
            delta=self.delta,
            data_fit=self.data_fit,
        )

    def get_c(self, S, delta):
        denom = S @ S
        if denom <= 0:
            return 0.0
        if self.data_fit == "quad":
            return (S @ self.y) / denom
        return self.c_huber(S, delta)

    def c_huber(self, S, delta):
        def f(c):
            R = self.y - S * c
            return abs((S * grad_huber(R, delta)).sum())

        nonzero = np.abs(S) > 1e-14
        if not np.any(nonzero):
            return 0.0
        yS = self.y[nonzero] / S[nonzero]
        return optimize.golden(f, brack=(min(yS), max(yS)))

    def get_reg_max(self, c):
        L = np.tri(self.A.shape[1])
        AL = self.A @ L
        z = np.zeros(self.A.shape[1])
        z[0] = c
        return np.max(abs(self.grad(AL, z)))

    def grad(self, A, u):
        R = self.y - A @ u
        if self.data_fit == "quad":
            return -A.T @ R
        return -A.T @ grad_huber(R, self.delta)
