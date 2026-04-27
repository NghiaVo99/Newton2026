import pathlib
import sys

from benchopt import BaseSolver
from benchopt import safe_import_context

BENCHMARK_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

with safe_import_context() as import_ctx:
    import numpy as np

    from benchmark_utils.tv1d_utils import can_materialize_design
    from benchmark_utils.tv1d_utils import compute_step_size
    from benchmark_utils.tv1d_utils import has_tv_prox_available
    from benchmark_utils.tv1d_utils import make_cached_tv_subproblem_solver
    from benchmark_utils.tv1d_utils import make_tv_prox
    from benchmark_utils.tv1d_utils import materialize_design
    from benchmark_utils.tv1d_utils import run_newton_ista

    if not has_tv_prox_available():
        raise ImportError("newton_ista requires TV prox support")


class Solver(BaseSolver):
    name = "newton_ista"
    sampling_strategy = "iteration"
    install_cmd = "conda"
    requirements = []
    support_sparse = False

    def skip(self, A, reg, y, c, delta, data_fit):
        if data_fit != "quad":
            return True, f"{self.name} currently supports only quadratic TV1D loss"
        if not has_tv_prox_available():
            return True, f"{self.name} requires TV prox support"
        if not can_materialize_design(A):
            return True, f"{self.name} materializes A densely for this benchmark"
        return False, None

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.A = materialize_design(A)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        self.reg = float(reg)
        self.c = float(c)
        self.u = self.c * np.ones(self.A.shape[1], dtype=float)
        self.step_size = compute_step_size(self.A)
        self.prox = make_tv_prox(self.A.shape[1])
        self.subproblem_solver = make_cached_tv_subproblem_solver(self.A)

    def run(self, n_iter):
        if int(n_iter) <= 0:
            return
        self.u = run_newton_ista(
            self.A,
            self.y,
            self.reg,
            self.c,
            int(n_iter),
            step_size=self.step_size,
            prox=self.prox,
            subproblem_solver=self.subproblem_solver,
        )

    @staticmethod
    def get_next(previous):
        if previous <= 0:
            return 1
        return 2 * previous

    def get_result(self):
        return dict(u=self.u)
