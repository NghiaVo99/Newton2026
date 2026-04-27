import pathlib
import sys

from benchopt import BaseSolver
from benchopt import safe_import_context

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

with safe_import_context() as import_ctx:
    import numpy as np

    from src.lasso.utils_lasso import solve_lasso_cvxpy


class Solver(BaseSolver):
    name = "cvxpy"
    sampling_strategy = "run_once"
    support_sparse = False
    install_cmd = "pip"
    requirements = ["cvxpy"]

    def skip(self, X, y, lmbd):
        try:
            import cvxpy  # noqa: F401
        except Exception as exc:  # pragma: no cover - runtime dependency check
            return True, f"{self.name} unavailable: {exc}"
        return False, None

    def set_objective(self, X, y, lmbd):
        self.X = X
        self.y = y
        self.lmbd = lmbd
        self.beta = np.zeros(X.shape[1], dtype=float)

    def run(self, _):
        self.beta = np.asarray(solve_lasso_cvxpy(self.X, self.y, self.lmbd), dtype=float)

    def get_result(self):
        return dict(beta=self.beta)
