import pathlib
import sys

from benchopt import BaseSolver
from benchopt import safe_import_context

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

with safe_import_context() as import_ctx:
    import numpy as np

    from src.lasso.utils_lasso import solve_lasso_gurobi


class Solver(BaseSolver):
    name = "gurobi"
    sampling_strategy = "run_once"
    support_sparse = False
    install_cmd = "pip"
    requirements = ["gurobipy"]

    def skip(self, X, y, lmbd):
        try:
            import gurobipy as gp

            model = gp.Model()
            model.setParam("OutputFlag", 0)
            x = model.addVar(name="x")
            model.setObjective(x, gp.GRB.MAXIMIZE)
            model.optimize()
        except Exception as exc:  # pragma: no cover - runtime dependency check
            return True, f"{self.name} unavailable: {exc}"
        return False, None

    def set_objective(self, X, y, lmbd):
        self.X = X
        self.y = y
        self.lmbd = lmbd
        self.beta = np.zeros(X.shape[1], dtype=float)

    def run(self, _):
        self.beta = np.asarray(
            solve_lasso_gurobi(self.X, self.y, self.lmbd, verbose=False)[0],
            dtype=float,
        )

    def get_result(self):
        return dict(beta=self.beta)
