import contextlib
import io
import unittest

import numpy as np

from benchmarks.benchmark_lasso.benchmark_utils.lasso_utils import DEFAULT_BT_BETA
from benchmarks.benchmark_lasso.benchmark_utils.lasso_utils import DEFAULT_FISTA_NEWTON_TOL
from benchmarks.benchmark_lasso.benchmark_utils.lasso_utils import DEFAULT_ISTA_NEWTON_TOL
from benchmarks.benchmark_lasso.benchmark_utils.lasso_utils import DEFAULT_NEWTON_STEP
from benchmarks.benchmark_lasso.benchmark_utils.lasso_utils import dense_lasso_newton_subproblem
from benchmarks.benchmark_lasso.benchmark_utils.lasso_utils import run_newton_bt_fista
from benchmarks.benchmark_lasso.benchmark_utils.lasso_utils import run_newton_bt_ista
from benchmarks.benchmark_lasso.benchmark_utils.lasso_utils import run_newton_fista
from benchmarks.benchmark_lasso.benchmark_utils.lasso_utils import run_newton_ista
from src.lasso.newton_lasso import Algo_Newton_BT_Fista_new
from src.lasso.newton_lasso import Algo_Newton_BT_Ista
from src.lasso.newton_lasso import Algo_Newton_Fista_new
from src.lasso.newton_lasso import Algo_Newton_Ista
from src.lasso.utils_lasso import cost_lasso
from src.lasso.utils_lasso import lipschitz_exact
from src.lasso.utils_lasso import proxL1


class BenchoptLassoUtilsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.X = rng.randn(20, 30)
        self.y = rng.randn(20)
        self.lmbd = 0.05 * np.abs(self.X.T @ self.y).max()
        self.x0 = np.zeros(self.X.shape[1], dtype=float)
        self.step_size = 1.0 / lipschitz_exact(self.X)
        self.n_iter = 5
        self.no_early_stop_tol = -1.0

    def _run_direct(self, func, *args):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            return func(*args)

    def test_newton_bt_ista_runner_matches_direct_call(self):
        _, beta_direct, _, _, _ = self._run_direct(
            Algo_Newton_BT_Ista,
            self.X,
            self.y,
            self.x0,
            self.lmbd,
            self.n_iter,
            DEFAULT_BT_BETA,
            DEFAULT_NEWTON_STEP,
            self.no_early_stop_tol,
            cost_lasso,
            proxL1,
            dense_lasso_newton_subproblem,
            DEFAULT_ISTA_NEWTON_TOL,
            0,
        )
        beta_runner = run_newton_bt_ista(self.X, self.y, self.lmbd, self.n_iter)
        np.testing.assert_allclose(beta_runner, beta_direct)

    def test_newton_ista_runner_matches_direct_call(self):
        _, beta_direct, _, _, _ = self._run_direct(
            Algo_Newton_Ista,
            self.X,
            self.y,
            self.x0,
            self.lmbd,
            self.n_iter,
            self.step_size,
            DEFAULT_BT_BETA,
            DEFAULT_NEWTON_STEP,
            self.no_early_stop_tol,
            cost_lasso,
            proxL1,
            dense_lasso_newton_subproblem,
            DEFAULT_ISTA_NEWTON_TOL,
            0,
        )
        beta_runner = run_newton_ista(self.X, self.y, self.lmbd, self.n_iter)
        np.testing.assert_allclose(beta_runner, beta_direct)

    def test_newton_bt_fista_runner_matches_direct_call(self):
        _, beta_direct, _, _, _ = self._run_direct(
            Algo_Newton_BT_Fista_new,
            self.X,
            self.y,
            self.x0,
            self.lmbd,
            self.n_iter,
            DEFAULT_BT_BETA,
            DEFAULT_NEWTON_STEP,
            self.no_early_stop_tol,
            cost_lasso,
            proxL1,
            dense_lasso_newton_subproblem,
            DEFAULT_FISTA_NEWTON_TOL,
            0,
        )
        beta_runner = run_newton_bt_fista(self.X, self.y, self.lmbd, self.n_iter)
        np.testing.assert_allclose(beta_runner, beta_direct)

    def test_newton_fista_runner_matches_direct_call(self):
        _, beta_direct, _, _, _ = self._run_direct(
            Algo_Newton_Fista_new,
            self.X,
            self.y,
            self.x0,
            self.lmbd,
            self.n_iter,
            self.step_size,
            DEFAULT_BT_BETA,
            DEFAULT_NEWTON_STEP,
            self.no_early_stop_tol,
            cost_lasso,
            proxL1,
            dense_lasso_newton_subproblem,
            DEFAULT_FISTA_NEWTON_TOL,
            0,
        )
        beta_runner = run_newton_fista(self.X, self.y, self.lmbd, self.n_iter)
        np.testing.assert_allclose(beta_runner, beta_direct)


if __name__ == "__main__":
    unittest.main()
