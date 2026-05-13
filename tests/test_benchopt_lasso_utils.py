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
from src.lasso.newton_lasso import Algo_Globalized_Effective_Subspace_Newton_Lasso
from src.lasso.newton_lasso import Algo_Newton_Ista
from src.lasso.newton_lasso import _solve_alg5_lasso_direction
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

    def _run_direct(self, func, *args, **kwargs):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            return func(*args, **kwargs)

    def _manual_dense_subproblem(self, x, y):
        gram = self.X.T @ self.X
        atb = self.X.T @ self.y
        kappa = np.where(np.abs(y) >= 0.999 * self.lmbd)[0]
        d_full = np.zeros_like(y, dtype=float)
        if kappa.size == 0:
            return d_full

        Q = gram[np.ix_(kappa, kappa)]
        rhs = (gram @ x - atb + y)[kappa]
        try:
            d_reduced = np.linalg.solve(Q, rhs)
        except np.linalg.LinAlgError:
            d_reduced = np.linalg.lstsq(Q, rhs, rcond=None)[0]
        d_full[kappa] = d_reduced
        return d_full

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

    def test_verbose_false_matches_verbose_true_for_newton_variants(self):
        _, beta_verbose, _, _, _ = self._run_direct(
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
        _, beta_quiet, _, _, _ = Algo_Newton_Ista(
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
            verbose=False,
        )
        np.testing.assert_allclose(beta_quiet, beta_verbose)

        _, beta_verbose, _, _, _ = self._run_direct(
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
        _, beta_quiet, _, _, _ = Algo_Newton_Fista_new(
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
            verbose=False,
        )
        np.testing.assert_allclose(beta_quiet, beta_verbose)

    def test_dense_lasso_newton_subproblem_matches_manual_cached_formula(self):
        x = np.linspace(-0.3, 0.4, self.X.shape[1])
        y = self.X.T @ (self.X @ x - self.y)
        y[:5] = np.sign(y[:5]) * max(2.0 * self.lmbd, 1e-8)

        actual = dense_lasso_newton_subproblem(self.X, x, y, self.y, self.lmbd)
        expected = self._manual_dense_subproblem(x, y)

        np.testing.assert_allclose(actual, expected)

    def test_algorithm5_lasso_returns_finite_histories_and_descends(self):
        costs, beta, n_iter, dists, times = (
            Algo_Globalized_Effective_Subspace_Newton_Lasso(
                self.X,
                self.y,
                self.x0,
                self.lmbd,
                10,
                self.step_size,
                self.no_early_stop_tol,
                cost_lasso,
                proxL1,
                approx_sol=0,
                epsilon=DEFAULT_ISTA_NEWTON_TOL,
                verbose=False,
            )
        )

        self.assertGreaterEqual(n_iter, 0)
        self.assertEqual(len(costs), len(dists))
        self.assertEqual(len(costs), len(times))
        self.assertTrue(np.all(np.isfinite(costs)))
        self.assertTrue(np.all(np.isfinite(beta)))
        self.assertTrue(np.all(np.isfinite(dists)))
        self.assertTrue(np.all(np.isfinite(times)))
        self.assertLessEqual(costs[-1], costs[0] + 1e-10)
        self.assertTrue(np.all(np.diff(costs) <= 1e-8))

    def test_algorithm5_empty_effective_subspace_direction_is_zero(self):
        x = np.linspace(-0.2, 0.2, self.X.shape[1])
        y = x.copy()
        z = np.zeros_like(x)

        direction = _solve_alg5_lasso_direction(
            self.X, y, z, self.y, self.lmbd, mu=1.0, active_tol=1e-12)

        np.testing.assert_allclose(direction, np.zeros_like(x))


if __name__ == "__main__":
    unittest.main()
