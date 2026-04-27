import numpy as np

import src.Gen_lasso.Gen_Lasso_algo as algo
from src.Gen_lasso.Gen_Lasso_algo import _accept_damped_newton_step
from src.Gen_lasso.Gen_Lasso_algo import Algo_Newton_Fista_new
from src.Gen_lasso.Gen_Lasso_algo import Algo_Newton_Ista
from src.Gen_lasso.Gen_Lasso_algo import BT_FISTA1
from src.Gen_lasso.Gen_Lasso_algo import BT_ISTA
from src.Gen_lasso.Gen_Lasso_utils import cost_generalized_lasso
from src.Gen_lasso.Gen_Lasso_utils import inactive_tv_constraint_indices


def _quadratic_cost(A, x, b, alpha, D):
    return float(np.dot(x, x))


def test_bt_ista_passes_gradient_before_prox_to_linesearch(monkeypatch):
    A = np.eye(2)
    D = np.eye(2)
    b = np.array([1.0, -2.0])
    x0 = np.zeros(2)

    def prox(x, tau):
        return x

    def fake_linesearch(A_arg, b_arg, x_arg, grad_arg, prox_arg, alpha_arg):
        assert isinstance(grad_arg, np.ndarray)
        assert callable(prox_arg)
        return 0.1

    monkeypatch.setattr(algo, "backtracking_linesearch", fake_linesearch)

    BT_ISTA(A, D, b, x0, 0.2, 1, -1.0, _quadratic_cost, prox)


def test_bt_fista_takes_gradient_step_before_prox(monkeypatch):
    A = np.eye(2)
    D = np.eye(2)
    b = np.array([1.0, -2.0])
    x0 = np.zeros(2)
    seen = []

    def prox(x, tau):
        seen.append(x.copy())
        return x

    monkeypatch.setattr(algo, "backtracking_linesearch", lambda *args, **kwargs: 0.1)

    BT_FISTA1(A, D, b, x0, 0.2, 1, -1.0, _quadratic_cost, prox)

    expected_grad = A.T @ (A @ x0 - b)
    np.testing.assert_allclose(seen[0], x0 - 0.1 * expected_grad)


def test_inactive_tv_indices_scale_with_alpha():
    zk = np.array([0.4, 0.2, 0.0, 0.0])

    idx_alpha_one = inactive_tv_constraint_indices(zk, alpha=1.0, n=4)
    idx_alpha_half = inactive_tv_constraint_indices(zk, alpha=0.5, n=4)

    np.testing.assert_array_equal(idx_alpha_one, np.array([0, 1, 2]))
    np.testing.assert_array_equal(idx_alpha_half, np.array([0]))


def test_damped_newton_rejects_objective_increase():
    x_hat = np.zeros(2)
    d = np.array([-1.0, 0.0])

    x_new, accepted = _accept_damped_newton_step(
        np.eye(2),
        np.eye(2),
        np.zeros(2),
        0.0,
        _quadratic_cost,
        x_hat,
        d,
        beta=0.5,
        newton_stepsize=1.0,
        max_backtracks=3,
    )

    assert not accepted
    np.testing.assert_allclose(x_new, x_hat)


def test_negative_tol_skips_extra_stopping_prox_for_newton_variants():
    A = np.eye(3)
    D = np.eye(3)
    b = np.array([1.0, -2.0, 0.5])
    x0 = np.zeros(3)
    alpha = 0.1
    step_size = 0.5
    max_iter = 4

    def subproblem_solver(A_arg, yk, zk, b_arg, alpha_arg):
        return np.zeros_like(yk)

    counts = {"ista": 0, "fista": 0}

    def prox_ista(x, tau):
        counts["ista"] += 1
        return x

    def prox_fista(x, tau):
        counts["fista"] += 1
        return x

    Algo_Newton_Ista(
        A, D, b, x0, alpha, max_iter, step_size, 0.5, 1.0, -1.0,
        cost_generalized_lasso, prox_ista, subproblem_solver, newt_tol=-1.0,
        verbose=False,
    )
    Algo_Newton_Fista_new(
        A, D, b, x0, alpha, max_iter, step_size, 0.5, 1.0, -1.0,
        cost_generalized_lasso, prox_fista, subproblem_solver, newt_tol=-1.0,
        verbose=False,
    )

    assert counts["ista"] == max_iter
    assert counts["fista"] == max_iter
