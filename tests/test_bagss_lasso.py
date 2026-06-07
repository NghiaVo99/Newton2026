import numpy as np
import pytest
from scipy import sparse

from src.lasso.BaGSS import BasGSSLasso
from src.lasso.utils_lasso import cost_lasso, solve_lasso_cvxpy


def test_bagss_dense_lasso_matches_cvxpy_solution():
    pytest.importorskip("cvxpy")
    rng = np.random.default_rng(0)
    m, n = 18, 30
    A = rng.normal(size=(m, n)) / np.sqrt(m)
    x_true = np.zeros(n)
    x_true[rng.choice(n, 5, replace=False)] = rng.normal(size=5)
    b = A @ x_true + 0.01 * rng.normal(size=m)
    lam = 0.05 * np.linalg.norm(A.T @ b, np.inf)

    x_ref = solve_lasso_cvxpy(A, b, lam)
    solver = BasGSSLasso(
        A, b, lam, lambda0=1e-2, lambda_bar=1.0,
        alpha=0.25, beta=0.5, sigma=0.5, rho_bar=10.0,
        eps=1e-8, max_iters=200, cg_tol=1e-10, cg_maxit=1000,
    )

    result = solver.solve(np.zeros(n), approx_solution=x_ref)

    assert cost_lasso(A, result["z"], b, lam) == pytest.approx(
        cost_lasso(A, x_ref, b, lam), abs=1e-7
    )
    assert np.linalg.norm(result["z"] - x_ref) < 1e-4
    assert result["history"]["r"][-1] < 1e-6


def test_bagss_sparse_csr_runs_without_sparse_multiply_bug():
    rng = np.random.default_rng(1)
    m, n = 20, 40
    A = sparse.random(
        m, n, density=0.2, format="csr", random_state=1,
        data_rvs=lambda size: rng.normal(size=size),
    )
    x_true = np.zeros(n)
    x_true[rng.choice(n, 4, replace=False)] = rng.normal(size=4)
    b = np.asarray(A @ x_true).reshape(-1) + 0.01 * rng.normal(size=m)
    lam = 0.05 * np.linalg.norm(np.asarray(A.T @ b).reshape(-1), np.inf)

    solver = BasGSSLasso(
        A, b, lam, lambda0=1e-2, lambda_bar=0.5,
        eps=1e-7, max_iters=50, rho_bar=5.0,
        cg_tol=1e-8, cg_maxit=300,
    )

    result = solver.solve(np.zeros(n))

    assert np.isfinite(result["history"]["phi_z"][-1])
    assert np.isfinite(result["history"]["r"][-1])
    assert result["history"]["r"][-1] <= result["history"]["r"][0]


def test_step7_lambda_growth_keeps_logged_fbe_consistent():
    A = np.zeros((2, 3))
    b = np.zeros(2)
    x0 = np.array([1.0, -2.0, 0.5])
    solver = BasGSSLasso(
        A, b, lambda_reg=0.1, lambda0=0.01, lambda_bar=0.08,
        search_dir="prox", eps=0.0, max_iters=1,
    )

    result = solver.solve(x0)
    hist = result["history"]
    fbe_recomputed, eta_recomputed = solver.fbe(
        result["x"], hist["lam"][-1], z=result["z"])

    assert hist["lam"][-1] > hist["lam"][0]
    assert hist["fbe"][-1] == pytest.approx(fbe_recomputed)
    assert hist["eta"][-1] == pytest.approx(eta_recomputed)


def test_lasso_boundary_zero_coordinates_are_inactive():
    lam_reg = 0.5
    A = np.eye(2)
    b = np.array([lam_reg, -lam_reg])
    solver = BasGSSLasso(A, b, lambda_reg=lam_reg)
    x = np.zeros(2)
    z = solver.T(x, lam=1.0)

    _, _, z_g_star, active, info = solver._gssn_direction_l1(
        x, z, lam=1.0, return_info=True)

    assert np.allclose(z, 0.0)
    assert np.allclose(np.abs(z_g_star), lam_reg)
    assert not np.any(active)
    assert info["active_size"] == 0


def test_prox_search_direction_is_forward_backward_fallback():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(10, 15)) / np.sqrt(10)
    b = rng.normal(size=10)
    lam = 0.1 * np.linalg.norm(A.T @ b, np.inf)
    solver = BasGSSLasso(
        A, b, lam, search_dir="prox", lambda0=1e-2,
        lambda_bar=0.5, eps=1e-8, max_iters=20,
    )

    result = solver.solve(np.zeros(15))

    assert np.isfinite(result["history"]["r"][-1])
    assert all(size == 0 for size in result["history"]["active_size"])
    assert all(iters == 0 for iters in result["history"]["cg_iters"])
