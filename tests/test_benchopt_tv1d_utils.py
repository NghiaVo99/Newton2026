import importlib
import pathlib
import sys
import types

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator

BENCHMARK_DIR = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_tv_1d"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_utils.tv1d_utils import dense_tv_newton_subproblem
from benchmark_utils.tv1d_utils import condat_tv1d
from benchmark_utils.tv1d_utils import compute_step_size
from benchmark_utils.tv1d_utils import has_tv_prox_available
from benchmark_utils.tv1d_utils import make_tv_prox
from benchmark_utils.tv1d_utils import materialize_design
from benchmark_utils.tv1d_utils import run_newton_fista
from benchmark_utils.tv1d_utils import run_newton_ista
from benchmark_utils.tv1d_utils import TV_PROX_NITER
from benchmark_utils.tv1d_utils import TV_PROX_RTOL


def test_newton_solvers_skip_huber_loss(monkeypatch):
    class DummySafeImportContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    dummy_benchopt = types.SimpleNamespace(
        BaseSolver=object,
        safe_import_context=lambda: DummySafeImportContext(),
    )
    monkeypatch.setitem(sys.modules, "benchopt", dummy_benchopt)
    sys.modules.pop("solvers.newton_ista", None)

    module = importlib.import_module("solvers.newton_ista")
    solver = module.Solver()

    should_skip, reason = solver.skip(
        np.eye(4),
        reg=0.1,
        y=np.ones(4),
        c=0.0,
        delta=0.9,
        data_fit="huber",
    )

    assert should_skip
    assert "quadratic" in reason


def test_materialize_design_matches_linear_operator():
    dense = np.arange(12, dtype=float).reshape(3, 4)
    op = LinearOperator(
        shape=dense.shape,
        matvec=lambda x: dense @ x,
        matmat=lambda X: dense @ X,
        rmatvec=lambda x: dense.T @ x,
        rmatmat=lambda X: dense.T @ X,
        dtype=np.float64,
    )

    np.testing.assert_allclose(materialize_design(op), dense)


def test_dense_tv_newton_subproblem_matches_gurobi_when_available():
    from src.Gen_lasso.Gen_Lasso_utils import sub_problem_gen_lasso

    rng = np.random.RandomState(0)
    A = rng.randn(7, 4)
    b = rng.randn(7)
    yk = rng.randn(4)
    zk = np.array([0.1, 0.2, -0.1, 0.0])
    alpha = 0.8

    actual = dense_tv_newton_subproblem(A, yk, zk, b, alpha)
    try:
        expected = sub_problem_gen_lasso(A, yk, zk, b, alpha, silent=True)
    except Exception as exc:
        pytest.skip(f"Gurobi reference unavailable: {exc}")

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_newton_tv_runners_return_finite_vectors():
    if not has_tv_prox_available():
        pytest.skip("prox_tv or pyproximal is required for TV prox")

    rng = np.random.RandomState(1)
    A = rng.randn(8, 5)
    y = rng.randn(8)
    reg = 0.05
    c = 0.0

    u_ista = run_newton_ista(A, y, reg, c, n_iter=2)
    u_fista = run_newton_fista(A, y, reg, c, n_iter=2)

    assert u_ista.shape == (5,)
    assert u_fista.shape == (5,)
    assert np.all(np.isfinite(u_ista))
    assert np.all(np.isfinite(u_fista))


def test_pyproximal_tv_prox_settings_are_tight():
    assert TV_PROX_NITER >= 200
    assert TV_PROX_RTOL <= 1e-8


def test_local_condat_tv_prox_basic_properties():
    y = np.array([3.0, -1.0, 2.0, 4.0])

    np.testing.assert_allclose(condat_tv1d(y, 0.0), y)
    np.testing.assert_allclose(condat_tv1d(y, 1e6), np.full_like(y, y.mean()))

    x = condat_tv1d(y, 0.7)
    assert x.shape == y.shape
    assert np.all(np.isfinite(x))


def test_tv_prox_keeps_ista_objective_monotone():
    rng = np.random.RandomState(42)
    A = rng.randn(30, 20)
    y = rng.randn(30)
    reg = 0.2
    step_size = compute_step_size(A)
    prox = make_tv_prox(A.shape[1])
    x = np.zeros(A.shape[1])
    values = []

    for _ in range(25):
        residual = A @ x - y
        values.append(0.5 * residual @ residual + reg * np.abs(np.diff(x)).sum())
        grad = A.T @ residual
        x = prox(x - step_size * grad, reg * step_size)

    assert np.all(np.diff(values) <= 1e-8)
