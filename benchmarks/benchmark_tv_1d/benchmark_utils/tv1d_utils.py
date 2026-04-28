import contextlib
import io
import pathlib
import sys

import numpy as np
from scipy import sparse

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.Gen_lasso.Gen_Lasso_algo import Algo_Newton_Fista_new
from src.Gen_lasso.Gen_Lasso_algo import Algo_Newton_Ista
from src.Gen_lasso.Gen_Lasso_utils import cost_generalized_lasso
from src.Gen_lasso.Gen_Lasso_utils import inactive_tv_constraint_indices
from src.Gen_lasso.Gen_Lasso_utils import make_forward_diff
from src.Gen_lasso.Gen_Lasso_utils import sub_problem_gen_lasso

DEFAULT_BT_BETA = 0.5
DEFAULT_NEWTON_STEP = 1.0
DEFAULT_NEWTON_TRIGGER_STEPS = 3
DEFAULT_ISTA_NEWTON_TOL = 1e-2
DEFAULT_FISTA_NEWTON_TOL = 1e-2
DEFAULT_NEWTON_REJECT_COOLDOWN = 8
DEFAULT_MAX_NEWTON_BACKTRACKS = 25
NO_EARLY_STOP_TOL = -1.0
MAX_DENSE_ELEMENTS = 5_000_000
TV_PROX_NITER = 100
TV_PROX_RTOL = 1e-10


def can_materialize_design(A, max_elements=MAX_DENSE_ELEMENTS):
    return int(A.shape[0]) * int(A.shape[1]) <= int(max_elements)


def materialize_design(A, max_elements=MAX_DENSE_ELEMENTS):
    if not can_materialize_design(A, max_elements=max_elements):
        raise ValueError(
            f"Dense materialization of A with shape {A.shape} exceeds "
            f"max_elements={max_elements}."
        )
    if sparse.issparse(A):
        A_dense = A.toarray()
    elif isinstance(A, np.ndarray):
        A_dense = A
    else:
        eye = np.eye(A.shape[1])
        A_dense = A @ eye

    A_dense = np.asarray(A_dense, dtype=float)
    if A_dense.shape != tuple(A.shape):
        raise ValueError(
            f"Materialized A has shape {A_dense.shape}, expected {tuple(A.shape)}."
        )
    if not np.all(np.isfinite(A_dense)):
        raise ValueError("Materialized A contains non-finite values.")
    return A_dense


def compute_step_size(A):
    A = np.asarray(A, dtype=float)
    L = np.linalg.norm(A, ord=2) ** 2
    return 1.0 / max(float(L), 1e-12)


def has_tv_prox_available():
    return True


def condat_tv1d(y, lam):
    """Exact prox of lam * sum_i |x[i+1] - x[i]| by Condat's 1D TV method."""
    y = np.asarray(y, dtype=float).reshape(-1)
    n = y.size
    lam = float(lam)
    if n == 0 or lam <= 0.0:
        return y.copy()

    x = np.empty(n, dtype=float)
    k = k0 = kplus = kminus = 0
    vmin = y[0] - lam
    vmax = y[0] + lam
    umin = lam
    umax = -lam

    while True:
        if k == n - 1:
            if umin < 0.0:
                while k0 <= kminus:
                    x[k0] = vmin
                    k0 += 1
                k = k0
                if k0 >= n:
                    break
                kplus = kminus = k0
                vmin = y[k0]
                vmax = y[k0] + 2.0 * lam
                umin = lam
                umax = -lam
            elif umax > 0.0:
                while k0 <= kplus:
                    x[k0] = vmax
                    k0 += 1
                k = k0
                if k0 >= n:
                    break
                kplus = kminus = k0
                vmin = y[k0] - 2.0 * lam
                vmax = y[k0]
                umin = lam
                umax = -lam
            else:
                vmin += umin / (k - k0 + 1)
                while k0 <= k:
                    x[k0] = vmin
                    k0 += 1
                break
        else:
            k += 1
            val = y[k]
            umin += val - vmin
            umax += val - vmax
            if umin < -lam:
                while k0 <= kminus:
                    x[k0] = vmin
                    k0 += 1
                k = k0
                if k0 >= n:
                    break
                kplus = kminus = k0
                vmin = y[k0]
                vmax = y[k0] + 2.0 * lam
                umin = lam
                umax = -lam
            elif umax > lam:
                while k0 <= kplus:
                    x[k0] = vmax
                    k0 += 1
                k = k0
                if k0 >= n:
                    break
                kplus = kminus = k0
                vmin = y[k0] - 2.0 * lam
                vmax = y[k0]
                umin = lam
                umax = -lam
            else:
                if umin >= lam:
                    vmin += (umin - lam) / (k - k0 + 1)
                    umin = lam
                    kminus = k
                if umax <= -lam:
                    vmax += (umax + lam) / (k - k0 + 1)
                    umax = -lam
                    kplus = k

    return x


def make_tv_prox(n_features):
    try:
        import prox_tv as ptv

        return lambda x, tau: ptv.tv1_1d(np.asarray(x, dtype=float), float(tau),
                                         method="condat")
    except ImportError:
        return lambda x, tau: condat_tv1d(np.asarray(x, dtype=float), float(tau))


def dense_tv_newton_subproblem(A, yk, zk, b, alpha, *, gram=None):
    A = np.asarray(A, dtype=float)
    yk = np.asarray(yk, dtype=float).reshape(-1)
    zk = np.asarray(zk, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    n_features = A.shape[1]
    if yk.shape[0] != n_features or zk.shape[0] != n_features:
        raise ValueError("yk and zk must have length A.shape[1].")

    if gram is None:
        gram = A.T @ A
    c = A.T @ (A @ yk - b) + zk

    inactive = set(inactive_tv_constraint_indices(zk, alpha, n_features).tolist())
    groups = []
    start = 0
    for edge in range(n_features - 1):
        if edge not in inactive:
            groups.append((start, edge))
            start = edge + 1
    groups.append((start, n_features - 1))

    Q = np.zeros((n_features, len(groups)), dtype=float)
    for col, (lo, hi) in enumerate(groups):
        length = hi - lo + 1
        Q[lo:hi + 1, col] = 1.0 / np.sqrt(length)

    reduced_hessian = Q.T @ gram @ Q
    reduced_rhs = Q.T @ c
    try:
        coeff = np.linalg.solve(reduced_hessian, reduced_rhs)
    except np.linalg.LinAlgError:
        coeff = np.linalg.lstsq(reduced_hessian, reduced_rhs, rcond=None)[0]
    d = Q @ coeff
    if not np.all(np.isfinite(d)):
        raise FloatingPointError("TV Newton subproblem produced non-finite values.")
    return np.asarray(d, dtype=float)


def make_cached_tv_subproblem_solver(A, *, use_gurobi_fallback=False):
    A = np.asarray(A, dtype=float)
    gram = A.T @ A

    def _solver(A_arg, yk, zk, b, alpha):
        try:
            return dense_tv_newton_subproblem(
                A, yk, zk, b, alpha, gram=gram
            )
        except Exception:
            if not use_gurobi_fallback:
                raise
            return sub_problem_gen_lasso(A, yk, zk, b, alpha, silent=True)

    return _solver


def _run_without_stdout(func, *args, **kwargs):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        return func(*args, **kwargs)


def _initial_point(c, n_features):
    return float(c) * np.ones(int(n_features), dtype=float)


def run_newton_ista(
    A,
    y,
    reg,
    c,
    n_iter,
    *,
    step_size=None,
    prox=None,
    subproblem_solver=None,
):
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    x0 = _initial_point(c, A.shape[1])
    D = make_forward_diff(A.shape[1])
    if step_size is None:
        step_size = compute_step_size(A)
    if prox is None:
        prox = make_tv_prox(A.shape[1])
    if subproblem_solver is None:
        subproblem_solver = make_cached_tv_subproblem_solver(A)

    _, u, _, _, _ = _run_without_stdout(
        Algo_Newton_Ista,
        A,
        D,
        y,
        x0,
        float(reg),
        int(n_iter),
        float(step_size),
        DEFAULT_BT_BETA,
        DEFAULT_NEWTON_STEP,
        NO_EARLY_STOP_TOL,
        cost_generalized_lasso,
        prox,
        subproblem_solver,
        DEFAULT_ISTA_NEWTON_TOL,
        0,
        DEFAULT_NEWTON_TRIGGER_STEPS,
        DEFAULT_NEWTON_REJECT_COOLDOWN,
        DEFAULT_MAX_NEWTON_BACKTRACKS,
        False,
    )
    return np.asarray(u, dtype=float)


def run_newton_fista(
    A,
    y,
    reg,
    c,
    n_iter,
    *,
    step_size=None,
    prox=None,
    subproblem_solver=None,
):
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    x0 = _initial_point(c, A.shape[1])
    D = make_forward_diff(A.shape[1])
    if step_size is None:
        step_size = compute_step_size(A)
    if prox is None:
        prox = make_tv_prox(A.shape[1])
    if subproblem_solver is None:
        subproblem_solver = make_cached_tv_subproblem_solver(A)

    _, u, _, _, _ = _run_without_stdout(
        Algo_Newton_Fista_new,
        A,
        D,
        y,
        x0,
        float(reg),
        int(n_iter),
        float(step_size),
        DEFAULT_BT_BETA,
        DEFAULT_NEWTON_STEP,
        NO_EARLY_STOP_TOL,
        cost_generalized_lasso,
        prox,
        subproblem_solver,
        DEFAULT_FISTA_NEWTON_TOL,
        0,
        DEFAULT_NEWTON_TRIGGER_STEPS,
        DEFAULT_NEWTON_REJECT_COOLDOWN,
        DEFAULT_MAX_NEWTON_BACKTRACKS,
        False,
    )
    return np.asarray(u, dtype=float)
