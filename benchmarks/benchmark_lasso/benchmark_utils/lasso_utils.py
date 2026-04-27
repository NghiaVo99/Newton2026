import contextlib
import io
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lasso.newton_lasso import Algo_Newton_BT_Fista_new
from src.lasso.newton_lasso import Algo_Newton_BT_Ista
from src.lasso.newton_lasso import Algo_Newton_Fista_new
from src.lasso.newton_lasso import Algo_Newton_Ista
from src.lasso.newton_lasso import FISTA1
from src.lasso.newton_lasso import ISTA
from src.lasso.utils_lasso import cost_lasso
from src.lasso.utils_lasso import grad_f
from src.lasso.utils_lasso import lipschitz_exact
from src.lasso.utils_lasso import proxL1
from src.lasso.utils_lasso import sub_problem_of_lasso


DEFAULT_BT_BETA = 0.5
DEFAULT_NEWTON_STEP = 1.0
DEFAULT_NEWTON_TRIGGER_STEPS = 2
# Keep separate Newton activation defaults for ISTA and FISTA wrappers:
# the FISTA variant should mirror the original comparison script more closely,
# while the ISTA variant benefits from an earlier trigger in short Benchopt runs.
DEFAULT_ISTA_NEWTON_TOL = 1e-2
# On the (500, 600) simulated benchmark, 1e-3 delays the Newton phase too much.
DEFAULT_FISTA_NEWTON_TOL = 1e-2
NO_EARLY_STOP_TOL = -1.0
USE_GUROBI_NEWTON_SUBPROBLEM = False

_DENSE_NEWTON_CACHE = {}


def _dense_cache_key(A, b):
    A_arr = np.asarray(A)
    b_arr = np.asarray(b).reshape(-1)
    return (id(A), id(b), A_arr.shape, b_arr.shape, A_arr.dtype.str, b_arr.dtype.str)


def _get_dense_newton_cache(A, b):
    key = _dense_cache_key(A, b)
    cache = _DENSE_NEWTON_CACHE.get(key)
    if cache is not None:
        return cache

    A_arr = np.asarray(A, dtype=float)
    b_arr = np.asarray(b, dtype=float).reshape(-1)
    gram = np.asarray(A_arr.T @ A_arr, dtype=float)
    atb = np.asarray(A_arr.T @ b_arr, dtype=float)
    cache = {"gram": gram, "atb": atb}
    _DENSE_NEWTON_CACHE[key] = cache
    return cache


def soft_threshold(x, lam):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def objective_value(X, y, beta, lmbd):
    beta = np.asarray(beta, dtype=float).reshape(-1)
    diff = X @ beta - y
    return 0.5 * float(diff @ diff) + lmbd * float(np.abs(beta).sum())


def compute_step_size(X):
    return 1.0 / max(lipschitz_exact(X), 1e-12)


def dense_lasso_newton_subproblem(A, x, y, b, alpha):
    """Solve the reduced Newton subproblem with dense NumPy linear algebra."""
    cache = _get_dense_newton_cache(A, b)
    gram = cache["gram"]
    atb = cache["atb"]

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    kappa = np.where(np.abs(y) >= 0.999 * alpha)[0]
    d_full = np.zeros_like(y, dtype=float)
    if kappa.size == 0:
        return d_full

    Q = gram[np.ix_(kappa, kappa)]
    rhs_full = gram @ x - atb + y
    rhs = rhs_full[kappa]

    try:
        d_reduced = np.linalg.solve(Q, rhs)
    except np.linalg.LinAlgError:
        d_reduced = np.linalg.lstsq(Q, rhs, rcond=None)[0]

    d_full[kappa] = d_reduced
    return d_full


def lasso_newton_subproblem(A, x, y, b, alpha):
    """Fast cached dense subproblem by default, optional Gurobi fallback."""
    d = None
    if USE_GUROBI_NEWTON_SUBPROBLEM:
        try:
            d = sub_problem_of_lasso(A, x, y, b, alpha)
        except Exception:
            d = None

    if d is None:
        d = dense_lasso_newton_subproblem(A, x, y, b, alpha)

    d = np.asarray(d, dtype=float)
    if not np.all(np.isfinite(d)):
        d = dense_lasso_newton_subproblem(A, x, y, b, alpha)
        d = np.asarray(d, dtype=float)
    if not np.all(np.isfinite(d)):
        return np.zeros_like(np.asarray(y, dtype=float))
    return d


def _run_without_stdout(func, *args, **kwargs):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        return func(*args, **kwargs)


def run_ista(X, y, lmbd, n_iter):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    x0 = np.zeros(X.shape[1], dtype=float)
    step_size = compute_step_size(X)
    _, beta, _, _, _ = _run_without_stdout(
        ISTA,
        X,
        y,
        x0,
        lmbd,
        int(n_iter),
        step_size,
        NO_EARLY_STOP_TOL,
        cost_lasso,
        proxL1,
        0,
    )
    return np.asarray(beta, dtype=float)


def run_fista(X, y, lmbd, n_iter):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    x0 = np.zeros(X.shape[1], dtype=float)
    step_size = compute_step_size(X)
    _, beta, _, _, _ = _run_without_stdout(
        FISTA1,
        X,
        y,
        x0,
        lmbd,
        int(n_iter),
        step_size,
        NO_EARLY_STOP_TOL,
        cost_lasso,
        proxL1,
        0,
    )
    return np.asarray(beta, dtype=float)


def run_fista_adaptive(X, y, lmbd, n_iter):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n_features = X.shape[1]
    step_size = compute_step_size(X)
    beta = np.zeros(n_features, dtype=float)
    beta_prev = beta.copy()
    z = beta.copy()
    t = 1.0
    best_value = objective_value(X, y, beta, lmbd)

    for _ in range(int(n_iter)):
        grad = grad_f(X, z, y)
        beta_next = proxL1(z - step_size * grad, lmbd * step_size)
        value = objective_value(X, y, beta_next, lmbd)

        if value > best_value:
            z = beta.copy()
            t = 1.0
            grad = grad_f(X, z, y)
            beta_next = proxL1(z - step_size * grad, lmbd * step_size)
            value = objective_value(X, y, beta_next, lmbd)

        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = beta_next + ((t - 1.0) / t_next) * (beta_next - beta_prev)
        beta_prev = beta.copy()
        beta = beta_next
        t = t_next
        best_value = min(best_value, value)

    return beta


def run_coordinate_descent(X, y, lmbd, n_iter):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    _, n_features = X.shape
    beta = np.zeros(n_features, dtype=float)
    residual = y.copy()
    col_norm_sq = np.sum(X * X, axis=0)
    col_norm_sq[col_norm_sq == 0.0] = 1.0

    for _ in range(int(n_iter)):
        for j in range(n_features):
            old_beta = beta[j]
            if old_beta != 0.0:
                residual += X[:, j] * old_beta

            rho = float(X[:, j] @ residual)
            beta[j] = soft_threshold(rho, lmbd) / col_norm_sq[j]
            residual -= X[:, j] * beta[j]

    return beta


def run_newton_bt_ista(X, y, lmbd, n_iter):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    x0 = np.zeros(X.shape[1], dtype=float)
    _, beta, _, _, _ = _run_without_stdout(
        Algo_Newton_BT_Ista,
        X,
        y,
        x0,
        lmbd,
        int(n_iter),
        DEFAULT_BT_BETA,
        DEFAULT_NEWTON_STEP,
        NO_EARLY_STOP_TOL,
        cost_lasso,
        proxL1,
        lasso_newton_subproblem,
        DEFAULT_ISTA_NEWTON_TOL,
        0,
        DEFAULT_NEWTON_TRIGGER_STEPS,
    )
    return np.asarray(beta, dtype=float)


def run_newton_ista(X, y, lmbd, n_iter):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    x0 = np.zeros(X.shape[1], dtype=float)
    step_size = compute_step_size(X)
    _, beta, _, _, _ = _run_without_stdout(
        Algo_Newton_Ista,
        X,
        y,
        x0,
        lmbd,
        int(n_iter),
        step_size,
        DEFAULT_BT_BETA,
        DEFAULT_NEWTON_STEP,
        NO_EARLY_STOP_TOL,
        cost_lasso,
        proxL1,
        lasso_newton_subproblem,
        DEFAULT_ISTA_NEWTON_TOL,
        0,
        DEFAULT_NEWTON_TRIGGER_STEPS,
    )
    return np.asarray(beta, dtype=float)


def run_newton_bt_fista(X, y, lmbd, n_iter):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    x0 = np.zeros(X.shape[1], dtype=float)
    _, beta, _, _, _ = _run_without_stdout(
        Algo_Newton_BT_Fista_new,
        X,
        y,
        x0,
        lmbd,
        int(n_iter),
        DEFAULT_BT_BETA,
        DEFAULT_NEWTON_STEP,
        NO_EARLY_STOP_TOL,
        cost_lasso,
        proxL1,
        lasso_newton_subproblem,
        DEFAULT_FISTA_NEWTON_TOL,
        0,
        DEFAULT_NEWTON_TRIGGER_STEPS,
    )
    return np.asarray(beta, dtype=float)


def run_newton_fista(X, y, lmbd, n_iter):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    x0 = np.zeros(X.shape[1], dtype=float)
    step_size = compute_step_size(X)
    _, beta, _, _, _ = _run_without_stdout(
        Algo_Newton_Fista_new,
        X,
        y,
        x0,
        lmbd,
        int(n_iter),
        step_size,
        DEFAULT_BT_BETA,
        DEFAULT_NEWTON_STEP,
        NO_EARLY_STOP_TOL,
        cost_lasso,
        proxL1,
        lasso_newton_subproblem,
        DEFAULT_FISTA_NEWTON_TOL,
        0,
        DEFAULT_NEWTON_TRIGGER_STEPS,
    )
    return np.asarray(beta, dtype=float)
