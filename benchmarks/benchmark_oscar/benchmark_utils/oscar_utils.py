import contextlib
import io
import pathlib
import sys

import numpy as np
from scipy import sparse
try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is in the benchopt env.
    njit = None

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.OSCAR.OSCAR_algo import Algo_Newton_BT_Fista_new
from src.OSCAR.OSCAR_algo import Algo_Newton_BT_Ista
from src.OSCAR.OSCAR_algo import Algo_Newton_Fista_new
from src.OSCAR.OSCAR_algo import Algo_Newton_Ista
from src.OSCAR.OSCAR_ultils_v1 import cost_oscar
from src.OSCAR.OSCAR_ultils_v1 import grad_f
from src.OSCAR.OSCAR_ultils_v1 import prox_oscar
from src.OSCAR.OSCAR_ultils_v1 import sub_problem_oscar

DEFAULT_BT_BETA = 0.5
DEFAULT_NEWTON_STEP = 1.0
DEFAULT_NEWTON_TRIGGER_STEPS = 2
DEFAULT_ISTA_NEWTON_TOL = 3e-2
DEFAULT_FISTA_NEWTON_TOL = 1e-2
DEFAULT_NEWTON_REJECT_STREAK_TRIGGER = 2
DEFAULT_NEWTON_REJECT_COOLDOWN = 8
DEFAULT_MAX_NEWTON_BACKTRACKS = 25
DEFAULT_COEF_CLEANUP_TOLS = (1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5)
NO_EARLY_STOP_TOL = -1.0


if njit is not None:

    @njit(cache=True)
    def _pav_nonincreasing_nonnegative_numba(u):
        n = u.size
        v = np.empty_like(u)
        starts = np.zeros(n, dtype=np.int64)
        ends = np.zeros(n, dtype=np.int64)
        means = np.zeros(n, dtype=np.float64)

        nb = 0
        for i in range(n):
            starts[nb] = i
            ends[nb] = i
            means[nb] = u[i] if u[i] > 0.0 else 0.0
            nb += 1

            while nb >= 2 and means[nb - 2] < means[nb - 1]:
                len1 = ends[nb - 2] - starts[nb - 2] + 1
                len2 = ends[nb - 1] - starts[nb - 1] + 1
                means[nb - 2] = (
                    len1 * means[nb - 2] + len2 * means[nb - 1]
                ) / (len1 + len2)
                ends[nb - 2] = ends[nb - 1]
                nb -= 1

        for block_id in range(nb):
            for j in range(starts[block_id], ends[block_id] + 1):
                v[j] = means[block_id]
        return v


    @njit(cache=True)
    def _prox_oscar_numba(y, tau, w1, w2, positive):
        n = y.size
        if n == 0:
            return y.copy()

        abs_y = np.empty(n, dtype=np.float64)
        signs = np.ones(n, dtype=np.float64)
        for i in range(n):
            if positive:
                abs_y[i] = y[i] if y[i] > 0.0 else 0.0
            else:
                if y[i] > 0.0:
                    signs[i] = 1.0
                    abs_y[i] = y[i]
                elif y[i] < 0.0:
                    signs[i] = -1.0
                    abs_y[i] = -y[i]
                else:
                    signs[i] = 0.0
                    abs_y[i] = 0.0

        order = np.argsort(-abs_y)
        z = abs_y[order]
        u = np.empty(n, dtype=np.float64)
        for i in range(n):
            u[i] = z[i] - tau * (w1 + w2 * (n - 1 - i))

        v = _pav_nonincreasing_nonnegative_numba(u)
        x = np.empty(n, dtype=np.float64)
        for i in range(n):
            value = v[i]
            if not positive:
                value *= signs[order[i]]
            x[order[i]] = value
        return x

else:
    _prox_oscar_numba = None


def alphas_to_w1_w2(alphas, *, rtol=1e-5, atol=1e-10):
    alphas = np.asarray(alphas, dtype=float).reshape(-1)
    if alphas.size == 0:
        raise ValueError("alphas must be non-empty")
    if not np.all(np.isfinite(alphas)):
        raise ValueError("alphas contains non-finite values")
    if np.any(alphas < -atol):
        raise ValueError("alphas must be nonnegative")

    w1 = float(alphas[-1])
    if alphas.size == 1:
        return w1, 0.0

    diffs = alphas[:-1] - alphas[1:]
    w2 = float(np.mean(diffs))

    if not np.allclose(diffs, w2, rtol=rtol, atol=atol):
        raise ValueError(
            "alphas are not OSCAR-equivalent (non-constant adjacent differences)."
        )

    n = alphas.size
    expected = w1 + w2 * (n - 1 - np.arange(n, dtype=float))
    if not np.allclose(alphas, expected, rtol=rtol, atol=atol):
        raise ValueError("alphas do not match OSCAR structure from (w1, w2).")

    return w1, w2


def compute_step_size(X):
    X = np.asarray(X, dtype=float)
    if sparse.issparse(X):
        L = sparse.linalg.svds(X, k=1)[1][0] ** 2
    else:
        L = np.linalg.norm(X, ord=2) ** 2
    return 1.0 / max(float(L), 1e-12)


def _run_without_stdout(func, *args, **kwargs):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        return func(*args, **kwargs)


def fast_prox_oscar(y, tau, w1, w2, positive=False):
    if _prox_oscar_numba is None:
        return prox_oscar(y, tau, w1, w2, positive=positive)
    y = np.asarray(y, dtype=float)
    return _prox_oscar_numba(y, float(tau), float(w1), float(w2), bool(positive))


def warm_up_fast_prox(n_features):
    fast_prox_oscar(np.zeros(n_features, dtype=float), 1.0, 0.1, 0.01)


def _pack_beta(beta, fit_intercept=False):
    beta = np.asarray(beta, dtype=float).reshape(-1)
    if fit_intercept:
        raise ValueError("Custom OSCAR Newton wrappers do not handle fit_intercept=True.")
    return np.r_[0.0, beta]


def _cleanup_near_zero_coefficients(X, y, beta, w1, w2):
    """Remove tiny numerical coefficients when this lowers the OSCAR objective."""
    beta = np.asarray(beta, dtype=float).reshape(-1)
    best_beta = beta
    best_cost = cost_oscar(X, beta, y, w1, w2)

    for tol in DEFAULT_COEF_CLEANUP_TOLS:
        cleaned = beta.copy()
        cleaned[np.abs(cleaned) < tol] = 0.0
        if np.array_equal(cleaned, beta):
            continue

        cleaned_cost = cost_oscar(X, cleaned, y, w1, w2)
        if cleaned_cost < best_cost:
            best_beta = cleaned
            best_cost = cleaned_cost

    return best_beta


def _make_cached_subproblem_solver(X):
    cache = {"H": None}

    def _solver(A, yk, zk, b, w1, w2):
        d = fast_oscar_newton_subproblem(A, yk, zk, b, w1, w2)
        if d is not None:
            return d
        if cache["H"] is None:
            cache["H"] = A.T @ A
        return sub_problem_oscar(A, yk, zk, b, w1, w2, H=cache["H"], silent=True)

    return _solver


def _fast_build_Q_from_oscar(n, z, w1, w2, *, atol=1e-9, rtol=1e-7):
    z = np.asarray(z, dtype=float).reshape(-1)
    if z.size != n:
        raise ValueError("n must equal len(z).")

    lambdas = w1 + w2 * np.arange(n - 1, -1, -1, dtype=float)
    active_pos = np.flatnonzero(
        np.isclose(np.cumsum(np.sort(np.abs(z))[::-1]), np.cumsum(lambdas),
                   rtol=rtol, atol=atol)
    )
    if active_pos.size == 0:
        return np.zeros((n, 0), dtype=float)

    perm = np.argsort(-np.abs(z), kind="mergesort")
    z_sorted = z[perm]
    r = np.abs(z_sorted)
    sigma = np.sign(z_sorted)
    sigma[sigma == 0] = 1.0

    adjacent_tie = np.isclose(r[1:], r[:-1], rtol=rtol, atol=atol)
    starts = np.r_[0, np.flatnonzero(~adjacent_tie) + 1]
    ends = np.r_[starts[1:] - 1, n - 1]
    active_block_ids = np.searchsorted(starts, active_pos, side="right") - 1
    active_block_ids = np.unique(active_block_ids)

    n_diff_generators = int(np.sum(ends[active_block_ids] - starts[active_block_ids]))
    G = np.zeros((n, active_pos.size + n_diff_generators), dtype=float)

    col = 0
    for pos in active_pos:
        idx = perm[: pos + 1]
        G[idx, col] = sigma[: pos + 1]
        col += 1

    for block_id in active_block_ids:
        start = starts[block_id]
        end = ends[block_id]
        anchor = perm[start]
        for pos in range(start + 1, end + 1):
            G[anchor, col] = -sigma[start]
            G[perm[pos], col] = sigma[pos]
            col += 1

    if G.size == 0:
        return np.zeros((n, 0), dtype=float)

    norms = np.linalg.norm(G, axis=0)
    G = G[:, norms > 1e-14 * np.sqrt(n)]
    if G.size == 0:
        return np.zeros((n, 0), dtype=float)

    Q, _ = np.linalg.qr(G, mode="reduced")
    return Q


def fast_oscar_newton_subproblem(A, yk, zk, b, w1, w2):
    """Solve the OSCAR Newton subproblem by reduced dense linear algebra."""
    zk = np.asarray(zk, dtype=float).reshape(-1)
    n_features = zk.shape[0]
    Q = _fast_build_Q_from_oscar(n_features, zk, w1, w2)
    if Q.shape[1] == 0:
        return np.zeros(n_features, dtype=float)

    c = grad_f(A, yk, b) + zk
    AQ = A @ Q
    reduced_hessian = AQ.T @ AQ
    reduced_rhs = Q.T @ c

    try:
        alpha = np.linalg.solve(reduced_hessian, reduced_rhs)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(reduced_hessian, reduced_rhs, rcond=None)[0]

    residual = reduced_hessian @ alpha - reduced_rhs
    scale = 1.0 + np.linalg.norm(reduced_rhs)
    if np.linalg.norm(residual) > 1e-7 * scale:
        return None

    d = Q @ alpha
    if not np.all(np.isfinite(d)):
        return None
    return np.asarray(d, dtype=float)


def _scaled_oscar_weights(X, alphas):
    """Convert normalized BenchOpt SLOPE weights to src/OSCAR weights."""
    w1, w2 = alphas_to_w1_w2(alphas)
    n_samples = X.shape[0]
    return n_samples * w1, n_samples * w2


def run_newton_ista(
    X,
    y,
    alphas,
    n_iter,
    fit_intercept=False,
    subproblem_solver=None,
    step_size=None,
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    w1, w2 = _scaled_oscar_weights(X, alphas)
    x0 = np.zeros(X.shape[1], dtype=float)
    if step_size is None:
        step_size = compute_step_size(X)
    if subproblem_solver is None:
        subproblem_solver = _make_cached_subproblem_solver(X)

    _, beta, _, _, _ = _run_without_stdout(
        Algo_Newton_Ista,
        X,
        y,
        x0,
        w1,
        w2,
        int(n_iter),
        step_size,
        DEFAULT_BT_BETA,
        DEFAULT_NEWTON_STEP,
        NO_EARLY_STOP_TOL,
        cost_oscar,
        fast_prox_oscar,
        subproblem_solver,
        DEFAULT_ISTA_NEWTON_TOL,
        0,
        False,
        DEFAULT_NEWTON_TRIGGER_STEPS,
        DEFAULT_NEWTON_REJECT_STREAK_TRIGGER,
        DEFAULT_NEWTON_REJECT_COOLDOWN,
        DEFAULT_MAX_NEWTON_BACKTRACKS,
    )
    beta = _cleanup_near_zero_coefficients(X, y, beta, w1, w2)
    return _pack_beta(beta, fit_intercept=fit_intercept)


def run_newton_fista(
    X,
    y,
    alphas,
    n_iter,
    fit_intercept=False,
    subproblem_solver=None,
    step_size=None,
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    w1, w2 = _scaled_oscar_weights(X, alphas)
    x0 = np.zeros(X.shape[1], dtype=float)
    if step_size is None:
        step_size = compute_step_size(X)
    if subproblem_solver is None:
        subproblem_solver = _make_cached_subproblem_solver(X)

    _, beta, _, _, _ = _run_without_stdout(
        Algo_Newton_Fista_new,
        X,
        y,
        x0,
        w1,
        w2,
        int(n_iter),
        step_size,
        DEFAULT_BT_BETA,
        DEFAULT_NEWTON_STEP,
        NO_EARLY_STOP_TOL,
        cost_oscar,
        fast_prox_oscar,
        subproblem_solver,
        DEFAULT_FISTA_NEWTON_TOL,
        0,
        False,
        DEFAULT_NEWTON_TRIGGER_STEPS,
        DEFAULT_NEWTON_REJECT_STREAK_TRIGGER,
        DEFAULT_NEWTON_REJECT_COOLDOWN,
        DEFAULT_MAX_NEWTON_BACKTRACKS,
    )
    beta = _cleanup_near_zero_coefficients(X, y, beta, w1, w2)
    return _pack_beta(beta, fit_intercept=fit_intercept)


def run_newton_bt_ista(
    X, y, alphas, n_iter, fit_intercept=False, subproblem_solver=None
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    w1, w2 = _scaled_oscar_weights(X, alphas)
    x0 = np.zeros(X.shape[1], dtype=float)
    if subproblem_solver is None:
        subproblem_solver = _make_cached_subproblem_solver(X)

    _, beta, _, _, _ = _run_without_stdout(
        Algo_Newton_BT_Ista,
        X,
        y,
        x0,
        w1,
        w2,
        int(n_iter),
        DEFAULT_BT_BETA,
        DEFAULT_NEWTON_STEP,
        NO_EARLY_STOP_TOL,
        cost_oscar,
        fast_prox_oscar,
        subproblem_solver,
        DEFAULT_ISTA_NEWTON_TOL,
        0,
        False,
        DEFAULT_NEWTON_TRIGGER_STEPS,
        DEFAULT_NEWTON_REJECT_STREAK_TRIGGER,
        DEFAULT_NEWTON_REJECT_COOLDOWN,
        DEFAULT_MAX_NEWTON_BACKTRACKS,
    )
    beta = _cleanup_near_zero_coefficients(X, y, beta, w1, w2)
    return _pack_beta(beta, fit_intercept=fit_intercept)


def run_newton_bt_fista(
    X, y, alphas, n_iter, fit_intercept=False, subproblem_solver=None
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    w1, w2 = _scaled_oscar_weights(X, alphas)
    x0 = np.zeros(X.shape[1], dtype=float)
    step_size = compute_step_size(X)
    if subproblem_solver is None:
        subproblem_solver = _make_cached_subproblem_solver(X)

    _, beta, _, _, _ = _run_without_stdout(
        Algo_Newton_BT_Fista_new,
        X,
        y,
        x0,
        w1,
        w2,
        int(n_iter),
        step_size,
        DEFAULT_BT_BETA,
        DEFAULT_NEWTON_STEP,
        NO_EARLY_STOP_TOL,
        cost_oscar,
        fast_prox_oscar,
        subproblem_solver,
        DEFAULT_FISTA_NEWTON_TOL,
        0,
        False,
        DEFAULT_NEWTON_TRIGGER_STEPS,
        DEFAULT_NEWTON_REJECT_STREAK_TRIGGER,
        DEFAULT_NEWTON_REJECT_COOLDOWN,
        DEFAULT_MAX_NEWTON_BACKTRACKS,
    )
    beta = _cleanup_near_zero_coefficients(X, y, beta, w1, w2)
    return _pack_beta(beta, fit_intercept=fit_intercept)
