from re import S
import numpy as np
from scipy import sparse

try:
    import cvxpy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover - optional dependency
    gp = None
    GRB = None


def cost_lasso(A, x, b, alpha):
    x = np.asarray(x).reshape(-1)
    b = np.asarray(b).reshape(-1)
    r = A @ x - b                     # (m,)
    # Avoid the sqrt: r @ r is ||r||_2^2
    return 0.5 * (r @ r) + float(alpha) * np.abs(x).sum()

def grad_f(A, x, b):
    # x = np.asarray(x).reshape(-1)
    # b = np.asarray(b).reshape(-1)
    r = A @ x - b                     # (m,)
    return A.T @ r                    # (n,)

def hessian_f(A):
    # Exact Hessian of 0.5||Ax-b||^2 is A^T A.
    # Be careful: forming A.T @ A can be expensive/dense.
    return A.T @ A


def proxL1(x, lam):
    # lam should be a scalar (or a 1-D vector for weighted L1)
    x = np.asarray(x)
    lam = float(lam)
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


# def backtracking_linesearch(A, b, x, grad, prox, alpha, L_prev = 1, beta=1.5):
#     """ Backtracking line search for step size selection """
#     L = L_prev
#     while True:
#         x_new = prox(x - (1/L) * grad,alpha * 1/L)
#         lhs = 0.5 * np.linalg.norm(A @ x_new - b)**2
#         rhs = 0.5 * np.linalg.norm(A @ x - b)**2 - grad.T @ (x - x_new) + (0.5 * L) * np.linalg.norm(x - x_new)**2
#         if lhs <= rhs:
#             break
#         L = L*beta
#     return 1/L

def backtracking_linesearch(A, b, x, grad_x, prox, alpha, L_prev=1.0, eta=2.0,
                            max_tries=50, eps=1e-12):
    """
    Beck–Teboulle backtracking for FISTA on:
        f(x) = 0.5 ||A x - b||^2,  g(x) = alpha * ||x||_1
    Accepts when:
        F(x_new) <= f(x) + grad_x^T (x_new - x) + (L/2)||x_new - x||^2 + g(x_new)

    Returns
    -------
    step_size : float   # = 1 / L_used
    L_used    : float   # curvature used this iteration (warm-start next time)
    """
    import numpy as np

    L = max(float(L_prev), eps)

    # Cache f(x) = 0.5||Ax-b||^2
    r = A @ x - b
    f_x = 0.5 * r.T @ r

    for _ in range(max_tries):
        t = 1.0 / L
        x_new = prox(x - t * grad_x, t * alpha)

        d = x_new - x
        # Quadratic upper bound Q_L(x_new, x) + g(x_new)
        quad = f_x + (grad_x.T @ d) + 0.5 * L * (d.T @ d) + alpha * np.linalg.norm(x_new, 1)

        # Composite objective at trial
        r_new = A @ x_new - b
        F_trial = 0.5 * (r_new.T @ r_new) + alpha * np.linalg.norm(x_new, 1)

        if F_trial <= quad + 1e-12:
            return t  # step_size, L_used

        L *= eta  # increase curvature (shrink step)

    # Fallback if not accepted within max_tries
    return 1.0/L


def solve_lasso_cvxpy(A, b, alpha):
    """
    Solve LASSO: min_x (1/2) * ||A x - b||^2 + lam * ||x||_1 using CVXPY

    Parameters:
        A (np.ndarray): Matrix A of shape (n, p)
        b (np.ndarray): Vector b of shape (n,)
        lam (float): Regularization parameter

    Returns:
        x_opt (np.ndarray): Solution vector x (estimated coefficients)
    """
    if cp is None:
        raise ImportError("cvxpy is required to use solve_lasso_cvxpy.")

    n, p = A.shape
    x = cp.Variable(p)

    # Define the objective
    objective = cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + alpha * cp.norm1(x))
    problem = cp.Problem(objective)

    # Solve the problem
    problem.solve()

    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
        return x.value
    else:
        raise RuntimeError("CVXPY failed to solve the LASSO problem.")
    
def solve_lasso_gurobi(A, b, alpha, time_limit=60, verbose=False, warm_start=None):
    """
    Solve LASSO: min_x (1/2)*||A x - b||^2 + alpha * ||x||_1 using Gurobi.

    Parameters
    ----------
    A : np.ndarray or scipy.sparse matrix, shape (n, p)
    b : np.ndarray, shape (n,)
    alpha : float
    time_limit : float or None
        Optional time limit in seconds.
    verbose : bool
        If False, suppress solver output.
    warm_start : dict or None
        Optional warm start, e.g. {"x": np.array(shape=(p,))}.

    Returns
    -------
    x_opt : np.ndarray, shape (p,)
    """

    if gp is None or GRB is None:
        raise ImportError("gurobipy is required to use solve_lasso_gurobi.")

    # Ensure shapes
    if sparse.issparse(A):
        A = sparse.csr_matrix(A)
        n, p = A.shape
    else:
        A = np.asarray(A)
        n, p = A.shape
    b = np.asarray(b).reshape(-1)
    assert b.shape[0] == n, "b must have length n"

    # Model
    env = gp.Env(empty=not verbose)
    if not verbose:
        env.setParam("OutputFlag", 0)
    m = gp.Model("lasso", env=env)
    if time_limit is not None:
        m.setParam("TimeLimit", float(time_limit))

    # Decision variables
    x = m.addMVar(shape=p, lb=-GRB.INFINITY, name="x")   # free variable
    r = m.addMVar(shape=n, lb=-GRB.INFINITY, name="r")   # residuals r = A x - b
    t = m.addMVar(shape=p, lb=0.0, name="t")             # absolute value auxiliaries

    # Constraints: r = A x - b  -->  [-A  I] [x; r] = -b (constant RHS needed)
    # Build a single block sparse matrix for addMConstr
    if sparse.issparse(A):
        block = sparse.hstack([-A, sparse.identity(n, format="csr")], format="csr")
    else:
        block = np.hstack([-A, np.eye(n)])

    z = m.addMVar(shape=p + n, lb=-GRB.INFINITY, name="z")  # z = [x; r]
    # Fix z's first p entries to x, last n to r via linear equality constraints:
    # We do this with two sets of simple identities: z[:p] = x, z[p:] = r
    # (Gurobi doesn't have direct MVar concatenation, so tie them with constraints.)
    m.addConstr(z[:p] == x, name="link_x")
    m.addConstr(z[p:] == r, name="link_r")

    # Now the single vectorized linear system
    m.addMConstr(A=block, x=z, sense="=", b=-b, name="residual_def")

    # |x| modeling: -t <= x <= t  <=>  t >= x and t >= -x
    m.addConstr(t >= x, name="t_ge_x")
    m.addConstr(t >= -x, name="t_ge_negx")

    # Objective: 0.5 * r^T r + alpha * sum(t)
    m.setObjective(0.5 * (r @ r) + float(alpha) * t.sum(), GRB.MINIMIZE)
    m.Params.OutputFlag = 0

    # Warm start (optional)
    if warm_start is not None and "x" in warm_start:
        x_start = np.asarray(warm_start["x"]).reshape(-1)
        if x_start.shape[0] == p:
            x.start = x_start

    m.optimize()

    x_opt = x.X.copy()
    opt_val = m.ObjVal
    return x_opt, opt_val
    
def sub_problem_of_lasso(A, x, y, b, alpha):
    if gp is None or GRB is None:
        raise ImportError("gurobipy is required to use sub_problem_of_lasso.")

    m = gp.Model()

    kappa = np.where(np.abs(y) >= 0.999*alpha)[0]   # keep only these
    if kappa.size == 0:                             # trivial case
        return np.zeros_like(y)

    # --- decision vector on the reduced space -----------------
    #d_k = m.addMVar(shape=(kappa.size,1), name="d_k", lb=-GRB.INFINITY)
    d_k = m.addMVar(shape=kappa.size, name="d_k", lb=-GRB.INFINITY)

    # --- reduced gradient / Hessian ---------------------------
    g   = grad_f(A, x, b)[kappa]                    # slice once
    Q   = hessian_f(A)[np.ix_(kappa, kappa)]        # sub-matrix only

    # print('Q_shape',Q.shape)
    # print('dk_shape',d_k.shape)
    # print('g_shape',g.shape)
    # print('y_kappa_shape',y[kappa].shape)
    m.setObjective(0.5 * d_k.T @ Q @ d_k  -  (g + y[kappa]).T @ d_k,
                   GRB.MINIMIZE)

    m.Params.OutputFlag = 0
    m.setParam('TimeLimit', 3)
    m.optimize()

    # --- embed back into full length --------------------------
    d_full       = np.zeros_like(y)
    if m.Status == GRB.OPTIMAL:
        d_full[kappa] = d_k.X
    elif m.Status == GRB.TIME_LIMIT:
        #print("Gurobi reached time limit, returning partial solution.")
        d_full[kappa] = d_k.X if d_k.X is not None else np.zeros_like(d_k.X)
    return d_full


_DENSE_NEWTON_CACHE = {}


def _dense_newton_cache_key(A, b):
    if sparse.issparse(A):
        shape = A.shape
        dtype = A.dtype.str
    else:
        A_arr = np.asarray(A)
        shape = A_arr.shape
        dtype = A_arr.dtype.str
    b_arr = np.asarray(b).reshape(-1)
    return (id(A), id(b), shape, b_arr.shape, dtype, b_arr.dtype.str)


def _get_dense_newton_cache(A, b):
    key = _dense_newton_cache_key(A, b)
    cache = _DENSE_NEWTON_CACHE.get(key)
    if cache is not None:
        return cache

    if sparse.issparse(A):
        A_arr = A.toarray().astype(float, copy=False)
    else:
        A_arr = np.asarray(A, dtype=float)
    b_arr = np.asarray(b, dtype=float).reshape(-1)
    cache = {
        "gram": np.asarray(A_arr.T @ A_arr, dtype=float),
        "atb": np.asarray(A_arr.T @ b_arr, dtype=float),
    }
    _DENSE_NEWTON_CACHE[key] = cache
    return cache


def dense_lasso_newton_subproblem(A, x, y, b, alpha):
    """Solve the reduced Lasso Newton subproblem with cached dense algebra."""
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
    rhs = (gram @ x - atb + y)[kappa]
    try:
        d_reduced = np.linalg.solve(Q, rhs)
    except np.linalg.LinAlgError:
        d_reduced = np.linalg.lstsq(Q, rhs, rcond=None)[0]

    d_full[kappa] = d_reduced
    return d_full


def lasso_newton_subproblem(A, x, y, b, alpha, use_gurobi=False):
    """Fast cached Newton subproblem, with optional Gurobi fallback."""
    d = None
    if use_gurobi:
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



import numpy as np

def column_norms_l2(A) -> np.ndarray:
    """
    Return 1-D array of column 2-norms of A (dense or scipy.sparse).
    """
    try:
        from scipy import sparse
        if sparse.isspmatrix(A):
            Ac = A.tocsc(copy=False)  # cheap view if already CSC
            # Square elementwise stays sparse; sum by column is efficient
            norms_mat = (Ac.power(2)).sum(axis=0)
            norms = np.asarray(norms_mat).ravel()  # 1-D float array
            # numerical guard
            norms = np.sqrt(norms, dtype=np.float64, where=~np.isnan(norms))
            return norms
    except ImportError:
        pass

    # Dense path
    A = np.asarray(A)
    # axis=0 -> column norms; keep float dtype
    norms = np.linalg.norm(A, axis=0)
    return norms


def scale_columns_at_most_unit(A):
    """
    Scale columns so that each column has Euclidean norm <= 1.

    Returns
    -------
    A_tilde : same type as A (dense ndarray or sparse matrix)
        Column-scaled A.
    s : np.ndarray, shape (n,)
        s_j = max(1, ||A[:, j]||_2) (the original column norms clipped at 1).
    """
    norms = column_norms_l2(A).astype(np.float64, copy=False)

    # Replace NaN/Inf with 1 so we don't up/downscale unpredictably
    bad = ~np.isfinite(norms)
    if bad.any():
        norms = norms.copy()
        norms[bad] = 1.0

    s = np.maximum(1.0, norms)
    inv_s = 1.0 / s

    try:
        from scipy import sparse
        if sparse.isspmatrix(A):
            # Right-scale sparse matrix by diagonal (column scaling)
            Ac = A.tocsc(copy=False)
            Dinv = sparse.diags(inv_s)  # n x n diagonal (sparse)
            A_tilde = Ac @ Dinv
            return A_tilde, s
    except ImportError:
        pass

    # Dense path: broadcast across columns
    A = np.asarray(A)
    A_tilde = np.multiply(A, inv_s, out=None)  # A * inv_s (broadcast on axis=0)
    return A_tilde, s

import numpy as np

def preprocess_A_scale1(A):
    """
    SSNAL-style column scaling (Ascale == 1):
      d_j = 1 / max(1, ||A[:,j]||_2)
      A_tilde = A * diag(d)
    Returns (A_tilde, d) where d is shape (n,).
    Works for dense NumPy arrays and SciPy sparse matrices.
    """
    try:
        from scipy import sparse
        is_sp = sparse.isspmatrix(A)
    except ImportError:
        is_sp = False

    if is_sp:
        Ac = A.tocsc(copy=False)
        norms = np.sqrt((Ac.power(2)).sum(axis=0)).A1   # column L2 norms
        d = 1.0 / np.maximum(1.0, norms)
        A_tilde = Ac @ sparse.diags(d)                  # right-scale columns
    else:
        A = np.asarray(A)
        norms = np.linalg.norm(A, axis=0)
        d = 1.0 / np.maximum(1.0, norms)
        A_tilde = A * d                                 # broadcast across cols
    return A_tilde, d

def postprocess_x_scale1(x_tilde, d):
    """
    Map solution back to original x: x = diag(d) * x_tilde.
    (Because A_tilde = A * diag(d).)
    """
    return x_tilde * d



def lambda_max_lasso(A_tilde, b):
    """λ_max for lasso on scaled data: ||A^T b||_∞."""
    return np.abs(A_tilde.T @ b).max()

def lipschitz_exact(A_tilde):
    """
    Exact L = ||A||_2^2.
    - Dense: uses NumPy 2-norm (largest singular value).
    - Sparse: uses svds(k=1) for the top singular value.
    """
    try:
        from scipy import sparse
        if sparse.isspmatrix(A_tilde):
            from scipy.sparse.linalg import svds
            # svds returns singular values in ascending order
            sigma_max = svds(A_tilde, k=1, return_singular_vectors=False)[0]
            return float(sigma_max**2)
    except ImportError:
        pass
    # Dense fallback (or if no SciPy): this *requires* A_tilde to be a NumPy array
    sigma_max = np.linalg.norm(np.asarray(A_tilde), 2)
    return float(sigma_max**2)
