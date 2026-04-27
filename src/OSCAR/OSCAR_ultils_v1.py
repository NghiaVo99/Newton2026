import numpy as np
import gurobipy as gp
from gurobipy import GRB
import cvxpy as cp
from scipy import sparse as sp

def grad_f(A,x,b):
  return A.T@(A@x-b)

def hessian_f(A):
  return A.T@A

def prox_oscar(y: np.ndarray,
        tau: float,
        lam1: float,
        lam2: float,
        positive: bool = False) -> np.ndarray:
    """
    Proximal operator of OSCAR at y:
        prox_{tau * OSCAR}(y) where
        OSCAR(x) = lam1 * ||x||_1 + lam2 * sum_{i<j} max(|x_i|, |x_j|)
                 = sum_{k=1}^n w_k * |x|_(k),  w_k = lam1 + lam2 * (n-k)

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Input vector
    tau : float
        Prox stepsize (must be >= 0)
    lam1, lam2 : float
        OSCAR parameters (assumed >= 0)
    positive : bool, default False
        If True, enforce nonnegativity of the solution (skip sign restoration)

    Returns
    -------
    x : np.ndarray, shape (n,)
        The proximal result
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y.copy()
    if tau < 0 or lam1 < 0 or lam2 < 0:
        raise ValueError("Require tau >= 0, lam1 >= 0, lam2 >= 0.")

    # 1) sort by magnitude (descending)
    if positive:
        abs_y = y.copy()
        abs_y[abs_y < 0] = 0.0
        signs = np.ones_like(y)
    else:
        signs = np.sign(y)
        abs_y = np.abs(y)

    order = np.argsort(-abs_y)         # indices for descending |y|
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n)

    z = abs_y[order]                   # sorted magnitudes

    # 2) build OSCAR / OWL weights in sorted order:
    #    w_k = lam1 + lam2 * (n-k), for k=1..n (k=0..n-1 in 0-based)
    w = lam1 + lam2 * np.arange(n-1, -1, -1, dtype=float)

    # 3) soft-threshold by tau*w then impose nonincreasing constraint via PAV
    u = z - tau * w

    # Pool-Adjacent-Violators (isotonic regression) for nonincreasing and >=0
    # We implement isotonic regression on -u for nondecreasing, then flip back,
    # or directly maintain nonincreasing by merging blocks when averages increase.
    v = _pav_nonincreasing_nonnegative(u)

    # 4) restore original order and signs
    s = v[inv_order]                   # magnitudes in original order
    if not positive:
        x = signs * s
    else:
        x = s
    return x

def _pav_nonincreasing_nonnegative(u: np.ndarray) -> np.ndarray:
    """
    Project vector u onto the cone {v : v >= 0, v1 >= v2 >= ... >= vn}
    using pool-adjacent-violators in O(n).
    Returns v (same shape as u).
    """
    n = u.size
    # We want v = argmin ||v - u||^2 s.t. v nonincreasing, v>=0
    # Implement by maintaining blocks with nonincreasing averages.
    v = np.empty_like(u)
    # block starts/ends and block means
    starts = np.zeros(n, dtype=int)
    ends = np.zeros(n, dtype=int)
    means = np.zeros(n, dtype=float)

    nb = 0  # number of blocks - 1 (index of last block)
    for i in range(n):
        # start new block [i,i] with mean=max(u[i], 0)
        nb += 1
        starts[nb-1] = i
        ends[nb-1] = i
        means[nb-1] = max(u[i], 0.0)

        # merge while the nonincreasing constraint is violated
        while nb >= 2 and means[nb-2] < means[nb-1]:
            # merge block nb-2 and nb-1
            new_start = starts[nb-2]
            new_end = ends[nb-1]
            # pooled average of the two blocks
            s1, e1, m1 = starts[nb-2], ends[nb-2], means[nb-2]
            s2, e2, m2 = starts[nb-1], ends[nb-1], means[nb-1]
            len1 = e1 - s1 + 1
            len2 = e2 - s2 + 1
            new_mean = (len1 * m1 + len2 * m2) / (len1 + len2)

            starts[nb-2] = new_start
            ends[nb-2] = new_end
            means[nb-2] = new_mean
            nb -= 1  # removed last block

    # write block means back
    idx = 0
    for b in range(nb):
        s = starts[b]
        e = ends[b]
        v[s:e+1] = means[b]
        idx = e + 1
    return v

def oscar_value(x: np.ndarray, w1: float, w2: float) -> float:
    """
    OSCAR(x) = lam1 * ||x||_1 + lam2 * sum_{i<j} max(|x_i|, |x_j|)
             = sum_{k=1}^n w_k * |x|_{(k)},  with  w_k = lam1 + lam2*(n-k)
    where |x|_{(1)} ≥ … ≥ |x|_{(n)}.
    """
    n = x.size
    sx = np.sort(np.abs(x))[::-1]            # descending
    w = w1 + w2 * (np.arange(n-1, -1, -1))  # [lam1+lam2*(n-1), …, lam1]
    return float(np.dot(w, sx))



def solve_oscar_gurobi(
    A: np.ndarray,
    b: np.ndarray,
    w1: float,
    w2: float,
    *,
    verbose=True,
    time_limit= 30.0,
    method = 2,       # 2=barrier (often fastest for large QP)
    crossover = 0,    # 0 to get a faster first primal solution
    threads = None,   # optionally limit threads
):
    """
    Solve OSCAR-regularized least squares:

        minimize_x   0.5 * ||A x - b||_2^2
                      + w1 * ||x||_1
                      + w2 * sum_{i<j} max(|x_i|, |x_j|)

    using Gurobi with vectorized model-building (no Python loops).
    Compatible with older Gurobi (no addMConstrs / lists-of-mats).

    Parameters
    ----------
    A : (m, n) array-like (dense or scipy.sparse)
    b : (m,) array-like
    w1, w2 : nonnegative floats
    verbose : bool
    time_limit : seconds or None
    method : Gurobi Method parameter (default 2=barrier)
    crossover : Gurobi Crossover (0 disables; often faster)
    threads : int or None

    Returns
    -------
    x_opt : (n,) np.ndarray
    obj_val : float  (recomputed numerically)
    model : gurobipy.Model  (in case you want logs/dual info)
    """
    # ---- sanitize inputs ----
    if sp.issparse(A):
        A = A.tocsr().astype(float, copy=False)
        m, n = A.shape
    else:
        A = np.asarray(A, dtype=float)
        m, n = A.shape

    b = np.asarray(b, dtype=float).reshape(-1)
    assert b.shape[0] == m, "Dimension mismatch: A is (m,n), b must be length m."
    if w1 < 0 or w2 < 0:
        raise ValueError("w1 and w2 must be nonnegative.")

    # ---- build model ----
    env = gp.Env(empty=not verbose)
    if not verbose:
        env.setParam("OutputFlag", 0)

    model = gp.Model("LS_OSCAR_vectorized", env=env)

    if time_limit is not None:
        model.setParam("TimeLimit", float(time_limit))
    if method is not None:
        model.setParam("Method", int(method))
    if crossover is not None:
        model.setParam("Crossover", int(crossover))
    if threads is not None:
        model.setParam("Threads", int(threads))

    # Decision variables
    x = model.addMVar(n, lb=-GRB.INFINITY, name="x")
    u = model.addMVar(n, lb=0.0, name="u")                 # u_i >= |x_i|
    r = model.addMVar(m, lb=-GRB.INFINITY, name="r")       # residuals: r = A x - b

    # |x| envelope (two vectorized inequalities)
    # u >=  x  and  u >= -x
    model.addConstr(u - x >= 0.0, name="u_ge_x")
    model.addConstr(u + x >= 0.0, name="u_ge_negx")

    # Residual equality (vectorized)
    # r == A @ x - b   ->   A@x - r == b
    # Works with A as dense or SciPy CSR
    model.addConstr(A @ x - r == b, name="residuals")

    # Pairwise max with vectorized selectors (no Python loops)
    Iu, Ju = np.triu_indices(n, k=1)   # all pairs i<j
    P = Iu.size
    if P > 0:
        mvar = model.addMVar(P, lb=0.0, name="m")

        rows = np.arange(P)
        # Sparse selector matrices: Ei picks u_i for each pair k=(i,j)
        Ei = sp.csr_matrix((np.ones(P), (rows, Iu)), shape=(P, n), dtype=float)
        Ej = sp.csr_matrix((np.ones(P), (rows, Ju)), shape=(P, n), dtype=float)

        # m >= Ei @ u   and   m >= Ej @ u
        model.addConstr(mvar - (Ei @ u) >= 0.0, name="m_ge_ui")
        model.addConstr(mvar - (Ej @ u) >= 0.0, name="m_ge_uj")

        # Objective
        obj = 0.5 * (r @ r) + w1 * u.sum() + w2 * mvar.sum()
    else:
        # n<=1: no pairwise term
        obj = 0.5 * (r @ r) + w1 * u.sum()

    model.setObjective(obj, GRB.MINIMIZE)
    model.setParam("OutputFlag", 1 if verbose else 0)

    # Optimize
    model.optimize()

    # Basic status check
    if model.status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError(f"Gurobi ended with status {model.status}.")

    # Extract solution
    x_opt = x.X.copy()

    # Recompute objective numerically (stable, independent of model internals)
    axmb = (A @ x_opt - b) if sp.issparse(A) else (A.dot(x_opt) - b)
    absx = np.abs(x_opt)
    if n > 1:
        pair_sum = np.maximum(absx[Iu], absx[Ju]).sum()
    else:
        pair_sum = 0.0
    obj_val = 0.5 * float(axmb @ axmb) + w1 * float(absx.sum()) + w2 * float(pair_sum)

    return x_opt, obj_val



# def solve_oscar_gurobi(
#     A: np.ndarray,
#     b: np.ndarray,
#     w1: float,
#     w2: float,
#     *,
#     verbose: bool = True,
#     time_limit = 20,
# ):
#     """
#     Solve: 0.5 * ||A x - b||_2^2 + lambda1 * ||x||_1
#            + lambda2 * sum_{i<j} max(|x_i|, |x_j|)
#     using Gurobi, with fast model building.

#     Key speedups:
#       - Model residuals r = A x - b and minimize 0.5 * ||r||^2 (no Q = A^T A).
#       - Vectorized constraints and variable creation (no Python double loops).

#     Parameters
#     ----------
#     A : (m, n) array-like (np.ndarray; scipy.sparse is OK if .toarray() is feasible, else pass dense)
#     b : (m,) array-like
#     lambda1, lambda2 : float, >= 0
#     verbose : bool
#     time_limit : float | None
#     warm_start : (n,) array-like | None
#     threads : int | None
#         Set Gurobi Threads param, if desired.

#     Returns
#     -------
#     x_opt : (n,) ndarray
#     obj_val : float
#     model : gurobipy.Model
#     """

#     A = np.asarray(A, dtype=float)
#     b = np.asarray(b, dtype=float).reshape(-1)
#     m, n = A.shape
#     assert b.shape[0] == m
#     if w1 < 0 or w2 < 0:
#         raise ValueError("lambda1 and lambda2 must be nonnegative.")

#     # ---- Build model ----
#     env = gp.Env(empty=not verbose)
#     if not verbose:
#         env.setParam("OutputFlag", 0)
#     model = gp.Model("LS_OSCAR_fast", env=env)

#     if time_limit is not None:
#         model.setParam("TimeLimit", float(time_limit))

#     # Decision variables
#     x = model.addMVar(n, lb=-GRB.INFINITY, name="x")
#     u = model.addMVar(n, lb=0.0, name="u")     # u_i >= |x_i|
#     r = model.addMVar(m, lb=-GRB.INFINITY, name="r")  # residuals r = A x - b

#     # Envelope for |x|
#     model.addConstr(u >= x,    name="u_ge_x")
#     model.addConstr(u >= -x,   name="u_ge_negx")

#     # Residual equality (vectorized)
#     # r == A @ x - b
#     model.addConstr(r == A @ x - b, name="residuals")

#     # Pairwise max via single addVars + addConstrs
#     # Build all (i, j) with i < j once using NumPy (fast)
#     I, J = np.triu_indices(n, k=1)
#     pairs = list(zip(I.tolist(), J.tolist()))
#     m_ij = model.addVars(pairs, lb=0.0, name="m")

#     # m_ij >= u_i and m_ij >= u_j  (vectorized via dict comprehensions)
#     model.addConstrs((m_ij[i, j] >= u[i] for (i, j) in pairs), name="m_ge_ui")
#     model.addConstrs((m_ij[i, j] >= u[j] for (i, j) in pairs), name="m_ge_uj")

#     # Optional warm start

#     # Objective: 0.5 * ||r||^2 + lambda1 * sum(u) + lambda2 * sum(m_ij)
#     quad_r = 0.5 * (r @ r)  # MQuadExpr with zero Python loops
#     obj = quad_r \
#           + w1 * gp.quicksum(u) \
#           + w2 * gp.quicksum(m_ij.values())
#     model.setObjective(obj, GRB.MINIMIZE)

#     # A couple of params that can help a bit in practice
#     model.setParam("PreSparsify", 1)  # tends to help big dense A
#     # model.setParam("Method", 1)     # dual simplex sometimes faster for LP relaxations
#     # model.setParam("Crossover", 0)  # try disabling if QP is big (trade-off)

#     model.optimize()

#     if model.status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
#         raise RuntimeError(f"Gurobi ended with status {model.status}.")

#     x_opt = x.X.copy()

#     # Compute full objective value numerically (stable and quick)
#     obj_val = 0.5 * np.sum((A @ x_opt - b) ** 2) \
#               + w1 * np.sum(np.abs(x_opt)) \
#               + w2 * np.sum([max(abs(x_opt[i]), abs(x_opt[j])) for (i, j) in pairs])

#     return x_opt, obj_val


def sub_problem_oscar(A, yk, zk, b, w1, w2, time_limit=10.0, silent=True, H=None):
    """
    Gurobi port of your CVXPY subproblem:
        minimize  0.5 * ||Q d||^2 - (g + z_k)^T d
        subject to d[i] == d[i+1]  for i with |sum_{j=1}^i y_k[j]| <= 1 - 1e-4
    where Q = hessian_f(A) in your code (be sure what Q represents; see note above).
    """
    # Inputs and shapes
    if H is None:
        H = np.asarray(hessian_f(A), dtype=float)  # (n,n)
    else:
        H = np.asarray(H, dtype=float)
    g  = np.asarray(grad_f(A, yk, b), dtype=float)  # (n,)
    zk = np.asarray(zk, dtype=float)
    n = zk.shape[0]
    Q, active_idx, lambdas, Lambda = build_Q_from_oscar(n, zk, w1, w2)
    #print('parallel space basis Q:', Q)
    num_col = Q.shape[1]
    c = g + zk

    # Build model
    m = gp.Model("sub_problem1")
    if silent:
        m.Params.OutputFlag = 0
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)

    d = m.addMVar(n, lb=-GRB.INFINITY, name="d")
    alpha = m.addMVar(num_col, lb=-GRB.INFINITY, name="alpha") 
    # Objective: 0.5 * d^T H d - c^T d
    # gurobipy supports @ with MVar and numpy arrays
    quad = 0.5 * (d @ H @ d)
    lin  = c @ d
    m.setObjective(quad - lin, GRB.MINIMIZE)

    # Constraints
    if num_col > 0:
        m.addConstr(d - Q @ alpha == 0, name="parallel_space")
    m.setParam('Outputflag', 0)  
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        return d.X
    elif m.Status == GRB.TIME_LIMIT:
        #print("Gurobi reached time limit, returning partial solution (zeros).")
        return np.zeros(n, dtype=float)
    else:
        #print(f"Gurobi status: {m.Status}. Returning zeros.")
        return np.zeros(n, dtype=float)


def cost_oscar(A, x, b, w1, w2) -> float:
    r = A @ x - b
    return 0.5 * float(r @ r) + oscar_value(x, w1, w2)

def backtracking_linesearch(
    A,
    b,
    x,
    grad,
    prox,
    w1,
    w2,
    beta=2,
    f_x=None,
    return_candidate=False,
):
    """ Backtracking line search for step size selection """
    if f_x is None:
        r_x = A @ x - b
        f_x = 0.5 * float(r_x @ r_x)

    L = 1
    while True:
        x_new = prox(x - (1/L) * grad,1/L,w1,w2, positive = False)
        r_new = A @ x_new - b
        lhs = 0.5 * float(r_new @ r_new)
        dx = x - x_new
        rhs = f_x - np.dot(grad, dx) + (0.5 * L) * float(dx @ dx)
        if lhs <= rhs:
            break
        L = L*beta
    step = 1 / L
    if return_candidate:
        return step, x_new
    return step


def build_Q_from_oscar(n, z, w1, w2, atol=1e-9, rtol=1e-7, verbose=True):
    """
    Build an orthonormal basis Q (columns) for
        P = span( ⋃_{k in A(z)} ∂ s_k(z) ),
    with OSCAR/SLOPE weights λ_i = w1 + w2*(n - i).

    Parameters
    ----------
    n : int
        Dimension of z.
    z : (n,) array_like
        Current iterate (real vector). (May be unsorted.)
    w1, w2 : float
        OSCAR/SLOPE parameters (nonnegative). λ_i = w1 + w2*(n - i).
    atol, rtol : float, optional
        Tolerances for equality tests and tie detection.
    verbose : bool
        If True, prints lambdas and cumulative sums.

    Returns
    -------
    Q : (n, r) ndarray
        Orthonormal basis for P (columns), in ORIGINAL coordinate order.
        If P = {0}, r = 0 and Q has shape (n, 0).
    active_idx : list of int
        1-based indices k where s_k(z) = Λ_k (within tolerances).
    lambdas : (n,) ndarray
        OSCAR weights (descending).
    Lambda : (n,) ndarray
        Cumulative prefix sums of lambdas (ascending).
    """
    z = np.asarray(z, dtype=float).reshape(-1)
    if z.size != n:
        raise ValueError("n must equal len(z).")
    if w1 < 0 or w2 < 0:
        raise ValueError("w1 and w2 must be nonnegative.")

    # OSCAR/SLOPE weights
    i = np.arange(1, n+1, dtype=float)
    lambdas = w1 + w2*(n - i)
    Lambda = np.cumsum(lambdas)

    # Sort by |z| descending
    perm = np.argsort(-np.abs(z), kind='mergesort')
    iperm = np.empty_like(perm); iperm[perm] = np.arange(n)
    z_sorted = z[perm]
    r = np.abs(z_sorted)
    sigma = np.sign(z_sorted); sigma[sigma == 0] = 1.0

    # Active set
    s_prefix = np.cumsum(r)
    active_mask = np.isclose(s_prefix, Lambda, rtol=rtol, atol=atol)
    active_k = np.where(active_mask)[0] + 1

   #  if verbose:
   #      print("lambdas =", lambdas)
   #      print("Lambda (cumulative) =", Lambda)
   #      print("s_prefix (from z) =", s_prefix)

    # Early exit
    if active_k.size == 0:
        return np.zeros((n, 0)), [], lambdas, Lambda

    # Tie blocks
    blocks = []
    a = 0
    while a < n:
        b = a
        while b+1 < n and np.isclose(r[b+1], r[a], rtol=rtol, atol=atol):
            b += 1
        blocks.append((a+1, b+1))
        a = b+1
    def block_of_k(k):
        for A, B in blocks:
            if A <= k <= B:
                return A, B
        raise RuntimeError("Internal error: k not in any block")
    seen=set(); active_blocks=[]
    for k in active_k:
        ab = block_of_k(k)
        if ab not in seen:
            active_blocks.append(ab); seen.add(ab)

    # Generators
    gens=[]
    for k in active_k:
        a_k,b_k = block_of_k(k)
        m_k = k-(a_k-1)
        g = np.zeros(n)
        if a_k>1: g[:a_k-1] = sigma[:a_k-1]
        if m_k>0: g[a_k-1:a_k-1+m_k] = sigma[a_k-1:a_k-1+m_k]
        gens.append(g)
    for a_b,b_b in active_blocks:
        for i in range(a_b+1, b_b+1):
            d = np.zeros(n)
            d[i-1]   =  sigma[i-1]
            d[a_b-1] = -sigma[a_b-1]
            gens.append(d)

    G_sorted = np.column_stack(gens) if gens else np.zeros((n,0))
    G = G_sorted[iperm,:]
    if G.size==0:
        Q = np.zeros((n,0))
    else:
        norms = np.linalg.norm(G, axis=0)
        keep = norms > 1e-14*np.sqrt(n)
        G = G[:,keep]
        if G.size==0:
            Q = np.zeros((n,0))
        else:
            Q, _ = np.linalg.qr(G, mode='reduced')

    return Q, active_k.tolist(), lambdas, Lambda


def _pd_factor(Sigma, jitter=1e-12):
    Sigma = 0.5*(Sigma + Sigma.T) + jitter*np.eye(Sigma.shape[0])
    try:
        return np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(Sigma)
        w = np.clip(w, 1e-15, None)
        L = (V * np.sqrt(w)) @ V.T
        return 0.5*(L + L.T)

def build_test_problem(n=200, sigma2=0.01, rho=0.7, seed=3):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    Sigma = rho ** np.abs(np.subtract.outer(idx, idx))
    L = _pd_factor(Sigma, jitter=1e-12)
    Z = rng.standard_normal((n, n))
    H = Z @ L.T
    H -= H.mean(axis=0, keepdims=True)
    H /= (H.std(axis=0, ddof=1, keepdims=True) + 1e-12)
    x_true = np.zeros(n)
    x_true[int(0.15*n):int(0.20*n)] =  3.0
    x_true[int(0.45*n):int(0.50*n)] = -4.0
    x_true[int(0.70*n):int(0.75*n)] =  6.0
    y = H @ x_true + np.sqrt(sigma2) * rng.standard_normal(n)
    return H, y, x_true

