import numpy as np
import gurobipy as gp
from gurobipy import GRB
import cvxpy as cp

def grad_f(A,x,b):
  return A.T@(A@x-b)

def hessian_f(A):
  return A.T@A

def make_forward_diff(n: int, dtype=float):
    """
    (n-1) x n forward-difference matrix D: (Dx)_i = x_{i+1} - x_i
    """
    if n < 2:
        raise ValueError("n must be >= 2 for forward differences.")
    D = np.zeros((n-1, n), dtype=dtype)
    idx = np.arange(n-1)
    D[idx, idx] = 1.0
    D[idx, idx+1] = -1.0
    return D


def solve_generalized_lasso_gurobi(A, b, lam: float, D,
                            time_limit: float = 20.0,
                             silent: bool = True):
    """
    Solve:  min_x  0.5*||A x - b||_2^2 + lam * ||D x||_1
    using Gurobi as a convex QP with linear constraints.

    Variables:
      x in R^n (free), r in R^m (free), t in R^p (>=0)
    Constraints:
      r = A x - b
      D x <=  t
     -D x <=  t
      t >= 0
    Objective:
      0.5 * r^T r + lam * 1^T t
    """
    # --- coerce to dense numpy (gurobi works great with numpy arrays) ---
    if hasattr(A, "toarray"): A = A.toarray()
    if hasattr(D, "toarray"): D = D.toarray()
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    D = np.asarray(D, dtype=float)

    m, n = A.shape
    p, nD = D.shape
    assert nD == n, f"D has {nD} columns but A has {n}"
    assert b.shape[0] == m, f"b has length {b.shape[0]} but A has {m} rows"
    assert lam >= 0.0, "lambda must be >= 0"

    # --- build model ---
    mdl = gp.Model("generalized_lasso")
    if silent:
        mdl.Params.OutputFlag = 0
    if time_limit is not None:
        mdl.Params.TimeLimit = float(time_limit)

    # decision variables
    x = mdl.addMVar(n, lb=-GRB.INFINITY, name="x")
    r = mdl.addMVar(m, lb=-GRB.INFINITY, name="r")
    t = mdl.addMVar(p, lb=0.0, name="t")

    # constraints: r = A x - b
    mdl.addConstr(r == A @ x - b, name="residual")

    # |D x| <= t  <=>  D x <= t and -D x <= t
    mdl.addConstr(D @ x <= t, name="abs_pos")
    mdl.addConstr(-D @ x <= t, name="abs_neg")

    # objective: 0.5*||r||^2 + lam*sum(t)
    obj = 0.5 * (r @ r) + lam * gp.quicksum(t)
    mdl.setObjective(obj, GRB.MINIMIZE)

    # optimize
    mdl.optimize()

    # retrieve solution (best available)
    if mdl.SolCount > 0:
        x_val = x.X
        return x_val
    else:
        print(f"Gurobi status: {mdl.Status}. No incumbent found; returning zeros.")
        return np.zeros(n, dtype=float)


def sub_problem_gen_lasso(A, yk, zk, b, alpha, time_limit=4.0, silent=True):
    """
    Gurobi port of your CVXPY subproblem:
        minimize  0.5 * ||Q d||^2 - (g + z_k)^T d
        subject to d[i] == d[i+1]  for i with |sum_{j=1}^i y_k[j]| <= 1 - 1e-4
    where Q = hessian_f(A) in your code (be sure what Q represents; see note above).
    """
    # Inputs and shapes
    Q = np.asarray(hessian_f(A), dtype=float)  # (n,n)
    g  = np.asarray(grad_f(A, yk, b), dtype=float)  # (n,)
    zk = np.asarray(zk, dtype=float)

    # Infer n and build H = Q^T Q to match 0.5*||Q d||^2
    if Q.ndim != 2:
        raise ValueError("Q must be a 2D array.")
    n = Q.shape[1]
    if g.shape[0] != n or zk.shape[0] != n:
        raise ValueError(f"Dimension mismatch: n={n}, g={g.shape}, zk={zk.shape}")

    H = Q
    
    c = g + zk

    # Prefix-sum "parallel-space" constraints (TV case)
    zk_vec = np.asarray(zk, float).reshape(-1)
    if zk_vec.size < n:
        raise ValueError(f"zk length {zk_vec.size} < n={n}.")
    s = np.cumsum(zk_vec[:n-1])     # length n-1
    idx = np.where(np.abs(s) < 1 - 1e-4)[0]  # indices i where d[i]==d[i+1]

    # Build model
    m = gp.Model("sub_problem1")
    if silent:
        m.Params.OutputFlag = 0
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)

    d = m.addMVar(n, lb=-GRB.INFINITY, name="d")

    # Objective: 0.5 * d^T H d - c^T d
    # gurobipy supports @ with MVar and numpy arrays
    quad = 0.5 * (d @ H @ d)
    lin  = c @ d
    m.setObjective(quad - lin, GRB.MINIMIZE)

    # Constraints: d[i] == d[i+1] for i in idx
    if idx.size:
        m.addConstrs((d[int(i)] - d[int(i)+1] == 0 for i in idx), name="parallel_space")

    m.optimize()

    if m.Status == GRB.OPTIMAL:
        return d.X
    elif m.Status == GRB.TIME_LIMIT:
        print("Gurobi reached time limit, returning partial solution (zeros).")
        return np.zeros(n, dtype=float)
    else:
        print(f"Gurobi status: {m.Status}. Returning zeros.")
        return np.zeros(n, dtype=float)


def cost_generalized_lasso(A, x, b, lam, D):
    """
    Objective: 0.5 * ||A x - b||^2 + lam * ||D x||_1
    """
    # print("A shape:", A.shape)
    # print("x shape:", x.shape)
    # print("b shape:", b.shape)
    r = A @ x - b
    Dx = D @ x
    return 0.5 * np.dot(r, r) + lam * np.linalg.norm(Dx, ord=1)

def backtracking_linesearch(A, b, x, grad, prox, alpha, beta=1.5):
    """ Backtracking line search for step size selection """
    L = 1
    while True:
        x_new = prox(x - (1/L) * grad,alpha * 1/L)
        lhs = 0.5 * np.linalg.norm(A @ x_new - b)**2
        rhs = 0.5 * np.linalg.norm(A @ x - b)**2 - np.dot(grad, x - x_new) + (0.5 * L) * np.linalg.norm(x - x_new)**2
        if lhs <= rhs:
            break
        L = L*beta
    return 1/L

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


