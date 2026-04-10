# ISTA for Generalized Lasso (TV) in pure NumPy
# minimize 0.5*||A x - b||^2 + lam * ||D x||_1
# - Uses NumPy spectral norm (no power iteration)
# - You plug your own TV prox: prox_{tau*lam * ||D·||_1}(x)

import numpy as np
import matplotlib.pyplot as plt
import pyproximal
from src.lasso.utils_lasso import grad_f, hessian_f
import gurobipy as gp
from gurobipy import GRB
import cvxpy as cp


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

def generalized_lasso_gurobi(A, b, D, lam: float,
                            time_limit: float = 60.0,
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
        t_val = t.X
        r_val = r.X
        # compute true objective value (no constants omitted here)
        obj_val = 0.5 * np.dot(r_val, r_val) + lam * np.sum(t_val)
        info = {
            "status": mdl.Status,
            "runtime": mdl.Runtime,
            "obj_val": obj_val,
            "gap": getattr(mdl, "MIPGap", 0.0) if hasattr(mdl, "IsMIP") and mdl.IsMIP else 0.0
        }
        return x_val, info
    else:
        print(f"Gurobi status: {mdl.Status}. No incumbent found; returning zeros.")
        return np.zeros(n, dtype=float), {"status": mdl.Status, "runtime": mdl.Runtime}


def sub_problem1_gurobi(A, yk, zk, b, alpha, time_limit=4.0, silent=True):
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




def obj_generalized_lasso(A, b, x, D, lam):
    """
    Objective: 0.5 * ||A x - b||^2 + lam * ||D x||_1
    """
    r = A @ x - b
    Dx = D @ x
    return 0.5 * np.dot(r, r) + lam * np.linalg.norm(Dx, ord=1)

def lipschitz_from_spectral_norm(A):
    """
    L = ||A||_2^2 using NumPy's spectral norm (largest singular value).
    """
    # np.linalg.norm(A, 2) computes the operator 2-norm (largest singular value)
    s_max = np.linalg.norm(A, 2)
    return float(s_max * s_max)

def ista_generalized_lasso(
    A, b, lam,
    prox,               # callable: x_next = prox_Dx_l1(y, tau, D, lam)
    x0=None,
    D=None,
    max_iter=500,
    step_scale=0.99,
    tol_rel=1e-6,
    record_every=1,
    verbose=False
):
    """
    ISTA: x^{k+1} = prox_{t*lam*||D·||_1}( x^k - t * A^T(Ax^k - b) )
    - t = step_scale / L, with L = ||A||_2^2 (NumPy spectral norm)
    Returns (x, obj_vals)
    """
    m, n = A.shape
    if x0 is None:
        x = np.zeros(n, dtype=A.dtype)
    else:
        x = x0.astype(A.dtype, copy=True)

    if D is None:
        D = make_forward_diff(n, dtype=A.dtype)

    L = lipschitz_from_spectral_norm(A)
    if L <= 0:
        raise ValueError("Nonpositive Lipschitz constant; check A.")
    t = step_scale / L

    obj_vals = []
    x_iters = []
    AT = A.T

    prev_val = obj_generalized_lasso(A, b, x, D, lam)
    obj_vals.append(prev_val)

    for k in range(1, max_iter+1):
        # gradient step
        grad = grad_f(A, x, b)
        y = x - t * grad

        # prox step (your TV prox)
        x_new = prox(y,lam * t)

        if k % record_every == 0:
            val = obj_generalized_lasso(A, b, x_new, D, lam)
            obj_vals.append(val)
            if verbose:
                print(f"iter {k:4d}: obj = {val:.6e}")
            # relative improvement stopping
            denom = max(1.0, abs(prev_val))
            if abs(prev_val - val) / denom < tol_rel:
                x = x_new
                break
            prev_val = val

        x = x_new
        x_iters.append(x)

    return x, obj_vals, x_iters


def ista_newton_generalized_lasso(
    A, b, lam,
    prox,               # callable: x_next = prox_Dx_l1(y, tau, D, lam)
    x0=None,
    D=None,
    beta = 0.5,
    max_iter=500,
    step_scale=0.99,
    tol_rel=1e-6,
    record_every=1,
    verbose=False
):
    """
    ISTA: x^{k+1} = prox_{t*lam*||D·||_1}( x^k - t * A^T(Ax^k - b) )
    - t = step_scale / L, with L = ||A||_2^2 (NumPy spectral norm)
    Returns (x, obj_vals)
    """
    m, n = A.shape
    if x0 is None:
        x = np.zeros(n, dtype=A.dtype)
    else:
        x = x0.astype(A.dtype, copy=True)

    if D is None:
        D = make_forward_diff(n, dtype=A.dtype)

    L = lipschitz_from_spectral_norm(A)
    if L <= 0:
        raise ValueError("Nonpositive Lipschitz constant; check A.")
    t = step_scale / L

    obj_vals = []
    x_iters = []
    AT = A.T
    do_newton = False  
    prev_val = obj_generalized_lasso(A, b, x, D, lam)
    obj_vals.append(prev_val)

    for k in range(1, max_iter+1):
        # gradient step
        grad = grad_f(A, x, b)
        y = x - t * grad

        # prox step (your TV prox)
        x_hat = prox(y,lam * t)

        if do_newton:
            zk = (x - x_hat) / t - grad
            #d_k = sub_problem1_cvxpy(A, x_hat, zk, b, lam)
            d_k = sub_problem1_gurobi(A, x_hat, zk, b, lam)
            print('norm d_k', np.linalg.norm(d_k))
            newton_stepsize = 1.0
            # while obj_generalized_lasso(A, b, x_hat - newton_stepsize * d_k, D, lam) > obj_generalized_lasso(A, b, x_hat, D, lam):
            #    newton_stepsize *= beta
            x_new = x_hat - newton_stepsize * d_k
            #do_newton = False
        else:
            x_new = x_hat

        #if obj_generalized_lasso(A, b, x_new, D, lam) - obj_generalized_lasso(A, b, x_hat, D, lam) < 1e-2:
                #if np.linalg.norm(x_new - x_hat) < 1e-2: 
        if np.linalg.norm(x_new - x) < 1e-3: 
            do_newton = True
        else:
            do_newton = False

        if k % record_every == 0:
            val = obj_generalized_lasso(A, b, x_new, D, lam)
            obj_vals.append(val)
            if verbose:
                print(f"iter {k:4d}: obj = {val:.6e}")
            # relative improvement stopping
            denom = max(1.0, abs(prev_val))
            if abs(prev_val - val) / denom < tol_rel:
                x = x_new
                break
            prev_val = val

        x = x_new
        x_iters.append(x)

    return x, obj_vals, x_iters

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Problem dimensions
    m, n = 256, 512
    max_iter = 500
    tol = 1e-6
    # Synthetic data
    A = rng.normal(size=(m, n)) / np.sqrt(m)
    x_true = np.random.standard_normal(n)
    b = A @ x_true + 0.001 * rng.normal(size=m)

    lam = 1 
    D = make_forward_diff(n)
    # TV prox from pyproximal package
    prox = pyproximal.TV(dims = (n,)).prox

    # ---- Replace this with YOUR TV proximal implementation ----
    def prox_Dx_l1_USER(x, tau, D, lam):
        """
        Must compute prox_{tau*lam*||D·||_1}(x).
        Replace with your package call, e.g.:
            return my_tv_prox(x, weight=tau*lam)
        """
        raise NotImplementedError("Plug your TV prox: prox_{tau*lam*||D·||_1}(x).")

    try:
        x_gurobi, info = generalized_lasso_gurobi(A, b, D, lam, silent=False)
        print(f"Gurobi solution: obj = {info['obj_val']:.6e}")
        x_hat, history, x_iters = ista_generalized_lasso(
            A, b, lam,
            prox=prox,  
            D=D,
            max_iter= max_iter,
            step_scale=0.99,
            tol_rel= tol,
            record_every=1,
            verbose=True
        )

        x_hat2, history2, x_iters2 = ista_newton_generalized_lasso(
            A, b, lam,
            prox=prox,  # Replace with prox_Dx_l1_USER to use your own
            D=D,
            beta= 0.9,
            max_iter= max_iter,
            step_scale=0.99,
            tol_rel= tol,
            record_every=1,
            verbose=True
        )
    except NotImplementedError as e:
        print(e)
        history = None

    # Plot objective history
    if history is not None and len(history) > 1:
        min_val = np.min(history + history2)
        plt.figure(1)
        plt.plot(np.array(history) - min_val, linewidth=2, label='ISTA')
        plt.plot(np.array(history2) - min_val, linewidth=2, label='ISTA with Newton')
        plt.xlabel("Iteration")
        plt.ylabel(r"Objective $0.5\|Ax-b\|^2 + \lambda\|Dx\|_1$")
        plt.yscale("log")
        plt.title("ISTA for Generalized Lasso (TV) — Objective vs Iterations")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        plt.figure(2)
        plt.plot(np.linalg.norm(x_iters - x_gurobi, axis=1), linewidth=2, label='ISTA')
        plt.plot(np.linalg.norm(x_iters2 - x_gurobi, axis=1), linewidth=2, label='ISTA with Newton')
        plt.xlabel("Iteration")
        plt.ylabel(r"$\|x-x^\ast\|$")
        plt.yscale("log")
        plt.title("ISTA for Generalized Lasso (TV) — Objective vs Iterations")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
