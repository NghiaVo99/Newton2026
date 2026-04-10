from re import S
import numpy as np
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
from scipy import sparse

import numpy as np

def make_groups_dict(n, group_len):

    if n % group_len != 0:
        raise ValueError("group_len must divide n evenly.")
    num = n // group_len
    return {i: slice(i*group_len, (i+1)*group_len) for i in range(num)}

def cost_group_lasso(A, x, b, alpha, groups):
    """0.5||Ax-b||^2 + alpha * sum_g ||x_g||_2  with equal-sized groups."""
    A = np.asarray(A, float); x = np.asarray(x, float); b = np.asarray(b, float)
    data_fit = 0.5 * np.linalg.norm(A @ x - b)**2
    reg = sum(np.linalg.norm(x[s]) for s in groups.values())
    return data_fit + float(alpha) * reg


def grad_f(A, x, b):
    # x = np.asarray(x).reshape(-1)
    # b = np.asarray(b).reshape(-1)
    r = A @ x - b                     # (m,)
    return A.T @ r                    # (n,)

def hessian_f(A):
    # Exact Hessian of 0.5||Ax-b||^2 is A^T A.
    # Be careful: forming A.T @ A can be expensive/dense.
    return A.T @ A


def proxL1_L2(x, lam, groups):
    # Prox of group lasso with all w_g = 1:
    #   prox_{lam * sum_g ||x_g||_2}(x)
    x = np.asarray(x, dtype=float)
    z = x.copy()
    lam = float(lam)
    if lam < 0:
        raise ValueError("lam must be nonnegative")

    for idx in groups.values():
        xg = z[idx]
        nrm = np.linalg.norm(xg)
        if nrm <= lam or nrm == 0.0:
            z[idx] = 0.0
        else:
            z[idx] = (1.0 - lam / nrm) * xg
    return z



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

def backtracking_linesearch(A, b, x, grad_x, prox, alpha, groups, L_prev=1.0, eta=2.0,
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
        x_new = prox(x - t * grad_x, t * alpha, groups)

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

def _get_index(A,target_set,group_len):
    n = A.shape[1]
    index = np.arange(n)
    target_index = []
    group_idx_dict = {}
    for i in target_set:
        tmp = np.where(np.floor(index/group_len)  == i)[0]
        target_index += list(tmp)
        group_idx_dict[i] = tmp
    return target_index, group_idx_dict
    
# def sub_problem_of_group_lasso(A, x, y, b, num_group, group_len, alpha):
#     m = gp.Model()
    
#     tmp = y.reshape(num_group,group_len)
#     group_norm = np.linalg.norm(tmp,axis = 0)
#     #print("Group norms:", group_norm)
#     active = np.where(group_norm >= 0.99*alpha)[0]
#     #print("Active groups in subproblem:", active.tolist())
#     num_variables = group_len * len(active)
#     active_idx, group_idx_dict = _get_index(A,active,group_len)

#     # --- decision vector on the reduced space -----------------
#     #d_k = m.addMVar(shape=(num_variables,1), name="d_k", lb=-GRB.INFINITY)
#     beta = m.addMVar(shape=(len(active),1), name="beta", lb=-GRB.INFINITY)
#     d_k = m.addMVar(shape=num_variables, name="d_k", lb=-GRB.INFINITY)

#     # --- reduced gradient / Hessian ---------------------------
#     g   = grad_f(A, x, b)[active_idx]                    # slice once
#     Q   = hessian_f(A)[np.ix_(active_idx, active_idx)]        # sub-matrix only

#     m.setObjective(0.5 * d_k.T @ Q @ d_k  -  (g + y[active_idx]).T @ d_k,
#                    GRB.MINIMIZE)
#     for i, group_idx in enumerate(group_idx_dict):
#         m.addConstr(d_k[i*group_len:(i+1)*group_len] - beta[i]*y[group_idx] == 0)

#     m.Params.OutputFlag = 0
#     m.setParam('TimeLimit', 3)
#     m.optimize()

#     # --- embed back into full length --------------------------
#     d_full       = np.zeros_like(y)
#     if m.Status == GRB.OPTIMAL:
#         d_full[active_idx] = d_k.X
#     elif m.Status == GRB.TIME_LIMIT:
#         #print("Gurobi reached time limit, returning partial solution.")
#         d_full[active_idx] = d_k.X if d_k.X is not None else np.zeros_like(d_k.X)
#     return d_full

def sub_problem_of_group_lasso_new(A, x, y, b, num_group, group_len, alpha):
    m = gp.Model()
    
    tmp = x.reshape(num_group,group_len)
    group_norm = np.linalg.norm(tmp,axis = 0)
    #print("Group norms:", group_norm)
    inactive = np.where(group_norm < 0.0001)[0]
    active = np.where(group_norm >= 0.0001)[0]
    #print("Active groups in subproblem:", active.tolist())
    num_variables = group_len * len(active)
    active_idx, group_idx_dict = _get_index(A,active,group_len)

    # --- decision vector on the reduced space -----------------
    #d_k = m.addMVar(shape=(num_variables,1), name="d_k", lb=-GRB.INFINITY)
    beta = m.addMVar(shape=(len(active),1), name="beta", lb=-GRB.INFINITY)
    d_k = m.addMVar(shape=num_variables, name="d_k", lb=-GRB.INFINITY)

    # --- reduced gradient / Hessian ---------------------------
    g   = grad_f(A, x, b)[active_idx]                    # slice once
    Q   = hessian_f(A)[np.ix_(active_idx, active_idx)]        # sub-matrix only

    m.setObjective(0.5 * d_k.T @ Q @ d_k  -  (g + y[active_idx]).T @ d_k,
                   GRB.MINIMIZE)
    for i, group_idx in enumerate(group_idx_dict):
        m.addConstr(d_k[i*group_len:(i+1)*group_len] - beta[i]*y[group_idx] == 0)

    m.Params.OutputFlag = 0
    m.setParam('TimeLimit', 3)
    m.optimize()

    # --- embed back into full length --------------------------
    d_full       = np.zeros_like(y)
    if m.Status == GRB.OPTIMAL:
        d_full[active_idx] = d_k.X
    elif m.Status == GRB.TIME_LIMIT:
        #print("Gurobi reached time limit, returning partial solution.")
        d_full[active_idx] = d_k.X if d_k.X is not None else np.zeros_like(d_k.X)
    return d_full


