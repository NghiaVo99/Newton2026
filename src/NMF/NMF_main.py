# Compare Jacobi (simultaneous) vs Gauss–Seidel (alternating) projected-gradient for NMF
import numpy as np, time, matplotlib.pyplot as plt
from numpy.linalg import norm
import gurobipy as gp
from gurobipy import GRB
from src.NMF.NMF_utils import *
import cvxpy as cp
from scipy import sparse
from scipy.sparse import linalg as spla


rng = np.random.default_rng(5)

# synthetic nonnegative matrix A = W*H + noise
m, n, r = 28, 28, 3

W_true = rng.random((m, r))
H_true = rng.random((r, n))
A = W_true @ H_true
A = np.clip(A + 0.001 * rng.random(A.shape), 0, None)

def sub_problem_cvx(A, W, H, Z1, Z2):
     
    m,n = A.shape
    r = W.shape[1]

    # --- decision vector on the reduced space -----------------
    D1 = cp.Variable(shape=(m, r), name="d1")
    D2 = cp.Variable(shape=(r, n), name="d2")
    R = W @ H - A
    # --- reduced gradient / Hessian ---------------------------
    gW, gH = grad_f(A, W, H)
    quadratic_term = cp.sum_squares(D1 @ H + W @ D2) + 2*cp.sum(cp.multiply(R, D1 @ D2))
    linear_term = cp.sum(cp.multiply((gW+Z1), D1)) + cp.sum(cp.multiply((gH+Z2), D2))
    subproblem_objective = cp.Minimize(0.5 * quadratic_term -  linear_term)
    # --- constraints ------------------------------------------
    kappa_plus1 = np.where(Z1 <= -1e-6)  
    kappa_plus2 = np.where(Z2 <= -1e-6)  
    constraints = []
    if len(kappa_plus1[0]) > 0 :
        constraints = [D1[kappa_plus1] == 0]
    elif len(kappa_plus2[0]) > 0:
        constraints = [D2[kappa_plus2] == 0]
    problem = cp.Problem(subproblem_objective, constraints)
    
    # Solve the problem
    problem.solve()

    if problem.status == cp.OPTIMAL:
        return D1.value, D2.value
    elif problem.status == cp.OPTIMAL_INACCURATE:
        print("Warning: subproblem solved to OPTIMAL_INACCURATE")
        D1 = np.zeros_like(W)
        D2 = np.zeros_like(H)
        return D1, D2
    

def alternate_sub_problem_cvx(A, W, H, Z1, Z2):
     
    m,n = A.shape
    r = W.shape[1]

    # --- decision vector on the reduced space -----------------
    D1 = cp.Variable(shape=(m, r), name="d1")
    D2 = cp.Variable(shape=(r, n), name="d2")
    R = W @ H - A
    # --- reduced gradient / Hessian ---------------------------
    gW, gH = grad_f(A, W, H)
    quadratic_term = cp.sum_squares(D1 @ H + W @ D2) + 2*cp.sum(cp.multiply(R, D1 @ D2))
    linear_term = cp.sum(cp.multiply((gW+Z1), D1)) + cp.sum(cp.multiply((gH+Z2), D2))
    subproblem_objective = cp.Minimize(0.5 * quadratic_term -  linear_term)
    # --- constraints ------------------------------------------
    kappa_plus1 = np.where(Z1 <= -1e-6)  
    kappa_plus2 = np.where(Z2 <= -1e-6)  
    constraints = []
    if len(kappa_plus1[0]) > 0 :
        constraints = [D1[kappa_plus1] == 0]
    elif len(kappa_plus2[0]) > 0:
        constraints = [D2[kappa_plus2] == 0]
    problem = cp.Problem(subproblem_objective, constraints)
    
    # Solve the problem
    problem.solve()

    if problem.status == cp.OPTIMAL:
        return D1.value, D2.value
    elif problem.status == cp.OPTIMAL_INACCURATE:
        print("Warning: subproblem solved to OPTIMAL_INACCURATE")
        D1 = np.zeros_like(W)
        D2 = np.zeros_like(H)
        return D1, D2


def sub_problem_gurobi(A, W, H, Z1, Z2, verbose=False):
    """
    Solve:
        min_{D1,D2} 0.5 * || D1 H + W D2 ||_F^2 - <gW+Z1, D1> - <gH+Z2, D2>
        s.t.        D1_ij = 0 for (i,j) with Z1_ij <= -1e-8
                    D2_ij = 0 for (i,j) with Z2_ij <= -1e-8
    Returns:
        D1_opt (m x r), D2_opt (r x n)
    """
    A = np.asarray(A, dtype=float)
    W = np.asarray(W, dtype=float)
    H = np.asarray(H, dtype=float)
    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)

    m, n = A.shape
    r = W.shape[1]
    assert W.shape == (m, r)
    assert H.shape == (r, n)
    assert Z1.shape == (m, r)
    assert Z2.shape == (r, n)

    # Gradients of f(W,H) = 0.5||A - WH||_F^2  (same as your grad_f)
    R  = W @ H - A           # m x n
    gW = R @ H.T             # m x r
    gH = W.T @ R             # r x n

    # Index sets for zeroing entries (same rule as CVXPY code)
    kappa_plus1 = np.where(Z1 <= -1e-8)     # tuple of arrays (rows, cols)
    kappa_plus2 = np.where(Z2 <= -1e-8)

    model = gp.Model("sub_problem_qp")
    model.Params.OutputFlag = 1 if verbose else 0
    # Keep NonConvex at default (0). The Hessian here is PSD since it's a squared norm of an affine map.

    # Decision variables
    D1 = model.addMVar((m, r), lb=-GRB.INFINITY, name="D1")
    D2 = model.addMVar((r, n), lb=-GRB.INFINITY, name="D2")

    # Linear expression for Y = D1 H + W D2  (m x n)
    # (MVar @ const) and (const @ MVar) produce linear expressions entrywise.
    Y = D1 @ H + W @ D2

    # Quadratic objective: 0.5 * sum_{i,j} Y[i,j]^2
    # (Building the sum-of-squares via a generator is standard practice in Gurobi.)
    quad = gp.quicksum(Y[i, j] * Y[i, j] for i in range(m) for j in range(n))

    # Linear part: - <gW+Z1, D1> - <gH+Z2, D2>
    # Use matrix entries directly; this is efficient and avoids Python loops over constraints.
    GWZ = gW + Z1
    GHZ = gH + Z2
    lin1 = gp.quicksum(float(GWZ[i, j]) * D1[i, j] for i in range(m) for j in range(r))
    lin2 = gp.quicksum(float(GHZ[i, j]) * D2[i, j] for i in range(r) for j in range(n))

    model.setObjective(0.5 * quad - (lin1 + lin2), GRB.MINIMIZE)

    # Zero constraints for the selected entries — do this by fixing bounds to 0 (no new rows in the LP/QP).
    if kappa_plus1[0].size > 0:
        sel = D1[kappa_plus1]     # this selects a 1-D view of those entries
        sel.lb = 0.0
        sel.ub = 0.0
    if kappa_plus2[0].size > 0:
        sel = D2[kappa_plus2]
        sel.lb = 0.0
        sel.ub = 0.0

    # Optimize
    model.setParam('TimeLimit', 4)  # 4 second time limit
    model.optimize()

    # Extract solution (handle non-optimal statuses similarly to your CVXPY code)
    if model.Status == GRB.OPTIMAL:
        D1_sol = D1.X.reshape(m, r)
        D2_sol = D2.X.reshape(r, n)
    else:
        # Mirror your OPTIMAL_INACCURATE fallback with safe zeros
        if not verbose:
            print(f"Warning: Gurobi status {model.Status}, returning zeros.")
        D1_sol = np.zeros((m, r), dtype=float)
        D2_sol = np.zeros((r, n), dtype=float)

    return D1_sol, D2_sol

import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB

def sub_problem_gurobi_fast(A, W, H, Z1, Z2, verbose=False):
    """
    Vectorized build (no per-term loops).
    Matches CVX objective:
        0.5 * ( ||D1 H||_F^2 + ||W D2||_F^2 + 2*tr(H^T D1^T W D2) + 2*<R, D1 D2> )
        - <gW+Z1, D1> - <gH+Z2, D2>
    with R = W H - A
    """
    m, n = A.shape
    r = W.shape[1]
    R = W @ H - A
    gW, gH = grad_f(A, W, H)

    # ---- Flatten variables  x = [vec(D1); vec(D2)]  ----
    # vec stacks columns: vec(M) shape is (rows*cols,)
    p1 = m * r
    p2 = r * n
    p  = p1 + p2

    # Quadratic blocks
    S1 = H @ H.T             # r x r
    S2 = W.T @ W             # r x r
    Q11 = sp.kron(sp.eye(m, format='csr'), sp.csr_matrix(S1))   # (m r) x (m r)
    Q22 = sp.kron(sp.eye(n, format='csr'), sp.csr_matrix(S2))   # (n r) x (n r)

    # Cross term  tr(H^T D1^T W D2)  => vec(D1)^T (H ⊗ W) vec(D2)
    B1  = sp.kron(sp.csr_matrix(H), sp.csr_matrix(W))           # (m r) x (n r)
    # Cross term  <R, D1 D2>         => vec(D1)^T (R ⊗ I_r) vec(D2)
    B2  = sp.kron(sp.csr_matrix(R), sp.eye(r, format='csr'))    # (m r) x (n r)
    B   = B1 + B2

    # Assemble symmetric Q = [[Q11, 2B],
    #                         [2B^T, Q22]]
    upper = sp.hstack([Q11, 2.0 * B], format='csr')
    lower = sp.hstack([2.0 * B.T, Q22], format='csr')
    Q = sp.vstack([upper, lower], format='csr')

    # Linear term  c^T x  with c = [ -vec(gW+Z1);  -vec(gH+Z2) ]
    c1 = -(gW + Z1).reshape(-1, order='F')   # vec column-major
    c2 = -(gH + Z2).reshape(-1, order='F')
    c  = np.concatenate([c1, c2])

    # ---- Variable bounds to encode "== 0" masks without constraints ----
    lb = np.full(p, -GRB.INFINITY)
    ub = np.full(p,  GRB.INFINITY)

    kappa_plus1 = np.where(Z1 <= -1e-6)   # indices for D1 (m x r)
    kappa_plus2 = np.where(Z2 <= -1e-6)   # indices for D2 (r x n)

    # Map (i,k) in D1 to position in vec(D1): i + k*m  (column-major)
    if kappa_plus1[0].size > 0:
        rows, cols = kappa_plus1
        idx = rows + cols * m
        lb[idx] = 0.0
        ub[idx] = 0.0
    elif kappa_plus2[0].size > 0:
        rows, cols = kappa_plus2
        # Map (k,j) in D2 to position in vec(D2): offset p1 + k + j*r
        idx = p1 + rows + cols * r
        lb[idx] = 0.0
        ub[idx] = 0.0

    # ---- Build and solve ----
    model = gp.Model()
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.NonConvex = 2
    model.Params.TimeLimit = 4.0  # 4 second time limit

    x = model.addMVar(p, lb=lb, ub=ub, name="x")

    # Set quadratic objective in one call
    # 0.5 * x^T Q x + c^T x
    model.setMObjective(Q, c, 0.0, x, sense=GRB.MINIMIZE)
    model.optimize()

    # ---- Extract solution back to D1, D2 ----
    if model.Status == GRB.OPTIMAL:
        xval = x.X
        D1 = xval[:p1].reshape((m, r), order='F')
        D2 = xval[p1:].reshape((r, n), order='F')
        return D1, D2
    else:
        print(f"Gurobi status: {model.Status} (returning zeros)")
        return np.zeros_like(W), np.zeros_like(H)



# Jacobi (simultaneous) projected gradient with backtracking safeguards
def nmf_jacobi_pg(A, W, H, r, iters=200, bt_max=5):
    m, n = A.shape
    hist = []
    t0 = time.time()
    for k in range(iters):
        R = W @ H - A
        gradW = R @ H.T
        gradH = W.T @ R
        LW = norm(H, 2)**2 
        LH = norm(W, 2)**2 
        tW, tH = 1.0 / LW, 1.0 / LH
        f_prev = nmf_objective(A,W, H)
        # Take simultaneous steps, backtrack if necessary
        for _ in range(bt_max):
            W_new = proj_plus(W - tW * gradW)
            H_new = proj_plus(H - tH * gradH)
            f_new = nmf_objective(A,W_new, H_new)
            if f_new <= f_prev:
                W, H = W_new, H_new
                break
            tW *= 0.9; tH *= 0.9
        hist.append((k+1, f_new, time.time() - t0))
    return W, H, np.array(hist)

# Gauss–Seidel (alternating) projected gradient with block Lipschitz steps
def nmf_gs_pg(A, W, H, r, iters=200):
    m, n = A.shape
    hist = []
    t0 = time.time()
    for k in range(iters):
        # update W with H fixed
        R = W @ H - A
        gradW = R @ H.T
        LW = norm(H, 2)**2 
        
        W = proj_plus(W - (1.0 / LW) * gradW)
        # update H with updated W
        R = W @ H - A
        gradH = W.T @ R
        LH = norm(W, 2)**2
        H = proj_plus(H - (1.0 / LH) * gradH)
        hist.append((k+1, nmf_objective(A,W, H), time.time() - t0))
    return W, H, np.array(hist)

def nmf_jacobi_pg_newton(A, W, H, r, iters=200, bt_max=5):
    m, n = A.shape
    hist = []
    do_newton = False
    consecutive_same = 0
    rank_W = np.linalg.matrix_rank(W)
    rank_H = np.linalg.matrix_rank(H)
    t0 = time.time()
    for k in range(iters):
        gradW, gradH = grad_f(A, W, H)
        f_prev = nmf_objective(A, W, H)
        LH = norm(W, 2)**2
        LW = norm(H, 2)**2
        tW, tH = 1.0 / LW, 1.0 / LH
        for _ in range(bt_max):
            W_hat = proj_plus(W - tW * gradW)
            H_hat = proj_plus(H - tH * gradH)
            f_new = nmf_objective(A, W_hat, H_hat)
            if f_new <= f_prev:
                break
            tW *= 0.9; tH *= 0.9

        if do_newton:
            # Solve the subproblem to get the Newton direction
            Z1 = (W - W_hat)/tW - gradW
            Z2 = (H - H_hat)/tH - gradH
            #dW, dH = sub_problem_cvx(A, W_hat, H_hat, Z1, Z2)
            dW, dH = sub_problem_gurobi(A, W_hat, H_hat, Z1, Z2, verbose=False)
            #dW, dH = sub_problem_gurobi_fast(A, W_hat, H_hat, Z1, Z2, verbose=False)
            # Line search on the full objective
            newton_stepsize = 1.0
            while nmf_objective(A, W_hat - newton_stepsize * dW, H_hat - newton_stepsize * dH) > nmf_objective(A, W_hat, H_hat):
                newton_stepsize *= 0.9
            W_new = W_hat - newton_stepsize * dW
            H_new = H_hat - newton_stepsize * dH
            do_newton = False
        else:
            W_new, H_new = W_hat, H_hat
        W, H = W_new, H_new
        f_new = nmf_objective(A, W, H)
        print(f"Iter {k+1}, f = {f_new:.6e}")
        hist.append((k+1, f_new, time.time() - t0))

        new_rank_W = np.linalg.matrix_rank(W)
        new_rank_H = np.linalg.matrix_rank(H)
        if new_rank_H == rank_H and new_rank_W == rank_W:
            consecutive_same += 1
        else:
            consecutive_same = 0
  
        if consecutive_same >= 10:
            do_newton = True
        rank_H = new_rank_H
        rank_W = new_rank_W
    return W, H, np.array(hist)

def nmf_alternate_pg_newton(A, W, H, r, iters=200, bt_max=5):
    m, n = A.shape
    hist = []
    do_newton = False
    consecutive_same = 0
    rank_W = np.linalg.matrix_rank(W)
    rank_H = np.linalg.matrix_rank(H)
    t0 = time.time()
    for k in range(iters):
        R = W @ H - A
        gradW = R @ H.T
        LW = norm(H, 2)**2 
        W_hat = proj_plus(W - (1.0 / LW) * gradW)

        # update H with updated W
        R = W_hat @ H - A
        gradH = W_hat.T @ R
        LH = norm(W_hat, 2)**2
        H_hat = proj_plus(H - (1.0 / LH) * gradH)
        tW, tH = 1.0 / LW, 1.0 / LH
        if do_newton:
            # Solve the subproblem to get the Newton direction
            Z1 = (W - W_hat)/tW - gradW
            Z2 = (H - H_hat)/tH - gradH
            #dW, dH = sub_problem_cvx(A, W_hat, H_hat, Z1, Z2)
            dW, dH = sub_problem_gurobi(A, W_hat, H_hat, Z1, Z2, verbose=False)
            #dW, dH = sub_problem_gurobi_fast(A, W_hat, H_hat, Z1, Z2, verbose=False)
            # Line search on the full objective
            newton_stepsize = 1.0
            while nmf_objective(A, W_hat - newton_stepsize * dW, H_hat - newton_stepsize * dH) > nmf_objective(A, W_hat, H_hat):
                newton_stepsize *= 0.9
            W_new = W_hat - newton_stepsize * dW
            H_new = H_hat - newton_stepsize * dH
            do_newton = False
        else:
            W_new, H_new = W_hat, H_hat
        W, H = W_new, H_new
        f_new = nmf_objective(A, W, H)
        print(f"Iter {k+1}, f = {f_new:.6e}")
        hist.append((k+1, f_new, time.time() - t0))

        new_rank_W = np.linalg.matrix_rank(W)
        new_rank_H = np.linalg.matrix_rank(H)
        if new_rank_H == rank_H and new_rank_W == rank_W:
            consecutive_same += 1
        else:
            consecutive_same = 0
  
        if consecutive_same >= 10:
            do_newton = True
        rank_H = new_rank_H
        rank_W = new_rank_W
    return W, H, np.array(hist)


iters = 50
W0 = rng.random((m, r))
H0 = rng.random((r, n))
# W0 = W_true + 0.001 * rng.random((m, r))
# H0 = H_true + 0.001 * rng.random((r, n))

Wj, Hj, hist_j = nmf_jacobi_pg(A, W0, H0, r, iters=iters)
Wg, Hg, hist_g = nmf_gs_pg(A, W0, H0, r, iters=iters)
Wj_newton, Hj_newton, hist_j_newton = nmf_jacobi_pg_newton(A, W0, H0, r, iters=iters)
W_gs_newton, H_gs_newton, hist_g_newton = nmf_alternate_pg_newton(A, W0, H0, r, iters=iters)
#Wj_newton, Hj_newton, hist_j_newton = nmf_alg1_newton(
#    A, W0, H0, iters=iters, tol_z=1e-10, alpha_damp=0.0)

min_fval = min(hist_j[:,1].min(), hist_g[:,1].min(), hist_j_newton[:,1].min(), hist_g_newton[:,1].min())


# Plot objective vs iteration
plt.figure(figsize=(7,5))
plt.plot(hist_j[:,0], hist_j[:,1] - min_fval, label="ISTA (simultaneous)", color='red')
plt.plot(hist_g[:,0], hist_g[:,1] - min_fval, label="ISTA (alternating)", color='blue')
plt.plot(hist_j_newton[:,0], hist_j_newton[:,1] - min_fval, label="ISTA (Newton)", color='green')
plt.plot(hist_g_newton[:,0], hist_g_newton[:,1] - min_fval, label="ISTA (Gauss-Seidel Newton)", color='orange')
plt.xlabel("Iteration")
plt.ylabel("Objective 0.5||A - WH||_F^2")
plt.yscale('log')
plt.grid()
plt.title("NMF: Jacobi vs Gauss–Seidel Projected Gradient vs Ista_newton")
plt.legend()
plt.tight_layout()

# Plot objective vs wall-clock time
plt.figure(figsize=(7,5))
plt.plot(hist_j[:,2], hist_j[:,1] - min_fval, label="ISTA (simultaneous)", color='red')
plt.plot(hist_g[:,2], hist_g[:,1] - min_fval, label="ISTA (alternating)", color='blue')
plt.plot(hist_j_newton[:,2], hist_j_newton[:,1] - min_fval, label="ISTA (Newton)", color='green')
plt.xlabel("Time (s)")
plt.ylabel("Objective 0.5||A - WH||_F^2")
plt.yscale('log')
plt.title("NMF: Jacobi vs Gauss–Seidel vs Ista_newton (objective vs time)")
plt.legend()
plt.tight_layout()
plt.show()
# Summaries
finals = {
    "method": ["Jacobi PG", "Gauss–Seidel PG", "Jacobi PG (Newton)"],
    "final_obj": [hist_j[-1,1], hist_g[-1,1], hist_j_newton[-1,1]],
    "iters": [int(hist_j[-1,0]), int(hist_g[-1,0]), int(hist_j_newton[-1,0])],
    "time_s": [float(hist_j[-1,2]), float(hist_g[-1,2]), float(hist_j_newton[-1,2])],
}
import pandas as pd
df = pd.DataFrame(finals)

