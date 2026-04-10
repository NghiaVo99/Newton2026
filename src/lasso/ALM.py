import numpy as np
import time
from gurobipy import GRB
import gurobipy as gp
import cvxpy as cp
from src.lasso.ultils_TV import *

import cvxpy as cp
from scipy.sparse import csr_matrix

def solve_TV_gurobi(A, D, b, alpha, verbose=False):
    """
    Solve TV with Gurobi:
        min_x 1/2||A x - b||^2 + alpha * ||D x||_1

    Inputs:
      A     (n×p) numpy array
      D     (m×p) numpy array or sparse matrix
      b     (n,)   numpy vector
      alpha scalar ≥0
    Returns:
      x_opt  (p,) optimal x
      objval scalar optimal objective
    """
    n, p = A.shape
    # number of TV terms
    m = D.shape[0]

    # Precompute Hessian and linear term
    H = A.T @ A           # p×p
    q = -A.T @ b          # p-vector

    # Build model
    model = gp.Model()
    model.Params.OutputFlag = 1 if verbose else 0

    # Decision vars
    x = model.addMVar(shape=p,
                      lb=-GRB.INFINITY,
                      ub= GRB.INFINITY,
                      name="x")
    t = model.addMVar(shape=m,
                      lb=0.0,
                      ub=GRB.INFINITY,
                      name="t")

    # TV constraints:  (D x)_i <= t_i  and  -(D x)_i <= t_i
    # Gurobi will automatically expand D @ x into linear exprs
    model.addConstr(D @ x <= t, name="pos_abs")
    model.addConstr(-D @ x <= t, name="neg_abs")

    # Objective: 0.5 x^T H x + q^T x + alpha * sum(t)
    # Gurobi’s Python API supports x @ H @ x → QuadExpr
    quad_term   = 0.5 * (x @ (H @ x))
    linear_term = q @ x + alpha * gp.quicksum(t)
    model.setObjective(quad_term + linear_term, GRB.MINIMIZE)

    # Optimize
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        return x.X, model.ObjVal
    else:
        raise RuntimeError(f"Gurobi failed with status {model.Status}")



def sub_problem1(A, D, x_hat, y_hat, z, b, alpha, rho, x_len):
    m = gp.Model()

    kappa = np.where(np.abs(y_hat) >= 0.999*alpha)[0]   # keep only these
    if kappa.size == 0:                             # trivial case
        return np.zeros_like(y_hat)

    # --- decision vector on the reduced space -----------------
    d_k = m.addMVar(shape=kappa.size, name="d_k", lb=-GRB.INFINITY)

    
    x = x_hat[:x_len]
    y = x_hat[x_len:]
    partial_x, partial_y = grad_f(A,x,y,z,b, rho, D)
    g = np.concatenate([partial_x, partial_y])  # full gradient
    # --- reduced gradient / Hessian ---------------------------
    g   = g[kappa]                    # slice once
    Q   = hessian_f(A,x,y,z, rho, D)[np.ix_(kappa, kappa)]        # sub-matrix only

    m.setObjective(0.5 * d_k @ Q @ d_k  -  (g + y_hat[kappa]) @ d_k,
                   GRB.MINIMIZE)

    m.Params.OutputFlag = 0
    m.optimize()

    # --- embed back into full length --------------------------
    d_full       = np.zeros_like(y_hat)
    if m.Status == GRB.OPTIMAL:
        d_full[kappa] = d_k.X
    return d_full


def Augmented_Lag_method(x0,y0,z0, A, D, b, alpha, rho, step_size, outer_max_iter, inner_max_iter, tol=1e-6):
    
    x = x0.astype(np.float64).copy()
    y = y0.astype(np.float64).copy()
    z = z0.astype(np.float64).copy()
    Dx = D @ x
    cost_val = cost(A, x, b, alpha, Dx)
    beta = 0.9
    cost_list = [cost_val]
    time_list = [0]
    time_start = time.time()
    inner_accumulator = [0]
    for i in range(outer_max_iter):
        for j in range(inner_max_iter):
            
            partial_x, partial_y = grad_f(A,x,y,z,b, rho, D)
            augmented_cost_val = augmented_cost(A, x, y, z, b, alpha, rho, D)
            # Update x
            x = x - step_size * partial_x
            # Update y
            y = prox(y - step_size*partial_y, step_size * alpha)
            augmented_cost_val_new = augmented_cost(A, x, y, z, b, alpha, rho, D)
            if abs(augmented_cost_val_new - augmented_cost_val) < tol:
                break
        inner_accumulator.append(j + inner_accumulator[-1])            
        Dx = D @ x
        # Update z
        z += rho * (Dx - y)
        cost_val = cost(A, x, b, alpha, Dx)
        time_list.append(time.time() - time_start)
        print('Outer Iteration:', i, 'Cost:', cost_val)
        cost_list.append(cost_val)
        # Check convergence
        primal_residual = np.linalg.norm(Dx - y)
        dual_residual = np.linalg.norm(A.T@(A@x - b) + D.T @ z + rho * D.T @ (D @ x - y))
        if primal_residual < tol or dual_residual < tol:
            break
    return x, y, z, i, j, cost_list, time_list, inner_accumulator


def Augmented_Lag_Newt_method(x0,y0,z0, A, D, b, alpha, rho, step_size, outer_max_iter, inner_max_iter, tol=1e-6):
    
    x = x0.astype(np.float64).copy()
    y = y0.astype(np.float64).copy()
    z = z0.astype(np.float64).copy()
    Dx = D @ x
    cost_val = cost(A, x, b, alpha, Dx)
    beta = 0.9
    cost_list = [cost_val]
    time_list = [0]
    time_start = time.time()
    inner_accumulator = [0]
    for i in range(outer_max_iter):
        for j in range(inner_max_iter):
            
            partial_x, partial_y = grad_f(A,x,y,z,b, rho, D)
            x_tilde = np.concatenate([x.copy(),y.copy()])
            grad = np.concatenate([partial_x, partial_y])
            augmented_cost_val = augmented_cost(A, x, y, z, b, alpha, rho, D)
            # Update x
            x = x - step_size * partial_x
            # Update y
            y = prox(y - step_size*partial_y, step_size * alpha)

            x_hat = np.concatenate([x,y])
            Gradient_map = (x_tilde-x_hat)/step_size
            y_hat = Gradient_map - grad
            d = sub_problem1(A, D, x_hat, y_hat, z, b, alpha, rho, x_len = len(x))
            dx = d[:len(x)]
            dy = d[len(x):]
            # d_norm.append(np.linalg.norm(d))
            newton_stepsize = 1
            while (augmented_cost(A, x - newton_stepsize*dx, y- newton_stepsize*dy, z, b, alpha, rho, D) 
                    > augmented_cost(A, x - newton_stepsize*dx, y- newton_stepsize*dy, z, b, alpha, rho, D)):
                 newton_stepsize = beta*newton_stepsize
            x = x - newton_stepsize*dx
            y = y - newton_stepsize*dy
            augmented_cost_val_new = augmented_cost(A, x, y, z, b, alpha, rho, D)
            print('Inner Iteration:', i, j, 'Cost:', augmented_cost_val_new)
            if abs(augmented_cost_val_new - augmented_cost_val) < tol:
                break
        # if j == 0:
        #     inner_accumulator.append(j + 1)  
        # else:
        inner_accumulator.append(j + inner_accumulator[-1])            
        Dx = D @ x
        # Update z
        z += rho * (Dx - y)
        cost_val = cost(A, x, b, alpha, Dx)
        time_list.append(time.time() - time_start)
        print('Outer Iteration:', i, 'Cost:', cost_val)
        cost_list.append(cost_val)
        # Check convergence
        primal_residual = np.linalg.norm(Dx - y)
        dual_residual = np.linalg.norm(A.T@(A@x - b) + D.T @ z + rho * D.T @ (D @ x - y))
        if primal_residual < tol or dual_residual < tol:
            break
    return x, y, z, i, j, cost_list, time_list, inner_accumulator