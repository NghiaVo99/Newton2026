from re import S
import numpy as np
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB


def cost_infinity(A,x,b,alpha):
  return 0.5*np.linalg.norm(A@x-b)**2 + alpha*np.linalg.norm(x,ord = np.inf)

def grad_f(A,x,b):
  return A.T@(A@x-b)

def hessian_f(A):
  return A.T@A

def project_l1_ball(x):
    """
    Projects the vector x onto the unit L1 norm ball using CVXPY.
    """
    n = len(x)

    # Define variable
    z = cp.Variable(n)

    # Define objective: minimize squared Euclidean distance to x
    objective = cp.Minimize(cp.sum_squares(z - x))

    # Define constraint: L1 norm should be <= 1
    constraints = [cp.norm1(z) <= 1]

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return z.value

def ProxL_infinity(x,lamda):
  if np.linalg.norm(x,1) <= lamda:
    return (1-lamda)*x
  else:
    return x - lamda*project_l1_ball(x/lamda)

def backtracking_linesearch(A, b, x, grad, prox, alpha, beta=0.9):
    """ Backtracking line search for step size selection """
    t = 1.0
    while True:
        x_new = prox(x - t * grad, t*alpha)
        lhs = 0.5 * np.linalg.norm(A @ x_new - b)**2
        rhs = 0.5 * np.linalg.norm(A @ x - b)**2 - np.dot(grad, x - x_new) + (0.5 / t) * np.linalg.norm(x - x_new)**2
        if lhs <= rhs:
            break
        t = beta*t
    return t

def solve_infinity_cvxpy(A, b, alpha):
    """
    Solve: min_x (1/2) * ||A x - b||^2 + lam * ||x||_infinity using CVXPY

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
    objective = cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + alpha * cp.norm_inf(x))
    problem = cp.Problem(objective)

    # Solve the problem
    problem.solve()

    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
        return x.value, problem.value
    else:
        raise RuntimeError("CVXPY failed to solve the infinity regularized problem.")
    

import numpy as np
import gurobipy as gp
from gurobipy import GRB

def solve_infinity_gurobi(A, b, alpha):
    """
    Solve: min_x (1/2) * ||A x - b||^2 + alpha * ||x||_infinity using Gurobi

    Parameters:
        A (np.ndarray): Matrix A of shape (n, p)
        b (np.ndarray): Vector b of shape (n,)
        alpha (float): Regularization parameter

    Returns:
        x_opt (np.ndarray): Solution vector x (estimated coefficients)
        obj_val (float): Optimal objective value
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n, p = A.shape

    # Create model
    model = gp.Model("infinity_reg")
    model.Params.OutputFlag = 0  # silence Gurobi output; remove if you want logs

    # Decision variables
    x = model.addMVar(shape=p, lb=-GRB.INFINITY, name="x")
    r = model.addMVar(shape=n, lb=-GRB.INFINITY, name="r")  # residual r = A x - b
    t = model.addVar(lb=0.0, name="t")  # ||x||_inf

    # Residual constraints: r = A x - b
    # Gurobi supports NumPy-style linear expressions with MVar
    model.addConstr(r == A @ x - b, name="residual_def")

    # Infinity norm constraints: -t <= x_j <= t  for all j
    for j in range(p):
        model.addConstr(x[j] <= t, name=f"x_le_t_{j}")
        model.addConstr(-x[j] <= t, name=f"-x_le_t_{j}")  # equivalent to x[j] >= -t

    # Objective: 0.5 * ||r||_2^2 + alpha * t
    quad_term = r @ r          # this is sum_i r_i^2 as a QuadExpr
    obj = 0.5 * quad_term + alpha * t
    model.setObjective(obj, GRB.MINIMIZE)

    # Optimize
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        x_opt = x.X.copy()          # x.X is a NumPy array
        obj_val = model.ObjVal
        return x_opt, obj_val
    else:
        raise RuntimeError(
            f"Gurobi failed to solve the infinity-regularized problem. "
            f"Model status: {model.Status}"
        )

    

def sub_problem_of_infinity(A, x, y, b, alpha):
    model = gp.Model()

    # Define variables
    d = model.addMVar(len(y), name="d", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    lamda = model.addMVar(1, name="lamda", lb=-GRB.INFINITY, ub=GRB.INFINITY)

    # Apply constraint
    supp_y_positive = np.where(y >= 1e-5)[0]
    supp_y_negative = np.where(-y >= 1e-5)[0]
    supp_y_complement = np.where(np.abs(y) < 1e-5)[0]

    for idx in supp_y_positive:
        model.addConstr(d[idx] == lamda)

    for idx in supp_y_negative:
        model.addConstr(d[idx] == -lamda)

    # Compute gradient and Hessian
    gradient = grad_f(A, x, b)
    hessian = hessian_f(A)

    # Set objective
    model.setObjective(0.5 * (d.T @ hessian @ d) - ((y + gradient).T @ d), GRB.MINIMIZE)

    # Suppress solver output
    model.setParam('OutputFlag', 0)

    # Optimize
    model.optimize()

    # Check if the model found an optimal solution
    if model.Status == GRB.OPTIMAL:
        optimal_d = d.X
    else:
        optimal_d = np.zeros(len(y))  # Set d to zero if the solver fails

    return optimal_d
