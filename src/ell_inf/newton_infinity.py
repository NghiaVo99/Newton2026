import numpy as np
import time
from src.ell_inf.ultils_infinity import *
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt




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
        return x.value
    else:
        raise RuntimeError("CVXPY failed to solve the LASSO problem.")
    

def sub_problem1(A, x, y, b, alpha):
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

    print('model_status', model.Status)

    # Check if the model found an optimal solution
    if model.Status == GRB.OPTIMAL:
        optimal_d = d.X
    else:
        optimal_d = np.zeros(len(y))  # Set d to zero if the solver fails

    return optimal_d



def sub_problem2(A, x, y, b, alpha):

    model = gp.Model()
    # Define variables
    d = model.addMVar(len(y), name="d", lb = - GRB.INFINITY, ub= GRB.INFINITY)
    #d = model.addMVar(len(y), name="d", lb = 0)
    kappa = np.where(np.abs(y) >= 0.999*alpha)[0]
    H = np.where(np.abs(y) < 0.999*alpha)[0]

    model.addConstr(d[H] == 0)

    gradient = grad_f(A, x, b)
    hessian = hessian_f(A)
    model.setObjective((d.T @ hessian @ hessian @ d) - 2*d.T @ hessian @ (y + gradient), GRB.MINIMIZE)
    model.setParam('OutputFlag', 0)
    #model.setParam('Presolve', 0)
    model.optimize()
    #print('model_status', model.Status)
    optimal_d = d.X

    return optimal_d


def ISTA(A,b,x0,alpha,max_iter, step_size, tol, approx_sol):

  x = x0
  cost_val = []
  norm_sol = np.linalg.norm(approx_sol)
  x_k = [np.linalg.norm(x0-approx_sol)]
  time_list = []
  #cost_diff = []
  for i in range(max_iter):
    start_time = time.time()
    x_old = x.copy()
    grad = grad_f(A,x,b)
    x = ProxL_infinity(x - step_size*grad, step_size*alpha)

    x_k.append(np.linalg.norm(x-approx_sol))
    cost_val.append(cost(A,x,b,alpha))
    time_list.append(time.time() - start_time)
    #cost_diff.append(cost(A,x,b,alpha) - cost(A,z,b,alpha))
    #if np.linalg.norm(grad_f(A,x,b)) < tol:
    #if abs(cost(A,x,b,alpha) - cost(A,x_old,b,alpha)) < tol:
    #if np.linalg.norm(x - x_old) < tol:
    if np.linalg.norm(x - approx_sol) < tol:
        return cost_val, x, i, x_k, time_list
  return cost_val, x, i, x_k, time_list

def BT_ISTA(A,b,x0,alpha,max_iter, tol, approx_sol):

  x = x0
  cost_val = []
  time_list = []
  norm_sol = np.linalg.norm(approx_sol)
  x_k = [np.linalg.norm(x0-approx_sol)]
  #cost_diff = []
  start_time = time.time()
  for i in range(max_iter):

    x_old = x.copy()
    grad = grad_f(A,x,b)
    step_size = backtracking_linesearch(A, b, x, grad, alpha)
    x = ProxL_infinity(x - step_size*grad, step_size*alpha)

    x_k.append(np.linalg.norm(x-approx_sol))
    cost_val.append(cost(A,x,b,alpha))
    time_list.append(time.time() - start_time)
    #cost_diff.append(cost(A,x,b,alpha) - cost(A,z,b,alpha))
    #if np.linalg.norm(grad_f(A,x,b)) < tol:
    #if abs(cost(A,x,b,alpha) - cost(A,x_old,b,alpha)) < tol:
    #if np.linalg.norm(x - x_old) < tol:
    #if np.linalg.norm(x - approx_sol) < tol:
    if abs(cost(A,x,b,alpha) - cost(A,approx_sol,b,alpha)) < tol:
        return cost_val, x, i, x_k, time_list
  return cost_val, x, i, x_k, time_list

def FISTA(A, b, x0, alpha, max_iter, step_size, tol):
    x = x0
    t = 1
    z = x0
    x_old = x0
    cost_val = []

    for i in range(max_iter):

        grad = grad_f(A,z,b)
        z = z - step_size*grad
        x = ProxL_infinity(z, alpha*step_size)
        t_old = t
        t = (1 + np.sqrt(1 + 4*(t_old**2)))/2
        z = x + ((t_old - 1) / t) * (x - x_old)
        #if np.linalg.norm(grad_f(A,x,b)) < tol: #
        #if abs(cost(A,x,b,alpha) - cost(A,x_old,b,alpha)) < tol:
        #print(f'iteration {i} cost {cost(A,x,b,alpha)}')
        if np.linalg.norm(x - x_old) < tol:
          return cost_val, x, i
        cost_val.append(cost(A,x,b,alpha))
        x_old = x

    return cost_val, x, i

def FISTA1(A, b, x0, alpha, max_iter, step_size, approx_sol):
    x = x0
    x_old = x0
    z = x0
    norm_sol = np.linalg.norm(approx_sol)
    x_k = [np.linalg.norm(x0-approx_sol)]
    cost_val = []
    time_list= []
    t = 1
    start_time = time.time()
    for i in range(max_iter):

        grad = grad_f(A,z,b)
        z = z - step_size*grad
        x = ProxL_infinity(z, alpha*step_size)
        x_k.append(np.linalg.norm(x-approx_sol))
        t_old = t
        t = (1 + np.sqrt(1 + 4*(t_old**2)))/2
        z = x + ((t_old - 1) / t) * (x - x_old)
        #if np.linalg.norm(grad_f(A,x,b)) < tol: #
        #if abs(cost(A,x,b,alpha) - cost(A,x_old,b,alpha)) < tol:
        #if np.linalg.norm(x - x_old) < tol:
        #if np.linalg.norm(x - approx_sol) < tol:
        #print(f'iteration {i} cost {cost(A,x,b,alpha)}')

        # if abs(cost(A,x,b,alpha) - cost(A,approx_sol,b,alpha)) < tol:
        #   return cost_val, x, i, x_k, time_list

        cost_val.append(cost(A,x,b,alpha))
        time_list.append(time.time() - start_time)
        x_old = x

    return cost_val, x, i, x_k, time_list

def BT_FISTA1(A, b, x0, alpha, max_iter, approx_sol):
    x = x0
    x_old = x0
    z = x0
    norm_sol = np.linalg.norm(approx_sol)
    x_k = [np.linalg.norm(x0-approx_sol)]
    cost_val = []
    time_list= []
    t = 1
    start_time = time.time()
    for i in range(max_iter):


        grad = grad_f(A,z,b)
        step_size = backtracking_linesearch(A, b, z, grad, alpha)
        z = z - step_size*grad
        x = ProxL_infinity(z, alpha*step_size)
        x_k.append(np.linalg.norm(x-approx_sol))
        t_old = t
        t = (1 + np.sqrt(1 + 4*(t_old**2)))/2
        z = x + ((t_old - 1) / t) * (x - x_old)
        #if np.linalg.norm(grad_f(A,x,b)) < tol: #
        #if abs(cost(A,x,b,alpha) - cost(A,x_old,b,alpha)) < tol:
        #if np.linalg.norm(x - x_old) < tol:
        #if np.linalg.norm(x - approx_sol) < tol:
        if abs(cost(A,x,b,alpha) - cost(A,approx_sol,b,alpha)) < tol:
          return cost_val, x, i, x_k, time_list
        cost_val.append(cost(A,x,b,alpha))
        time_list.append(time.time() - start_time)
        x_old = x

    return cost_val, x, i, x_k, time_list


def Algo_Newton_Ista(A,b,x0,alpha,max_iter, step_size, beta, newton_stepsize, tol, approx_sol):

  x = x0
  m = A.shape[0]
  cost_val = []
  time_list = []
  #norm_sol = np.linalg.norm(approx_sol)
  x_k = [np.linalg.norm(x0-approx_sol)]
  #x_hat_k = [np.linalg.norm(x0-approx_sol)/norm_sol]
  d_norm = []
  hessian = hessian_f(A)
  start_time = time.time()
  for i in range(max_iter):


    grad = grad_f(A,x,b)
    x_hat = ProxL_infinity(x - step_size*grad, step_size*alpha)
    #x_hat_k.append(np.linalg.norm(x_hat-approx_sol)/norm_sol)
    y = (x-x_hat)/step_size - grad

    if np.linalg.norm(x_hat - x) < 1e-3:
    #if abs(cost(A,x_hat,b,alpha) - cost(A,x,b,alpha)) < 1e-3:
      d = sub_problem1(A,x_hat,y,b, alpha)
      d_norm.append(np.linalg.norm(d))
      newton_stepsize = 1
      while cost(A,x_hat - newton_stepsize*d,b,alpha) > cost(A,x_hat,b,alpha):
          newton_stepsize = beta*newton_stepsize
      x_new = x_hat - newton_stepsize*d

    else:
      x_new = x_hat
      d_norm.append(0)

    # J = np.where(np.abs(y) >= 0.99*alpha)[0]
    # x_new = x_hat.copy()
    # if len(J) <= m:
    #   H_J = hessian[np.ix_(J,J)]
    #   d_J = np.linalg.inv(H_J) @ ((y + grad_f(A,x_hat,b))[J])# (y+A.T @ (A @ x_hat - b))[J] #
    #   d_norm.append(np.linalg.norm(d_J))
    #   x_new[J] = x_hat[J] - newton_stepsize*d_J

    #   while cost(A, x_new, b, alpha) > cost(A, x_hat, b, alpha):
    #     newton_stepsize = beta*newton_stepsize
    #     x_new[J] = x_hat[J] - newton_stepsize*d_J
    # else:
    #   d_norm.append(0)

    x_k.append(np.linalg.norm(x_new-approx_sol))
    cost_val.append(cost(A,x_new,b,alpha))
    time_list.append(time.time() - start_time)
    #print("Iteration:", i, "cardinality of J", len(J))
    #print(f'iteration {i} cost {cost(A,x_new,b,alpha)}')
    #if np.linalg.norm(grad_f(A,x,b)) < tol:
    #if abs(cost(A,x_new,b,alpha) - cost(A,x,b,alpha)) < tol:
    if np.linalg.norm(x_new - x) < tol:
    #if np.linalg.norm(x_new - approx_sol) < tol:
    #if abs(cost(A,x_new,b,alpha) - cost(A,approx_sol,b,alpha)) < tol:
        return cost_val, x, d_norm, i, x_k, time_list

    x = x_new

  return cost_val, x, d_norm, i, x_k, time_list

def Algo_Newton_BT_Ista(A,b,x0,alpha,max_iter, beta, newton_stepsize, tol, approx_sol):

  x = x0
  m = A.shape[0]
  cost_val = []
  time_list = []
  #norm_sol = np.linalg.norm(approx_sol)
  x_k = [np.linalg.norm(x0-approx_sol)]
  #x_hat_k = [np.linalg.norm(x0-approx_sol)/norm_sol]
  d_norm = []
  hessian = hessian_f(A)
  start_time = time.time()
  for i in range(max_iter):


    grad = grad_f(A,x,b)
    step_size = backtracking_linesearch(A, b, x, grad, alpha)
    x_hat = ProxL_infinity(x - step_size*grad, step_size*alpha)
    #x_hat_k.append(np.linalg.norm(x_hat-approx_sol)/norm_sol)
    y = (x-x_hat)/step_size - grad

    # d = sub_problem1(A,x_hat,y,b, alpha)
    # d_norm.append(np.linalg.norm(d))
    # while cost(A,x_hat - newton_stepsize*d,b,alpha) > cost(A,x_hat,b,alpha):
    #     newton_stepsize = beta*newton_stepsize
    # x = x_hat - newton_stepsize*d
    J = np.where(np.abs(y) >= 0.99*alpha)[0]
    x_new = x_hat.copy()
    if len(J) <= m:
      H_J = hessian[np.ix_(J,J)]
      d_J = np.linalg.inv(H_J) @ ((y + grad_f(A,x_hat,b))[J])# (y+A.T @ (A @ x_hat - b))[J] #
      d_norm.append(np.linalg.norm(d_J))
      x_new[J] = x_hat[J] - newton_stepsize*d_J

      while cost(A, x_new, b, alpha) > cost(A, x_hat, b, alpha):
        newton_stepsize = beta*newton_stepsize
        x_new[J] = x_hat[J] - newton_stepsize*d_J
    else:
      d_norm.append(0)

    x_k.append(np.linalg.norm(x_new-approx_sol))
    cost_val.append(cost(A,x_new,b,alpha))
    time_list.append(time.time() - start_time)
    print("Iteration:", i, "cardinality of J", len(J))
    #if np.linalg.norm(grad_f(A,x,b)) < tol:
    #if abs(cost(A,x_new,b,alpha) - cost(A,x,b,alpha)) < tol:
    #if np.linalg.norm(x_new - x) < tol:
    #if np.linalg.norm(x_new - approx_sol) < tol:
    if abs(cost(A,x_new,b,alpha) - cost(A,approx_sol,b,alpha)) < tol:
        return cost_val, x, d_norm, i, x_k, time_list

    x = x_new

  return cost_val, x, d_norm, i, x_k, time_list

# def Algo_Newton_Fista(A,b,x0,alpha,max_iter, step_size, beta, newton_stepsize, tol, approx_sol):

#   x = x0
#   z = x.copy()
#   #x_hat = x0
#   x_old = x0
#   m = A.shape[0]
#   hessian = hessian_f(A)
#   #norm_sol = np.linalg.norm(approx_sol)
#   x_k = [np.linalg.norm(x0-approx_sol)]
#   #x_hat_k = [np.linalg.norm(x0-approx_sol)/norm_sol]

#   t = 1
#   cost_val = []
#   d_norm = []
#   time_list = []
#   start_time = time.time()
#   for i in range(max_iter):


#     grad = grad_f(A,z,b)
#     x_hat = ProxL_infinity(z - step_size*grad, alpha*step_size)
#     #x_hat_k.append(np.linalg.norm(x_hat-approx_sol)/norm_sol)
#     y = (z-x_hat)/step_size - grad

#     #print('d_2iter', np.linalg.norm(x_hat - x_old))
#     #print('d_2cost',abs(cost(A,x_hat,b,alpha) - cost(A,x_old,b,alpha)))
#     if np.linalg.norm(x_hat - x_old) < 1e-3:
#     #if abs(cost(A,x_hat,b,alpha) - cost(A,x_old,b,alpha)) < 1e-3:
#       x_hat_tmp = ProxL_infinity(x_hat - step_size* grad_f(A,x_hat,b), alpha*step_size)
#       y = (x_hat-x_hat_tmp)/step_size - grad_f(A,x_hat,b)
#       d = sub_problem1(A,x_hat_tmp,y,b,alpha)
#       ls_newton_step = newton_stepsize
#       while cost(A,x_hat - ls_newton_step*d,b,alpha) > cost(A,x_hat,b,alpha):
#           ls_newton_step = beta*ls_newton_step
#       d_norm.append(np.linalg.norm(d))
#       print('newton_fista_d_norm',np.linalg.norm(d))
#       x_new = x_hat - ls_newton_step*d

#     else:
#       x_new = x_hat
#       d_norm.append(0)


#     x_k.append(np.linalg.norm(x_new-approx_sol))
#     cost_val.append(cost(A,x_new,b,alpha))
#     time_list.append(time.time() - start_time)
#     #print("Iteration:", i, "cardinality of J", len(J))
#     print(f'iteration {i} cost {cost(A,x_new,b,alpha)}')
#     #if np.linalg.norm(grad_f(A,x,b)) < tol: #
#     #if abs(cost(A,x_new,b,alpha) - cost(A,x_hat,b,alpha)) < tol:
#     if np.linalg.norm(x_new - x_old) < tol:
#     #if np.linalg.norm(x_new - approx_sol) < tol:
#     #if abs(cost(A,x_new,b,alpha) - cost(A,approx_sol,b,alpha)) < tol:
#         return cost_val, x, d_norm, i, x_k, time_list
#     x = x_new
#     t_old = t
#     t = (1 + np.sqrt(1 + 4*(t_old**2)))/2
#     z = x + ((t_old - 1) / t) * (x - x_old)
#     x_old = x

#   return cost_val, x, d_norm, i, x_k, time_list

def Algo_Newton_Fista(A,b,x0,alpha,max_iter, step_size, beta, newton_stepsize, tol, approx_sol):

  x = x0
  z = x.copy()
  x_hat = x0
  x_old = x0
  m = A.shape[0]
  hessian = hessian_f(A)
  #norm_sol = np.linalg.norm(approx_sol)
  x_k = [np.linalg.norm(x0-approx_sol)]
  #x_hat_k = [np.linalg.norm(x0-approx_sol)/norm_sol]

  t = 1
  cost_val = []
  time_list = []
  d_norm = []
  start_time = time.time()
  for i in range(max_iter):


    grad = grad_f(A,z,b)
    #step_size = backtracking_linesearch(A, b, z, grad, alpha)
    x_hat = ProxL_infinity(z - step_size*grad, alpha*step_size)
    #y = (z-x_hat)/step_size - grad

    x_new = x_hat.copy()
    if np.linalg.norm(x_hat - x_old) < 1e-3:
      break

    x_k.append(np.linalg.norm(x_new-approx_sol))
    cost_val.append(cost(A,x_new,b,alpha))
    time_list.append(time.time() - start_time)

    #if np.linalg.norm(grad_f(A,x,b)) < tol: #
    #if abs(cost(A,x_new,b,alpha) - cost(A,z,b,alpha)) < tol:
    #if np.linalg.norm(x_new - z) < tol:
    # if np.linalg.norm(x_new - approx_sol) < tol:
    #     return cost_val, x, d_norm, i, x_k
    x = x_new
    t_old = t
    t = (1 + np.sqrt(1 + 4*(t_old**2)))/2
    z = x + ((t_old - 1) / t) * (x - x_old)
    x_old = x
  #tmp = time.time() - start_time
  cost_val_newton_ista, x1, d_norm1, i1, x_k1, time_k1 = Algo_Newton_Ista(A,b_new,x_hat, alpha, max_iter, step_size, beta, newton_stepsize, tol, approx_sol)

  cost_val= cost_val + cost_val_newton_ista
  d_norm = d_norm + d_norm1
  x_k= x_k +x_k1
  x = x1
  i = i + i1
  #time1 = np.array(time1) + tmp
  #time_list = time_list + list(time1)


  return cost_val, x, d_norm, i, x_k#, time_list




def Algo_Newton_BT_Fista(A,b,x0,alpha,max_iter, beta, newton_stepsize, tol, approx_sol):

  x = x0
  z = x.copy()
  x_hat = x0
  x_old = x0
  m = A.shape[0]
  hessian = hessian_f(A)
  #norm_sol = np.linalg.norm(approx_sol)
  x_k = [np.linalg.norm(x0-approx_sol)]
  #x_hat_k = [np.linalg.norm(x0-approx_sol)/norm_sol]

  t = 1
  cost_val = []
  time_list = []
  d_norm = []
  start_time = time.time()
  for i in range(max_iter):

    grad = grad_f(A,z,b)
    step_size = backtracking_linesearch(A, b, x, grad, alpha)
    x_hat = ProxL_infinity(z - step_size*grad, alpha*step_size)
    #x_hat_k.append(np.linalg.norm(x_hat-approx_sol)/norm_sol)
    y = (z-x_hat)/step_size - grad


    # d = sub_problem1(A,x_hat,y,b, alpha)
    # d_norm.append(np.linalg.norm(d))
    # while cost(A,x_hat - newton_stepsize*d,b,alpha) > cost(A,x_hat,b,alpha):
    #     newton_stepsize = beta*newton_stepsize

    # x = x_hat - newton_stepsize*d

    J = np.where(np.abs(y) >= 0.99*alpha)[0]
    x_new = x_hat.copy()
    if len(J) < m/3:
      H_J = hessian[np.ix_(J,J)]
      d_J = np.linalg.inv(H_J) @ ((y + grad_f(A,x_hat,b))[J])# (y+A.T @ (A @ x_hat - b))[J] #
      #print('cond_num of H_J ', np.linalg.cond(H_J))
      d_norm.append(np.linalg.norm(d_J))
      x_new[J] = x_hat[J] - newton_stepsize*d_J

      while cost(A, x_new, b, alpha) > cost(A, x_hat, b, alpha):
        newton_stepsize = beta*newton_stepsize
        x_new[J] = x_hat[J] - newton_stepsize*d_J
    else:
      d_norm.append(0)

    x_k.append(np.linalg.norm(x_new-approx_sol))
    cost_val.append(cost(A,x_new,b,alpha))
    time_list.append(time.time() - start_time)
    print("Iteration:", i, "cardinality of J", len(J))

    #if np.linalg.norm(grad_f(A,x,b)) < tol: #
    #if abs(cost(A,x_new,b,alpha) - cost(A,z,b,alpha)) < tol:
    #if np.linalg.norm(x_new - z) < tol:
    #if np.linalg.norm(x_new - approx_sol) < tol:
    if abs(cost(A,x_new,b,alpha) - cost(A,approx_sol,b,alpha)) < tol:
        return cost_val, x, d_norm, i, x_k, time
    x = x_new
    t_old = t
    t = (1 + np.sqrt(1 + 4*(t_old**2)))/2
    z = x + ((t_old - 1) / t) * (x - x_old)
    x_old = x

  return cost_val, x, d_norm, i, x_k, time_list


def Hybrid_FNewton_ista(A,b,x0,alpha,max_iter, beta, newton_stepsize, tol, approx_sol):

  x = x0
  z = x.copy()
  x_hat = x0
  x_old = x0
  m = A.shape[0]
  hessian = hessian_f(A)
  #norm_sol = np.linalg.norm(approx_sol)
  x_k = [np.linalg.norm(x0-approx_sol)]
  #x_hat_k = [np.linalg.norm(x0-approx_sol)/norm_sol]

  t = 1
  cost_val = []
  time_list = []
  d_norm = []
  start_time = time.time()
  for i in range(max_iter):


    grad = grad_f(A,z,b)
    step_size = backtracking_linesearch(A, b, z, grad, alpha)
    x_hat = ProxL_infinity(z - step_size*grad, alpha*step_size)
    #x_hat_k.append(np.linalg.norm(x_hat-approx_sol)/norm_sol)
    y = (z-x_hat)/step_size - grad


    # d = sub_problem1(A,x_hat,y,b, alpha)
    # d_norm.append(np.linalg.norm(d))
    # while cost(A,x_hat - newton_stepsize*d,b,alpha) > cost(A,x_hat,b,alpha):
    #     newton_stepsize = beta*newton_stepsize

    # x = x_hat - newton_stepsize*d

    J = np.where(np.abs(y) >= 0.99*alpha)[0]
    x_new = x_hat.copy()
    if len(J) <= m:
      break

    x_k.append(np.linalg.norm(x_new-approx_sol))
    cost_val.append(cost(A,x_new,b,alpha))
    time_list.append(time.time() - start_time)
    print("Iteration:", i, "cardinality of J", len(J))

    #if np.linalg.norm(grad_f(A,x,b)) < tol: #
    #if abs(cost(A,x_new,b,alpha) - cost(A,z,b,alpha)) < tol:
    #if np.linalg.norm(x_new - z) < tol:
    # if np.linalg.norm(x_new - approx_sol) < tol:
    #     return cost_val, x, d_norm, i, x_k
    x = x_new
    t_old = t
    t = (1 + np.sqrt(1 + 4*(t_old**2)))/2
    z = x + ((t_old - 1) / t) * (x - x_old)
    x_old = x
  tmp = time.time() - start_time
  cost_val_newton_ista, x1, d_norm1, i1, x_k1, time1 = Algo_Newton_BT_Ista(A,b_new, x, alpha, max_iter, beta, newton_stepsize, tol, approx_sol)

  cost_val= cost_val + cost_val_newton_ista
  d_norm = d_norm + d_norm1
  x_k= x_k +x_k1
  x = x1
  i = i + i1
  time1 = np.array(time1) + tmp
  time_list = time_list + list(time1)
  #print(time1)

  return cost_val, x, d_norm, i, x_k, time_list

def Hybrid_Fista_ista(A,b,x0,alpha,max_iter, beta, newton_stepsize, tol, approx_sol):

  x = x0
  z = x.copy()
  x_hat = x0
  x_old = x0
  m = A.shape[0]
  hessian = hessian_f(A)
  #norm_sol = np.linalg.norm(approx_sol)
  x_k = [np.linalg.norm(x0-approx_sol)]
  #x_hat_k = [np.linalg.norm(x0-approx_sol)/norm_sol]

  t = 1
  cost_val = []
  time_list = []
  d_norm = []
  start_time = time.time()
  for i in range(max_iter):


    grad = grad_f(A,z,b)
    step_size = backtracking_linesearch(A, b, z, grad, alpha)
    x_hat = ProxL_infinity(z - step_size*grad, alpha*step_size)
    #x_hat_k.append(np.linalg.norm(x_hat-approx_sol)/norm_sol)
    y = (z-x_hat)/step_size - grad


    # d = sub_problem1(A,x_hat,y,b, alpha)
    # d_norm.append(np.linalg.norm(d))
    # while cost(A,x_hat - newton_stepsize*d,b,alpha) > cost(A,x_hat,b,alpha):
    #     newton_stepsize = beta*newton_stepsize

    # x = x_hat - newton_stepsize*d

    J = np.where(np.abs(y) >= 0.99*alpha)[0]
    x_new = x_hat.copy()
    if len(J) <= m:
      break

    x_k.append(np.linalg.norm(x_new-approx_sol))
    cost_val.append(cost(A,x_new,b,alpha))
    time_list.append(time.time() - start_time)
    print("Iteration:", i, "cardinality of J", len(J))

    #if np.linalg.norm(grad_f(A,x,b)) < tol: #
    #if abs(cost(A,x_new,b,alpha) - cost(A,z,b,alpha)) < tol:
    #if np.linalg.norm(x_new - z) < tol:
    # if np.linalg.norm(x_new - approx_sol) < tol:
    #     return cost_val, x, d_norm, i, x_k
    x = x_new
    t_old = t
    t = (1 + np.sqrt(1 + 4*(t_old**2)))/2
    z = x + ((t_old - 1) / t) * (x - x_old)
    x_old = x

  cost_val_newton_ista, x1, i1, x_k1, time1 = BT_ISTA(A,b_new,x,alpha, max_iter, tol, approx_sol)

  cost_val= cost_val + cost_val_newton_ista
  x_k= x_k +x_k1
  x = x1
  i = i + i1
  # print(time1)
  # time1 = np.array(time1) + time_list[-1]
  # time_list = time_list + list(time1)

  return cost_val, x, d_norm, i, x_k, time_list


m = 63
n = 64
#np.random.seed(42)
z = np.zeros(n) #create z with full zeros before assignment
sparsity = 8
nonzero_indices = np.random.choice(n, sparsity, replace=False)
z[nonzero_indices] = np.ones(sparsity)

A = np.random.randn(m,n)
step_size = 1/(np.linalg.norm(A,2)**2)

beta, newton_stepsize = 0.5, 1
tol = 1e-7
b = A@z
noise = np.random.randn(m) * 0.001 #sigma = 0.01
b_new = b + noise
#alpha = 0.1*np.linalg.norm(A.T @ b_new,np.inf)
alpha = 1
max_iter = 800
x0 = np.zeros(n)

# z = np.array([0,1,0])
# A = np.array([[1,1,0],
#               [1,0,-1]])

step_size = 1/np.linalg.norm(A,2)**2
# # step_size = 0.0001
# alpha = 1
# beta, newton_stepsize = 0.9, 1
# b = np.array([2,-1])
# noise = np.random.randn(*b.shape) * 0.01
# b_new = b + noise
# max_iter = 2000
# tol = 1e-5
# x0 = np.array([5,5,5])

approx_sol = solve_infinity_cvxpy(A,b_new,alpha)
cost_val_newton_ista, x1, d_norm1, i1, x_k1, time_k1 = Algo_Newton_Ista(A,b_new,x0, alpha, max_iter, step_size, beta, newton_stepsize, tol, approx_sol)
cost_val_newton_fista, x2, d_norm2, i2, x_k2 = Algo_Newton_Fista(A,b_new,x0, alpha, max_iter, step_size, beta, newton_stepsize, tol, approx_sol)
cost_val_ista, x3, i3, x_k3, time_k3 = ISTA(A,b_new,x0,alpha, max_iter, step_size, tol, approx_sol)
cost_val_fista, x4, i4, x_k4, time_k4 = FISTA1(A,b_new,x0,alpha, max_iter, step_size, approx_sol)


print(f'Algo_Newton_Ista converge in {i1} iteration')
print(f'Algo_Newton_Fista converge in {i2} iteration')
print(f'Ista converge in {i3} iteration')
print(f'Fista converge in {i4} iteration')
optimal_cost = cost(A,approx_sol,b,alpha)

fig, axs = plt.subplots(1, 2, figsize=(20, 5))

width = 2.5
axs[0].plot(abs(cost_val_newton_ista - optimal_cost),label = 'Newton_Ista', color = 'r',linewidth = width)
axs[0].plot(abs(cost_val_newton_fista - optimal_cost),label = 'Newton_Fista', color = 'm', linewidth = width)
axs[0].plot(abs(cost_val_ista - optimal_cost), label = 'ISTA', color = 'b')
axs[0].plot(abs(cost_val_fista - optimal_cost), label = 'FISTA', color = 'k', linewidth = width)

axs[0].legend()
axs[0].grid()
axs[0].set_ylim(bottom = 1e-6)
axs[0].set_xlabel('Iterations')
axs[0].set_yscale("log")
axs[0].set_ylabel(r'$|f(x_k) - f(x^*)|$')


axs[1].plot(x_k1,label = 'Newton_Ista', color = 'r', linewidth = width)
axs[1].plot(x_k2,label = 'Newton_Fista', color = 'm', linewidth = width)
axs[1].plot(x_k3, label = 'ISTA', color = 'b')
axs[1].plot(x_k4, label = 'FISTA', color = 'k', linewidth = width)

axs[1].legend()
axs[1].grid()
axs[1].set_ylim(bottom = 1e-6)
axs[1].set_yscale("log")
axs[1].set_xlabel('Iterations')
axs[1].set_ylabel(r'$||x_k - x^*||$')

plt.show()
