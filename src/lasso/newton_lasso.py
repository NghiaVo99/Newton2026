import numpy as np
import time
from src.lasso.utils_lasso import *
#from ultils_TV import *

# -------------------------------- ISTA --------------------------------------

def ISTA(A,b,x0,alpha,max_iter, step_size, tol, cost, prox, approx_sol = 0):
  x = x0.copy()
  AT = A.T

  cost_val = np.empty(max_iter + 1, dtype=float)
  x_k      = np.empty(max_iter + 1, dtype=float)
  time_list= np.empty(max_iter + 1, dtype=float)

  cost_val[0] = cost(A,x,b,alpha)
  x_k[0] = np.linalg.norm(x-approx_sol)
  time_list[0] = 0.0
  start_time = time.time()

  r = A @ x - b
  grad = AT @ r

  for i in range(max_iter):
    x = prox(x - step_size*grad, step_size*alpha)

    time_list[i+1] = time.time() - start_time
    x_k[i+1] = np.linalg.norm(x-approx_sol)
    cost_val[i+1] = cost(A,x,b,alpha)
    print('Iteration:', i, 'Cost:', cost_val[i+1])

    r = A @ x - b
    grad_new = AT @ r
    stopping_criteria = np.linalg.norm(x - prox(x - step_size*grad_new, step_size*alpha))/ (1 + np.linalg.norm(x) + np.linalg.norm(r))
    if stopping_criteria < tol or abs(cost_val[i+1] - cost_val[i]) < tol:
        print(f'Algo_Ista converge in {i} iteration')
        return cost_val[:i+2].tolist(), x, i, x_k[:i+2].tolist(), time_list[:i+2].tolist()
    grad = grad_new
  print(f'Algo_Ista converge in {i} iteration')
  return cost_val.tolist(), x, i, x_k.tolist(), time_list.tolist()

# ------------------------------- BT_ISTA ------------------------------------

def BT_ISTA(A,b,x0,alpha,max_iter, tol, cost, prox, approx_sol = 0):
  x = x0.copy()
  AT = A.T

  cost_val = np.empty(max_iter + 1, dtype=float)
  time_list= np.empty(max_iter + 1, dtype=float)
  x_k      = np.empty(max_iter + 1, dtype=float)

  cost_val[0] = cost(A,x,b,alpha)
  time_list[0] = 0.0
  x_k[0] = np.linalg.norm(x-approx_sol)
  start_time = time.time()

  r = A @ x - b
  grad = AT @ r

  for i in range(max_iter):
    step_size = backtracking_linesearch(A, b, x, grad, prox, alpha)

    x = prox(x - step_size*grad, step_size*alpha)

    x_k[i+1] = np.linalg.norm(x-approx_sol)
    cost_val[i+1] = cost(A,x,b,alpha)
    time_list[i+1] = time.time() - start_time
    print('Iteration:', i, 'Cost:', cost_val[i+1])

    r = A @ x - b
    grad_new = AT @ r
    stopping_criteria = np.linalg.norm(x - prox(x - step_size*grad_new, step_size*alpha))/ (1 + np.linalg.norm(x) + np.linalg.norm(r))
    if stopping_criteria < tol or abs(cost_val[i+1] - cost_val[i]) < tol:
        print(f'Algo_BT_Ista converge in {i} iteration')
        return cost_val[:i+2].tolist(), x, i, x_k[:i+2].tolist(), time_list[:i+2].tolist()
    grad = grad_new
  print(f'Algo_BT_Ista converge in {i} iteration')
  return cost_val.tolist(), x, i, x_k.tolist(), time_list.tolist()

# -------------------------------- FISTA1 ------------------------------------

def FISTA1(A, b, x0, alpha, max_iter, step_size, tol, cost, prox, approx_sol = 0):
    x = x0.copy()
    x_old = x0.copy()
    z = x0.copy()
    AT = A.T

    x_k = np.empty(max_iter + 1, dtype=float)
    cost_val = np.empty(max_iter + 1, dtype=float)
    time_list= np.empty(max_iter + 1, dtype=float)

    x_k[0] = np.linalg.norm(x-approx_sol)
    cost_val[0] = cost(A,x,b,alpha)
    time_list[0] = 0.0
    t = 1.0
    start_time = time.time()

    for i in range(max_iter):
        rz = A @ z - b
        grad = AT @ rz
        x = prox(z - step_size*grad, alpha*step_size)

        t_old = t
        t = (1 + np.sqrt(1 + 4*(t_old**2)))/2
        z = x + ((t_old - 1) / t) * (x - x_old)

        time_list[i+1] = time.time() - start_time
        x_k[i+1] = np.linalg.norm(x-approx_sol)
        cost_val[i+1] = cost(A,x,b,alpha)
        print('Iteration:', i, 'Cost:', cost_val[i+1])

        r = A @ x - b
        grad_xnew = AT @ r
        stopping_criteria = np.linalg.norm(x - prox(x - step_size*grad_xnew, step_size*alpha))/ (1 + np.linalg.norm(x) + np.linalg.norm(r))
        if stopping_criteria < tol or abs(cost_val[i+1] - cost_val[i]) < tol:
            print(f'Algo_Fista converge in {i} iteration')
            return cost_val[:i+2].tolist(), x, i, x_k[:i+2].tolist(), time_list[:i+2].tolist()
        x_old = x
    print(f'Algo_Fista converge in {i} iteration')
    return cost_val.tolist(), x, i, x_k.tolist(), time_list.tolist()

# ------------------------------- BT_FISTA1 ----------------------------------

def BT_FISTA1(A, b, x0, alpha, max_iter, tol, cost, prox, approx_sol = 0):
    x = x0.copy()
    x_old = x0.copy()
    z = x0.copy()
    AT = A.T

    x_k = np.empty(max_iter + 1, dtype=float)
    cost_val = np.empty(max_iter + 1, dtype=float)
    time_list= np.empty(max_iter + 1, dtype=float)

    x_k[0] = np.linalg.norm(x0-approx_sol)
    cost_val[0] = cost(A,x,b,alpha)
    time_list[0] = 0.0
    t = 1.0
    start_time = time.time()

    for i in range(max_iter):
        rz = A @ z - b
        grad = AT @ rz
        step_size = backtracking_linesearch(A, b, z, grad, prox, alpha)

        z = z - step_size*grad
        x = prox(z, alpha*step_size)
        x_k[i+1] = np.linalg.norm(x-approx_sol)
        t_old = t
        t = (1 + np.sqrt(1 + 4*(t_old**2)))/2
        z = x + ((t_old - 1) / t) * (x - x_old)
        cost_val[i+1] = cost(A,x,b,alpha)
        time_list[i+1] = time.time() - start_time

        r = A @ x - b
        grad_xnew = AT @ r
        stopping_criteria = np.linalg.norm(x - prox(x - step_size*grad_xnew, step_size*alpha))/ (1 + np.linalg.norm(x) + np.linalg.norm(r))
        if stopping_criteria < tol or abs(cost_val[i+1] - cost_val[i]) < tol:
            print(f'Algo_BT_Fista converge in {i} iteration')
            return cost_val[:i+2].tolist(), x, i, x_k[:i+2].tolist(), time_list[:i+2].tolist()

        x_old = x
    print(f'Algo_BT_Fista converge in {i} iteration')
    return cost_val.tolist(), x, i, x_k.tolist(), time_list.tolist()

# ----------------------------- Algo_Newton_Ista -----------------------------

def Algo_Newton_Ista(A,b,x0,alpha,max_iter, step_size, beta, newton_stepsize, tol, cost,
                     prox, subproblem_solver, newt_tol = 1e-3, approx_sol = 0):
  x = x0.copy()
  AT = A.T

  cost_val = np.empty(max_iter + 1, dtype=float)
  time_list= np.empty(max_iter + 1, dtype=float)
  x_k      = np.empty(max_iter + 1, dtype=float)

  cost_val[0] = cost(A,x,b,alpha)
  time_list[0] = 0.0
  x_k[0] = np.linalg.norm(x-approx_sol)
  start_time = time.time()
  do_newton = 0

  r = A @ x - b
  grad = AT @ r

  for i in range(max_iter):
    x_hat = prox(x - step_size*grad, step_size*alpha)

    if do_newton:
      Gradient_map = (x-x_hat)/step_size
      y = Gradient_map - grad
      d = subproblem_solver(A,x_hat,y,b, alpha)
      newton_stepsize = 1
      x_new = x_hat - newton_stepsize*d
      if cost(A,x_new,b,alpha) >= cost(A,x_hat,b,alpha):
         x_new = x_hat
    else:
      x_new = x_hat

    time_list[i+1] = time.time() - start_time
    x_k[i+1] = np.linalg.norm(x_new-approx_sol)
    cost_val[i+1] = cost(A,x_new,b,alpha)
    print('Iteration:', i, 'Cost:', cost_val[i+1])

    if np.linalg.norm(x_new - x) < newt_tol: 
       do_newton = True
    else:
       do_newton = False

    x = x_new
    r = A @ x_new - b
    grad_xnew = AT @ r
    stopping_criteria = np.linalg.norm(x_new - prox(x_new - step_size*grad_xnew, step_size*alpha))/ (1 + np.linalg.norm(x_new) + np.linalg.norm(r))
    if stopping_criteria < tol or abs(cost_val[i+1] - cost_val[i]) < tol:
        print(f'Algo_Newton_Ista converge in {i} iteration')
        return cost_val[:i+2].tolist(), x, i, x_k[:i+2].tolist(), time_list[:i+2].tolist()

    grad = grad_xnew
  print(f'Algo_Newton_Ista converge in {i} iteration')
  return cost_val.tolist(), x, i, x_k.tolist(), time_list.tolist()

# --------------------------- Algo_Newton_BT_Ista ----------------------------

def Algo_Newton_BT_Ista(A,b,x0,alpha,max_iter, beta, newton_stepsize, tol, cost,
                        prox, subproblem_solver, newt_tol = 1e-3, approx_sol = 0):
  x = x0.copy()
  AT = A.T

  cost_val = np.empty(max_iter + 1, dtype=float)
  time_list= np.empty(max_iter + 1, dtype=float)
  x_k      = np.empty(max_iter + 1, dtype=float)

  cost_val[0] = cost(A,x,b,alpha)
  time_list[0] = 0.0
  x_k[0] = np.linalg.norm(x-approx_sol)
  start_time = time.time()
  do_newton = 0

  r = A @ x - b
  grad = AT @ r

  for i in range(max_iter):
    step_size = backtracking_linesearch(A, b, x, grad, prox, alpha)
    x_hat = prox(x - step_size*grad, step_size*alpha)

    if do_newton:
      Gradient_map = (x-x_hat)/step_size
      y = Gradient_map - grad
      d = subproblem_solver(A,x_hat,y,b, alpha)
      newton_stepsize = 1
      x_new = x_hat - newton_stepsize*d
      if cost(A,x_new,b,alpha) >= cost(A,x_hat,b,alpha):
        x_new = x_hat
    else:
      x_new = x_hat

    time_list[i+1] = time.time() - start_time
    x_k[i+1] = np.linalg.norm(x_new-approx_sol)
    cost_val[i+1] = cost(A,x_new,b,alpha)
    print('Iteration:', i, 'Cost:', cost_val[i+1])

    if np.linalg.norm(x_new - x) < newt_tol: 
       do_newton = True
    else:
       do_newton = False

    x = x_new
    r = A @ x - b
    grad_xnew = AT @ r
    stopping_criteria = np.linalg.norm(x - prox(x - step_size*grad_xnew, step_size*alpha))/ (1 + np.linalg.norm(x) + np.linalg.norm(r))
    if stopping_criteria < tol or abs(cost_val[i+1] - cost_val[i]) < tol:
        print(f'Algo_Newton_BT_Ista converge in {i} iteration')
        return cost_val[:i+2].tolist(), x, i, x_k[:i+2].tolist(), time_list[:i+2].tolist()

    grad = grad_xnew
  print(f'Algo_Newton_BT_Ista converge in {i} iteration')
  return cost_val.tolist(), x, i, x_k.tolist(), time_list.tolist()

# -------------------------- Algo_Newton_Fista_new ---------------------------

def Algo_Newton_Fista_new(A,b,x0,alpha,max_iter, step_size, beta, newton_stepsize, tol, cost,
                          prox, subproblem_solver, newt_tol = 1e-3, approx_sol = 0):
  x = x0.copy()
  z = x.copy()
  x_hat = x0.copy()
  x_old = x0.copy()
  x_k = np.empty(max_iter + 1, dtype=float)

  AT = A.T

  t = 1.0
  t_old = 1.0
  cost_val = np.empty(max_iter + 1, dtype=float)
  time_list= np.empty(max_iter + 1, dtype=float)
  start_time = time.time()
  do_newton = 0

  x_k[0] = np.linalg.norm(x0-approx_sol)
  cost_val[0] = cost(A,x0,b,alpha)
  time_list[0] = 0.0

  for i in range(max_iter):
    grad = AT @ (A @ z - b)
    x_hat = prox(z - step_size*grad, alpha*step_size)

    if do_newton:
      Gradient_map = (z-x_hat)/step_size
      y = Gradient_map - grad
      d =  subproblem_solver(A,x_hat,y,b, alpha)
      newton_stepsize = 1
      x_new = x_hat - newton_stepsize*d
      if cost(A,x_new,b,alpha) >= cost(A,x_hat,b,alpha):
        x_new = x_hat
    else:
      x_new = x_hat

    time_list[i+1] = time.time() - start_time
    x_k[i+1] = np.linalg.norm(x_new-approx_sol)
    cost_val[i+1] = cost(A,x_new,b,alpha)
    print('Iteration:', i, 'Cost:', cost_val[i+1])

    if np.linalg.norm(x_new - z) < newt_tol: 
      do_newton = True
    else:
      do_newton = False

    x = x_new
    t = (0.99 + np.sqrt(1 + 4*(t_old**2)))/2
    z = x + ((t_old - 1) / t) * (x - x_old)
    x_old = x
    t_old = t

    r = A @ x - b
    grad_xnew = AT @ r
    stopping_criteria = np.linalg.norm(x - prox(x - step_size*grad_xnew, step_size*alpha))/ (1 + np.linalg.norm(x) + np.linalg.norm(r))
    if stopping_criteria < tol: #or abs(cost_val[i+1] - cost_val[i]) < tol:
        print(f'Algo_Newton_Fista converge in {i} iteration')
        return cost_val[:i+2].tolist(), x, i, x_k[:i+2].tolist(), time_list[:i+2].tolist()
  print(f'Algo_Newton_Fista converge in {i} iteration')
  return cost_val.tolist(), x, i, x_k.tolist(), time_list.tolist()

# ------------------------ Algo_Newton_BT_Fista_new --------------------------

def Algo_Newton_BT_Fista_new(A,b,x0,alpha,max_iter, beta, newton_stepsize, tol, cost,
                             prox, subproblem_solver, newt_tol = 1e-3, approx_sol = 0):
  x = x0.copy()
  z = x.copy()
  x_hat = x0.copy()
  x_old = x0.copy()
  x_k = np.empty(max_iter + 1, dtype=float)

  AT = A.T

  t = 1.0
  t_old = 1.0
  cost_val = np.empty(max_iter + 1, dtype=float)
  time_list= np.empty(max_iter + 1, dtype=float)
  start_time = time.time()
  do_newton = 0

  x_k[0] = np.linalg.norm(x0-approx_sol)
  cost_val[0] = cost(A,x0,b,alpha)
  time_list[0] = 0.0

  for i in range(max_iter):
    grad = AT @ (A @ z - b)
    step_size = backtracking_linesearch(A, b, z, grad, prox, alpha)
    x_hat = prox(z - step_size*grad, alpha*step_size)

    if do_newton:
      Gradient_map = (z-x_hat)/step_size
      y = Gradient_map - grad
      d = subproblem_solver(A,x_hat,y,b, alpha)
      newton_stepsize = 1
      x_new = x_hat - newton_stepsize*d
      # if cost(A,x_new,b,alpha) >= cost(A,x_hat,b,alpha):
      #    x_new = x_hat
    else:
      x_new = x_hat

    time_list[i+1] = time.time() - start_time
    x_k[i+1] = np.linalg.norm(x_new-approx_sol)
    cost_val[i+1] = cost(A,x_new,b,alpha)
    print('Iteration:', i, 'Cost:', cost_val[i+1])

    if np.linalg.norm(x_new - z) < newt_tol: 
      do_newton = True
    else:
      do_newton = False

    x = x_new
    t = (0.99 + np.sqrt(1 + 4*(t_old**2)))/2
    z = x + ((t_old - 1) / t) * (x - x_old)
    x_old = x
    t_old = t

    r = A @ x - b
    grad_xnew = AT @ r
    stopping_criteria = np.linalg.norm(x - prox(x - step_size*grad_xnew, step_size*alpha))/ (1 + np.linalg.norm(x) + np.linalg.norm(r))
    if stopping_criteria < tol: # or abs(cost_val[i+1] - cost_val[i]) < tol:
        print(f'Algo_Newton_BT_Fista converge in {i} iteration')
        return cost_val[:i+2].tolist(), x, i, x_k[:i+2].tolist(), time_list[:i+2].tolist()
  print(f'Algo_Newton_BT_Fista converge in {i} iteration')
  return cost_val.tolist(), x, i, x_k.tolist(), time_list.tolist()

