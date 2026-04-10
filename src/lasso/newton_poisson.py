import numpy as np
import time
from src.lasso.Poisson_utils import *

def BT_ISTA(A,AT,b,x0, noisy_z,
                        alpha,max_iter, beta, newton_stepsize, tol, cost,
                        prox, subproblem_solver, approx_sol = 0):
  x = x0.copy()
  z = noisy_z.copy()

  cost_val = np.empty(max_iter + 1, dtype=float)
  time_list= np.empty(max_iter + 1, dtype=float)
  x_k      = np.empty(max_iter + 1, dtype=float)

  cost_val[0] = cost(A, x0, z, b, alpha)
  time_list[0] = 0.0
  x_k[0] = np.linalg.norm(x-approx_sol)
  start_time = time.time()

  grad = grad_KL(A, AT, x, z,b)

  for i in range(max_iter):
    step_size = backtracking_linesearch(A,AT,z,b,f_KL, grad_KL, prox, x, alpha=alpha)
    x_hat = prox(x - step_size*grad, step_size*alpha)

    x_new = x_hat

    time_list[i+1] = time.time() - start_time
    x_k[i+1] = np.linalg.norm(x_new-approx_sol)
    cost_val[i+1] = cost(A, x_new, z, b, alpha)
    print('Iteration:', i, 'Cost:', cost_val[i+1])


    x = x_new

    grad_xnew = grad_KL(A, AT, x, z,b)
    # f_term =  f_KL(A, x, z, b)
    #stopping_criteria = np.linalg.norm(x - prox(x - step_size*grad_xnew, step_size*alpha))/ (1 + np.linalg.norm(x) + f_term)
    #if stopping_criteria < tol or abs(cost_val[i+1] - cost_val[i]) < tol:
    if abs(cost_val[i+1] - cost_val[i]) < tol:
        print(f'BT_Ista converge in {i} iteration')
        print('Iteration:', i, 'Cost:', cost_val[i+1])
        return cost_val[:i+2].tolist(), x, i, x_k[:i+2].tolist(), time_list[:i+2].tolist()

    grad = grad_xnew
  print(f'BT_Ista converge in {i} iteration')
  print('Iteration:', i, 'Cost:', cost_val[-1])
  return cost_val.tolist(), x, i, x_k.tolist(), time_list.tolist()

def BT_FISTA1(A,AT,b,x0, noisy_z,
                        alpha,max_iter, beta, newton_stepsize, tol, cost,
                        prox, subproblem_solver, approx_sol = 0):
    x = x0.copy()
    x_old = x0.copy()
    w = x0.copy()
    z = noisy_z.copy()

    x_k = np.empty(max_iter + 1, dtype=float)
    cost_val = np.empty(max_iter + 1, dtype=float)
    time_list= np.empty(max_iter + 1, dtype=float)

    x_k[0] = np.linalg.norm(x0-approx_sol)
    cost_val[0] = cost(A, x, z, b, alpha)
    time_list[0] = 0.0
    t = 1.0
    start_time = time.time()

    for i in range(max_iter):
        grad = grad_KL(A, AT, w, z,b)
        step_size = backtracking_linesearch(A,AT,z,b,f_KL, grad_KL, prox, x, alpha=alpha)

        x = prox(w - step_size*grad, alpha*step_size)
        x_k[i+1] = np.linalg.norm(x-approx_sol)
        t_old = t
        t = (1 + np.sqrt(1 + 4*(t_old**2)))/2
        w = x + ((t_old - 1) / t) * (x - x_old)
        cost_val[i+1] = cost(A, x, z, b, alpha)
        time_list[i+1] = time.time() - start_time

        # grad_xnew = grad_KL(A, AT, x, z,b)
        # f_term =  f_KL(A, x, z, b)
        #stopping_criteria = np.linalg.norm(x - prox(x - step_size*grad_xnew, step_size*alpha))/ (1 + np.linalg.norm(x) + f_term)
        #if stopping_criteria < tol or abs(cost_val[i+1] - cost_val[i]) < tol:
        if abs(cost_val[i+1] - cost_val[i]) < tol:
            print(f'Algo_BT_Fista converge in {i} iteration')
            return cost_val[:i+2].tolist(), x, i, x_k[:i+2].tolist(), time_list[:i+2].tolist()

        x_old = x
    print(f'Algo_BT_Fista converge in {i} iteration')
    return cost_val.tolist(), x, i, x_k.tolist(), time_list.tolist()


# --------------------------- Algo_Newton_BT_Ista ----------------------------

def Algo_Newton_BT_Ista(A,AT,b,x0, noisy_z,
                        alpha,max_iter, beta, newton_stepsize, tol, cost,
                        prox, subproblem_solver, newt_tol = 1e-3, approx_sol = 0):
  x = x0.copy()
  z = noisy_z.copy()

  cost_val = np.empty(max_iter + 1, dtype=float)
  time_list= np.empty(max_iter + 1, dtype=float)
  x_k      = np.empty(max_iter + 1, dtype=float)

  cost_val[0] = cost(A, x0, z, b, alpha)
  time_list[0] = 0.0
  x_k[0] = np.linalg.norm(x-approx_sol)
  start_time = time.time()
  do_newton = 0

  grad = grad_KL(A, AT, x, z,b)

  for i in range(max_iter):
    step_size = backtracking_linesearch(A,AT,z,b,f_KL, grad_KL, prox, x, alpha=alpha)
    x_hat = prox(x - step_size*grad, step_size*alpha)

    if do_newton:
      Gradient_map = (x-x_hat)/step_size
      y = Gradient_map - grad
      d = subproblem_solver(A,x_hat,y,b, alpha)
      newton_stepsize = 1
      x_new = x_hat - newton_stepsize*d
      if cost(A, x_new, z, b, alpha) >= cost(A, x_hat, z, b, alpha):
        x_new = x_hat
    else:
      x_new = x_hat

    time_list[i+1] = time.time() - start_time
    x_k[i+1] = np.linalg.norm(x_new-approx_sol)
    cost_val[i+1] = cost(A, x_new, z, b, alpha)
    print('Iteration:', i, 'Cost:', cost_val[i+1])

    if np.linalg.norm(x_new - x) < newt_tol: 
       do_newton = True
    else:
       do_newton = False

    x = x_new

    grad_xnew = grad_KL(A, AT, x, z,b)
    # f_term =  f_KL(A, x, z, b)
    #stopping_criteria = np.linalg.norm(x - prox(x - step_size*grad_xnew, step_size*alpha))/ (1 + np.linalg.norm(x) + f_term)
    #if stopping_criteria < tol or abs(cost_val[i+1] - cost_val[i]) < tol:
    if abs(cost_val[i+1] - cost_val[i]) < tol:
        print(f'Algo_Newton_BT_Ista converge in {i} iteration')
        print('Iteration:', i, 'Cost:', cost_val[i+1])
        return cost_val[:i+2].tolist(), x, i, x_k[:i+2].tolist(), time_list[:i+2].tolist()

    grad = grad_xnew
  print(f'Algo_Newton_BT_Ista converge in {i} iteration')
  print('Iteration:', i, 'Cost:', cost_val[-1])
  return cost_val.tolist(), x, i, x_k.tolist(), time_list.tolist()

# ------------------------ Algo_Newton_BT_Fista_new --------------------------

def Algo_Newton_BT_Fista_new(A,AT,b,x0, noisy_z,
                             alpha,max_iter, beta, newton_stepsize, tol, cost,
                             prox, subproblem_solver, newt_tol = 1e-3, approx_sol = 0):
  x = x0.copy()
  w = x0.copy()
  z = noisy_z.copy()
  x_hat = x0.copy()
  x_old = x0.copy()
  x_k = np.empty(max_iter + 1, dtype=float)


  t = 1.0
  t_old = 1.0
  cost_val = np.empty(max_iter + 1, dtype=float)
  time_list= np.empty(max_iter + 1, dtype=float)
  start_time = time.time()
  do_newton = 0

  x_k[0] = np.linalg.norm(x0-approx_sol)
  cost_val[0] = cost(A, x0, z, b, alpha)
  time_list[0] = 0.0

  for i in range(max_iter):
    grad = grad_KL(A, AT, w, z,b)
    step_size = backtracking_linesearch(A,AT,z,b,f_KL, grad_KL, prox, w, alpha=alpha)
    x_hat = prox(w - step_size*grad, alpha*step_size)

    if do_newton:
      Gradient_map = (w-x_hat)/step_size
      y = Gradient_map - grad
      #d = subproblem_solver(A,x_hat,y,b, alpha)
      ops_dict = {"A": A, "AT": AT, "z": noisy_z, "b": b, "lam": alpha}
      d = subproblem_solver(ops_dict, yk=x_hat, zk=y, b=b, alpha=alpha)
      newton_stepsize = 1
      x_new = x_hat - newton_stepsize*d
      if cost(A, x_new, z, b, alpha) >= cost(A, x_hat, z, b, alpha):
         x_new = x_hat
    else:
      x_new = x_hat

    time_list[i+1] = time.time() - start_time
    x_k[i+1] = np.linalg.norm(x_new-approx_sol)
    cost_val[i+1] = cost(A, x_new, z, b, alpha)
    print('Iteration:', i, 'Cost:', cost_val[i+1])
    

    if np.linalg.norm(x_new - w) < newt_tol: 
      do_newton = True
    else:
      do_newton = False

    x = x_new
    t = (1 + np.sqrt(1 + 3.9*(t_old**2)))/2
    w = x + ((t_old - 1) / t) * (x - x_old)
    x_old = x
    t_old = t

    # grad_xnew = grad_KL(A, AT, w, z,b)
    # f_term =  f_KL(A, x, z, b)
    #stopping_criteria = np.linalg.norm(x - prox(x - step_size*grad_xnew, step_size*alpha))/ (1 + np.linalg.norm(x) + f_term)
    #if stopping_criteria < tol or abs(cost_val[i+1] - cost_val[i]) < tol:
    if abs(cost_val[i+1] - cost_val[i]) < tol:
        print(f'Algo_Newton_BT_Fista converge in {i} iteration')
        print('Iteration:', i, 'Cost:', cost_val[i+1])
        return cost_val[:i+2].tolist(), x, i, x_k[:i+2].tolist(), time_list[:i+2].tolist()
  print(f'Algo_Newton_BT_Fista converge in {i} iteration')
  print('Iteration:', i, 'Cost:', cost_val[-1])
  return cost_val.tolist(), x, i, x_k.tolist(), time_list.tolist()

