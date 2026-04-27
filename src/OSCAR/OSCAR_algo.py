import numpy as np
import time
from src.OSCAR.OSCAR_ultils_v1 import *


def _accept_damped_newton_step(
    A,
    b,
    w1,
    w2,
    cost,
    x_hat,
    d,
    beta,
    newton_stepsize,
    max_backtracks=25,
    min_step=1e-12,
):
  """Try a damped Newton correction and keep the prox point on rejection."""
  if not np.all(np.isfinite(d)):
    return x_hat, False

  step = float(newton_stepsize)
  shrink = float(beta)
  cost_hat = cost(A, x_hat, b, w1, w2)
  if not np.isfinite(cost_hat):
    return x_hat, False

  for _ in range(int(max_backtracks)):
    if step <= float(min_step):
      break

    x_trial = x_hat - step * d
    if not np.all(np.isfinite(x_trial)):
      step *= shrink
      continue

    cost_trial = cost(A, x_trial, b, w1, w2)
    if np.isfinite(cost_trial) and cost_trial < cost_hat:
      return x_trial, True

    step *= shrink

  return x_hat, False


def ISTA(A,b,x0,w1,w2,max_iter, step_size, tol, cost, prox, approx_sol = 0):
  x = x0
  cost_val = [cost(A,x0,b,w1,w2)]
  #norm_sol = np.linalg.norm(approx_sol)
  x_k = [np.linalg.norm(x0-approx_sol)]
  time_list = [0]
  start_time = time.time()
  for i in range(max_iter):
    
    #x_old = x.copy()
    grad = grad_f(A,x,b)
    # x = Prox_func(x - step_size*grad, step_size*alpha)
    x = prox(x - step_size*grad, step_size, w1, w2, positive = False)

    time_list.append(time.time() - start_time)
    x_k.append(np.linalg.norm(x-approx_sol))
    cost_val.append(cost(A,x,b,w1,w2))
    #print('Iteration:', i, 'Cost:', cost_val[-1])
    
    stopping_criteria = np.linalg.norm(x - prox(x - step_size*grad, step_size,w1,w2,positive = False))/ (1 + np.linalg.norm(x) + np.linalg.norm(grad))
    if stopping_criteria < tol:
    #if abs(cost_val[-1] - optim_cost) < tol:# or abs(cost_val[-1] - cost_val[-2]) < tol:
        print(f'Algo_Ista converge in {i} iteration')
        return cost_val, x, i, x_k, time_list
  print(f'Algo_Ista converge in {i} iteration')
  return cost_val, x, i, x_k, time_list

def BT_ISTA(A,b,x0,w1,w2,max_iter, tol, cost, prox, approx_sol = 0):
  x = x0
  cost_val = [cost(A,x0,b,w1,w2)]
  time_list = [0]
  x_k = [np.linalg.norm(x0-approx_sol)]
  start_time = time.time()
  for i in range(max_iter):

    grad = grad_f(A,x,b)
    step_size, x = backtracking_linesearch(
        A, b, x, grad, prox, w1, w2, return_candidate=True
    )

    x_k.append(np.linalg.norm(x-approx_sol))
    cost_val.append(cost(A,x,b,w1,w2))
    time_list.append(time.time() - start_time)
    #print('Iteration:', i, 'Cost:', cost_val[-1])

    stopping_criteria = np.linalg.norm(x - prox(x - step_size*grad, step_size,w1,w2,positive = False))/ (1 + np.linalg.norm(x) + np.linalg.norm(grad))
    if stopping_criteria < tol:
    #if abs(cost_val[-1] - optim_cost) < tol: # or abs(cost_val[-1] - cost_val[-2]) < tol:
        print(f'Algo_BT_Ista converge in {i} iteration')
        print(f'optim_cost: {cost_val[-1]}')
        return cost_val, x, i, x_k, time_list
  print(f'Algo_BT_Ista converge in {i} iteration')
  return cost_val, x, i, x_k, time_list


def FISTA1(A, b, x0, w1, w2, max_iter, step_size, tol, cost, prox, approx_sol = 0):
    x = x0
    x_old = x0
    z = x0
    x_k = [np.linalg.norm(x0-approx_sol)]
    cost_val = [cost(A,x0,b,w1,w2)]
    time_list= [0]
    t = 1
    start_time = time.time()
    for i in range(max_iter):

        grad = grad_f(A,z,b)
        #x = Prox_func(z - step_size*grad, alpha*step_size)
        x = prox(z - step_size*grad, step_size, w1,w2)
        
        t_old = t
        t = (1 + np.sqrt(1 + 4*(t_old**2)))/2
        z = x + ((t_old - 1) / t) * (x - x_old)

        time_list.append(time.time() - start_time)
        x_k.append(np.linalg.norm(x-approx_sol))
        cost_val.append(cost(A,x,b,w1,w2))
        #print('Iteration:', i, 'Cost:', cost_val[-1])
        stopping_criteria = np.linalg.norm(x - prox(x - step_size*grad, step_size,w1,w2,positive = False))/ (1 + np.linalg.norm(x) + np.linalg.norm(grad))
        if stopping_criteria < tol:
        #if abs(cost_val[-1] - optim_cost) < tol:# or abs(cost_val[-1] - cost_val[-2]) < tol:
            print(f'Algo_Fista converge in {i} iteration')
            print(f'optim_cost: {cost_val[-1]}')
            return cost_val, x, i, x_k, time_list
        x_old = x
    print(f'Algo_Fista converge in {i} iteration')
    return cost_val, x, i, x_k, time_list

def BT_FISTA1(A, b, x0, w1, w2, max_iter, tol, cost, prox, approx_sol = 0):
    x = x0
    x_old = x0
    z = x0
    #norm_sol = np.linalg.norm(approx_sol)
    x_k = [np.linalg.norm(x0-approx_sol)]
    cost_val = [cost(A,x0,b,w1,w2)]
    time_list= []
    t = 1
    start_time = time.time()
    for i in range(max_iter):

        grad = grad_f(A,z,b)
        step_size, x = backtracking_linesearch(
            A, b, z, grad, prox, w1, w2, return_candidate=True
        )
        x_k.append(np.linalg.norm(x-approx_sol))
        t_old = t
        t = (1 + np.sqrt(1 + 4*(t_old**2)))/2
        z = x + ((t_old - 1) / t) * (x - x_old)

        cost_val.append(cost(A,x,b,w1,w2))
        time_list.append(time.time() - start_time)
        stopping_criteria = np.linalg.norm(x - prox(x - step_size*grad, step_size,w1,w2,positive = False))/ (1 + np.linalg.norm(x) + np.linalg.norm(grad))
        if stopping_criteria < tol:
        #if abs(cost(A,x,b,alpha) - optim_cost) < tol or abs(cost_val[-1] - cost_val[-2]) < tol:
            print(f'Algo_BT_Fista converge in {i} iteration')
            print(f'optim_cost: {cost_val[-1]}')
            return cost_val, x, i, x_k, time_list

        x_old = x
    print(f'Algo_BT_Fista converge in {i} iteration')
    return cost_val, x, i, x_k, time_list


def Algo_Newton_Ista(A,b,x0,w1,w2,max_iter, step_size, beta, newton_stepsize, tol, cost,
                     prox, subproblem_solver, newt_tol = 1e-3, approx_sol = 0,
                     verbose=False, newton_trigger_steps=3,
                     newton_reject_streak_trigger=2,
                     newton_reject_cooldown=8, max_newton_backtracks=25):
  x = x0
  cost_val = [cost(A,x0,b,w1,w2)]
  time_list = [0]
  x_k = [np.linalg.norm(x0-approx_sol)]
  start_time = time.time()
  do_newton = 0
  close_count = 0
  reject_streak = 0
  newton_cooldown = 0
  grad = grad_f(A,x,b)
  for i in range(max_iter):
    
    x_hat = prox(x - step_size*grad, step_size,w1,w2, positive = False)
    #x_hat_k.append(np.linalg.norm(x_hat-approx_sol)/norm_sol)
    Gradient_map = (x-x_hat)/step_size
    x_new = x_hat
    new_cost = cost(A, x_hat, b, w1, w2)

    attempted_newton = False
    if do_newton:
      attempted_newton = True
  
      y = Gradient_map - grad
      d = subproblem_solver(A,x_hat,y,b, w1, w2)
      if verbose:
          print('norm_d:', np.linalg.norm(d))
      x_new, accepted_newton = _accept_damped_newton_step(
          A, b, w1, w2, cost, x_hat, d, beta, newton_stepsize,
          max_backtracks=max_newton_backtracks
      )
      # do_newton = 0
      # if np.linalg.norm(d) <= 1e-4:
      #   time_list.append(time.time() - start_time)
      #   x_k.append(np.linalg.norm(x_new-approx_sol))
      #   cost_val.append(cost(A,x_new,b,w1,w2))
      #   return cost_val, x_new, i, x_k, time_list
      new_cost = cost(A, x_new, b, w1, w2)
      if accepted_newton:
          reject_streak = 0
      else:
          reject_streak += 1
          if reject_streak >= int(newton_reject_streak_trigger):
              newton_cooldown = int(newton_reject_cooldown)
              reject_streak = 0

    time_list.append(time.time() - start_time)
    x_k.append(np.linalg.norm(x_new-approx_sol))
    cost_val.append(new_cost)
    #print('Iteration:', i, 'Cost:', cost_val[-1])    

    if attempted_newton:
       do_newton = False
       close_count = 0
    elif newton_cooldown > 0:
       newton_cooldown -= 1
       close_count = 0
       do_newton = False
    else:
       if np.linalg.norm(x_new - x) < newt_tol:
          close_count += 1
       else:
          close_count = 0
       do_newton = close_count >= int(newton_trigger_steps)
    
    x = x_new
    grad_xnew = grad_f(A,x_new,b)
    if tol >= 0:
      stopping_criteria = np.linalg.norm(x_new - prox(x_new - step_size*grad_xnew, step_size,w1,w2,positive = False))/ (1 + np.linalg.norm(x_new) + np.linalg.norm(grad_xnew))
      if stopping_criteria < tol:
      #if abs(cost_val[-1] - optim_cost) < tol: # or abs(cost_val[-1] - cost_val[-2]) < tol:
          if verbose:
              print(f'Algo_Newton_Ista converge in {i} iteration')
              print(f'optim_cost: {cost_val[-1]}')
          return cost_val, x, i, x_k, time_list

    grad = grad_xnew
    
  if verbose:
      print(f'Algo_Newton_Ista converge in {i} iteration')
  return cost_val, x, i, x_k, time_list


def Algo_Newton_BT_Ista(A,b,x0,w1,w2,max_iter, beta, newton_stepsize, tol, cost,
                        prox, subproblem_solver, newt_tol = 1e-3, approx_sol = 0,
                        verbose=False, newton_trigger_steps=3,
                        newton_reject_streak_trigger=2,
                        newton_reject_cooldown=8, max_newton_backtracks=25):
  x = x0
  cost_val = [cost(A,x0,b,w1,w2)]
  time_list = [0]
  x_k = [np.linalg.norm(x0-approx_sol)]
  start_time = time.time()
  do_newton = 0
  close_count = 0
  reject_streak = 0
  newton_cooldown = 0
  grad = grad_f(A,x,b)
  for i in range(max_iter):
    
    step_size, x_hat = backtracking_linesearch(
        A, b, x, grad, prox, w1, w2, return_candidate=True
    )
    Gradient_map = (x-x_hat)/step_size

    x_new = x_hat
    new_cost = cost(A, x_hat, b, w1, w2)

    attempted_newton = False
    if do_newton:
      attempted_newton = True
      
      y = Gradient_map - grad
      d = subproblem_solver(A,x_hat,y,b, w1, w2)
      if verbose:
          print('norm_d:', np.linalg.norm(d))

      #d_norm.append(np.linalg.norm(d))
      x_new, accepted_newton = _accept_damped_newton_step(
          A, b, w1, w2, cost, x_hat, d, beta, newton_stepsize,
          max_backtracks=max_newton_backtracks
      )
      #do_newton = 0
      # if np.linalg.norm(d) <= 1e-4:
      #   time_list.append(time.time() - start_time)
      #   x_k.append(np.linalg.norm(x_new-approx_sol))
      #   cost_val.append(cost(A,x_new,b,w1,w2))
      #   return cost_val, x_new, i, x_k, time_list
      new_cost = cost(A, x_new, b, w1, w2)
      if accepted_newton:
          reject_streak = 0
      else:
          reject_streak += 1
          if reject_streak >= int(newton_reject_streak_trigger):
              newton_cooldown = int(newton_reject_cooldown)
              reject_streak = 0

    time_list.append(time.time() - start_time)
    x_k.append(np.linalg.norm(x_new-approx_sol))
    cost_val.append(new_cost)
    #print('Iteration:', i, 'Cost:', cost_val[-1])    

    if attempted_newton:
       do_newton = False
       close_count = 0
    elif newton_cooldown > 0:
       newton_cooldown -= 1
       close_count = 0
       do_newton = False
    else:
       if np.linalg.norm(x_new - x) < newt_tol:
          close_count += 1
       else:
          close_count = 0
       do_newton = close_count >= int(newton_trigger_steps)
    
    x = x_new
    grad_xnew = grad_f(A,x_new,b)
    if tol >= 0:
      stopping_criteria = np.linalg.norm(x_new - prox(x_new - step_size*grad_xnew, step_size,w1,w2,positive = False))/ (1 + np.linalg.norm(x_new) + np.linalg.norm(grad_xnew))
      if stopping_criteria < tol:
      #if abs(cost_val[-1] - optim_cost) < tol: # or abs(cost_val[-1] - cost_val[-2]) < tol:
          if verbose:
              print(f'Algo_Newton_BT_Ista converge in {i} iteration')
              print(f'optim_cost: {cost_val[-1]}')
          return cost_val, x, i, x_k, time_list

    grad = grad_xnew
    
  if verbose:
      print(f'Algo_Newton_BT_Ista converge in {i} iteration')
  return cost_val, x, i, x_k, time_list



def Algo_Newton_Fista_new(A,b,x0,w1,w2,max_iter, step_size, beta, newton_stepsize, tol, cost,
                             prox, subproblem_solver, newt_tol = 1e-3, approx_sol = 0,
                             verbose=False, newton_trigger_steps=3,
                             newton_reject_streak_trigger=2,
                             newton_reject_cooldown=8, max_newton_backtracks=25):
    x = x0
    z = x.copy()
    x_hat = x0
    x_old = x0
    x_k = [np.linalg.norm(x0-approx_sol)]

    t = 1
    t_old = 1
    cost_val = [cost(A,x0,b,w1,w2)]
    time_list = [0]
    start_time = time.time()
    do_newton = 0
    close_count = 0
    reject_streak = 0
    newton_cooldown = 0
    
    for i in range(max_iter):

        # FISTA + backtracking step
        grad = grad_f(A, z, b)
        #print('iteration:', i, 'stepsize:', step_size)
        x_hat = prox(z - step_size*grad, step_size, w1, w2, positive=False)
        Gradient_map = (z - x_hat) / step_size

        # Start from pure FISTA step
        x_new = x_hat
        new_cost = cost(A, x_hat, b, w1, w2)

        attempted_newton = False
        accepted_newton = False
        if do_newton:
            attempted_newton = True
            # Newton refinement around x_hat
            y = Gradient_map - grad
            d = subproblem_solver(A, x_hat, y, b, w1, w2)
            if verbose:
                print('norm_d:', np.linalg.norm(d))

            x_new, accepted_newton = _accept_damped_newton_step(
                A, b, w1, w2, cost, x_hat, d, beta, newton_stepsize,
                max_backtracks=max_newton_backtracks
            )
            new_cost = cost(A, x_new, b, w1, w2)
            if accepted_newton:
                reject_streak = 0
            else:
                reject_streak += 1
                if reject_streak >= int(newton_reject_streak_trigger):
                    newton_cooldown = int(newton_reject_cooldown)
                    reject_streak = 0

        time_list.append(time.time() - start_time)
        x_k.append(np.linalg.norm(x_new - approx_sol))
        cost_val.append(new_cost)

        # Decide whether to enable Newton next iteration
        if attempted_newton:
            do_newton = False
            close_count = 0
        elif newton_cooldown > 0:
            newton_cooldown -= 1
            close_count = 0
            do_newton = False
        else:
            if np.linalg.norm(x_new - x) < newt_tol:
                close_count += 1
            else:
                close_count = 0
            do_newton = close_count >= int(newton_trigger_steps)

        # FISTA extrapolation
        x = x_new
        if accepted_newton or (cost_val[-1] > cost_val[-2]):
            t = 1
            t_old = 1
            z = x.copy()
            x_old = x.copy()
        else:
            t = (0.99 + np.sqrt(1 + 4*(t_old**2))) / 2
            z = x + ((t_old - 1) / t) * (x - x_old)
            x_old = x
            t_old = t

        # Stopping criterion
        if tol >= 0:
            grad_xnew = grad_f(A, x_new, b)
            stopping_criteria = np.linalg.norm(
                x - prox(x - step_size * grad_xnew, step_size, w1, w2, positive=False)
            ) / (1 + np.linalg.norm(x_new) + np.linalg.norm(grad_xnew))

            if stopping_criteria < tol:
                if verbose:
                    print(f'Algo_Newton_Fista converge in {i} iteration')
                    print(f'optim_cost: {new_cost}')
                return cost_val, x, i, x_k, time_list
    
    if verbose:
        print(f'Algo_Newton_Fista converge in {i} iteration')
    return cost_val, x, i, x_k, time_list



def Algo_Newton_BT_Fista_new(A,b,x0,w1,w2,max_iter, step_size, beta, newton_stepsize, tol, cost,
                             prox, subproblem_solver, newt_tol = 1e-3, approx_sol = 0,
                             verbose=False, newton_trigger_steps=3,
                             newton_reject_streak_trigger=2,
                             newton_reject_cooldown=8, max_newton_backtracks=25):
    x = x0
    z = x.copy()
    x_hat = x0
    x_old = x0
    x_k = [np.linalg.norm(x0-approx_sol)]

    t = 1
    t_old = 1
    cost_val = [cost(A,x0,b,w1,w2)]
    time_list = [0]
    start_time = time.time()
    do_newton = 0
    close_count = 0
    reject_streak = 0
    newton_cooldown = 0
    
    for i in range(max_iter):

        # FISTA + backtracking step
        grad = grad_f(A, z, b)
        step_size, x_hat = backtracking_linesearch(
            A, b, z, grad, prox, w1, w2, return_candidate=True
        )
        #print('iteration:', i, 'stepsize:', step_size)
        Gradient_map = (z - x_hat) / step_size

        # Start from pure FISTA step
        x_new = x_hat
        new_cost = cost(A, x_hat, b, w1, w2)

        attempted_newton = False
        accepted_newton = False
        if do_newton:
            attempted_newton = True
            # Newton refinement around x_hat
            y = Gradient_map - grad
            d = subproblem_solver(A, x_hat, y, b, w1, w2)
            if verbose:
                print('norm_d:', np.linalg.norm(d))

            x_new, accepted_newton = _accept_damped_newton_step(
                A, b, w1, w2, cost, x_hat, d, beta, newton_stepsize,
                max_backtracks=max_newton_backtracks
            )
            new_cost = cost(A, x_new, b, w1, w2)
            if accepted_newton:
                reject_streak = 0
            else:
                reject_streak += 1
                if reject_streak >= int(newton_reject_streak_trigger):
                    newton_cooldown = int(newton_reject_cooldown)
                    reject_streak = 0

        time_list.append(time.time() - start_time)
        x_k.append(np.linalg.norm(x_new - approx_sol))
        cost_val.append(new_cost)

        # Decide whether to enable Newton next iteration
        if attempted_newton:
            do_newton = False
            close_count = 0
        elif newton_cooldown > 0:
            newton_cooldown -= 1
            close_count = 0
            do_newton = False
        else:
            if np.linalg.norm(x_new - x) < newt_tol:
                close_count += 1
            else:
                close_count = 0
            do_newton = close_count >= int(newton_trigger_steps)

        # FISTA extrapolation
        x = x_new
        if accepted_newton or (cost_val[-1] > cost_val[-2]):
            t = 1
            t_old = 1
            z = x.copy()
            x_old = x.copy()
        else:
            t = (0.99 + np.sqrt(1 + 4*(t_old**2))) / 2
            z = x + ((t_old - 1) / t) * (x - x_old)
            x_old = x
            t_old = t

        # Stopping criterion
        if tol >= 0:
            grad_xnew = grad_f(A, x_new, b)
            stopping_criteria = np.linalg.norm(
                x - prox(x - step_size * grad_xnew, step_size, w1, w2, positive=False)
            ) / (1 + np.linalg.norm(x_new) + np.linalg.norm(grad_xnew))

            if stopping_criteria < tol:
                if verbose:
                    print(f'Algo_Newton_BT_Fista converge in {i} iteration')
                    print(f'optim_cost: {new_cost}')
                return cost_val, x, i, x_k, time_list
    
    if verbose:
        print(f'Algo_Newton_BT_Fista converge in {i} iteration')
    return cost_val, x, i, x_k, time_list



def Algo_FastADMM_Lasso(A,b,x0,alpha,max_iter, tol, cost,
                             prox, approx_sol = 0):
    """
    Fast ADMM with Restart (Goldstein et al., Alg. 8) specialized to:
        min_x 0.5*||A x - b||^2 + alpha*||x||_1

    Signature & returns match your Algo_* functions:
      returns: cost_val, x, iters, x_k, time_list
    """
    m, n = A.shape
    AT = A.T
    ATb = AT @ b

    # Heuristic penalty; works well in practice. You can tune if desired.
    # L = ||A||_2^2
    smax = np.linalg.svd(A, compute_uv=False)[0]
    L = (smax**2) if smax is not None else np.linalg.norm(A, 2)**2
    tau = 0.5 * L + 1e-8
    eta = 0.999  # restart threshold from Alg. 8

    # Pre-factorize (A^T A + tau I)
    M = AT @ A + tau * np.eye(n)
    try:
        Lc = np.linalg.cholesky(M)
        def solve_M(rhs):
            y = np.linalg.solve(Lc, rhs)
            return np.linalg.solve(Lc.T, y)
    except np.linalg.LinAlgError:
        def solve_M(rhs):
            return np.linalg.solve(M, rhs)

    # Initialize
    x = x0.copy()
    v = x0.copy()               # ADMM "v" (will be the returned x)
    lam = np.zeros_like(v)      # Lagrange multiplier (scaled)
    v_hat = v.copy()
    lam_hat = lam.copy()

    v_prev = v.copy()
    lam_prev = lam.copy()

    alpha_k = 1.0
    c_prev = np.inf

    cost_val = [cost(A, x0, b, alpha)]
    x_k = [np.linalg.norm(x0 - approx_sol)]
    time_list = [0.0]
    start_time = time.time()

    # Main loop
    for i in range(max_iter):
        # u-update: (A^T A + tau I) u = A^T b + tau v_hat + lam_hat
        rhs = ATb + tau * v_hat + lam_hat
        u = solve_M(rhs)

        # v-update: prox of alpha||.||_1
        v_new = prox(u - lam_hat / tau, alpha / tau)

        # lambda-update
        lam_new = lam_hat + tau * (v_new - u)

        # Combined residual (Alg. 8): ck = (1/tau)||λ-λ̂||^2 + tau||v-v̂||^2
        c_k = (1.0 / tau) * np.sum((lam_new - lam_hat)**2) + tau * np.sum((v_new - v_hat)**2)

        # Restart test
        if c_k < eta * c_prev:
            # accelerate
            alpha_kp1 = (1.0 + np.sqrt(1.0 + 4.0 * alpha_k**2)) / 2.0
            gamma = (alpha_k - 1.0) / alpha_kp1
            v_hat_new = v_new + gamma * (v_new - v_prev)
            lam_hat_new = lam_new + gamma * (lam_new - lam_prev)
        else:
            # restart
            alpha_kp1 = 1.0
            v_hat_new = v_prev.copy()
            lam_hat_new = lam_prev.copy()
            # keep monotonicity per Alg. 8
            c_k = (1.0 / eta) * c_prev

        # Book-keeping for your harness
        x = v_new
        time_list.append(time.time() - start_time)
        x_k.append(np.linalg.norm(x - approx_sol))
        cost_val.append(cost(A, x, b, alpha))

        # Stopping: use combined residual (Alg. 8)
        if c_k <= tol:
            print(f'Alg8-FastADMM converged in {i} iterations (ck={c_k:.3e}).')
            return cost_val, x, i, x_k, time_list

        # Next iter
        v_prev, v = v.copy(), v_new
        lam_prev, lam = lam.copy(), lam_new
        v_hat, lam_hat = v_hat_new, lam_hat_new
        alpha_k = alpha_kp1
        c_prev = c_k

    print(f'Algo-FastADMM reached max_iter={max_iter} (ck={c_prev:.3e}).')
    return cost_val, x, max_iter, x_k, time_list
