import numpy as np
import time
from src.lasso.utils_lasso import *
#from ultils_TV import *


def _accept_damped_newton_step(
    A,
    b,
    alpha,
    cost,
    x_hat,
    d,
    beta,
    newton_stepsize,
    max_backtracks=25,
    min_step=1e-12,
):
  """Try a damped Newton correction.

  Returns
  -------
  x_new : np.ndarray
      Accepted iterate, or x_hat if no acceptable damped step is found.
  accepted : bool
      True iff a damped Newton correction was accepted.
  """
  if not np.all(np.isfinite(d)):
    return x_hat, False

  step = float(newton_stepsize)
  shrink = float(beta)
  cost_hat = cost(A, x_hat, b, alpha)
  if not np.isfinite(cost_hat):
    return x_hat, False

  for _ in range(int(max_backtracks)):
    if step <= float(min_step):
      break
    x_trial = x_hat - step * d
    if not np.all(np.isfinite(x_trial)):
      step *= shrink
      continue

    cost_trial = cost(A, x_trial, b, alpha)
    if np.isfinite(cost_trial) and (cost_trial < cost_hat):
      return x_trial, True
    step *= shrink

  return x_hat, False

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
                     prox, subproblem_solver, newt_tol = 1e-3, approx_sol = 0,
                     newton_trigger_steps=3, newton_reject_streak_trigger=2,
                     newton_reject_cooldown=8, max_newton_backtracks=25):
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
  close_count = 0
  reject_streak = 0
  newton_cooldown = 0

  r = A @ x - b
  grad = AT @ r

  for i in range(max_iter):
    x_hat = prox(x - step_size*grad, step_size*alpha)

    attempted_newton = False
    if do_newton:
      attempted_newton = True
      Gradient_map = (x-x_hat)/step_size
      y = Gradient_map - grad
      d = subproblem_solver(A,x_hat,y,b, alpha)
      x_new, accepted_newton = _accept_damped_newton_step(
          A, b, alpha, cost, x_hat, d, beta, newton_stepsize,
          max_backtracks=max_newton_backtracks
      )
      if accepted_newton:
        reject_streak = 0
      else:
        reject_streak += 1
        if reject_streak >= int(newton_reject_streak_trigger):
          newton_cooldown = int(newton_reject_cooldown)
          reject_streak = 0
    else:
      x_new = x_hat

    time_list[i+1] = time.time() - start_time
    x_k[i+1] = np.linalg.norm(x_new-approx_sol)
    cost_val[i+1] = cost(A,x_new,b,alpha)
    print('Iteration:', i, 'Cost:', cost_val[i+1])

    if attempted_newton:
      # Cooldown: after one Newton step, re-arm only after consecutive close
      # proximal iterations again.
      do_newton = False
      close_count = 0
    else:
      if newton_cooldown > 0:
        newton_cooldown -= 1
        close_count = 0
      else:
        if np.linalg.norm(x_new - x) < newt_tol:
          close_count += 1
        else:
          close_count = 0
        if close_count >= int(newton_trigger_steps):
          do_newton = True

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
                        prox, subproblem_solver, newt_tol = 1e-3, approx_sol = 0,
                        newton_trigger_steps=3, newton_reject_streak_trigger=2,
                        newton_reject_cooldown=8, max_newton_backtracks=25):
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
  close_count = 0
  reject_streak = 0
  newton_cooldown = 0

  r = A @ x - b
  grad = AT @ r

  for i in range(max_iter):
    step_size = backtracking_linesearch(A, b, x, grad, prox, alpha)
    x_hat = prox(x - step_size*grad, step_size*alpha)

    attempted_newton = False
    if do_newton:
      attempted_newton = True
      Gradient_map = (x-x_hat)/step_size
      y = Gradient_map - grad
      d = subproblem_solver(A,x_hat,y,b, alpha)
      x_new, accepted_newton = _accept_damped_newton_step(
          A, b, alpha, cost, x_hat, d, beta, newton_stepsize,
          max_backtracks=max_newton_backtracks
      )
      if accepted_newton:
        reject_streak = 0
      else:
        reject_streak += 1
        if reject_streak >= int(newton_reject_streak_trigger):
          newton_cooldown = int(newton_reject_cooldown)
          reject_streak = 0
    else:
      x_new = x_hat

    time_list[i+1] = time.time() - start_time
    x_k[i+1] = np.linalg.norm(x_new-approx_sol)
    cost_val[i+1] = cost(A,x_new,b,alpha)
    print('Iteration:', i, 'Cost:', cost_val[i+1])

    if attempted_newton:
      do_newton = False
      close_count = 0
    else:
      if newton_cooldown > 0:
        newton_cooldown -= 1
        close_count = 0
      else:
        if np.linalg.norm(x_new - x) < newt_tol:
          close_count += 1
        else:
          close_count = 0
        if close_count >= int(newton_trigger_steps):
          do_newton = True

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
                          prox, subproblem_solver, newt_tol = 1e-3, approx_sol = 0,
                          newton_trigger_steps=3, newton_reject_streak_trigger=2,
                          newton_reject_cooldown=8, max_newton_backtracks=25):
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
  close_count = 0
  reject_streak = 0
  newton_cooldown = 0

  x_k[0] = np.linalg.norm(x0-approx_sol)
  cost_val[0] = cost(A,x0,b,alpha)
  time_list[0] = 0.0

  for i in range(max_iter):
    grad = AT @ (A @ z - b)
    x_hat = prox(z - step_size*grad, alpha*step_size)
    attempted_newton = False
    accepted_newton = False

    if do_newton:
      attempted_newton = True
      Gradient_map = (z-x_hat)/step_size
      y = Gradient_map - grad
      d =  subproblem_solver(A,x_hat,y,b, alpha)
      x_new, accepted_newton = _accept_damped_newton_step(
          A, b, alpha, cost, x_hat, d, beta, newton_stepsize,
          max_backtracks=max_newton_backtracks
      )
      if accepted_newton:
        reject_streak = 0
      else:
        reject_streak += 1
        if reject_streak >= int(newton_reject_streak_trigger):
          newton_cooldown = int(newton_reject_cooldown)
          reject_streak = 0
    else:
      x_new = x_hat

    time_list[i+1] = time.time() - start_time
    x_k[i+1] = np.linalg.norm(x_new-approx_sol)
    cost_val[i+1] = cost(A,x_new,b,alpha)
    print('Iteration:', i, 'Cost:', cost_val[i+1])

    if attempted_newton:
      do_newton = False
      close_count = 0
    else:
      if newton_cooldown > 0:
        newton_cooldown -= 1
        close_count = 0
      else:
        # Use successive iterates for trigger consistency with ISTA-style rule.
        if np.linalg.norm(x_new - x) < newt_tol:
          close_count += 1
        else:
          close_count = 0
        if close_count >= int(newton_trigger_steps):
          do_newton = True

    x = x_new
    # Restart momentum when Newton step is effectively accepted or
    # when acceleration causes a temporary objective increase.
    if accepted_newton or (cost_val[i+1] > cost_val[i]):
      t = 1.0
      t_old = 1.0
      z = x.copy()
      x_old = x.copy()
    else:
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
                             prox, subproblem_solver, newt_tol = 1e-3, approx_sol = 0,
                             newton_trigger_steps=3, newton_reject_streak_trigger=2,
                             newton_reject_cooldown=8, max_newton_backtracks=25):
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
  close_count = 0
  reject_streak = 0
  newton_cooldown = 0

  x_k[0] = np.linalg.norm(x0-approx_sol)
  cost_val[0] = cost(A,x0,b,alpha)
  time_list[0] = 0.0

  for i in range(max_iter):
    grad = AT @ (A @ z - b)
    step_size = backtracking_linesearch(A, b, z, grad, prox, alpha)
    x_hat = prox(z - step_size*grad, alpha*step_size)
    attempted_newton = False
    accepted_newton = False

    if do_newton:
      attempted_newton = True
      Gradient_map = (z-x_hat)/step_size
      y = Gradient_map - grad
      d = subproblem_solver(A,x_hat,y,b, alpha)
      x_new, accepted_newton = _accept_damped_newton_step(
          A, b, alpha, cost, x_hat, d, beta, newton_stepsize,
          max_backtracks=max_newton_backtracks
      )
      if accepted_newton:
        reject_streak = 0
      else:
        reject_streak += 1
        if reject_streak >= int(newton_reject_streak_trigger):
          newton_cooldown = int(newton_reject_cooldown)
          reject_streak = 0
    else:
      x_new = x_hat

    time_list[i+1] = time.time() - start_time
    x_k[i+1] = np.linalg.norm(x_new-approx_sol)
    cost_val[i+1] = cost(A,x_new,b,alpha)
    print('Iteration:', i, 'Cost:', cost_val[i+1])

    if attempted_newton:
      do_newton = False
      close_count = 0
    else:
      if newton_cooldown > 0:
        newton_cooldown -= 1
        close_count = 0
      else:
        if np.linalg.norm(x_new - x) < newt_tol:
          close_count += 1
        else:
          close_count = 0
        if close_count >= int(newton_trigger_steps):
          do_newton = True

    x = x_new
    if accepted_newton or (cost_val[i+1] > cost_val[i]):
      t = 1.0
      t_old = 1.0
      z = x.copy()
      x_old = x.copy()
    else:
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
