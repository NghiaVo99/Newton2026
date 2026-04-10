import numpy as np
from src.OSCAR.SSNAL_OSCAR import *
import matplotlib.pyplot as plt
from src.OSCAR.OSCAR_ultils_v1 import build_test_problem, solve_oscar_gurobi

n = 500
A, b_new, x_true = build_test_problem(n=n, sigma2=0.01, rho=0.7, seed=10)

step_size = 1.0 / (np.linalg.norm(A, 2) ** 2)
beta, newton_stepsize = 0.5, 1.0
tol = 1e-8
newt_tol = 4e-5
print(np.linalg.norm(A.T @ b_new, np.inf))
#w1 = 1e-6 * np.linalg.norm(A.T @ b_new, np.inf)
w1 = 1e-5 * np.linalg.norm(A.T @ b_new, np.inf)
#w1 = 0.2
w2 = 0.5*w1

max_iter = 100
x0 = np.zeros(n)


# ────────────────────────────────────────────────────────────
# Benchmarks
# ────────────────────────────────────────────────────────────
approx_sol, optimal_cost = solve_oscar_gurobi(A, b_new, w1, w2)

solver = NewtALM_OSCAR(A, b_new, w1, w2, tol_kkt=tol, max_outer=max_iter, max_inner=80, x_ref = approx_sol)
x_hat, log = solver.solve()   # prints obj every iteration
cost_val_osnal = log['obj']
x_k7 = log['dist']
time_k7 = [log['total_time_sec']]

print(cost_val_osnal- optimal_cost)
print(x_k7)

# print(type(cost_val_osnal), type(x_k7['dist']), type(time_k7))
plt.figure(figsize=(10, 6))
plt.plot(cost_val_osnal - optimal_cost, label='SSNAL-OSCAR')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Objective Value')


plt.figure(figsize=(10, 6))
plt.plot(x_k7, label='SSNAL-OSCAR')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Distance to Approximate Solution')

plt.show()



