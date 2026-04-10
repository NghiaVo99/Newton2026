import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt
from src.lasso.utils_lasso import *
from src.lasso.newton_lasso import *

# -----------------------------
# Problem setup
# -----------------------------
m, n = 48, 128
rng = np.random.default_rng(42)

z = np.zeros(n)
sparsity = 8
z[rng.choice(n, sparsity, replace=False)] = rng.normal(size=sparsity)

A = rng.normal(size=(m, n))
step_size = 1.0 / (np.linalg.norm(A, 2) ** 2)

beta, newton_stepsize = 0.5, 1.0
tol = 1e-9
newt_tol = 1e-3
b = A @ z
noise = rng.normal(scale=1e-3, size=m)
b_new = b + noise

alpha_c = 0.1
alpha = alpha_c * np.linalg.norm(A.T @ b_new, np.inf)
print('alpha', alpha)
max_iter = 1000
x0 = np.zeros(n)

prox = prox
subproblem_solver = sub_problem_of_lasso
# -----------------------------
# Solve (CVXPY reference) + run algorithms
# -----------------------------
approx_sol = solve_lasso_cvxpy(A, b_new, alpha)

cost_val_newton_ista, x1, i1, x_k1, time_k1 = Algo_Newton_Ista(
    A, b_new, x0, alpha, max_iter, step_size, beta, newton_stepsize, tol,
    prox, subproblem_solver, newt_tol, approx_sol)

cost_val_newton_fista, x2, i2, x_k2, time_k2 = Algo_Newton_Fista_new(
    A, b_new, x0, alpha, max_iter, step_size, beta, newton_stepsize, tol,
    prox, subproblem_solver, newt_tol, approx_sol)

cost_val_ista, x3, i3, x_k3, time_k3 = ISTA(
    A, b_new, x0, alpha, max_iter, step_size, tol, prox, approx_sol)

cost_val_fista, x4, i4, x_k4, time_k4 = FISTA1(
    A, b_new, x0, alpha, max_iter, step_size, tol, prox, approx_sol)

cost_val_newton_bt_ista, x5, i5, x_k5, time_k5 = Algo_Newton_BT_Ista(
    A, b_new, x0, alpha, max_iter, beta, newton_stepsize, tol,
    prox, subproblem_solver, newt_tol, approx_sol)

cost_val_newton_bt_fista, x6, i6, x_k6, time_k6 = Algo_Newton_BT_Fista_new(
    A, b_new, x0, alpha, max_iter, beta, newton_stepsize, tol,
    prox, subproblem_solver, newt_tol, approx_sol)

# rho = 1.0
# cost_val_admm, x_admm, i_admm, xk_admm, time_admm = ADMM_accel_lasso(
#     A, b_new, x0, alpha, max_iter=max_iter, rho=rho, tol=tol,
#     approx_sol=approx_sol, accel=False, restart=True, over_relax=1.2)


print(f'Algo_Newton_Ista converged in {i1} iterations')
print(f'Algo_Newton_Fista converged in {i2} iterations')
print(f'ISTA converged in {i3} iterations')
print(f'FISTA converged in {i4} iterations')

# Use the noisy b_new to match the problems solved above
optimal_cost = cost(A, approx_sol, b_new, alpha)

# -----------------------------
# Plotting
# -----------------------------
fig, axs = plt.subplots(1, 3, figsize=(22, 4))
width = 2

# Define a style dictionary for each method
series = {
    'Newton_ISTA':  {'color': 'r', 'marker': 'o', 'costs': cost_val_newton_ista, 'xerrs': x_k1, 'time': time_k1[-1]},
    'Newton_FISTA': {'color': 'm', 'marker': 's', 'costs': cost_val_newton_fista, 'xerrs': x_k2, 'time': time_k2[-1]},
    'ISTA':         {'color': 'b', 'marker': '^', 'costs': cost_val_ista,        'xerrs': x_k3, 'time': time_k3[-1]},
    'FISTA':        {'color': 'k', 'marker': 'D', 'costs': cost_val_fista,       'xerrs': x_k4, 'time': time_k4[-1]},
    'Newton_BT_ISTA':  {'color': 'c', 'marker': 'v', 'costs': cost_val_newton_bt_ista, 'xerrs': x_k5, 'time': time_k5[-1]},
    'Newton_BT_FISTA': {'color': 'y', 'marker': 'P', 'costs': cost_val_newton_bt_fista, 'xerrs': x_k6, 'time': time_k6[-1]},
}
# series['ADMM(accel)'] = {
#     'color': 'g', 'marker': 'x',
#     'costs': np.array(cost_val_admm),
#     'xerrs': np.array(xk_admm),
#     'time':  time_admm[-1]
# }

width = 2.0

# (0) Cost gap vs iterations
for name, s in series.items():
    axs[0].plot(np.abs(s['costs'] - optimal_cost), 
                label=name, 
                color=s['color'], 
                marker=s['marker'], 
                markevery=10,      # show marker every 30 points
                linewidth=width)
axs[0].set_yscale('log')
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel(r'$|f(x_k) - f(x^*)|$')
axs[0].set_ylim(bottom=1e-10)
axs[0].grid(True)

# (1) Norm error vs iterations
for name, s in series.items():
    axs[1].plot(s['xerrs'], 
                label=name, 
                color=s['color'], 
                marker=s['marker'], 
                markevery=10, 
                linewidth=width)
axs[1].set_yscale('log')
axs[1].set_xlabel('Iterations')
axs[1].set_ylabel(r'$\|x_k - x^*\|$')
axs[1].set_ylim(bottom=1e-10)
axs[1].grid(True)

# (2) Final runtimes, sorted (descending for visual emphasis)
methods  = list(series.keys())
times    = [series[m]['time'] for m in methods]
colors   = [series[m]['color'] for m in methods]
order    = np.argsort(times)[::-1]  # descending
axs[2].bar([methods[i] for i in order], [times[i] for i in order],
            color=[colors[i] for i in order])
axs[2].set_ylabel('Runtime (seconds)')
axs[2].set_yscale('log')
axs[2].set_xticks([])  # no xticks


handles, labels = axs[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    ncol=4,
    frameon=True,
    fontsize=10
)

plt.tight_layout(rect=[0, 0.05, 1, 1], pad=0.3)  # leave 5% space at bottom
plt.subplots_adjust(left=0.05,bottom=0.20, top=0.95, wspace=0.18)
plt.show()
