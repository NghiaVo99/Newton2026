import numpy as np
import matplotlib.pyplot as plt
import pyproximal
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.OSCAR.OSCAR_ultils_v1 import *
from src.OSCAR.OSCAR_algo import *
from src.OSCAR.SSNAL_OSCAR import *

# m, n = 150, 300  # problem dimensions
rng = np.random.default_rng(1)
# z = np.random.standard_normal(n)

# # Design matrix and noisy observations
# A = rng.standard_normal((m, n))
n = 500
A, b_new, x_true = build_test_problem(n=n, sigma2=0.01, rho=0.7, seed=1)

step_size = 1.0 / (np.linalg.norm(A, 2) ** 2)
#A = A / np.linalg.norm(A, 2)
beta, newton_stepsize = 0.5, 1.0
tol = 1e-8
newt_tol = 1e-3
print(np.linalg.norm(A.T @ b_new, np.inf))
#w1 = 1e-6 * np.linalg.norm(A.T @ b_new, np.inf)
w1 = 1e-5 * np.linalg.norm(A.T @ b_new, np.inf)
#w1 = 0.2
w2 = w1

max_iter = 1000
x0 = np.zeros(n)

prox = prox_oscar
subproblem_solver = sub_problem_oscar
cost = cost_oscar

# ────────────────────────────────────────────────────────────
# Benchmarks
# ────────────────────────────────────────────────────────────
approx_sol, optimal_cost = solve_oscar_gurobi(A, b_new, w1, w2)
#approx_sol = z

cost_val_newton_ista, x1, i1, x_k1, time_k1 = Algo_Newton_Ista(
    A, b_new, x0, w1, w2, max_iter, step_size, beta, newton_stepsize, tol, cost,
    prox, subproblem_solver, newt_tol, approx_sol)

cost_val_newton_fista, x2, i2, x_k2, time_k2 = Algo_Newton_Fista_new(
    A, b_new, x0, w1, w2, max_iter, step_size, beta, newton_stepsize, tol, cost,
    prox, subproblem_solver, newt_tol, approx_sol)

cost_val_ista, x3, i3, x_k3, time_k3 = ISTA(
    A, b_new, x0, w1, w2, max_iter, step_size, tol, cost, prox, approx_sol)

cost_val_fista, x4, i4, x_k4, time_k4 = FISTA1(
    A, b_new, x0, w1, w2, max_iter, step_size, tol, cost, prox, approx_sol)

cost_val_newton_bt_ista, x5, i5, x_k5, time_k5 = Algo_Newton_BT_Ista(
    A, b_new, x0, w1, w2, max_iter, beta, newton_stepsize, tol, cost,
     prox, subproblem_solver, newt_tol, approx_sol)

cost_val_newton_bt_fista, x6, i6, x_k6, time_k6 = Algo_Newton_BT_Fista_new(
    A, b_new, x0, w1, w2, max_iter,step_size, beta, newton_stepsize, tol, cost,
    prox, subproblem_solver, newt_tol, approx_sol)

# cost_val_fastADMM, x7, i7, x_k7, time_k7 = Algo_FastADMM_Lasso(
#     A, b_new, x0, alpha, max_iter, tol, cost,
#     prox, approx_sol)

solver = NewtALM_OSCAR(A, b_new, w1, w2, tol_kkt=tol, max_outer=max_iter, max_inner=100, x_ref = approx_sol)
x_hat, log = solver.solve(verbose=True)   # prints obj every iteration
cost_val_osnal = log['obj']
x_k7 = log['dist']
time_k7 = [log['total_time_sec']]

# print(cost_val_osnal)
# print(x_k7)


# Use the *noisy* problem for the optimal cost reference
#optimal_cost = cost_oscar(A, approx_sol, b_new, w1, w2)
# ────────────────────────────────────────────────────────────
# Series registry (add/remove methods here)
# ────────────────────────────────────────────────────────────
series = [
    {
        "name": "Newton_ISTA", "color": "r", "marker": "o",
        "costs": np.asarray(cost_val_newton_ista),
        "dists": np.asarray(x_k1),
        "times": np.asarray(time_k1),
    },
    {
        "name": "Newton_FISTA", "color": "m", "marker": "^",
        "costs": np.asarray(cost_val_newton_fista),
        "dists": np.asarray(x_k2),
        "times": np.asarray(time_k2),
    },
    {
        "name": "ISTA", "color": "b", "marker": "d",
        "costs": np.asarray(cost_val_ista),
        "dists": np.asarray(x_k3),
        "times": np.asarray(time_k3),
    },
    {
        "name": "FISTA", "color": "k", "marker": "v",
        "costs": np.asarray(cost_val_fista),
        "dists": np.asarray(x_k4),
        "times": np.asarray(time_k4),
    },
    {
        "name": "Newton_BT_ISTA", "color": "c", "marker": "<",
        "costs": np.asarray(cost_val_newton_bt_ista),
        "dists": np.asarray(x_k5),
        "times": np.asarray(time_k5),
    },
    {
        "name": "Newton_BT_FISTA", "color": "y", "marker": "P",
        "costs": np.asarray(cost_val_newton_bt_fista),
        "dists": np.asarray(x_k6),
        "times": np.asarray(time_k6),
    },
    {
        "name": "SSNAL", "color": "saddlebrown", "marker": "X",
        "costs": np.asarray(cost_val_osnal),
        "dists": np.asarray(x_k7),
        "times": np.asarray(time_k7),
    }

]

# ────────────────────────────────────────────────────────────
# Figure: 2-row layout (bottom row only for legend)
# ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 4))
gs  = fig.add_gridspec(2, 3, height_ratios=[20, 1], hspace=0.25, wspace=0.25)
axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
ax_leg = fig.add_subplot(gs[1, :])  # legend axis (spans all columns)
ax_leg.axis('off')

linewidth = 2.5
mark_every_cost = 10
mark_every_dist = 10
mark_every_time = 10

# tmp=[]
# tmp += [min(s["costs"]) for s in series]
# print(len(tmp))
# optimal_cost = np.min(np.array(tmp))
# ────────────────────────────────────────────────────────────
# 2) PLOT EACH METHOD: objective gap vs iterations + dist vs iterations
# ────────────────────────────────────────────────────────────
for s in series:
    # (a) objective gap vs iterations
    axs[0].plot(
        s["costs"] - optimal_cost,
        color=s["color"], linestyle='-', linewidth=linewidth,
        marker=s["marker"], markevery=mark_every_cost, markersize=6,
        label=s["name"]
    )

    # (b) distance vs iterations
    axs[1].plot(
        s["dists"],
        color=s["color"], linestyle='-', linewidth=linewidth,
        marker=s["marker"], markevery=mark_every_dist, markersize=6
    )

axs[0].set_yscale('log')
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel(r'$|\varphi(x_k) - \varphi(\overline{x})|$')
#axs[0].set_ylim(top=0,bottom=1e-1)
axs[0].grid(True)

axs[1].set_yscale('log')
axs[1].set_xlabel('Iterations')
axs[1].set_ylabel(r'$\|x_k - \overline{x}\|$')
#axs[1].set_ylim(top=0,bottom=1e-1)
axs[1].grid(True)
# ────────────────────────────────────────────────────────────
# 3) Replace time plot with horizontal bar chart of final runtimes
# ────────────────────────────────────────────────────────────
methods = [s["name"] for s in series]
final_times = [s["times"][-1] for s in series]
colors = [s["color"] for s in series]

# Sort by runtime (shortest at top)
order = np.argsort(final_times)[::-1]  # descending
methods_sorted = [methods[i] for i in order]
times_sorted   = [final_times[i] for i in order]
colors_sorted  = [colors[i] for i in order]

axs[2].bar(methods_sorted, times_sorted, color=colors_sorted)
axs[2].set_yscale('log')
axs[2].set_ylabel("log(Runtime)")        
axs[2].set_xticks("")            
#axs[2].tick_params(axis='x', labelrotation=45)

num_methods = len(methods)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    ncol=num_methods,
    frameon=True,
    fontsize=10
)

plt.tight_layout(rect=[0, 0.05, 1, 1], pad=0.2)  # leave space at bottom
plt.subplots_adjust(left=0.05,bottom=0.1, top=0.95, wspace=0.05)
plt.show()
