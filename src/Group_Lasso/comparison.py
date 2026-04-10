import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.io import loadmat

from src.Group_Lasso.utils_group_lasso import *
from src.Group_Lasso.newton_group_lasso import *


# ────────────────────────────────────────────────────────────
# Synthetic Problem setup
# ────────────────────────────────────────────────────────────
m, n = 200, 600
rng = np.random.default_rng(42)

beta, newton_stepsize = 0.5, 1.0
tol = 1e-6
newt_tol = 1e-3
x0 = np.zeros((n,))

# Problem dimensions and group structure

num_groups = 60                         # number of groups
group_len = n // num_groups                # group size (assume divisible)
print("Group length:", group_len)
groups_dict_idx = make_groups_dict(n, group_len)   # <-- build from group_len

# Design matrix
A = rng.normal(size=(m, n))
A = A / np.linalg.norm(A, 2)    # normalize A
step_size = 1.0 / (np.linalg.norm(A, 2) ** 2)

# Ground-truth x with a few active groups
num_active = 4
true_active = np.random.choice(np.arange(1, num_groups), size=num_active, replace=False).tolist()
#print("True active groups:", true_active)
x_true = np.zeros(n)
for gi in true_active:
    idx = groups_dict_idx[gi]
    #print("Active group index:", gi, "-> variable indices:", idx)
    block = rng.normal(size=(idx.stop - idx.start))
    #print(block)
    #print(idx.stop)
    #print(idx.start)
    x_true[idx] = block
#print("True x:", x_true)

# Observations with small noise
noise = 0.001 * rng.normal(size=m)
b_new = A @ x_true + noise

# alpha_c = 1e-3
# alpha = alpha_c * np.linalg.norm(A.T @ b_new, np.inf)
alpha = 0.001
max_iter = 300


prox = proxL1_L2
subproblem_solver = sub_problem_of_group_lasso_new
cost = cost_group_lasso

# ────────────────────────────────────────────────────────────
# Benchmarks
# ────────────────────────────────────────────────────────────
# approx_sol, optimal_cost = solve_lasso_gurobi(A, b_new, alpha,verbose=True)
approx_sol = x_true  # use true solution as approx_sol for distance calc
optimal_cost = cost_group_lasso(A, approx_sol, b_new, alpha, groups_dict_idx)

cost_val_newton_ista, x1, i1, x_k1, time_k1 = Algo_Newton_Ista(
    A, b_new, x0, alpha, max_iter, step_size, beta, newton_stepsize, tol, cost,
    prox, subproblem_solver,  group_len, groups_dict_idx, newt_tol, approx_sol)

cost_val_newton_fista, x2, i2, x_k2, time_k2 = Algo_Newton_Fista_new(
    A, b_new, x0, alpha, max_iter, step_size, beta, newton_stepsize, tol, cost,
    prox, subproblem_solver, group_len, groups_dict_idx, newt_tol,  approx_sol)

cost_val_ista, x3, i3, x_k3, time_k3 = ISTA(
    A, b_new, x0, alpha, max_iter, step_size, tol, cost, prox, group_len, groups_dict_idx, approx_sol)

cost_val_fista, x4, i4, x_k4, time_k4 = FISTA1(
    A, b_new, x0, alpha, max_iter, step_size, tol, cost, prox, group_len,groups_dict_idx, approx_sol)

cost_val_newton_bt_ista, x5, i5, x_k5, time_k5 = Algo_Newton_BT_Ista(
    A, b_new, x0, alpha, max_iter, beta, newton_stepsize, tol, cost,
     prox, subproblem_solver, group_len, groups_dict_idx, newt_tol, approx_sol)

cost_val_newton_bt_fista, x6, i6, x_k6, time_k6 = Algo_Newton_BT_Fista_new(
    A, b_new, x0, alpha, max_iter, beta, newton_stepsize, tol, cost,
    prox, subproblem_solver, group_len, groups_dict_idx, newt_tol, approx_sol)



# Use the *noisy* problem for the optimal cost reference
# optimal_cost = min(cost_val_newton_ista[-1], cost_val_newton_fista[-1],
#                    cost_val_ista[-1], cost_val_fista[-1],
#                    cost_val_newton_bt_ista[-1], cost_val_newton_bt_fista[-1])
#optimal_cost = cost_lasso(A, approx_sol, b_new, alpha)
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
axs[0].set_ylabel(r'$|f(x_k) - f(x^*)|$')
axs[0].set_ylim(bottom=1e-10)
axs[0].grid(True)

axs[1].set_yscale('log')
axs[1].set_xlabel('Iterations')
axs[1].set_ylabel(r'$\|x_k - x^*\|$')
axs[1].set_ylim(bottom=1e-8)
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
plt.subplots_adjust(left=0.05,bottom=0.08, top=0.95, wspace=0.05)


plt.show()