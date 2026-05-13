import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lasso.newton_lasso import Algo_Globalized_Effective_Subspace_Newton_Lasso
from src.lasso.newton_lasso import Algo_Newton_Ista
from src.lasso.utils_lasso import cost_lasso
from src.lasso.utils_lasso import proxL1
from src.lasso.utils_lasso import solve_lasso_gurobi
from src.lasso.utils_lasso import sub_problem_of_lasso


# Synthetic Problem setup
m, n = 300, 500
rng = np.random.default_rng(0)

z = np.zeros(n)
sparsity = 20
z[rng.choice(n, sparsity, replace=False)] = rng.normal(size=sparsity)

A = rng.normal(size=(m, n))
step_size = 1.0 / (np.linalg.norm(A, 2) ** 2)
beta, newton_stepsize = 0.5, 1.0
tol = 1e-8
newton_ista_tol = 1e-3
x0 = np.zeros((n,))

b = A @ z
noise = rng.normal(scale=0.001, size=m)
b_new = b + noise

alpha_c = 1e-3
alpha = alpha_c * np.linalg.norm(A.T @ b_new, np.inf)
max_iter = 1000

prox = proxL1
subproblem_solver = sub_problem_of_lasso
cost = cost_lasso


# Benchmarks
approx_sol, optimal_cost = solve_lasso_gurobi(A, b_new, alpha, verbose=True)

cost_val_newton_ista, x1, i1, x_k1, time_k1 = Algo_Newton_Ista(
    A, b_new, x0, alpha, max_iter, step_size, beta, newton_stepsize, tol, cost,
    prox, subproblem_solver, newton_ista_tol, approx_sol)

cost_val_alg5, x_alg5, i_alg5, x_k_alg5, time_k_alg5 = (
    Algo_Globalized_Effective_Subspace_Newton_Lasso(
        A, b_new, x0, alpha, max_iter, step_size, tol, cost, prox,
        approx_sol=approx_sol, epsilon=1e-10)
)


# Series registry
series = [
    {
        "name": "Newton_ISTA", "color": "r", "marker": "o",
        "costs": np.asarray(cost_val_newton_ista),
        "dists": np.asarray(x_k1),
        "times": np.asarray(time_k1),
    },
    {
        "name": "Globalized_ES_Newton", "color": "tab:green", "marker": "*",
        "costs": np.asarray(cost_val_alg5),
        "dists": np.asarray(x_k_alg5),
        "times": np.asarray(time_k_alg5),
    },
]


# Figure: 2-row layout (bottom row only for legend)
fig = plt.figure(figsize=(20, 4))
gs = fig.add_gridspec(2, 3, height_ratios=[20, 1], hspace=0.25, wspace=0.25)
axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
ax_leg = fig.add_subplot(gs[1, :])
ax_leg.axis("off")

linewidth = 2.5
mark_every_cost = 10
mark_every_dist = 10

for s in series:
    gap = np.maximum(np.abs(s["costs"] - optimal_cost), 1e-16)
    axs[0].plot(
        gap,
        color=s["color"], linestyle="-", linewidth=linewidth,
        marker=s["marker"], markevery=mark_every_cost, markersize=6,
        label=s["name"]
    )

    axs[1].plot(
        s["dists"],
        color=s["color"], linestyle="-", linewidth=linewidth,
        marker=s["marker"], markevery=mark_every_dist, markersize=6
    )

axs[0].set_yscale("log")
axs[0].set_xlabel("Iterations")
axs[0].set_ylabel(r"$|\varphi(x_k) - \varphi(\overline{x})|$")
axs[0].set_ylim(bottom=1e-10)
axs[0].grid(True)

axs[1].set_yscale("log")
axs[1].set_xlabel("Iterations")
axs[1].set_ylabel(r"$\|x_k - \overline{x}\|$")
axs[1].grid(True)

methods = [s["name"] for s in series]
final_times = [s["times"][-1] for s in series]
colors = [s["color"] for s in series]

order = np.argsort(final_times)[::-1]
methods_sorted = [methods[i] for i in order]
times_sorted = [final_times[i] for i in order]
colors_sorted = [colors[i] for i in order]

axs[2].bar(methods_sorted, times_sorted, color=colors_sorted)
axs[2].set_yscale("log")
axs[2].set_ylabel("log(Runtime)")
axs[2].set_xticks([])

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    ncol=2,
    frameon=True,
    fontsize=10
)

plt.tight_layout(rect=[0, 0.05, 1, 1], pad=0.2)
plt.subplots_adjust(left=0.05, bottom=0.15, top=0.95, wspace=0.05)

plt.show()
