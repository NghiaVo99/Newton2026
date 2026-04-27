import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.io import loadmat

# from utils_lasso import *
from src.lasso.untils_infinity import *
from src.lasso.newton_lasso import *
from src.lasso.lasso_GDNM import *
from src.lasso.lasso_GDFBE_LM import *
from src.lasso.test_ClassicLasso_random import *
from src.lasso.BaGSS import BasGSSLasso

# ────────────────────────────────────────────────────────────
# Read libsvmdata Problem setup
# ────────────────────────────────────────────────────────────

# d1 = 'abalone_scale_expanded7.mat'
# d2 = 'bodyfat_scale_expanded7.mat'
# d3 = 'housing_scale_expanded7.mat'
# d4 = 'mpg_scale_expanded7.mat'
# d5 = 'space_ga_scale_expanded9.mat'

# mat = loadmat(d4)  # e.g., contains A, b
# A = mat["A"]               # numpy array or scipy.sparse
# b_new = mat["b"]
# m, n = A.shape
# print('A_shape', m,n)
# print('b_shape', b_new.shape)
# step_size = 1.0 / (np.linalg.norm(A, 2) ** 2)
# beta, newton_stepsize = 0.5, 1.0
# tol = 1e-6
# newt_tol = 1e-3
# x0 = np.zeros((n,1))


# ────────────────────────────────────────────────────────────
# Synthetic Problem setup
# ────────────────────────────────────────────────────────────
m, n = 48, 128
rng = np.random.default_rng(42)

#z = np.random.rand(n)
z = np.zeros(n)
sparsity = 8
nonzero_indices = np.random.choice(n, sparsity, replace=False)
#z[nonzero_indices] = np.ones(sparsity)
z[rng.choice(n, sparsity, replace=False)] = rng.normal(size=sparsity)

A = rng.normal(size=(m, n)) 
step_size = 1.0 / (np.linalg.norm(A, 2) ** 2)
beta, newton_stepsize = 0.5, 1.0
tol = 1e-8
newt_tol = 1e-3
x0 = np.zeros((n,))

b = A @ z
noise = rng.normal(scale=0.001, size=m)
b_new = b + noise

alpha_c = 1e-1
alpha = alpha_c * np.linalg.norm(A.T @ b_new, np.inf)
max_iter = 1000


# prox = ProxL_infinity
# subproblem_solver = sub_problem_of_infinity
# cost = cost_infinity

prox = proxL1
subproblem_solver = sub_problem_of_lasso
cost = cost_lasso

# ────────────────────────────────────────────────────────────
# Benchmarks
# ────────────────────────────────────────────────────────────
approx_sol, optimal_cost = solve_lasso_gurobi(A, b_new, alpha,verbose=True)
#approx_sol, optimal_cost = solve_infinity_gurobi(A, b_new, alpha)

cost_val_newton_ista, x1, i1, x_k1, time_k1 = Algo_Newton_Ista(
    A, b_new, x0, alpha, max_iter, step_size, beta, newton_stepsize, tol, cost,
    prox, subproblem_solver, newt_tol, approx_sol)

cost_val_newton_fista, x2, i2, x_k2, time_k2 = Algo_Newton_Fista_new(
    A, b_new, x0, alpha, max_iter, step_size, beta, newton_stepsize, tol, cost,
    prox, subproblem_solver, newt_tol, approx_sol)

cost_val_ista, x3, i3, x_k3, time_k3 = ISTA(
    A, b_new, x0, alpha, max_iter, step_size, tol, cost, prox, approx_sol)

cost_val_fista, x4, i4, x_k4, time_k4 = FISTA1(
    A, b_new, x0, alpha, max_iter, step_size, tol, cost, prox, approx_sol)

cost_val_newton_bt_ista, x5, i5, x_k5, time_k5 = Algo_Newton_BT_Ista(
    A, b_new, x0, alpha, max_iter, beta, newton_stepsize, tol, cost,
     prox, subproblem_solver, newt_tol, approx_sol)

cost_val_newton_bt_fista, x6, i6, x_k6, time_k6 = Algo_Newton_BT_Fista_new(
    A, b_new, x0, alpha, max_iter, beta, newton_stepsize, tol, cost,
    prox, subproblem_solver, newt_tol, approx_sol)

# cost_val_fastADMM, x7, i7, x_k7, time_k7 = Algo_FastADMM_Lasso(
#     A, b_new, x0, alpha, max_iter, tol, cost,
#     prox, approx_sol)

solver_BaGSS = BasGSSLasso(A, b_new, lambda_reg=alpha,
                            lambda0=1e-2, lambda_bar=1.0,
                            alpha=0.25, beta=0.5, sigma=0.5, rho_bar=1.0,
                            eps=tol, max_iters=max_iter)

result_BaGSS = solver_BaGSS.solve(x0, approx_solution=approx_sol)

cost_val_BaGSS = result_BaGSS["history"]["phi_x"]
x_k_BaGSS = result_BaGSS["history"]["dist_x"]
time_k_BaGSS = result_BaGSS["history"]["time"]

# print(type(cost_val_BaGSS), type(x_k_BaGSS), type(time_k_BaGSS))
# print(len(cost_val_BaGSS), len(x_k_BaGSS), len(time_k_BaGSS))

# # Optional additional solvers you call:
x_est, cost_GDFBE, x_hist_GDFBE_LM, time_k8 = lasso_GDFBE_LM(A, b_new, alpha, approx_sol, tol=tol)
#x_GDNM, cost_GDNM, x_hist_GDNM, time_k9 = lasso_GDNM(A, b_new, alpha, approx_sol, tol=tol)
x_SSNAL, cost_SSNAL, x_hist_SSNAL, time_k10 = test_ClassicLasso_random_wrapper(A, b_new, x0, alpha, max_iter, step_size, beta, newton_stepsize, tol, approx_sol)
x_hist_SSNAL = np.linalg.norm(x_hist_SSNAL - approx_sol, axis=1)

# print(f'Newton_ISTA converged in {i1} iterations')
# print(f'Newton_BT_ISTA converged in {i11} iterations')
# print(f'Newton_FISTA converged in {i2} iterations')
# print(f'ISTA converged in {i3} iterations')
# print(f'FISTA converged in {i4} iterations')
# print(f'Newton_BT_FISTA (mod) converged in {i2_mod} iterations')

# Use the *noisy* problem for the optimal cost reference
#optimal_cost = cost_infinity(A, approx_sol, b_new, alpha)
# optimal_cost = cost_lasso(A, approx_sol, b_new, alpha)
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
    # {
    #     "name": "Fast-ADMM-R", "color": "pink", "marker": "h",
    #     "costs": np.asarray(cost_val_fastADMM),
    #     "dists": np.asarray(x_k7),
    #     "times": np.asarray(time_k7),
    # },
    {
        "name": "GRNM", "color": "orange", "marker": "P",
        "costs": np.asarray(cost_GDFBE),
        "dists": np.asarray(x_hist_GDFBE_LM),
        "times": np.asarray(time_k8),
    },
    # {
    #     "name": "GDNM", "color": "cyan", "marker": "<",
    #     "costs": np.asarray(cost_GDNM),
    #     "dists": np.asarray(x_hist_GDNM),
    #     "times": np.asarray(time_k9),
    # },
    {
        "name": "SSNAL", "color": "saddlebrown", "marker": "X",
        "costs": np.asarray(cost_SSNAL),
        "dists": np.asarray(x_hist_SSNAL),
        "times": np.asarray(time_k10),
    },
    {
        "name": "GSSN", "color": "green", "marker": "*",
        "costs": np.asarray(cost_val_BaGSS),
        "dists": np.asarray(x_k_BaGSS),
        "times": np.asarray(time_k_BaGSS),
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
axs[0].set_ylabel(r'$|\varphi(x_k) - \varphi(\overline{x})|$')
axs[0].set_ylim(bottom=1e-10)
axs[0].grid(True)

axs[1].set_yscale('log')
axs[1].set_xlabel('Iterations')
axs[1].set_ylabel(r'$\|x_k - \overline{x}\|$')
#axs[1].set_ylim(bottom=1e-8)
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
axs[2].set_xticks([])            
#axs[2].tick_params(axis='x', labelrotation=45)

num_methods = len(methods)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    ncol= 5,
    frameon=True,
    fontsize=10
)

plt.tight_layout(rect=[0, 0.05, 1, 1], pad=0.2)  # leave space at bottom
plt.subplots_adjust(left=0.05,bottom=0.15, top=0.95, wspace=0.05)


plt.show()