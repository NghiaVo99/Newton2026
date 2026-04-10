# ISTA for Generalized Lasso (TV) in pure NumPy
# minimize 0.5*||A x - b||^2 + lam * ||D x||_1
# - Uses NumPy spectral norm (no power iteration)
# - You plug your own TV prox: prox_{tau*lam * ||D·||_1}(x)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from matplotlib.ticker import ScalarFormatter
#from tick.prox import ProxTV
import pyproximal
from src.Gen_lasso.Gen_Lasso_utils import *
from src.Gen_lasso.Gen_Lasso_algo import *
from src.Gen_lasso.test_prob_gpt import generate_gen_lasso_toy


# ────────────────────────────────────────────────────────────
# Problem setup
# ────────────────────────────────────────────────────────────
m, n = 20, 90
rng = np.random.default_rng(10)
D = make_forward_diff(n)
data = generate_gen_lasso_toy(n=n, m=m, alpha=0.15, snr_db=25.0, seed=42)
A, b_new, z, alpha = data["A"], data["b"], data["x_true"], data["alpha"]

# z = rng.standard_normal(n)
# A = rng.normal(size=(m, n)) / np.sqrt(m)
# b_new = A @ z + 0.001 * rng.normal(size=m)

step_size = 1.0 / (np.linalg.norm(A, 2) ** 2)
beta, newton_stepsize = 0.5, 1.0
tol = 1e-8
newt_tol = 1e-4

alpha_c = 0.03 #1/np.linalg.norm(A.T @ b_new, np.inf)
print('alpha_c', alpha_c)
alpha = alpha_c * np.linalg.norm(A.T @ b_new, np.inf)
#print('alpha', alpha)
max_iter = 1000
x0 = np.zeros(n)

prox = pyproximal.TV(dims = (n,)).prox
#prox = ProxTV
subproblem_solver = sub_problem_gen_lasso
#subproblem_solver = sub_problem_gen_lasso_cvxpy
cost = cost_generalized_lasso

# ────────────────────────────────────────────────────────────
# Benchmarks
# ────────────────────────────────────────────────────────────
approx_sol = solve_generalized_lasso_gurobi(A, b_new, alpha, D = D)

cost_val_newton_ista, x1, i1, x_k1, time_k1 = Algo_Newton_Ista(
    A, D, b_new, x0, alpha, max_iter, step_size, beta, newton_stepsize, tol, cost,
    prox, subproblem_solver, newt_tol, approx_sol)

cost_val_newton_fista, x2, i2, x_k2, time_k2 = Algo_Newton_Fista_new(
    A, D, b_new, x0, alpha, max_iter, step_size, beta, newton_stepsize, tol, cost,
    prox, subproblem_solver, newt_tol, approx_sol)

cost_val_ista, x3, i3, x_k3, time_k3 = ISTA(
    A, D, b_new, x0, alpha, max_iter, step_size, tol, cost, prox, approx_sol)

cost_val_fista, x4, i4, x_k4, time_k4 = FISTA1(
    A, D, b_new, x0, alpha, max_iter, step_size, tol, cost, prox, approx_sol)

cost_val_newton_bt_ista, x5, i5, x_k5, time_k5 = Algo_Newton_BT_Ista(
    A, D, b_new, x0, alpha, max_iter, beta, newton_stepsize, tol, cost,
     prox, subproblem_solver, newt_tol, approx_sol)

cost_val_newton_bt_fista, x6, i6, x_k6, time_k6 = Algo_Newton_BT_Fista_new(
    A, D, b_new, x0, alpha, max_iter, beta, newton_stepsize, tol, cost,
    prox, subproblem_solver, newt_tol, approx_sol)

# cost_val_fastADMM, x7, i7, x_k7, time_k7 = Algo_FastADMM_Lasso(
#     A, b_new, x0, alpha, max_iter, tol, cost,
#     prox, approx_sol)


# Use the *noisy* problem for the optimal cost reference
optimal_cost = cost_generalized_lasso(A, approx_sol, b_new, alpha, D=D)
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
    # }
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
mark_every_cost = 30
mark_every_dist = 30
mark_every_time = 30

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
axs[0].set_ylabel(r'$|f(x_k) - f(x^*)|$')
#axs[0].set_ylim(top=0,bottom=1e-1)
axs[0].grid(True)

axs[1].set_yscale('log')
axs[1].set_xlabel('Iterations')
axs[1].set_ylabel(r'$\|x_k - x^*\|$')
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
    


handles, labels = axs[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    ncol=len(methods),
    frameon=True,
    fontsize=10
)

plt.tight_layout(rect=[0, 0.05, 1, 1], pad=0.2)  # leave space at bottom
plt.subplots_adjust(left=0.05,bottom=0.08, top=0.95, wspace=0.03)
plt.show()

# def ista_generalized_lasso(
#     A, b, lam,
#     TV,              
#     x0=None,
#     D=None,
#     max_iter=500,
#     step_scale=0.99,
#     tol_rel=1e-6,
#     record_every=1,
#     verbose=False
# ):
#     """
#     ISTA: x^{k+1} = prox_{t*lam*||D·||_1}( x^k - t * A^T(Ax^k - b) )
#     - t = step_scale / L, with L = ||A||_2^2 (NumPy spectral norm)
#     Returns (x, obj_vals)
#     """
#     m, n = A.shape
#     if x0 is None:
#         x = np.zeros(n, dtype=A.dtype)
#     else:
#         x = x0.astype(A.dtype, copy=True)

#     if D is None:
#         D = make_forward_diff(n, dtype=A.dtype)

#     L = lipschitz_from_spectral_norm(A)
#     if L <= 0:
#         raise ValueError("Nonpositive Lipschitz constant; check A.")
#     t = step_scale / L

#     obj_vals = []
#     x_iters = []
#     AT = A.T

#     prev_val = obj_generalized_lasso(A, b, x, D, lam)
#     obj_vals.append(prev_val)

#     for k in range(1, max_iter+1):
#         # gradient step
#         grad = grad_f(A, x, b)
#         y = x - t * grad

#         # prox step (your TV prox)
#         #x_new = prox(strength=lam * t).call(y)
#         x_new = TV.prox(y, lam * t)

#         if k % record_every == 0:
#             val = obj_generalized_lasso(A, b, x_new, D, lam)
#             obj_vals.append(val)
#             if verbose:
#                 print(f"iter {k:4d}: obj = {val:.6e}")
#             # relative improvement stopping
#             denom = max(1.0, abs(prev_val))
#             if abs(prev_val - val) / denom < tol_rel:
#                 x = x_new
#                 break
#             prev_val = val

#         x = x_new
#         x_iters.append(x)

#     return x, obj_vals, x_iters


# def ista_newton_generalized_lasso(
#     A, b, lam,
#     TV,               # callable: x_next = prox_Dx_l1(y, tau, D, lam)
#     x0=None,
#     D=None,
#     beta = 0.5,
#     max_iter=500,
#     step_scale=0.99,
#     tol_rel=1e-6,
#     record_every=1,
#     verbose=False
# ):
#     """
#     ISTA: x^{k+1} = prox_{t*lam*||D·||_1}( x^k - t * A^T(Ax^k - b) )
#     - t = step_scale / L, with L = ||A||_2^2 (NumPy spectral norm)
#     Returns (x, obj_vals)
#     """
#     m, n = A.shape
#     if x0 is None:
#         x = np.zeros(n, dtype=A.dtype)
#     else:
#         x = x0.astype(A.dtype, copy=True)

#     if D is None:
#         D = make_forward_diff(n, dtype=A.dtype)

#     L = lipschitz_from_spectral_norm(A)
#     if L <= 0:
#         raise ValueError("Nonpositive Lipschitz constant; check A.")
#     t = step_scale / L

#     obj_vals = []
#     x_iters = []
#     AT = A.T
#     do_newton = False  
#     prev_val = obj_generalized_lasso(A, b, x, D, lam)
#     obj_vals.append(prev_val)

#     for k in range(1, max_iter+1):
#         # gradient step
#         grad = grad_f(A, x, b)
#         y = x - t * grad

#         # prox step (your TV prox)
#         #x_hat = prox(strength=lam * t).call(y)
#         x_hat = TV.prox(y, lam * t)

#         if do_newton:
#             zk = (x - x_hat) / t - grad
#             #d_k = sub_problem1_cvxpy(A, x_hat, zk, b, lam)
#             d_k = sub_problem1_gurobi(A, x_hat, zk, b, lam)
#             print('norm d_k', np.linalg.norm(d_k))
#             newton_stepsize = 1.0
#             while obj_generalized_lasso(A, b, x_hat - newton_stepsize * d_k, D, lam) > obj_generalized_lasso(A, b, x_hat, D, lam):
#                newton_stepsize *= beta
#             x_new = x_hat - newton_stepsize * d_k
#             #do_newton = False
#         else:
#             x_new = x_hat

#         #if obj_generalized_lasso(A, b, x_new, D, lam) - obj_generalized_lasso(A, b, x_hat, D, lam) < 1e-2:
#         if np.linalg.norm(x_new - x_hat) < 1e-3: 
#             do_newton = True
#         else:
#             do_newton = False

#         if k % record_every == 0:
#             val = obj_generalized_lasso(A, b, x_new, D, lam)
#             obj_vals.append(val)
#             if verbose:
#                 print(f"iter {k:4d}: obj = {val:.6e}")
#             # relative improvement stopping
#             denom = max(1.0, abs(prev_val))
#             if abs(prev_val - val) / denom < tol_rel:
#                 x = x_new
#                 break
#             prev_val = val

#         x = x_new
#         x_iters.append(x)

#     return x, obj_vals, x_iters

# # ----------------------------
# # Example usage
# # ----------------------------
# if __name__ == "__main__":
#     rng = np.random.default_rng(0)

#     # Problem dimensions
#     m, n = 256, 512
#     max_iter = 500
#     tol = 1e-5
#     # Synthetic data
#     A = rng.normal(size=(m, n)) / np.sqrt(m)
#     x_true = np.random.standard_normal(n)
#     b = A @ x_true + 0.001 * rng.normal(size=m)

#     lam = 1
#     D = make_forward_diff(n)
#     TV = pyproximal.TV(dims = (n,))
#     try:
#         x_gurobi, info = generalized_lasso_gurobi(A, b, D, lam, silent=False)
#         print(f"Gurobi solution: obj = {info['obj_val']:.6e}")
#         x_hat, history, x_iters = ista_generalized_lasso(
#             A, b, lam,
#             TV=TV,  
#             D=D,
#             max_iter= max_iter,
#             step_scale=0.99,
#             tol_rel= tol,
#             record_every=1,
#             verbose=True
#         )

#         x_hat2, history2, x_iters2 = ista_newton_generalized_lasso(
#             A, b, lam,
#             TV = TV,  # Replace with prox_Dx_l1_USER to use your own
#             D=D,
#             beta= 0.9,
#             max_iter= max_iter,
#             step_scale=0.99,
#             tol_rel= tol,
#             record_every=1,
#             verbose=True
#         )
#     except NotImplementedError as e:
#         print(e)
#         history = None

#     # Plot objective history
#     if history is not None and len(history) > 1:
#         min_val = np.min(history + history2)
#         plt.figure(1)
#         plt.plot(np.array(history) - min_val, linewidth=2, label='ISTA')
#         plt.plot(np.array(history2) - min_val, linewidth=2, label='ISTA with Newton')
#         plt.xlabel("Iteration")
#         plt.ylabel(r"Objective $0.5\|Ax-b\|^2 + \lambda\|Dx\|_1$")
#         plt.yscale("log")
#         plt.title("ISTA for Generalized Lasso (TV) — Objective vs Iterations")
#         plt.grid(True, linestyle="--", alpha=0.5)
#         plt.legend()
#         plt.tight_layout()

#         plt.figure(2)
#         plt.plot(np.linalg.norm(x_iters - x_gurobi, axis=1), linewidth=2, label='ISTA')
#         plt.plot(np.linalg.norm(x_iters2 - x_gurobi, axis=1), linewidth=2, label='ISTA with Newton')
#         plt.xlabel("Iteration")
#         plt.ylabel(r"$\|x-x^\ast\|$")
#         plt.yscale("log")
#         plt.title("ISTA for Generalized Lasso (TV) — Objective vs Iterations")
#         plt.grid(True, linestyle="--", alpha=0.5)
#         plt.legend()
#         plt.tight_layout()
#         plt.show()