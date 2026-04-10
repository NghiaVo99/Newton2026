import numpy as np
import imageio.v3 as iio
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.io import loadmat
import scipy
import pathlib
from src.lasso.newton_poisson import *
from src.lasso.Poisson_utils import *
import imageio.v2 as imageio



# -------------------- I/O & dataset --------------------
FOLDER = pathlib.Path("sequence")
files = sorted(
    [*FOLDER.glob("*.tif"), *FOLDER.glob("*.tiff"),
     *FOLDER.glob("*.png"), *FOLDER.glob("*.jpg"), *FOLDER.glob("*.jpeg")]
)

z_vec = iio.imread(files[0], index=0)
b_vec = estimate_background_from_stack(files, pattern="*.tif", method="median").reshape(-1)


scale = 4                      # upsampling
lr_px_nm = 100
hr_px_nm = lr_px_nm / scale    # 25 nm/pixel

psf_hr = gaussian_psf_hr_cel0(fwhm_nm=258.2, hr_px_nm=hr_px_nm, size=33)
alpha = 0.5
max_iter = 210

ops = make_problem(psf_hr, scale=scale,
                   z_lr_2d=np.asarray(z_vec, np.float64),
                   b=b_vec,
                   x0_mode="backproj")


# ────────────────────────────────────────────────────────────

# Back-projection init (already provided as ops["x0"])
x0 = ops["x0"]           # = max(0, AT(z-b))
# plt.imshow(x0.reshape(256,256), cmap='gist_heat')
rng = np.random.default_rng(42)

A = ops["A"]
AT = ops["AT"]
beta, newton_stepsize = 0.5, 1.0
tol = 1e-8
newt_tol = 0.1


prox = ops['prox_g']
subproblem_solver = sub_problem_of_poisson
cost = cost_poisson

# ────────────────────────────────────────────────────────────
# Benchmarks
# ────────────────────────────────────────────────────────────
#approx_sol, optimal_cost = solve_lasso_gurobi(A, b_new, alpha,verbose=True)
approx_sol = 0

cost_val_ista, x3, i3, x_k3, time_k3 = BT_ISTA(A,AT,b_vec,x0, z_vec,
                        alpha,max_iter, beta, newton_stepsize, tol, cost,
                        prox, subproblem_solver, approx_sol = 0)

cost_val_fista, x4, i4, x_k4, time_k4 = BT_FISTA1(A,AT,b_vec,x0, z_vec,
                        alpha,max_iter, beta, newton_stepsize, tol, cost,
                        prox, subproblem_solver, approx_sol = 0)


cost_val_newton_bt_ista, x5, i5, x_k5, time_k5 = Algo_Newton_BT_Fista_new(A,AT,b_vec,x0, z_vec,
                             alpha,max_iter, beta, newton_stepsize, tol, cost,
                             prox, subproblem_solver, newt_tol = newt_tol, approx_sol = approx_sol)

cost_val_newton_bt_fista, x6, i6, x_k6, time_k6 = Algo_Newton_BT_Fista_new(A,AT,b_vec,x0, z_vec,
                             alpha,max_iter, beta, newton_stepsize, tol, cost,
                             prox, subproblem_solver, newt_tol = newt_tol, approx_sol = approx_sol)



# Use the *noisy* problem for the optimal cost reference
#optimal_cost = cost_infinity(A, approx_sol, b_new, alpha)
#optimal_cost = cost_lasso(A, approx_sol, b_new, alpha)
optimal_cost = min(cost_val_fista[ -1], cost_val_ista[ -1], 
                   cost_val_newton_bt_ista[ -1], cost_val_newton_bt_fista[ -1]) 
# ────────────────────────────────────────────────────────────
# Series registry (add/remove methods here)
# ────────────────────────────────────────────────────────────
series = [
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
mark_every_cost = 30
mark_every_dist = 30
mark_every_time = 30


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

    # # (b) distance vs iterations
    # axs[1].plot(
    #     s["dists"],
    #     color=s["color"], linestyle='-', linewidth=linewidth,
    #     marker=s["marker"], markevery=mark_every_dist, markersize=6
    # )

axs[0].set_yscale('log')
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel(r'$|f(x_k) - f(x^*)|$')
axs[0].set_ylim(bottom=1e-10)
axs[0].grid(True)

# axs[1].imshow(x6.reshape(ops['hr_shape']), cmap='gist_heat')
# axs[1].set_title('Reconstructed HR image')

# axs[2].imshow(z_vec.reshape(ops['lr_shape']), cmap='gist_heat')
# axs[2].set_title('Observed LR image')

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
axs[2].tick_params(axis='x', labelrotation=45)

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


