import numpy as np
import matplotlib.pyplot as plt

def compute_performance_profile(gaps, tau_max=20.0, n_tau=300):
    eps       = 1e-16
    gaps      = np.maximum(gaps, eps)
    best_gap  = gaps.min(axis=1, keepdims=True)
    ratios    = gaps / best_gap
    taus      = np.linspace(1.0, tau_max, n_tau)
    n_prob, n_sol = ratios.shape

    profiles = {}
    for s in range(n_sol):
        profiles[s] = np.array([
            (ratios[:, s] <= τ).sum() / n_prob
            for τ in taus
        ])
    return taus, profiles

# --- load your data ---
noise_level = 0.01
data        = np.load(f'results/lasso_benchmark_noise={noise_level}.npz', allow_pickle=True)
F           = data['f_vals']    # shape (P, S)
f_star      = data['f_star']    # shape (P,)
solver_names= data['solver_names']
# decode bytes→str if needed
solver_names = [n.decode('utf-8') if isinstance(n, bytes) else n
                for n in solver_names]
gaps        = np.abs(F - f_star[:, None])

# --- build a name→(color,marker) map matching your original script ---
mapping = {
    'Newton_Ista':            ('r',     'o'),
    'Newton_BT_Ista':         ('pink',  's'),
    'Newton_Fista':           ('m',     '^'),
    'ISTA':                   ('b',     'd'),
    'FISTA':                  ('k',     'v'),
    'GDFBE_LM':               ('orange','P'),
    'SSNAL':                  ('brown', 'X'),
    # your last solver was Algo_Newton_Fista_restart:
    'Newton_Fista_restart':   ('g',     '*'),
    # in case you ever use the old label:
    'Newton_BT_Fista':        ('g',     '*'),
}

# --- compute profile ---
taus, profiles = compute_performance_profile(gaps, tau_max=60.0, n_tau=60)

# --- plot with exact colors & markers ---
plt.figure(figsize=(8, 6))
for s, name in enumerate(solver_names):
    color, mkr = mapping[name]
    plt.plot(
        taus,
        profiles[s],
        label=name,
        color=color,
        marker=mkr,
        markevery=int(len(taus)/15),
        linewidth=1.8,
        markersize=6,
    )

plt.xlabel(r'Performance ratio $\tau$')
plt.ylabel(r'Proportion of problems $\rho_s(\tau)$')
plt.title('Performance Profile (objective‐gap)')
plt.grid(True, ls='--', alpha=0.5)
plt.legend(loc='lower right', fontsize='small', ncol=2)
plt.tight_layout()
plt.show()
# A = F-f_star[:, None]  # shape (P, S)
# mask_neg = (A < 0)

# plt.figure(figsize=(6,6))
# plt.spy(mask_neg, markersize=2)
# plt.title("Pattern of Negative Entries in A")

# eps       = 1e-16
# gaps      = np.maximum(A, eps)
# mask_neg_Aeps = (gaps < 0)
# plt.figure(figsize=(6,6))
# plt.spy(mask_neg_Aeps, markersize=2)
# plt.title("Pattern of Negative Entries in A_eps")
# plt.show()