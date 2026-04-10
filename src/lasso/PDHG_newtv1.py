# Two-phase hybrid: run PDHG until support (#nonzeros) stabilizes, then switch to Newton.
# Compare vs FISTA and full PDHG. Plot objective gap vs iteration.
import numpy as np
import time, math
import matplotlib.pyplot as plt
from collections import deque

np.random.seed(4)

# ----- Problem -----
m, n = 250, 800
density = 0.1
x_true = np.zeros(n)
supp = np.random.choice(n, int(density*n), replace=False)
x_true[supp] = np.random.randn(len(supp))

A = np.random.randn(m, n) / math.sqrt(m)
b = A @ x_true + 0.01*np.random.randn(m)

lam = 0.1  # as per your snippet

def F_val(x):
    r = A @ x - b
    return 0.5 * float(r @ r) + lam * np.linalg.norm(x, 1)

def grad_f(x):
    return A.T @ (A @ x - b)

def power_iteration_AT_A(A, iters=20):
    n = A.shape[1]
    v = np.random.randn(n); v /= (np.linalg.norm(v)+1e-16)
    for _ in range(iters):
        v = A.T @ (A @ v)
        nv = np.linalg.norm(v)+1e-16
        v /= nv
    return float(v @ (A.T @ (A @ v)))

L = power_iteration_AT_A(A, iters=20)

def soft_thresh(x, t):
    return np.sign(x)*np.maximum(np.abs(x)-t, 0.0)

# ----- Baselines -----
def run_fista(x0, iters=100):
    x = x0.copy(); y = x0.copy(); t = 1.0
    step = 1.0/L
    vals = []
    for k in range(iters):
        y_grad = y - step*grad_f(y)
        x_new = soft_thresh(y_grad, step*lam)
        t_new = 0.5*(1+math.sqrt(1+4*t*t))
        y = x_new + ((t-1)/t_new)*(x_new - x)
        x, t = x_new, t_new
        vals.append(F_val(x))
    return x, np.array(vals)

def prox_f_star(w, sigma):
    return (w - sigma*b)/(1.0 + sigma)

def run_pdhg_full(x0, iters=100):
    L_A = math.sqrt(L)
    tau = 0.9 / (L_A + 1e-12)
    sigma = 0.9 / (tau * (L_A**2 + 1e-12))
    theta = 1.0
    x = x0.copy(); x_bar = x.copy(); p = np.zeros(m)
    vals = []
    for k in range(iters):
        p = prox_f_star(p + sigma*(A @ x_bar), sigma)
        x_prev = x.copy()
        x = soft_thresh(x - tau*(A.T @ p), tau*lam)
        x_bar = x + theta*(x - x_prev)
        vals.append(F_val(x))
    return x, np.array(vals)

# ----- Two-phase Hybrid (PDHG then Newton) -----
def run_pdhg_then_newton(x0, pdhg_max_iters=100, stable_window=2, tol_support=1e-3,
                         newton_max_iters=60, alpha=0.9/L, eps_act=1e-12, backtrack=True):
    # Phase 1: PDHG until #nonzeros stable
    L_A = math.sqrt(L)
    tau = 0.9 / (L_A + 1e-12)
    sigma = 0.9 / (tau * (L_A**2 + 1e-12))
    theta = 1.0

    x = x0.copy(); x_bar = x.copy(); p = np.zeros(m)
    vals = []
    nnz_hist = deque(maxlen=stable_window)
    switched = False

    for k in range(pdhg_max_iters):
        # one PDHG iteration
        p = prox_f_star(p + sigma*(A @ x_bar), sigma)
        x_prev = x.copy()
        x = soft_thresh(x - tau*(A.T @ p), tau*lam)
        x_bar = x + theta*(x - x_prev)
        vals.append(F_val(x))

        # support stability check
        support_mask = np.abs(x) > tol_support
        nnz = int(np.count_nonzero(support_mask))
        nnz_hist.append(nnz)
        if len(nnz_hist) == stable_window and len(set(nnz_hist)) == 1:
            switched = True
            break

    # If never stabilized, return PDHG history
    if not switched:
        return x, np.array(vals)

    # Phase 2: Newton iterations using Algorithm-1 style (with exact prox for l1)
    H = A.T @ A
    for t in range(newton_max_iters):
        v = x - alpha*grad_f(x)
        y = soft_thresh(v, alpha*lam)
        z = (v - y)/alpha

        # active subspace from saturated dual
        active = (np.abs(z) >= lam - eps_act)
        if not np.any(active):
            # nothing active: just take prox update and stop
            x = y.copy()
            vals.append(F_val(x))
            break

        S = np.where(active)[0]
        H_SS = H[np.ix_(S, S)].copy()
        H_SS.flat[::H_SS.shape[0]+1] += 1e-10 + 1e-6*np.mean(np.diag(H_SS)) if H_SS.size>0 else 0.0
        rhs = z + grad_f(y)
        rhs_S = rhs[S]

        try:
            d_S = np.linalg.solve(H_SS, rhs_S)
        except np.linalg.LinAlgError:
            d_S = np.linalg.lstsq(H_SS, rhs_S, rcond=None)[0]

        d = np.zeros_like(x); d[S] = d_S

        Fy = F_val(y); gamma = 1.0; accepted = False
        if backtrack:
            while gamma > 1e-6:
                x_cand = y - gamma*d
                if F_val(x_cand) <= Fy - 1e-12:
                    x = x_cand; accepted = True; break
                gamma *= 0.5
            if not accepted:
                x = y.copy()
        else:
            x = y - d

        vals.append(F_val(x))

        # Optional: stop if small improvement
        if len(vals) >= 2 and abs(vals[-2] - vals[-1]) <= 1e-10 * (1 + vals[-2]):
            break

    return x, np.array(vals)

# ----- Run & Plot (objective gap vs iteration) -----
x0 = np.zeros(n)

xf_fista, vals_fista = run_fista(x0, iters=100)
xf_pdhg,  vals_pdhg  = run_pdhg_full(x0, iters=100)
xh_hyb,   vals_hyb   = run_pdhg_then_newton(x0, pdhg_max_iters=100, stable_window=2, tol_support=1e-3,
                                            newton_max_iters=80, alpha=0.9/L, eps_act=1e-6)

# normalize by subtracting the best final value among methods
sub_optf = min(vals_fista[-1], vals_pdhg[-1], vals_hyb[-1])
vals_fista_gap = vals_fista - sub_optf
vals_pdhg_gap  = vals_pdhg  - sub_optf
vals_hyb_gap   = vals_hyb   - sub_optf

plt.figure()
plt.plot(vals_fista_gap, label="FISTA")
plt.plot(vals_pdhg_gap,  label="PDHG (full)")
plt.plot(vals_hyb_gap,   label="Hybrid: PDHG → Newton")
plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("Objective gap F(x) - F* (up to const)")
plt.title("L1 least squares: PDHG until support stabilizes, then Newton")
plt.legend()
plt.tight_layout()
plt.show()
