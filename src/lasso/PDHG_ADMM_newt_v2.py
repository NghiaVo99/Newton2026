import numpy as np
import time, math
import matplotlib.pyplot as plt
from collections import deque

np.random.seed(5)

# ----- Problem -----
m, n = 200, 800
density = 0.1
x_true = np.zeros(n)
supp = np.random.choice(n, int(density*n), replace=False)
x_true[supp] = np.random.randn(len(supp))

A = np.random.randn(m, n) / math.sqrt(m)
b = A @ x_true + 0.01*np.random.randn(m)

lam = 0.1

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

def run_fista(x0, iters=120):
    x = x0.copy(); y = x0.copy(); t = 1.0
    step = 1.0/L
    vals, times = [], []
    t0 = time.time()
    for k in range(iters):
        y_grad = y - step*grad_f(y)
        x_new = soft_thresh(y_grad, step*lam)
        t_new = 0.5*(1+math.sqrt(1+4*t*t))
        y = x_new + ((t-1)/t_new)*(x_new - x)
        x, t = x_new, t_new
        vals.append(F_val(x)); times.append(time.time()-t0)
    return x, np.array(vals), np.array(times)

def prox_f_star(w, sigma):
    return (w - sigma*b)/(1.0 + sigma)

def run_pdhg_full(x0, iters=120):
    L_A = math.sqrt(L)
    tau = 0.9 / (L_A + 1e-12)
    sigma = 0.9 / (tau * (L_A**2 + 1e-12))
    theta = 1.0
    x = x0.copy(); x_bar = x.copy(); p = np.zeros(A.shape[0])
    vals, times = [], []; t0 = time.time()
    for k in range(iters):
        p = prox_f_star(p + sigma*(A @ x_bar), sigma)
        x_prev = x.copy()
        x = soft_thresh(x - tau*(A.T @ p), tau*lam)
        x_bar = x + theta*(x - x_prev)
        vals.append(F_val(x)); times.append(time.time()-t0)
    return x, np.array(vals), np.array(times)

def run_pdhg_then_newton(x0, pdhg_max_iters=120, stable_window=3, tol_support=1e-3,
                         newton_max_iters=80, alpha=0.9/L, eps_act=1e-12, backtrack=True):
    L_A = math.sqrt(L)
    tau = 0.9 / (L_A + 1e-12)
    sigma = 0.9 / (tau * (L_A**2 + 1e-12))
    theta = 1.0
    x = x0.copy(); x_bar = x.copy(); p = np.zeros(A.shape[0])
    vals, times = [], []; t0 = time.time()
    from collections import deque
    nnz_hist = deque(maxlen=stable_window)
    switched = False
    for k in range(pdhg_max_iters):
        p = prox_f_star(p + sigma*(A @ x_bar), sigma)
        x_prev = x.copy()
        x = soft_thresh(x - tau*(A.T @ p), tau*lam)
        x_bar = x + theta*(x - x_prev)
        vals.append(F_val(x)); times.append(time.time()-t0)
        nnz_hist.append(int(np.count_nonzero(np.abs(x) > tol_support)))
        if len(nnz_hist)==stable_window and len(set(nnz_hist))==1:
            switched = True
            break
    if not switched:
        return x, np.array(vals), np.array(times)

    H = A.T @ A
    for t in range(newton_max_iters):
        v = x - alpha*grad_f(x)
        y = soft_thresh(v, alpha*lam)
        z = (v - y)/alpha
        active = (np.abs(z) >= lam - eps_act)
        if not np.any(active):
            x = y.copy()
            vals.append(F_val(x)); times.append(time.time()-t0)
            break
        S = np.where(active)[0]
        H_SS = H[np.ix_(S, S)].copy()
        if H_SS.size>0:
            H_SS.flat[::H_SS.shape[0]+1] += 1e-10 + 1e-6*np.mean(np.diag(H_SS))
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
        vals.append(F_val(x)); times.append(time.time()-t0)
        if len(vals)>=2 and abs(vals[-2]-vals[-1]) <= 1e-10*(1+vals[-2]):
            break
    return x, np.array(vals), np.array(times)

def run_admm_full(x0, iters=120, rho=1.0):
    n = x0.size
    x = x0.copy()
    z = x0.copy()
    u = np.zeros_like(x0)
    M = A.T @ A + rho * np.eye(n)
    vals, times = [], []
    t0 = time.time()
    for k in range(iters):
        q = A.T @ b + rho * (z - u)
        x = np.linalg.solve(M, q)
        z = soft_thresh(x + u, lam / rho)
        u = u + x - z
        vals.append(F_val(x)); times.append(time.time()-t0)
    return x, np.array(vals), np.array(times)

def run_admm_then_newton(x0, admm_max_iters=120, stable_window=3, tol_support=1e-3,
                         newton_max_iters=80, alpha=0.9/L, rho=1.0, eps_act=1e-12, backtrack=True):
    n = x0.size
    x = x0.copy()
    z = x0.copy()
    u = np.zeros_like(x0)
    M = A.T @ A + rho * np.eye(n)
    vals, times = [], []; t0 = time.time()
    from collections import deque
    nnz_hist = deque(maxlen=stable_window)
    switched = False
    for k in range(admm_max_iters):
        q = A.T @ b + rho * (z - u)
        x = np.linalg.solve(M, q)
        z = soft_thresh(x + u, lam / rho)
        u = u + x - z
        vals.append(F_val(x)); times.append(time.time()-t0)
        nnz_hist.append(int(np.count_nonzero(np.abs(x) > tol_support)))
        if len(nnz_hist)==stable_window and len(set(nnz_hist))==1:
            switched = True
            break
    if not switched:
        return x, np.array(vals), np.array(times)

    H = A.T @ A
    for t in range(newton_max_iters):
        v = x - alpha*grad_f(x)
        y = soft_thresh(v, alpha*lam)
        z_dual = (v - y)/alpha
        active = (np.abs(z_dual) >= lam - eps_act)
        if not np.any(active):
            x = y.copy()
            vals.append(F_val(x)); times.append(time.time()-t0)
            break
        S = np.where(active)[0]
        H_SS = H[np.ix_(S, S)].copy()
        if H_SS.size>0:
            H_SS.flat[::H_SS.shape[0]+1] += 1e-10 + 1e-6*np.mean(np.diag(H_SS))
        rhs = z_dual + grad_f(y)
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
        vals.append(F_val(x)); times.append(time.time()-t0)
        if len(vals)>=2 and abs(vals[-2]-vals[-1]) <= 1e-10*(1+vals[-2]):
            break
    return x, np.array(vals), np.array(times)

def main():
    x0 = np.zeros(A.shape[1])
    xf_fista, vals_fista, times_fista = run_fista(x0, iters=120)
    xf_pdhg,  vals_pdhg,  times_pdhg  = run_pdhg_full(x0, iters=120)
    xh_pn,    vals_pn,    times_pn    = run_pdhg_then_newton(x0, pdhg_max_iters=120, stable_window=2, tol_support=1e-3,
                                                              newton_max_iters=80, alpha=0.9/L)
    xa_admm,  vals_admm,  times_admm  = run_admm_full(x0, iters=120, rho=1.0)
    xh_an,    vals_an,    times_an    = run_admm_then_newton(x0, admm_max_iters=120, stable_window=2, tol_support=1e-3,
                                                             newton_max_iters=80, alpha=0.9/L, rho=1.0)

    best_final = min(vals_fista[-1], vals_pdhg[-1], vals_pn[-1], vals_admm[-1], vals_an[-1])
    gap_fista = vals_fista - best_final
    gap_pdhg  = vals_pdhg  - best_final
    gap_pn    = vals_pn    - best_final
    gap_admm  = vals_admm  - best_final
    gap_an    = vals_an    - best_final

    plt.figure()
    plt.plot(gap_fista, label="FISTA")
    plt.plot(gap_pdhg,  label="PDHG (full)")
    plt.plot(gap_pn,    label="PDHG → Newton")
    plt.plot(gap_admm,  label="ADMM (full)")
    plt.plot(gap_an,    label="ADMM → Newton")
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Objective gap (≈ F(x) - F*)")
    plt.title("L1 least squares: All methods, gap vs iteration")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(times_fista, gap_fista, label="FISTA")
    plt.plot(times_pdhg,  gap_pdhg,  label="PDHG (full)")
    plt.plot(times_pn,    gap_pn,    label="PDHG → Newton")
    plt.plot(times_admm,  gap_admm,  label="ADMM (full)")
    plt.plot(times_an,    gap_an,    label="ADMM → Newton")
    plt.yscale('log')
    plt.xlabel("Time (s)")
    plt.ylabel("Objective gap (≈ F(x) - F*)")
    plt.title("L1 least squares: All methods, gap vs time")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()