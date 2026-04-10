import numpy as np, time

# ---------- DWSL1 (OSCAR) utilities ----------

def dws_value(x, lam):
    return float(np.dot(np.sort(np.abs(x))[::-1], lam))  # κ_λ(x)

def prox_dws_l1_unscaled(z, lam):
    """
    Prox of κ_λ at z (UNSCALED). Returns u = Prox_κ(z) and 'work' for a Jacobian–vec product.
    Optimized: vectorized expand via np.repeat; compact block metadata.
    """
    z = np.asarray(z, float)
    lam = np.asarray(lam, float)
    n = z.size

    idx = np.argsort(-np.abs(z))
    z_sorted = z[idx]
    sgn = np.sign(z_sorted)
    a = np.abs(z_sorted)

    # raw soft-threshold in sorted domain
    u = a - lam

    # PAVA (nonincreasing levels)
    v = np.empty_like(u)
    w = np.empty(u.size, dtype=int)
    k = 0
    for ui in u:  # linear-time stack PAVA
        v[k] = ui
        w[k] = 1
        while k > 0 and v[k-1] < v[k]:
            tw = w[k-1] + w[k]
            v[k-1] = (w[k-1]*v[k-1] + w[k]*v[k]) / tw
            w[k-1] = tw
            k -= 1
        k += 1

    # truncate stacks
    v = v[:k]
    w = w[:k]

    # clip levels at 0, then expand
    v = np.maximum(v, 0.0)
    zlev = np.repeat(v, w)  # length n

    # build compact block metadata for M_times_vec
    block_sizes = w
    block_starts = np.empty_like(block_sizes)
    np.cumsum(np.r_[0, block_sizes[:-1]], out=block_starts)
    # boolean per-block "active" and per-entry active mask on demand
    active_blocks = v > 0.0

    u_sorted = sgn * zlev
    u_full = np.zeros_like(z)
    u_full[idx] = u_sorted

    work = {
        "idx": idx,
        "sgn_sorted": sgn,
        "block_starts": block_starts,
        "block_sizes": block_sizes,
        "active_blocks": active_blocks
    }
    return u_full, work

def prox_dws_l1_scaled(z, lam, sigma):
    """Prox of σ κ_λ at z (via homogeneity)."""
    u, _ = prox_dws_l1_unscaled(z, sigma * lam)
    return u

def M_times_vec(delta, work, out=None, tmp=None):
    """
    Apply one generalized-Jacobian element of Prox_κ at the point encoded by 'work'.
    Optimized: compact block loop; optional out/tmp buffers to avoid allocs.
    """
    idx = work["idx"]; sgn_sorted = work["sgn_sorted"]
    Ls = work["block_starts"]; Ws = work["block_sizes"]; actB = work["active_blocks"]

    if tmp is None or tmp.shape != (idx.size,):
        tmp = np.empty(idx.size, dtype=float)
    if out is None or out.shape != (idx.size,):
        out = np.zeros(idx.size, dtype=float)
    else:
        out.fill(0.0)

    d_sorted = delta[idx] * sgn_sorted
    for j, (L, m) in enumerate(zip(Ls, Ws)):
        if m == 1:
            if actB[j]:
                out[L] = d_sorted[L]
            # else zero stays zero
        else:
            if actB[j]:
                R = L + m
                mean = d_sorted[L:R].mean()
                out[L:R] = mean
            # inactive block -> zeros (already set)

    out_final = np.empty_like(out)
    out_final[idx] = out * sgn_sorted
    return out_final

# ---------- ALM subproblem: Ψ_k and its gradient (UNSCALED prox) ----------

def grad_Psi(y, A, b, xk, sigma, lam):
    """
    Use z = (1/σ) xk - A^T y.
    ∇Ψ_k(y) = y + b - A * (σ * Prox_κ(z)).
    Also return work of Prox_κ(z) for the Jacobian M in V = I + σ A M A^T.
    """
    ATy = A.T @ y
    z = (xk / sigma) - ATy
    u, work = prox_dws_l1_unscaled(z, lam)  # UNscaled prox at z
    x_next = sigma * u                       # σ Prox_κ(z) == Prox_{σ κ}(σ z)
    g = y + b - A @ x_next
    return g, x_next, work, z

def Psi_exact(y, A, b, xk, sigma, lam):
    z = (xk / sigma) - (A.T @ y)
    p = prox_dws_l1_scaled(z, lam, sigma)        # p = Prox_{σ κ}(z)
    env = sigma * dws_value(p, lam) + 0.5 * np.dot(p - z, p - z)
    return 0.5 * np.dot(y, y) + float(b @ y) + 0.5 * sigma * np.dot(z, z) - env

def cg_solve(matvec, rhs, tol=1e-10, maxit=500):
    x = np.zeros_like(rhs)
    r = rhs - matvec(x)
    p = r.copy()
    rr = float(r @ r)
    if rr <= tol * tol:
        return x
    for _ in range(maxit):
        Ap = matvec(p)
        denom = float(p @ Ap)
        if denom <= 1e-30:
            break
        alpha = rr / denom
        x += alpha * p
        r -= alpha * Ap
        rr_new = float(r @ r)
        if rr_new <= tol * tol:
            break
        p = r + (rr_new / (rr + 1e-300)) * p
        rr = rr_new
    return x

def ssn_step(y, A, b, xk, sigma, lam,
             mu=1e-4, etabar=1e-2, delta=0.5, tau=0.5,
             cg_rtol=1e-12, cg_maxit=1000, damping=1e-8):
    """
    One SSN step for Ψ_k(y) with merit φ(y)=0.5||∇Ψ_k(y)||^2 and Armijo backtracking.
    Returns: (y_new, ||∇Ψ(y_new)||, x_next_new) so caller avoids extra prox/grad.
    """
    # gradient & linearization at current y
    g, x_next, work, z = grad_Psi(y, A, b, xk, sigma, lam)

    # closure for V d = d + σ A M A^T d (with tiny damping)
    # reuse small temporaries to cut allocs
    tmpJ = {"out": None, "tmp": None}
    def matvec(d):
        ATd = A.T @ d
        Md = M_times_vec(ATd, work, out=tmpJ["out"], tmp=tmpJ["tmp"])
        tmpJ["out"], tmpJ["tmp"] = None, None  # ensure no accidental reuse across calls
        return d + sigma * (A @ Md) + damping * d

    rhs = -g
    gnorm = float(np.linalg.norm(g))
    eta = min(etabar, max(cg_rtol, gnorm**(1.0 + tau)))
    d = cg_solve(matvec, rhs, tol=eta, maxit=cg_maxit)

    gd = float(g @ d)
    if gd >= -1e-14 or not np.isfinite(gd):
        d = -g
        gd = float(g @ d)

    phi0 = 0.5 * gnorm * gnorm
    alpha = 1.0

    # backtracking; reuse ATd to avoid re-mults in line search
    ATd = A.T @ d
    for _ in range(50):
        y_try = y + alpha * d
        # fast grad eval: z_try = z - alpha*ATd
        z_try = z - alpha * ATd
        u_try, _ = prox_dws_l1_unscaled(z_try, lam)
        x_next_try = sigma * u_try
        g_try = y_try + b - A @ x_next_try
        gnorm_try = float(np.linalg.norm(g_try))
        phi_try = 0.5 * gnorm_try * gnorm_try
        if phi_try <= (1.0 - mu * alpha) * phi0:
            return y_try, gnorm_try, x_next_try
        alpha *= delta

    return y, gnorm, x_next  # very rare

# ---------- Outer ALM (Algorithm 1) ----------

class NewtALM_OSCAR:
    """
    Algorithm 1 (Newt-ALM) for OSCAR/SLOPE.
    """
    def __init__(self, A, b, w1, w2,
                 sigma0=1.0, sigma_rate=2.0, sigma_max=1e6,
                 tol_kkt=1e-8, max_outer=200, max_inner=50, x_ref=None):
        self.A = np.asarray(A, float)
        self.b = np.asarray(b, float)
        self.m, self.n = self.A.shape

        lam = w1 + w2 * (self.n - 1 - np.arange(self.n))  # λ_i = w1 + w2(n-i)
        lam = np.clip(lam, 0.0, None)
        if not np.all(lam[:-1] >= lam[1:]):  # ensure nonincreasing
            lam = np.sort(lam)[::-1]
        self.lam = lam

        self.sigma0 = float(sigma0)
        self.sigma_rate = float(sigma_rate)
        self.sigma_max = float(sigma_max)
        self.tol_kkt = float(tol_kkt)
        self.max_outer = int(max_outer)
        self.max_inner = int(max_inner)

        self.x_ref = None if x_ref is None else np.asarray(x_ref, float).reshape(-1)
        if self.x_ref is not None and self.x_ref.size != self.n:
            raise ValueError(f"x_ref size {self.x_ref.size} != problem size {self.n}")
        self.history = None

    def _objective_fast(self, x, r):
        return 0.5 * float(r @ r) + dws_value(x, self.lam)

    def _kkt_residual_fast(self, x, r):
        g = self.A.T @ r
        z = x - g
        p, _ = prox_dws_l1_unscaled(z, self.lam)
        num = np.linalg.norm(x - p)
        den = 1 + np.linalg.norm(x) + np.linalg.norm(g)
        return num / den

    def solve(self, x0=None, verbose=True, print_every=1):
        A, b, lam = self.A, self.b, self.lam
        m, n = A.shape
        x = np.zeros(n) if x0 is None else np.asarray(x0, float).copy()
        sigma = self.sigma0
        y = np.zeros(m)  # warm-start across σ

        iters, sigmas, inner_list, objs, kkts = [], [], [], [], []
        times_cum, times_iter, xs, dists = [], [], [], []
        t0 = time.time(); t_prev = t0
        stopped_on = "max_outer"; stop_iter = -1
        r = A @ x - b
        kkt_k = self._kkt_residual_fast(x, r)
        dist_k = np.nan if self.x_ref is None else float(np.linalg.norm(x - self.x_ref))
        objs.append(self._objective_fast(x, r)); kkts.append(kkt_k); xs.append(x.copy()); dists.append(dist_k)

        for k in range(self.max_outer):
            # Step 1: solve min Ψ_k(y) until (A) and (B1) hold
            eps_k = 1e-2 / (k+1)**1.2
            del_k = 5e-1 / (k+1)**1.2
            tol_A = eps_k / np.sqrt(sigma if sigma >= 1.0 else 1.0)

            inner_iters = 0
            satisfied = False
            grad_norm = np.inf
            x_cand = x  # placeholder

            while inner_iters < self.max_inner:
                y, grad_norm, x_cand = ssn_step(y, A, b, x, sigma, lam)
                inner_iters += 1

                # conditions using returned ||∇Ψ|| and x_cand
                gnorm = grad_norm
                condA  = (gnorm <= tol_A)
                condB1 = (gnorm <= (del_k / np.sqrt(sigma if sigma >= 1.0 else 1.0)) * np.linalg.norm(x_cand - x))
                if condA and condB1:
                    satisfied = True
                    break
                # else: keep iterating SSN on same subproblem; x stays for now

            # Step 2: update x (always once per outer loop in this variant)
            x = x_cand

            # Logging (compute r once; reuse)
            r = A @ x - b
            obj_k = self._objective_fast(x, r)
            kkt_k = self._kkt_residual_fast(x, r)
            dist_k = np.nan if self.x_ref is None else float(np.linalg.norm(x - self.x_ref))

            t_now = time.time()
            times_cum.append(t_now - t0)
            times_iter.append(t_now - t_prev)
            t_prev = t_now

            iters.append(k); sigmas.append(sigma); inner_list.append(inner_iters)
            objs.append(obj_k); kkts.append(kkt_k); xs.append(x.copy()); dists.append(dist_k)

            if verbose and (k % print_every == 0):
                extra = "" if np.isnan(dist_k) else f"  dist={dist_k:.3e}"
                print(f"[Iter {k:03d}] σ={sigma:.2e}  inner={inner_iters:02d}  "
                      f"obj={obj_k:.6e}  KKT={kkt_k:.2e}{extra}  "
                      f"||∇Ψ||={grad_norm:.2e}  t_iter={times_iter[-1]:.3f}s  t_total={times_cum[-1]:.3f}s")

            # Stop on primal KKT
            if kkt_k < self.tol_kkt:
                stopped_on = "kkt_tol"; stop_iter = k; break

            # Step 3: increase σ ONLY if Step 1 satisfied (A)&(B1)
            if satisfied:
                sigma = min(self.sigma_max, sigma * self.sigma_rate)
                # keep y to warm-start the next subproblem

        total_time_sec = time.time() - t0
        if verbose:
            note = stop_iter if stop_iter >= 0 else iters[-1]
            print(f"Stopped on: {stopped_on} at iter {note}; total runtime = {total_time_sec:.3f}s")

        self.history = {
            "iter": np.array(iters, int),
            "sigma": np.array(sigmas, float),
            "inner_iters": np.array(inner_list, int),
            "obj": np.array(objs, float),
            "kkt": np.array(kkts, float),
            "time_sec": np.array(times_cum, float),
            "iter_time_sec": np.array(times_iter, float),
            "x": np.vstack(xs) if xs else np.empty((0, n)),
            "dist": np.array(dists, float),
            "total_time_sec": float(total_time_sec),
            "stopped_on": stopped_on,
            "stop_iter": int(stop_iter if stop_iter >= 0 else iters[-1]),
        }
        return x, self.history
