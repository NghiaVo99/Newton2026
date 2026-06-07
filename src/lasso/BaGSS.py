import numpy as np, time

try:
    from scipy import sparse
except Exception:  # pragma: no cover - scipy is optional for dense-only use
    sparse = None

def soft_threshold(v, kappa):
    return np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)

class BasGSSLasso:
    """
    BasGSS/GSSN for Lasso: min 0.5*||Ax-b||^2 + lambda_reg*||x||_1

    The globalization follows BasGSSN from Section 4 of the GSSN paper.  The
    default direction is the Section 5 SCD semismooth* Newton direction,
    computed on the Lasso active subspace by trust-region CG.

    direction:
      - 'prox'    : pure forward-backward fallback, x_new = z
      - 'gssn_l1' : SCD semismooth* Newton direction specialized to l1
    """
    def __init__(self, A, b, lambda_reg,
                 lambda0=1e-2, lambda_bar=1.0,
                 alpha=0.25, beta=0.5, sigma=0.5, rho_bar=1e6,
                 eps=1e-6, max_iters=10000, search_dir='gssn_l1', newton_tol=1e-10,
                 # ---- CG controls ----
                 cg_tol=1e-8, cg_maxit=5000, cg_precond='jacobi',
                 cg_ridge=0.0, dense_fallback_max_size=500,
                 seed=None, verbose=False,
                 active_tol=1e-12, rho_min=1e-12, trust_radius0=None,
                 xi_min=1e-12, xi_max=0.5, max_backtracks=1000):
        self.A = A
        self.b = np.asarray(b, dtype=float).reshape(-1)
        self.lambda_reg = float(lambda_reg)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.sigma = float(sigma)
        self.rho_bar = float(rho_bar)
        self.rho_min = float(rho_min)
        rho0 = self.rho_bar if trust_radius0 is None else float(trust_radius0)
        self.rho = min(self.rho_bar, max(self.rho_min, rho0))
        self.eps = float(eps)
        self.max_iters = int(max_iters)
        self.lambda0 = float(lambda0)
        self.lambda_bar = float(lambda_bar)
        if search_dir not in {'gssn_l1', 'prox'}:
            raise ValueError("search_dir must be either 'gssn_l1' or 'prox'")
        self.search_dir = search_dir
        self.newton_tol = float(newton_tol)
        self.verbose = bool(verbose)
        self.active_tol = float(active_tol)
        self.xi_min = float(xi_min)
        self.xi_max = float(xi_max)
        self.max_backtracks = int(max_backtracks)
        # CG params
        self.cg_tol = float(cg_tol)
        self.cg_maxit = int(cg_maxit)
        self.cg_precond = str(cg_precond) if cg_precond is not None else None
        self.cg_ridge = float(cg_ridge)
        self.dense_fallback_max_size = int(dense_fallback_max_size)
        if seed is not None:
            np.random.seed(seed)
        # Precompute
        self.ATA = A.T @ A
        self.ATb = np.asarray(A.T @ self.b).reshape(-1)

    # Smooth part: f(x) = 0.5||Ax-b||^2, grad f = A^T(Ax-b)
    def f(self, x):
        r = np.asarray(self.A @ x - self.b).reshape(-1)
        return 0.5 * np.dot(r, r)

    def gradf(self, x):
        return np.asarray(self.ATA @ x - self.ATb).reshape(-1)

    # Non-smooth: g(x) = lambda * ||x||_1
    def g(self, x):
        return self.lambda_reg * np.linalg.norm(x, 1)

    # Forward–backward operator T_λ(x) = prox_{λ g}(x - λ∇f(x))
    def T(self, x, lam):
        return soft_threshold(x - lam * self.gradf(x), lam * self.lambda_reg)

    # FBE at (x,λ) evaluated at z = T_λ(x)
    def fbe(self, x, lam, z=None, gradfx=None, fx=None):
        if z is None:
            z = self.T(x, lam)
        if gradfx is None:
            gradfx = self.gradf(x)
        if fx is None:
            fx = self.f(x)
        eta = 0.5 / lam * np.linalg.norm(z - x) ** 2
        return fx + np.dot(gradfx, z - x) + eta + self.g(z), eta

    def _is_sparse(self, A):
        return sparse is not None and sparse.issparse(A)

    def _column_sq_norms(self, A_S):
        if self._is_sparse(A_S):
            return np.asarray(A_S.multiply(A_S).sum(axis=0)).reshape(-1)
        return np.sum(np.asarray(A_S) * np.asarray(A_S), axis=0)

    def _xi_from_subgrad_norm(self, subgrad_norm):
        if subgrad_norm <= 0.0:
            return self.xi_min
        t = min(float(subgrad_norm), 1.0)
        xi = 0.1 / (1.0 - np.log(max(t, np.finfo(float).tiny)))
        return float(np.clip(xi, self.xi_min, self.xi_max))

    def _solve_small_trust_region(self, H, g, rho):
        """Dense exact trust-region solve for dimension <= 2."""
        if rho <= 0.0 or g.size == 0:
            return np.zeros_like(g)

        H = 0.5 * (H + H.T)
        eigvals, eigvecs = np.linalg.eigh(H)
        g_hat = eigvecs.T @ g

        if eigvals[0] > 0.0:
            y_unc = -eigvecs @ (g_hat / eigvals)
            if np.linalg.norm(y_unc) <= rho:
                return y_unc

        lam_low = max(0.0, -float(eigvals[0]))

        def y_at(lam):
            denom = eigvals + lam
            return -eigvecs @ (g_hat / denom)

        finite_at_low = True
        denom_low = eigvals + lam_low
        if np.any(np.abs(denom_low) <= 1e-14):
            zero_denom = np.abs(denom_low) <= 1e-14
            finite_at_low = np.all(np.abs(g_hat[zero_denom]) <= 1e-12)

        if finite_at_low:
            inv = np.zeros_like(g_hat)
            nonzero = np.abs(denom_low) > 1e-14
            inv[nonzero] = -g_hat[nonzero] / denom_low[nonzero]
            y_low = eigvecs @ inv
            y_low_norm = np.linalg.norm(y_low)
            if y_low_norm <= rho:
                min_eig_space = np.where(
                    np.abs(eigvals - eigvals[0]) <= 1e-12)[0]
                direction = eigvecs[:, min_eig_space[0]]
                return y_low + np.sqrt(max(0.0, rho * rho - y_low_norm * y_low_norm)) * direction

        lam_hi = max(1.0, lam_low + 1.0)
        while True:
            denom = eigvals + lam_hi
            if np.all(denom > 0.0):
                y_hi = y_at(lam_hi)
                if np.linalg.norm(y_hi) <= rho:
                    break
            lam_hi *= 2.0

        lam_lo = lam_low
        if not np.all(eigvals + lam_lo > 0.0):
            lam_lo = lam_low + 1e-14
        for _ in range(100):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            y_mid = y_at(lam_mid)
            if np.linalg.norm(y_mid) > rho:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
        return y_at(lam_hi)

    def _solve_span_trust_region(self, basis_cols, matvec, g, rho):
        """Solve (5.42) in the span of the supplied reduced-space columns."""
        B = np.column_stack([c for c in basis_cols if np.linalg.norm(c) > 0.0])
        if B.size == 0:
            return np.zeros_like(g)

        Q, R = np.linalg.qr(B, mode='reduced')
        keep = np.abs(np.diag(R)) > 1e-12
        Q = Q[:, keep]
        if Q.size == 0:
            return np.zeros_like(g)

        HQ = np.column_stack([matvec(Q[:, j]) for j in range(Q.shape[1])])
        H_small = Q.T @ HQ
        g_small = Q.T @ g
        y = self._solve_small_trust_region(H_small, g_small, rho)
        return Q @ y

    def _trcg_AtA_S(self, S, z_star, rho):
        """Algorithm 2 from Section 5 on the reduced Lasso subspace.

        The Lasso active basis Z is the coordinate selector for S, so Z*u is
        represented directly by the reduced vector u.
        """
        subgrad_norm = float(np.linalg.norm(z_star))
        xi = self._xi_from_subgrad_norm(subgrad_norm)
        target = xi * subgrad_norm

        if S.size == 0:
            return np.zeros(0), {
                "converged": True, "iters": 0, "res_norm": 0.0,
                "rel_res_norm": 0.0, "xi": xi, "status": "empty_active",
            }

        A_S = self.A[:, S]  # shape (m, |S|)

        def matvec(v):
            return np.asarray(A_S.T @ (A_S @ v)).reshape(-1)

        if self.cg_precond == 'jacobi':
            diag = self._column_sq_norms(A_S)
            diag = diag + max(0.0, self.cg_ridge)
            diag = np.where(diag > np.finfo(float).eps, diag, 1.0)
            def M_inv(v):
                return v / diag
        else:
            def M_inv(v):
                return v

        nS = S.size
        u = np.zeros(nS)
        g_S = np.asarray(z_star[S], dtype=float).reshape(-1)
        r = g_S.copy()
        res_norm = float(np.linalg.norm(r))
        if res_norm <= target or rho <= 0.0:
            status = "residual" if res_norm <= target else "zero_radius"
            rel = res_norm / max(subgrad_norm, np.finfo(float).tiny)
            return u, {
                "converged": res_norm <= target, "iters": 0,
                "res_norm": res_norm, "rel_res_norm": rel,
                "xi": xi, "status": status,
            }

        Cinv_r = M_inv(r)
        p = -Cinv_r
        p0 = p.copy()
        rz_old = float(np.dot(r, Cinv_r))
        converged = False
        it = 0
        status = "maxit"
        for it in range(1, self.cg_maxit + 1):
            Hp = matvec(p)
            pHp = float(np.dot(p, Hp))
            if pHp <= np.finfo(float).eps:
                u = self._solve_span_trust_region([p0, p], matvec, g_S, rho)
                status = "negative_curvature"
                break

            alpha_cg = rz_old / pHp
            u = u + alpha_cg * p
            r = r + alpha_cg * Hp
            res_norm = float(np.linalg.norm(r))
            if res_norm <= target:
                converged = True
                status = "residual"
                break

            if np.linalg.norm(u) > rho:
                status = "boundary"
                break

            Cinv_r = M_inv(r)
            rz_new = float(np.dot(r, Cinv_r))
            if rz_new <= 0.0:
                status = "nonpositive_preconditioned_residual"
                break
            beta_cg = rz_new / max(rz_old, np.finfo(float).tiny)
            p = -Cinv_r + beta_cg * p
            rz_old = rz_new

        u_norm = float(np.linalg.norm(u))
        if u_norm > rho:
            u = (rho / u_norm) * u

        final_r = matvec(u) + g_S
        res_norm = float(np.linalg.norm(final_r))
        rel = res_norm / max(subgrad_norm, np.finfo(float).tiny)
        return u, {
            "converged": converged, "iters": it, "res_norm": res_norm,
            "rel_res_norm": rel, "xi": xi, "status": status,
        }

    def _gssn_direction_l1(self, x, z, lam, rho=None, return_info=False):
        """
        SCD semismooth* Newton direction specialized to l1:
        1) z_g^* from (4.23)
        2) z^* = grad f(z) + z_g^* from (4.24)
        3) For q=1 Lasso, rge P is spanned by nonzero components of z.
        4) Solve the reduced trust-region quadratic subproblem from (5.40).
        """
        z_g_star = -self.gradf(x) - (1.0 / lam) * (z - x)  # ∈ ∂g(z)
        z_star = self.gradf(z) + z_g_star                  # ∈ ∂ϕ(z)
        subgrad_norm = float(np.linalg.norm(z_star))

        if self.search_dir == 'prox':
            active = np.zeros_like(z, dtype=bool)
            info = {
                "converged": True, "iters": 0, "res_norm": 0.0,
                "rel_res_norm": 0.0, "xi": 0.0, "status": "prox",
                "active_size": 0, "subgrad_norm": subgrad_norm,
            }
            result = (np.zeros_like(z), z_star, z_g_star, active, info)
            return result if return_info else result[:4]

        active = np.abs(z) > self.active_tol
        if not np.any(active):
            info = {
                "converged": True, "iters": 0, "res_norm": 0.0,
                "rel_res_norm": 0.0, "xi": self._xi_from_subgrad_norm(subgrad_norm),
                "status": "empty_active", "active_size": 0,
                "subgrad_norm": subgrad_norm,
            }
            result = (np.zeros_like(z), z_star, z_g_star, active, info)
            return result if return_info else result[:4]

        S = np.where(active)[0]
        rho = self.rho if rho is None else float(rho)
        s_S, info = self._trcg_AtA_S(S, z_star, rho)

        s = np.zeros_like(z)
        s[S] = s_S
        info = dict(info)
        info["active_size"] = int(S.size)
        info["subgrad_norm"] = subgrad_norm
        result = (s, z_star, z_g_star, active, info)
        return result if return_info else result[:4]

    def _update_trust_radius(self, tau, s, rho_used):
        s_norm = float(np.linalg.norm(s))
        if tau < 0.25:
            self.rho = max(self.rho_min, 0.5 * s_norm)
        elif tau == 1.0 and np.isclose(s_norm, rho_used, rtol=1e-8, atol=1e-12):
            self.rho = min(self.rho_bar, max(self.rho_min, 1.5 * s_norm))
        return self.rho

    def solve(self, x0, approx_solution=None, typ_val=None, typ_subgr=None):
        """
        Returns a dict with:
          x, z, iters, history (dict of lists)
        """
        t0 = time.perf_counter()
        x = np.asarray(x0, dtype=float).reshape(-1).copy()
        lam = min(self.lambda0, self.lambda_bar)

        # Steps 1–2: backtrack λ until local-model inequality holds
        z = self.T(x, lam)
        gradfx = self.gradf(x)
        fx = self.f(x)
        fbe_x, eta = self.fbe(x, lam, z=z, gradfx=gradfx, fx=fx)
        while self.f(z) > (fx + np.dot(gradfx, z - x) + self.alpha * eta):
            lam *= 0.5
            z = self.T(x, lam)
            gradfx = self.gradf(x)
            fx = self.f(x)
            fbe_x, eta = self.fbe(x, lam, z=z, gradfx=gradfx, fx=fx)

        # Init residual, typical magnitudes
        r = (1.0 + 1.0 / lam) * np.linalg.norm(x - z)
        r0 = r
        if typ_val is None:
            typ_val = r0

        # Typical subgradient magnitude from z^* (4.24)
        z_g_star0 = -self.gradf(x) - (1.0 / lam) * (z - x)  # (4.23)
        z_star0 = self.gradf(z) + z_g_star0                 # (4.24)
        if typ_subgr is None:
            typ_subgr = np.linalg.norm(z_star0)

        # Logs
        hist = dict(
            time=[], phi_x=[], phi_z=[], r=[], dist_x=[], dist_z=[],
            lam=[], tau=[], fbe=[], eta=[], rho=[], cg_iters=[],
            cg_residual=[], active_size=[], subgrad_norm=[],
        )

        def log_state(tau, dir_info=None):
            if dir_info is None:
                dir_info = {}
            hist['time'].append(time.perf_counter() - t0)
            hist['phi_x'].append(self.f(x) + self.g(x))
            hist['phi_z'].append(self.f(z) + self.g(z))
            hist['r'].append((1.0 + 1.0 / lam) * np.linalg.norm(x - z))
            hist['lam'].append(lam)
            hist['tau'].append(tau)
            hist['fbe'].append(fbe_x)
            hist['eta'].append(eta)
            hist['rho'].append(self.rho)
            hist['cg_iters'].append(int(dir_info.get("iters", 0)))
            hist['cg_residual'].append(float(dir_info.get("res_norm", 0.0)))
            hist['active_size'].append(int(dir_info.get("active_size", 0)))
            hist['subgrad_norm'].append(float(dir_info.get("subgrad_norm", 0.0)))
            if approx_solution is not None:
                hist['dist_x'].append(np.linalg.norm(x - approx_solution))
                hist['dist_z'].append(np.linalg.norm(z - approx_solution))
            else:
                hist['dist_x'].append(np.nan)
                hist['dist_z'].append(np.nan)

        log_state(tau=1.0)

        # ---- Main loop ----
        for i in range(self.max_iters):
            r = hist['r'][-1]
            if r <= self.eps * max(typ_val, r0):
                break

            # Direction (GSSN)
            rho_used = self.rho
            s, z_star, z_g_star, active, dir_info = self._gssn_direction_l1(
                x, z, lam, rho=rho_used, return_info=True)

            # Trial step
            tau = 1.0
            x_new = z + tau * s
            lam_new = lam
            z_new = self.T(x_new, lam_new)
            gradfx_new = self.gradf(x_new)
            fx_new = self.f(x_new)
            fbe_new, eta_new = self.fbe(
                x_new, lam_new, z=z_new, gradfx=gradfx_new, fx=fx_new)

            # Backtracking on τ and/or λ
            backtracks = 0
            while (fbe_new > fbe_x - self.beta * (1.0 - self.alpha) * eta) or \
                  (self.f(z_new) > (fx_new + np.dot(gradfx_new, z_new - x_new) + self.alpha * eta_new)):

                if (fbe_new > fbe_x - self.beta * (1.0 - self.alpha) * eta):
                    tau *= 0.5
                    x_new = z + tau * s
                else:
                    lam_new *= 0.5

                z_new = self.T(x_new, lam_new)
                gradfx_new = self.gradf(x_new)
                fx_new = self.f(x_new)
                fbe_new, eta_new = self.fbe(
                    x_new, lam_new, z=z_new, gradfx=gradfx_new, fx=fx_new)
                backtracks += 1
                if backtracks > self.max_backtracks:
                    raise RuntimeError("BasGSS backtracking exceeded max_backtracks")

            # Accept
            x, z, lam = x_new, z_new, lam_new
            fbe_x, eta = fbe_new, eta_new
            gradfx = self.gradf(x)
            fx = self.f(x)
            if self.verbose:
                print('iter', i, 'cost', fx + self.g(x))

            # Step 7: λ growth (BaGSS)
            while (self.f(z) <= (fx + np.dot(gradfx, z - x)) + self.sigma * self.alpha * eta) \
                  and (2.0 * lam <= self.lambda_bar):

                lam_trial = 2.0 * lam
                z_trial   = self.T(x, lam_trial)
                eta_trial = 0.5 / lam_trial * np.linalg.norm(z_trial - x)**2

                if self.f(z_trial) > (fx + np.dot(gradfx, z_trial - x)) + self.alpha * eta_trial:
                    break
                else:
                    lam = lam_trial
                    z   = z_trial
                    eta = eta_trial

            fbe_x, eta = self.fbe(x, lam, z=z, gradfx=gradfx, fx=fx)
            self._update_trust_radius(tau, s, rho_used)
            log_state(tau=tau, dir_info=dir_info)

        return dict(x=x, z=z, iters=len(hist['r']) - 1, history=hist)
