import numpy as np, time

def soft_threshold(v, kappa):
    return np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)

class BasGSSLasso:
    """
    BasGSS/GSSN for LASSO:  min 0.5*||Ax-b||^2 + lambda_reg*||x||_1
    direction:
      - 'prox'    : simple clipped step toward T_lambda(x)
      - 'gssn_l1' : SCD semismooth* Newton direction specialized to l1 (Example 3.8)
    Logs: objective values, residual r^k, accumulated time, and (optionally) distances to an approx solution.
    """
    def __init__(self, A, b, lambda_reg,
                 lambda0=1e-2, lambda_bar=1.0,
                 alpha=0.25, beta=0.5, sigma=0.5, rho_bar=1e6,
                 eps=1e-6, max_iters=10000, search_dir='gssn_l1', newton_tol=1e-10,
                 # ---- CG controls ----
                 cg_tol=1e-8, cg_maxit=5000, cg_precond='jacobi',
                 seed=None):
        self.A = A
        self.b = b
        self.lambda_reg = float(lambda_reg)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.sigma = float(sigma)
        self.rho_bar = float(rho_bar)
        self.eps = float(eps)
        self.max_iters = int(max_iters)
        self.lambda0 = float(lambda0)
        self.lambda_bar = float(lambda_bar)
        self.search_dir = search_dir
        self.newton_tol = float(newton_tol)
        # CG params
        self.cg_tol = float(cg_tol)
        self.cg_maxit = int(cg_maxit)
        self.cg_precond = str(cg_precond) if cg_precond is not None else None
        if seed is not None:
            np.random.seed(seed)
        # Precompute
        self.ATA = A.T @ A
        self.ATb = A.T @ b

    # Smooth part: f(x) = 0.5||Ax-b||^2, grad f = A^T(Ax-b)
    def f(self, x):
        r = self.A @ x - self.b
        return 0.5 * np.dot(r, r)

    def gradf(self, x):
        return self.ATA @ x - self.ATb

    # Non-smooth: g(x) = lambda * ||x||_1
    def g(self, x):
        return self.lambda_reg * np.linalg.norm(x, 1)

    # Forward–backward operator T_λ(x) = prox_{λ g}(x - λ∇f(x))
    def T(self, x, lam):
        return soft_threshold(x - lam * self.gradf(x), lam * self.lambda_reg)

    # FBE at (x,λ) evaluated at z = T_λ(x)
    def fbe(self, x, lam, z=None, gradfx=None):
        if z is None:
            z = self.T(x, lam)
        if gradfx is None:
            gradfx = self.gradf(x)
        eta = 0.5 / lam * np.linalg.norm(z - x) ** 2
        return self.f(x) + np.dot(gradfx, z - x) + eta + self.g(z), eta

    # ---------------------------
    # CG solver for (A_S^T A_S) s_S = rhs, without forming A_S^T A_S
    # ---------------------------
    def _cg_AtA_S(self, S, rhs, x0=None):
        """Conjugate Gradient on normal equations restricted to active set S.
        Solves: (A_S^T A_S) s_S = rhs
        Uses Jacobi preconditioning by default: M^{-1} = diag(A_S^T A_S)^{-1}.
        Returns s_S, info dict.
        """
        if S.size == 0:
            return np.zeros_like(rhs), {"converged": True, "iters": 0, "res_norm": 0.0}

        A_S = self.A[:, S]  # shape (m, |S|)
        # matvec: v -> A_S^T (A_S v)
        def matvec(v):
            return A_S.T @ (A_S @ v)

        # Preconditioner (Jacobi) on AtA: diag = sum of column squares of A_S
        if self.cg_precond == 'jacobi':
            diag = np.sum(A_S * A_S, axis=0)  # |S|
            # Guard against zeros (fully dependent columns / empty rows)
            diag = np.where(diag > 0, diag, 1.0)
            def M_inv(v):  # apply M^{-1}
                return v / diag
        else:
            def M_inv(v):
                return v  # identity (no preconditioning)

        # Initialize
        nS = S.size
        x = np.zeros(nS) if x0 is None else x0.copy()
        r = rhs - matvec(x)
        z = M_inv(r)
        p = z.copy()
        rz_old = float(np.dot(r, z))
        tol = self.cg_tol
        maxit = self.cg_maxit

        # Early exit
        if np.sqrt(rz_old) <= tol:
            return x, {"converged": True, "iters": 0, "res_norm": float(np.sqrt(rz_old))}

        converged = False
        it = 0
        for it in range(1, maxit + 1):
            Ap = matvec(p)
            pAp = float(np.dot(p, Ap))
            # Numerical safe-guard
            if pAp <= 0:
                # fallback to steepest descent step
                alpha = rz_old / (np.dot(p, p) + 1e-16)
            else:
                alpha = rz_old / pAp
            x += alpha * p
            r -= alpha * Ap
            # Check residual in M^{-1}-norm
            z = M_inv(r)
            rz_new = float(np.dot(r, z))
            if np.sqrt(rz_new) <= tol:
                converged = True
                break
            beta = rz_new / (rz_old + 1e-30)
            p = z + beta * p
            rz_old = rz_new

        res_norm = float(np.linalg.norm(r))
        return x, {"converged": converged, "iters": it, "res_norm": res_norm}

    def _gssn_direction_l1(self, x, z, lam):
        """
        SCD semismooth* Newton direction specialized to l1:
        1) z_g^* from (4.23)
        2) z^* = ∇f(z) + z_g^*  (4.24)
        3) Active set S (Example 3.8 logic)
        4) Solve (A_S^T A_S) s_S = - z^*_S with CG; set s_{S^c}=0
        """
        # (4.23) and (4.24)
        z_g_star = -self.gradf(x) - (1.0 / lam) * (z - x)  # ∈ ∂g(z)
        z_star = self.gradf(z) + z_g_star                  # ∈ ∂ϕ(z)

        # Active set: always active if z_i ≠ 0;
        # at zeros, active if |z_g_star_i| ≥ λ_reg (boundary)
        tol = 1e-12
        active = (np.abs(z) > 0) | (np.abs(z_g_star) >= self.lambda_reg - tol)
        if not np.any(active):
            return np.zeros_like(z), z_star, z_g_star, active

        S = np.where(active)[0]
        rhs = -z_star[S]  # Newton RHS on active coordinates

        # ---- CG solve on restricted normal equations ----
        s_S, info = self._cg_AtA_S(S, rhs)
        if (not info.get("converged", False)) and np.isfinite(info.get("res_norm", np.inf)):
            # Optional: light fallback if CG stalls (ill-conditioning / tiny |S|)
            try:
                A_S = self.A[:, S]
                # solve small system with lstsq as last resort
                s_S, *_ = np.linalg.lstsq(A_S.T @ A_S, rhs, rcond=None)
            except np.linalg.LinAlgError:
                pass  # keep CG output

        s = np.zeros_like(z)
        s[S] = s_S

        # Optional trust radius (||s|| ≤ ρ̄)
        nrm = np.linalg.norm(s)
        if nrm > self.rho_bar:
            s *= (self.rho_bar / nrm)

        return s, z_star, z_g_star, active

    def solve(self, x0, approx_solution=None, typ_val=None, typ_subgr=None):
        """
        Returns a dict with:
          x, z, iters, history (dict of lists): time, phi_x, phi_z, r, dist_x, dist_z, lambda, tau
        """
        t0 = time.perf_counter()
        x = x0.copy()
        lam = min(self.lambda0, self.lambda_bar)

        # Steps 1–2: backtrack λ until local-model inequality holds
        z = self.T(x, lam)
        gradfx = self.gradf(x)
        fbe_x, eta = self.fbe(x, lam, z=z, gradfx=gradfx)
        while self.f(z) > (self.f(x) + np.dot(gradfx, z - x) + self.alpha * eta):
            lam *= 0.5
            z = self.T(x, lam)
            gradfx = self.gradf(x)
            fbe_x, eta = self.fbe(x, lam, z=z, gradfx=gradfx)

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
        hist = dict(time=[], phi_x=[], phi_z=[], r=[], dist_x=[], dist_z=[], lam=[], tau=[])

        def log_state(tau):
            hist['time'].append(time.perf_counter() - t0)
            hist['phi_x'].append(self.f(x) + self.g(x))
            hist['phi_z'].append(self.f(z) + self.g(z))
            hist['r'].append((1.0 + 1.0 / lam) * np.linalg.norm(x - z))
            hist['lam'].append(lam)
            hist['tau'].append(tau)
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
            s, z_star, z_g_star, active = self._gssn_direction_l1(x, z, lam)

            # Trial step
            tau = 1.0
            x_new = z + tau * s
            lam_new = lam
            z_new = self.T(x_new, lam_new)
            gradfx_new = self.gradf(x_new)
            fbe_new, eta_new = self.fbe(x_new, lam_new, z=z_new, gradfx=gradfx_new)

            # Backtracking on τ and/or λ
            while (fbe_new > fbe_x - self.beta * (1.0 - self.alpha) * eta) or \
                  (self.f(z_new) > (self.f(x_new) + np.dot(gradfx_new, z_new - x_new) + self.alpha * eta_new)):

                if (fbe_new > fbe_x - self.beta * (1.0 - self.alpha) * eta):
                    tau *= 0.5
                    x_new = z + tau * s
                else:
                    lam_new *= 0.5

                z_new = self.T(x_new, lam_new)
                gradfx_new = self.gradf(x_new)
                fbe_new, eta_new = self.fbe(x_new, lam_new, z=z_new, gradfx=gradfx_new)

            # Accept
            x, z, lam = x_new, z_new, lam_new
            fbe_x, eta = fbe_new, eta_new
            gradfx = self.gradf(x)
            print('iter', i, 'cost', self.f(x) + self.g(x))

            # Step 7: λ growth (BaGSS)
            while (self.f(z) <= (self.f(x) + np.dot(gradfx, z - x)) + self.sigma * self.alpha * eta) \
                  and (2.0 * lam <= self.lambda_bar):

                lam_trial = 2.0 * lam
                z_trial   = self.T(x, lam_trial)
                eta_trial = 0.5 / lam_trial * np.linalg.norm(z_trial - x)**2

                if self.f(z_trial) > (self.f(x) + np.dot(gradfx, z_trial - x)) + self.alpha * eta_trial:
                    break
                else:
                    lam = lam_trial
                    z   = z_trial
                    eta = eta_trial

            log_state(tau=tau)

        return dict(x=x, z=z, iters=len(hist['r']) - 1, history=hist)
