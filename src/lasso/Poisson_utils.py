# poisson_sr_utils.py  — adheres to:  min_x  D_KL(z || M H x + b) + λ||x||_1 + δ_{x≥0}(x)
import numpy as np
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
import imageio.v2 as imageio
from numpy.fft import fft2, ifft2, fftshift
from numpy.fft import rfftn, irfftn


# ---------- vectorization helpers ----------
def im2vec(X):  return np.asarray(X, dtype=np.float64).ravel()
def vec2im(x, shape): return np.asarray(x, dtype=np.float64).reshape(shape)

# ---------- H and H^T (linear SAME convolution on HR grid; adjoint = flip) ----------
def conv2_same_linear(x, k):
    """
    Linear (zero-padded) convolution producing a same-size result.
    Pads to (H+Kh-1, W+Kw-1), multiplies in Fourier, inverse, then center-crops.
    """
    x = np.asarray(x, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    H, W = x.shape
    Kh, Kw = k.shape
    shp = (H + Kh - 1, W + Kw - 1)

    X = rfftn(x, s=shp)
    K = rfftn(k, s=shp)
    y = irfftn(X * K, s=shp)

    sh, sw = (Kh - 1) // 2, (Kw - 1) // 2
    return y[sh:sh + H, sw:sw + W]

def make_H_ops(psf_hr, hr_shape):
    """
    Returns H, HT that act on HR images (hr_shape) using linear convolution.
    - H  : X -> k * X (zero-padded, same size)
    - HT : Y -> k^T * Y, where k^T is the 180°-flipped kernel
    PSF is flux-normalized (sum=1).
    """
    psf = np.asarray(psf_hr, dtype=np.float64)
    psf = psf / max(psf.sum(), 1e-12)  # flux-preserving
    k = psf
    kT = psf[::-1, ::-1]

    # Bind to hr_shape for light sanity (optional)
    Hh, Wh = hr_shape
    def _chk(A):
        if A.shape != (Hh, Wh):
            raise ValueError(f"H/HT expects arrays of shape {hr_shape}, got {A.shape}")

    def H(X):
        X = np.asarray(X, dtype=np.float64)
        # _chk(X)   # uncomment if you want strict shape checks
        return conv2_same_linear(X, k)

    def HT(Y):
        Y = np.asarray(Y, dtype=np.float64)
        # _chk(Y)
        return conv2_same_linear(Y, kT)

    return H, HT


#---------- M and M^T (binning downsample & uniform unbinning) ----------
def make_M_ops(scale, hr_shape):
    s = int(scale)
    Hh, Wh = hr_shape
    assert Hh % s == 0 and Wh % s == 0
    Hl, Wl = Hh // s, Wh // s

    def M(X):         # HR -> LR (counts): sum over s×s blocks
        X4 = X.reshape(Hl, s, Wl, s)
        return X4.sum(axis=(1,3))
    def MT(Y):        # LR -> HR: replicate uniformly over s×s blocks
        return np.kron(Y, np.ones((s, s)))
    return M, MT, (Hl, Wl)

# def make_M_ops(M):

#     def M_op(x):
#         return M@x@M.T
#     def MT_op(x):
#         return M.T@x@M
#     return M_op, MT_op

# ---------- Build A = M H and A^T = H^T M^T on VECTORS ----------
def build_ops(psf_hr, scale, lr_shape):
    hr_shape = (lr_shape[0]*scale, lr_shape[1]*scale)
    H, HT = make_H_ops(psf_hr, hr_shape)
    M_op, MT_op, lr_shape_chk = make_M_ops(scale, hr_shape)
    
    #M_op, MT_op = make_M_ops(M)

    def A(x):
        X = vec2im(x, hr_shape)
        return im2vec(M_op(H(X)))
        #return M_op(H(x))
    def AT(y):
        Y = vec2im(y, lr_shape)
        return im2vec(HT(MT_op(Y)))
        #return HT(MT_op(y))
    return A, AT, H, HT, M_op, MT_op, hr_shape, lr_shape

# ---------- Poisson KL data term and gradient ----------
def f_KL(A, x, z, b, eps=1e-12):
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    lam = np.asarray(A(x), dtype=np.float64).reshape(-1) + np.asarray(b, dtype=np.float64).reshape(-1)
    lam = np.clip(lam, eps, None)  # avoid log(0)

    term = np.where(z > 0, z * np.log(z / lam), 0.0) + lam - z
    return float(term.sum())

def grad_KL(A, AT, x, z, b=None, eps=1e-12):
    lam = A(x) + (0.0 if b is None else b)
    lam = np.clip(lam, eps, None)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    grad = AT(1.0 - z/lam)
    return grad


# ---------- Reduced Hessian block on L_k (no explicit A) ----------
def _hessian_vector_block_on_active(A_op, AT_op, x, z, b, d_k, active_idx, eps=1e-12, ridge=1e-10):
    """
    Q = ∇²f(x)[κ, κ] assembled via |κ| Hessian–vec products. κ = active_idx.
    """
    kappa = np.asarray(active_idx, int).ravel()
    k = kappa.size
    Q = np.empty((k, k), float)

    x = np.asarray(x, float).reshape(-1)
    z = np.asarray(z, float).reshape(-1)
    lam = A_op(x)
    if b is not None:
        lam = lam + (float(b) if np.isscalar(b) else np.asarray(b, float).reshape(-1))
    lam = np.clip(lam, eps, None)
    w = z / (lam * lam)

    d_full = np.zeros_like(x)
    d_full[kappa] = d_k
    Hv_full = AT_op(w * A_op(d_full))     # one A, one AT

    return Hv_full[kappa]

# ---------- Effective subspace L_k (parallel space) for g = λ||x||_1 + δ_{R^n_+} ----------
def _Lk_indices_from_zk(zk, lam, tol=1e-12):
    """
    L_k = par ∂g*(z_k) for g as above -> free coords I = { i : (z_k)_i = λ }.
    We use a tolerant test: z_k[i] >= λ - tol.
    Returns the active index set κ.
    """
    zk = np.asarray(zk, float).reshape(-1)
    return np.where(zk >= float(lam) - tol)[0]

# ===================== DROP-IN SUBPROBLEM (same name) =======================

def sub_problem_of_poisson(A, yk, zk, b, alpha, ops=None,
                           active_idx=None, time_limit=3, verbose=False,
                           eps=1e-12, ridge=1e-10):
    """
    Solve:  min_{d in L_k}  0.5 d^T H_f(yk) d - (zk + grad_f(yk))^T d
    where L_k = par ∂ g*(zk) for g(x)=alpha||x||_1 + δ_{x>=0}.
    Returns d_full in R^n with support only on κ (zeros elsewhere).
    """
    # --- Resolve ops / operators
    if isinstance(A, dict):
        ops = A
    if ops is None:
        raise ValueError("Pass your `ops` dict as A or via ops=.")
    A_op  = ops["A"]    # R^n -> R^m
    AT_op = ops["AT"]   # R^m -> R^n
    z_lr  = ops["z"].reshape(-1)
    b_lr  = ops["b"] if b is None else (np.asarray(b, float).reshape(-1) if not np.isscalar(b) else float(b))
    lam   = ops.get("lam", float(alpha)) if alpha is None else float(alpha)

    # --- Prep vectors
    yk = np.asarray(yk, float).reshape(-1)
    zk = np.asarray(zk, float).reshape(-1)

    # λ_y = A yk + b   (intensity); clip for stability
    lam_y = A_op(yk)
    if b_lr is not None:
        lam_y = lam_y + (float(b_lr) if np.isscalar(b_lr) else np.asarray(b_lr, float).reshape(-1))
    lam_y = np.clip(lam_y, eps, None)

    # ∇f(yk) = A^T(1 - z/λ_y)
    grad_f_yk = AT_op(1.0 - (z_lr / lam_y))

    # Effective subspace κ = L_k = {i: z_k[i] >= α}  (for g* of nonneg-ℓ1)
    if active_idx is None:
        kappa = _Lk_indices_from_zk(zk, lam, tol=1e-12)
    else:
        kappa = np.asarray(active_idx, int).ravel()

    n = zk.size
    if kappa.size == 0:
        return np.zeros(n, float)

    k = kappa.size

    # Hessian–vector product: Hf(yk) v = A^T( diag(z/λ_y^2) * (A v) )
    w_diag = (z_lr / (lam_y * lam_y))  # length m

    def Hf_times(v_full):
        Av = A_op(v_full)               # m
        return AT_op(w_diag * Av)       # n

    # Assemble reduced Hessian Q = Hf(yk)[κ,κ] by applying to basis on κ
    Q = np.empty((k, k), float)
    for j in range(k):
        e_full = np.zeros(n, float)
        e_full[kappa[j]] = 1.0
        He = Hf_times(e_full)[kappa]    # take κ-entries only
        Q[:, j] = He
    # Symmetrize + ridge for stability
    Q = 0.5 * (Q + Q.T) + ridge * np.eye(k)

    # Linear term on κ: (zk + ∇f(yk))_κ
    lin = (zk + grad_f_yk)[kappa]

    # Small QP in Gurobi
    env = gp.Env(empty=not verbose)
    if not verbose:
        env.setParam("OutputFlag", 0)
    # >>> ADD THIS <<<
    env.start()                     # <-- start the environment
    # (Optional: if you want to be extra safe with path)
    # env.setParam("LICENSEFILE", "/Users/nghiavo/gurobi.lic")

    m = gp.Model("sub_prob_poisson_effective", env=env)
    m.setParam("TimeLimit", float(time_limit))


    d = m.addMVar(shape=k, lb=-GRB.INFINITY, name="d")

    quad = gp.QuadExpr()
    for i in range(k):
        for j in range(i, k):
            qij = float(Q[i, j])
            if qij == 0.0:
                continue
            coeff = qij if i == j else 2.0 * qij   # we'll multiply the whole quad by 0.5 outside
            quad += coeff * d[i] * d[j]            # <-- use operator overloading, not quad.add(...)

    linexpr = gp.quicksum(float(lin[i]) * d[i] for i in range(k))

    m.setObjective(0.5 * quad - linexpr, GRB.MINIMIZE)
    m.optimize()

    d_full = np.zeros(n, float)
    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        d_sol = d.X if d.X is not None else np.zeros(k)
        d_full[kappa] = d_sol
    return d_full



# ---------- Regularizer g and its prox (λ||x||_1 + δ_{x≥0}) ----------
def prox_g(x, lam):
    # prox_{t (λ||·||1 + δ_{+})}(v) = max(0, v - λ)
    v = np.asarray(x, dtype=np.float64)
    return np.maximum(0.0, v - lam)

def g_val(x, lam):
    x = np.asarray(x, dtype=np.float64)
    if (x < 0).any():  # δ_{+}(x) = +∞ if any negative
        return np.inf
    return float(lam) * np.abs(x).sum()

# ---------- Initialization: back-projection MT HT z, clamped to ≥0 ----------
def init_x0(AT, z, b=None):
    z0 = np.asarray(z, dtype=np.float64).reshape(-1)
    if b is not None:
        z0 = np.maximum(z0 - b, 0.0)
    x0 = AT(z0)
    return np.maximum(x0, 0.0)

def backtracking_linesearch(A,AT,z,b,f, grad_f, prox, x, alpha,
                            L_prev=1.0, eta=2.0, max_tries=50, eps=1e-12):
    """
    Backtracking for the Poisson-KL composite:
        f(x)=D_KL(z || A x + b),  g(x)=alpha*||x||_1 + δ_{x>=0}(x).

    Accept step t=1/L when:
        f(x_new)+g(x_new) <= f(x) + <grad_f(x), d> + 0.5*L*||d||^2 + g(x_new)

    Parameters
    ----------
    f       : callable, f(x)
    grad_f  : callable, grad_f(x)
    prox    : callable, prox_g(v, t*alpha) = max(0, v - t*alpha)
    x       : current point (n,)
    alpha   : scalar λ
    L_prev  : previous L (warm start)
    eta     : backtracking growth factor (>1)
    max_tries : cap on backtracking iterations
    eps     : small positive lower bound on L

    Returns
    -------
    t : accepted step size (= 1/L)
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    L = max(float(L_prev), eps)

    f_x = f_KL(A, x, z, b)
    g_x = g_val(x, alpha)  # optional; not used in the test but cheap to keep around
    #grad_x = np.asarray(grad_f(A, AT, x, z,b), dtype=np.float64).reshape(-1)
    grad_x = np.asarray(grad_KL(A, AT, x, z,b), dtype=np.float64).reshape(-1)

    for _ in range(max_tries):
        t = 1.0 / L
        x_new = prox(x - t * grad_x, t * alpha)     # respects x_new >= 0
        d = x_new - x

        # Quadratic upper bound of f at x plus exact g(x_new)
        Q = f_x + float(grad_x @ d) + 0.5 * L * float(d @ d) + g_val(x_new, alpha)

        # True composite at trial point
        F_trial = cost_poisson(A, x_new, z, b, alpha)

        if F_trial <= Q + 1e-12:
            return t

        L *= eta  # increase curvature (shrink t)

    # conservative fallback
    return 1.0 / L


def cost_poisson(A, x, z, b, lam):
    f_val = f_KL(A, x, z, b)
    g_val_ = g_val(x, lam)
    return f_val + g_val_

import numpy as np
import imageio.v2 as imageio
from pathlib import Path
import glob

def estimate_background_from_stack(lr_folder_or_files, pattern="*.tif", method="median", q=0.1):
    """
    Returns b (LR image) estimated across time from raw LR frames.
    method: "median" (default) or "quantile" (use q in (0,1))
    """
    if isinstance(lr_folder_or_files, (list, tuple)):
        files = list(lr_folder_or_files)
    else:
        files = sorted(glob.glob(str(Path(lr_folder_or_files)/pattern)))
    if not files:
        raise ValueError("No LR frames found for background estimation.")

    b = None
    stack_vals = []
    # Stream-friendly robust estimators:
    if method == "median":
        # Two-pass: collect per-pixel values in chunks if needed
        # (for typical 64x64 LR and ~hundreds of frames, this is fine in memory)
        arrs = [np.asarray(imageio.imread(f), dtype=np.float64) for f in files]
        Y = np.stack(arrs, axis=0)          # (T, H, W)
        b = np.median(Y, axis=0)
    elif method == "quantile":
        arrs = [np.asarray(imageio.imread(f), dtype=np.float64) for f in files]
        Y = np.stack(arrs, axis=0)
        b = np.quantile(Y, q, axis=0)
    else:
        raise ValueError("method must be 'median' or 'quantile'.")

    # Ensure nonnegativity
    b = np.maximum(b, 0.0)
    return b.astype(np.float64)

def estimate_background_scalar(b_map):
    """Collapse a per-pixel b to a single scalar if you prefer uniform background."""
    b_map = np.asarray(b_map, float)
    return float(np.median(b_map))


def make_problem(psf_hr, scale, z_lr_2d, b, x0_mode="backproj", noisy_init_hr=None):
    """
    x0_mode:
        "backproj"     -> x0 = A^T(max(z - b, 0))
        "noisy_lr"     -> x0 = M^T(max(z - b, 0))      (uniform unbin only)
        "noisy_hr"     -> x0 = max(noisy_init_hr, 0)   (caller provides HR image)
    """
    z_lr_2d = np.asarray(z_lr_2d, dtype=np.float64)
    lr_shape = z_lr_2d.shape
    A, AT, H, HT, M, MT, hr_shape, _ = build_ops(psf_hr, scale, lr_shape)

    z_vec = im2vec(z_lr_2d)
    b_vec = b

    f      = lambda x: f_KL(A, x, z_vec, b=b_vec)
    grad_f = lambda x: grad_KL(A, AT, x, z_vec, b=b_vec)
    prox   = lambda v, lam: prox_g(v, lam)
    g      = lambda x, lam: g_val(x, lam)

    # ----- x0 selection -----
    if x0_mode == "backproj":
        # original choice (A^T back-projection)
        x0 = init_x0(AT, z_vec, b=b_vec)

    elif x0_mode == "noisy_lr":
        # use *current noisy LR image* lifted to HR via M^T (no deblur)
        Y = z_lr_2d.astype(np.float64)
        if b is not None:
            Y = np.maximum(Y - b, 0.0)
        X0 = MT(Y)                           # HR shape
        x0 = np.maximum(0.0, im2vec(X0))

    elif x0_mode == "noisy_hr":
        if noisy_init_hr is None:
            raise ValueError("noisy_init_hr must be provided for x0_mode='noisy_hr'.")
        X0 = np.asarray(noisy_init_hr, dtype=np.float64).reshape(hr_shape)
        x0 = np.maximum(0.0, im2vec(X0))

    else:
        raise ValueError("x0_mode must be one of {'backproj','noisy_lr','noisy_hr'}.")

    return dict(
        f=f, grad_f=grad_f, g=g, prox_g=prox,
        A=A, AT=AT,
        hr_shape=hr_shape, lr_shape=lr_shape,
        x0=x0,
        H=H, HT=HT, M=M, MT=MT,
        b=b_vec, z=z_vec
    )


def gaussian_psf_hr_cel0(fwhm_nm=258.2, hr_px_nm=25.0, size=25):
    """
    CEL0-like Gaussian PSF on the HR grid (sum=1, odd size).
    Default matches ISBI 2013 '8 tubes': FWHM=258.2 nm, HR pixel 25 nm.
    """
    import numpy as np
    assert size % 2 == 1, "PSF size must be odd."
    sigma_nm = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_px = sigma_nm / hr_px_nm
    ax = np.arange(-(size//2), size//2 + 1)
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    K = np.exp(-(X**2 + Y**2) / (2.0 * sigma_px**2))
    K /= K.sum() + 1e-12
    return K.astype(np.float64)

def lambda_max(A, AT, z, b, eps=1e-12):
    z = np.asarray(z, float).ravel()
    b = np.asarray(b, float).ravel()
    grad0 = AT(1.0 - z/np.maximum(b, eps))   # ∇f(0)
    return float(np.max(np.maximum(grad0, 0.0)))





