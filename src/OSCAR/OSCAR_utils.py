import numpy as np


def prox_oscar(y: np.ndarray,
               tau: float,
               lam1: float,
               lam2: float,
               positive: bool = False) -> np.ndarray:
    """
    Proximal operator of OSCAR at y:
        prox_{tau * OSCAR}(y) where
        OSCAR(x) = lam1 * ||x||_1 + lam2 * sum_{i<j} max(|x_i|, |x_j|)
                 = sum_{k=1}^n w_k * |x|_(k),  w_k = lam1 + lam2 * (n-k)

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Input vector
    tau : float
        Prox stepsize (must be >= 0)
    lam1, lam2 : float
        OSCAR parameters (assumed >= 0)
    positive : bool, default False
        If True, enforce nonnegativity of the solution (skip sign restoration)

    Returns
    -------
    x : np.ndarray, shape (n,)
        The proximal result
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y.copy()
    if tau < 0 or lam1 < 0 or lam2 < 0:
        raise ValueError("Require tau >= 0, lam1 >= 0, lam2 >= 0.")

    # 1) sort by magnitude (descending)
    if positive:
        abs_y = y.copy()
        abs_y[abs_y < 0] = 0.0
        signs = np.ones_like(y)
    else:
        signs = np.sign(y)
        abs_y = np.abs(y)

    order = np.argsort(-abs_y)         # indices for descending |y|
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n)

    z = abs_y[order]                   # sorted magnitudes

    # 2) build OSCAR / OWL weights in sorted order:
    #    w_k = lam1 + lam2 * (n-k), for k=1..n (k=0..n-1 in 0-based)
    w = lam1 + lam2 * np.arange(n-1, -1, -1, dtype=float)

    # 3) soft-threshold by tau*w then impose nonincreasing constraint via PAV
    u = z - tau * w

    # Pool-Adjacent-Violators (isotonic regression) for nonincreasing and >=0
    # We implement isotonic regression on -u for nondecreasing, then flip back,
    # or directly maintain nonincreasing by merging blocks when averages increase.
    v = _pav_nonincreasing_nonnegative(u)

    # 4) restore original order and signs
    s = v[inv_order]                   # magnitudes in original order
    if not positive:
        x = signs * s
    else:
        x = s
    return x


def _pav_nonincreasing_nonnegative(u: np.ndarray) -> np.ndarray:
    """
    Project vector u onto the cone {v : v >= 0, v1 >= v2 >= ... >= vn}
    using pool-adjacent-violators in O(n).
    Returns v (same shape as u).
    """
    n = u.size
    # We want v = argmin ||v - u||^2 s.t. v nonincreasing, v>=0
    # Implement by maintaining blocks with nonincreasing averages.
    v = np.empty_like(u)
    # block starts/ends and block means
    starts = np.zeros(n, dtype=int)
    ends = np.zeros(n, dtype=int)
    means = np.zeros(n, dtype=float)

    nb = 0  # number of blocks - 1 (index of last block)
    for i in range(n):
        # start new block [i,i] with mean=max(u[i], 0)
        nb += 1
        starts[nb-1] = i
        ends[nb-1] = i
        means[nb-1] = max(u[i], 0.0)

        # merge while the nonincreasing constraint is violated
        while nb >= 2 and means[nb-2] < means[nb-1]:
            # merge block nb-2 and nb-1
            new_start = starts[nb-2]
            new_end = ends[nb-1]
            # pooled average of the two blocks
            s1, e1, m1 = starts[nb-2], ends[nb-2], means[nb-2]
            s2, e2, m2 = starts[nb-1], ends[nb-1], means[nb-1]
            len1 = e1 - s1 + 1
            len2 = e2 - s2 + 1
            new_mean = (len1 * m1 + len2 * m2) / (len1 + len2)

            starts[nb-2] = new_start
            ends[nb-2] = new_end
            means[nb-2] = new_mean
            nb -= 1  # removed last block

    # write block means back
    idx = 0
    for b in range(nb):
        s = starts[b]
        e = ends[b]
        v[s:e+1] = means[b]
        idx = e + 1
    return v


def lipschitz_from_spectral_norm(A):
    """
    L = ||A||_2 using NumPy's spectral norm (largest singular value).
    """
    # np.linalg.norm(A, 2) computes the operator 2-norm (largest singular value)
    s_max = np.linalg.norm(A, 2)
    return s_max**2

def oscar_value(x: np.ndarray, w1: float, w2: float) -> float:
    """
    OSCAR(x) = lam1 * ||x||_1 + lam2 * sum_{i<j} max(|x_i|, |x_j|)
             = sum_{k=1}^n w_k * |x|_{(k)},  with  w_k = lam1 + lam2*(n-k)
    where |x|_{(1)} ≥ … ≥ |x|_{(n)}.
    """
    n = x.size
    sx = np.sort(np.abs(x))[::-1]            # descending
    w = w1 + w2 * (np.arange(n-1, -1, -1))  # [lam1+lam2*(n-1), …, lam1]
    return float(np.dot(w, sx))

def obj_least_squares_plus_oscar(A, b, x, w1, w2) -> float:
    r = A @ x - b
    return 0.5 * float(r @ r) + oscar_value(x, w1, w2)


def build_Q_from_oscar(n, z, w1, w2, atol=1e-9, rtol=1e-7, verbose=True):
    """
    Build an orthonormal basis Q (columns) for
        P = span( ⋃_{k in A(z)} ∂ s_k(z) ),
    with OSCAR/SLOPE weights λ_i = w1 + w2*(n - i).

    Parameters
    ----------
    n : int
        Dimension of z.
    z : (n,) array_like
        Current iterate (real vector). (May be unsorted.)
    w1, w2 : float
        OSCAR/SLOPE parameters (nonnegative). λ_i = w1 + w2*(n - i).
    atol, rtol : float, optional
        Tolerances for equality tests and tie detection.
    verbose : bool
        If True, prints lambdas and cumulative sums.

    Returns
    -------
    Q : (n, r) ndarray
        Orthonormal basis for P (columns), in ORIGINAL coordinate order.
        If P = {0}, r = 0 and Q has shape (n, 0).
    active_idx : list of int
        1-based indices k where s_k(z) = Λ_k (within tolerances).
    lambdas : (n,) ndarray
        OSCAR weights (descending).
    Lambda : (n,) ndarray
        Cumulative prefix sums of lambdas (ascending).
    """
    z = np.asarray(z, dtype=float).reshape(-1)
    if z.size != n:
        raise ValueError("n must equal len(z).")
    if w1 < 0 or w2 < 0:
        raise ValueError("w1 and w2 must be nonnegative.")

    # OSCAR/SLOPE weights
    i = np.arange(1, n+1, dtype=float)
    lambdas = w1 + w2*(n - i)
    Lambda = np.cumsum(lambdas)

    # Sort by |z| descending
    perm = np.argsort(-np.abs(z), kind='mergesort')
    iperm = np.empty_like(perm); iperm[perm] = np.arange(n)
    z_sorted = z[perm]
    r = np.abs(z_sorted)
    sigma = np.sign(z_sorted); sigma[sigma == 0] = 1.0

    # Active set
    s_prefix = np.cumsum(r)
    active_mask = np.isclose(s_prefix, Lambda, rtol=rtol, atol=atol)
    active_k = np.where(active_mask)[0] + 1

   #  if verbose:
   #      print("lambdas =", lambdas)
   #      print("Lambda (cumulative) =", Lambda)
   #      print("s_prefix (from z) =", s_prefix)

    # Early exit
    if active_k.size == 0:
        return np.zeros((n, 0)), [], lambdas, Lambda

    # Tie blocks
    blocks = []
    a = 0
    while a < n:
        b = a
        while b+1 < n and np.isclose(r[b+1], r[a], rtol=rtol, atol=atol):
            b += 1
        blocks.append((a+1, b+1))
        a = b+1
    def block_of_k(k):
        for A, B in blocks:
            if A <= k <= B:
                return A, B
        raise RuntimeError("Internal error: k not in any block")
    seen=set(); active_blocks=[]
    for k in active_k:
        ab = block_of_k(k)
        if ab not in seen:
            active_blocks.append(ab); seen.add(ab)

    # Generators
    gens=[]
    for k in active_k:
        a_k,b_k = block_of_k(k)
        m_k = k-(a_k-1)
        g = np.zeros(n)
        if a_k>1: g[:a_k-1] = sigma[:a_k-1]
        if m_k>0: g[a_k-1:a_k-1+m_k] = sigma[a_k-1:a_k-1+m_k]
        gens.append(g)
    for a_b,b_b in active_blocks:
        for i in range(a_b+1, b_b+1):
            d = np.zeros(n)
            d[i-1]   =  sigma[i-1]
            d[a_b-1] = -sigma[a_b-1]
            gens.append(d)

    G_sorted = np.column_stack(gens) if gens else np.zeros((n,0))
    G = G_sorted[iperm,:]
    if G.size==0:
        Q = np.zeros((n,0))
    else:
        norms = np.linalg.norm(G, axis=0)
        keep = norms > 1e-14*np.sqrt(n)
        G = G[:,keep]
        if G.size==0:
            Q = np.zeros((n,0))
        else:
            Q, _ = np.linalg.qr(G, mode='reduced')

    return Q, active_k.tolist(), lambdas, Lambda
