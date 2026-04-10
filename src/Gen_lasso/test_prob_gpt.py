import numpy as np

# ---------- Generalized Lasso toy data ----------
def generate_gen_lasso_toy(n=20, m=15, alpha=0.1, snr_db=20.0, seed=0):
    """
    Create a small generalized lasso instance:
        min_x 0.5 ||A x - b||^2 + alpha ||L x||_1
    with L = 1D first-difference operator (fused lasso). Returns dict.
    """
    rng = np.random.default_rng(seed)

    # Ground-truth piecewise-constant x*
    x_true = np.zeros(n)
    # 3 segments with small jumps (change as you like)
    x_true[:n//3]         = 0.5
    x_true[n//3:2*n//3]   = -0.3
    x_true[2*n//3:]       = 0.8

    # Design A: Gaussian, columns normalized
    A = rng.normal(size=(m, n))
    A /= np.linalg.norm(A, axis=0, keepdims=True) + 1e-12

    y_clean = A @ x_true

    b = y_clean + 0.001 * rng.normal(size=m)

    # Generalized lasso penalty matrix L = D (first difference)
    # Shape (n-1, n)
    D = np.zeros((n-1, n))
    for i in range(n-1):
        D[i, i]   = 1.0
        D[i, i+1] = -1.0

    # A simple initial iterate and a dual-like vector for your subproblem API
    yk = np.zeros(n)           # you can also try yk = x_true + small noise
    zk = np.zeros(n)           # placeholder; adjust to your algorithm

    return {
        "A": A, "b": b, "alpha": float(alpha),
        "x_true": x_true, "L": D,   # L is the generalized lasso matrix
        "yk": yk, "zk": zk
    }

# ---------- Functions matching your subproblem ----------
def hessian_f(A):
    """
    For subproblem objective 0.5 ||Q d||^2, we take Q := A.
    This matches f(x) = 0.5||A x - b||^2 where ∇f(x) = A^T(Ax - b).
    """
    return np.asarray(A, dtype=float)

def grad_f(A, x, b):
    """
    Gradient of f(x) = 0.5||A x - b||^2.
    """
    A = np.asarray(A, float)
    x = np.asarray(x, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)
    return A.T @ (A @ x - b)

# ---------- Index set helper for your equality constraints ----------
def prefix_equalities_from_yk(yk, tol=1 - 1e-4):
    """
    Build idx = { i : |sum_{j=1}^i yk[j]| <= tol }, i = 0..n-2
    used to enforce d[i] == d[i+1] in your subproblem.
    """
    yk = np.asarray(yk, float).reshape(-1)
    s = np.cumsum(yk[:-1])  # length n-1
    idx = np.where(np.abs(s) <= tol)[0]
    return idx

# ---------- Tiny end-to-end example ----------
# if __name__ == "__main__":
#     data = generate_gen_lasso_toy(n=20, m=15, alpha=0.15, snr_db=25.0, seed=42)
#     A, b, L, yk, zk, alpha = data["A"], data["b"], data["L"], data["yk"], data["zk"], data["alpha"]

#     # Subproblem pieces
#     Q = hessian_f(A)          # here, Q = A
#     g = grad_f(A, yk, b)      # gradient at yk
#     c = g + zk                # your linear term in the subproblem
#     idx = prefix_equalities_from_yk(yk)  # start with empty set since yk=0

#     print("Shapes: A", A.shape, "b", b.shape, "L", L.shape)
#     print("alpha:", alpha)
#     print("#equalities for d[i]==d[i+1]:", idx.size)
#     print("‖x_true‖₂ =", np.linalg.norm(data["x_true"]))