import numpy as np
import time

def proj(y, l, L):
    return np.minimum(L, np.maximum(l, y))

def fval(A, b, x):
    return 0.5 * x.T @ A @ x + b.T @ x

def psi_qb(A, b, B, x, gamma, l, L):
    p = fval(A, b, x)
    q = -0.5 * gamma * np.linalg.norm(A @ x + b)**2
    y = B @ x - gamma * b
    r = (0.5/gamma) * np.linalg.norm(proj(y, l, L) - y)**2
    return p + q + r

def qb_glmm(A, b, tol, l, L):
    start_time = time.time()
    m, n = A.shape
    x = np.ones(n)
    gamma = 0.5 / np.linalg.norm(A, 2)
    I = np.eye(n)
    B = I - gamma * A

    iter_count = 0
    xlast = np.zeros(n)
    error = abs(fval(A, b, xlast) - fval(A, b, x)) / (1 + abs(fval(A, b, xlast)))

    while error > tol:
        xlast = x.copy()
        iter_count += 1
        u = B @ x - gamma * b
        v = proj(u, l, L)
        psi0 = psi_qb(A, b, B, x, gamma, l, L)
        grad = (1/gamma) * B @ (x - v)
        mu = 0.01 * np.linalg.norm(grad)
        diff = u - v
        zeros_index = np.where(diff == 0)[0]

        X = I.copy()
        for i in zeros_index:
            X[i, :] = gamma * A[i, :]

        M = B @ X + gamma * mu * I
        RHS = B @ (v - x)
        d = np.linalg.solve(M, RHS)

        tau = 1.0
        while psi_qb(A, b, B, x + tau * d, gamma, l, L) > psi0 + 0.1 * tau * grad.dot(d):
            tau /= 2

        x = x + tau * d
        error = abs(fval(A, b, xlast) - fval(A, b, x)) / (1 + abs(fval(A, b, xlast)))

    elapsed = time.time() - start_time
    print(f"Iterations: {iter_count}, Time: {elapsed:.6f}s")
    return x

# Second test example: lower-bound binding
A = np.array([[3.0, 0.0],
              [0.0, 1.0]])
x_true = np.array([0.0, 1.2])
b = -A @ x_true
l = np.array([0.0, 1.0])
L = np.array([2.0, 2.0])
tol = 1e-8

x_est = qb_glmm(A, b, tol, l, L)
print("Recovered x:", np.round(x_est, 6))
print("True x:     ", x_true)
print("Error norm: ", np.linalg.norm(x_est - x_true))