import numpy as np
import time

def prox(y, mu, gamma):
    return np.sign(y) * np.maximum(np.abs(y) - mu * gamma, 0)

def value_of_function(A, b, mu, x):
        return 0.5 * np.linalg.norm(A @ x - b)**2 + mu * np.linalg.norm(x, 1)

def psi(z, gamma, A, b, mu):
    p = A @ z - b
    q = A.T @ p
    r = prox(z - gamma * q, mu, gamma)
    return 0.5 * np.linalg.norm(p) ** 2 - 0.5 * gamma * np.linalg.norm(q) ** 2 + mu * np.linalg.norm(r, 1) + (0.5 / gamma) * np.linalg.norm(z - gamma * q - r) ** 2

def lasso_GDFBE_LM(A, b, mu, approx_sol, tol):
    start_time = time.time()
    m, n = A.shape
    x = np.zeros(n)
    ATA = A.T @ A
    c = A.T @ b
    gamma = 0.5 / np.linalg.norm(ATA, 2)
    I = np.eye(n)
    B = I - gamma * ATA
    C = np.linalg.inv(B)
    optim_cost = value_of_function(A,b,mu,approx_sol)

    iter_count = 1
    cost_hist, x_hist, time_hist = [], [], []
    residual = np.linalg.norm(x - prox(x - ATA @ x + A.T @ b, mu, 1)) / (1 + np.linalg.norm(x) + np.linalg.norm(A @ x - b))

    while residual > tol and time.time() - start_time < 10000:
        #if abs(value_of_function(A,b,mu,x) - optim_cost) > tol:
        iter_count += 1
        u = x - gamma * (ATA @ x - c)
        v = prox(u, mu, gamma)
        mu_k = (1 / gamma) * np.linalg.norm(B @ (x - v))
        psi0 = psi(x, gamma, A, b, mu)
        grad = (1 / gamma) * B @ (x - v)
        zeros_index = np.where(v == 0)[0]

        X = gamma * ATA + gamma * mu_k * C
        for i in zeros_index:
            X[i, :] = I[i, :] + gamma * mu_k * C[i, :]

        d = np.linalg.solve(X, v - x)

        tau = 1.0
        while psi(x + tau * d, gamma, A, b, mu) > psi0 + 0.1 * tau * grad @ d:
            tau /= 2

        x = x + tau * d
        time_hist.append(time.time() - start_time)
        x_hist.append(np.linalg.norm(x - approx_sol))
        cost_hist.append(value_of_function(A,b,mu,x))
        residual = np.linalg.norm(x - prox(x - ATA @ x + A.T @ b, mu, 1)) / (1 + np.linalg.norm(x) + np.linalg.norm(A @ x - b))

    # print(f"\nIterations: {iter_count}")
    # print(f"Time: {time.time() - start_time:.2f} seconds")
    # print(f"Value_GCNM_LM: {0.5 * np.linalg.norm(A @ x - b) ** 2 + mu * np.linalg.norm(x, 1)}")
    # print(f"Residual: {residual}")
    # print(f"Norm: {np.linalg.norm(x)}")

    return x, cost_hist, x_hist, time_hist




# A = np.array([[1.0, 2.0, 0.0],
#               [0.0, 3.0, 4.0],
#               [5.0, 0.0, 6.0]])

# x_true = np.array([1.0, 0.0, -1.0])
# b = A @ x_true

# mu = 0.01
# tol = 1e-8

# x_est = lasso_GDFBE_LM(A, b, mu, tol)
# print("Recovered x:", np.round(x_est, 6))
# print("True x:     ", x_true)
# print("Error norm: ", np.linalg.norm(x_est - x_true))

