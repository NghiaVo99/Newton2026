import numpy as np
import time
from scipy.linalg import solve
from scipy.sparse.linalg import eigs

def lasso_GDNM(A, b, mu, approx_sol, max_iter=100, tol=1e-10):
    ATA = A.T @ A
    gamma = 0.5 / eigs(ATA, k=1, which='LM', return_eigenvectors=False)[0].real

    def prox_of_function(gamma, y, mu):
        value = []
        zeros_index = []
        for i in range(len(y)):
            if y[i] > mu * gamma:
                value.append(y[i] - mu * gamma)
            elif y[i] < -mu * gamma:
                value.append(y[i] + mu * gamma)
            else:
                value.append(0)
                zeros_index.append(i)
        return np.array(value), zeros_index

    Q = np.linalg.inv(np.eye(ATA.shape[0]) - gamma * ATA)
    P = Q - np.eye(Q.shape[0])

    c = -gamma * Q @ A.T @ b

    def value_of_function(A, b, mu, x):
        return 0.5 * np.linalg.norm(A @ x - b)**2 + mu * np.linalg.norm(x, 1)

    def phi(A, b, gamma, mu, y):
        prox_y, zeros_index = prox_of_function(gamma, y, mu)
        return 0.5 * (P @ y).T @ y + c.T @ y + gamma * mu * np.linalg.norm(prox_y, 1) + 0.5 * np.linalg.norm(y - prox_y)**2

    def finding_gradient(A, b, gamma, mu, y):
        prox_y, _ = prox_of_function(gamma, y, mu)
        return Q @ y - prox_y + c

    def finding_dk(A, b, gamma, mu, y):
        grad = finding_gradient(A, b, gamma, mu, y)
        prox_y, zeros_index = prox_of_function(gamma, y, mu)
        X = P.copy()
        for i in zeros_index:
            X[i, :] = Q[i, :]
        return -solve(X, grad)

    def finding_tk(A, b, gamma, mu, y):
        tau = 1.0
        dk = finding_dk(A, b, gamma, mu, y)
        grad = finding_gradient(A, b, gamma, mu, y)
        while phi(A, b, gamma, mu, y + tau * dk) - phi(A, b, gamma, mu, y) - 0.1 * tau * grad.T @ dk > 0:
            tau /= 2
        return tau

    y = np.zeros_like(c)
    optim_cost = value_of_function(A, b, mu, approx_sol)
    cost_hist, x_hist, time_hist = [], [], []
    start_time = time.time()
    for iter in range(max_iter):
        tk = finding_tk(A, b, gamma, mu, y)
        dk = finding_dk(A, b, gamma, mu, y)
        y = y + tk * dk
        x = Q @ y + c
        val = value_of_function(A, b, mu, x)
        time_hist.append(time.time() - start_time)
        cost_hist.append(val)
        x_hist.append(np.linalg.norm(x - approx_sol))

        if abs(val - optim_cost) < tol:
            return x, cost_hist, x_hist, time_hist

        #print(f"Iter {iter}: Objective value = {val}")

    #x = Q @ y + c
    return x, cost_hist, x_hist, time_hist


