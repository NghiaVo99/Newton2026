import numpy as np
from src.lasso.BaGSS import BasGSSLasso
import matplotlib.pyplot as plt

# Example setup
rng = np.random.default_rng(42)
m, n = 48, 128
A = np.random.randn(m, n) #/ np.sqrt(m)
x_true = np.zeros(n)
sparsity = 8
x_true[rng.choice(n, sparsity, replace=False)] = rng.normal(size=sparsity)
b = A @ x_true + 0.001*np.random.randn(m)

solver = BasGSSLasso(A, b, lambda_reg=1e-2,
                     lambda0=1e-2, lambda_bar=1.0,
                     alpha=0.25, beta=0.5, sigma=0.5, rho_bar=1.0,
                     eps=1e-6, max_iters=200)

x0 = np.zeros(n)
# If you have an approximate solution later, pass approx_solution=... to log distances
result_BaGSS = solver.solve(x0, approx_solution=None)

cost_val_BaGSS = result_BaGSS["history"]["phi_x"]
x_k_BaGSS = result_BaGSS["history"]["dist_x"]
time_k_BaGSS = result_BaGSS["history"]["time"]

print(type(cost_val_BaGSS), type(x_k_BaGSS), type(time_k_BaGSS))
print(len(cost_val_BaGSS), len(x_k_BaGSS), len(time_k_BaGSS))

plt.plot(cost_val_BaGSS, label='Objective Value')
plt.title('BaGSS Objective Value over Iterations')
plt.xlabel('Iterations')
plt.ylabel('f(x_k)')
plt.show()
