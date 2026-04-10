import numpy as np
import time
from matplotlib import pyplot as plt
from src.lasso.Classic_Lasso_SSNAL_main import *
from src.lasso.Classic_Lasso_SSNAL import classic_lasso_ssnal

# 1. Build a simple test problem
Ainput = np.array([[3, 1],
                   [1, 5]])          # 2×2 matrix
b      = np.array([1.0, 1.5])         # length‐2 vector
lam    = 1.5
n      = 2

# 2. Assemble your options dict, making sure to include tstart & orgojbconst
options = {
    'maxiter':   10,     # just for a quick test
    'stoptol':   1e-6,
    'printyes':  1,
    'rescale':   1,
    'Lip':       1.0,     # Lipschitz constant (must be > 0)
    'orgojbconst': 0.0,   # if you have no constant shift in the objective
    'tstart':    time.time(),
    # (you can add 'sigma' if you want to override the default)
}

# 3. Call the top‐level wrapper
obj, y, xi, x, info, runhist, x_hist = classic_lasso_ssnal(Ainput, b, n, lam, options)

# 4. Inspect results
print("\nFinal x:", x)
print("Final ξ:", xi)
print("Final y:", y)
print("Primal obj, Dual obj:", obj)
print("Info keys:", info.keys())
print("Runhist keys:", runhist.keys())
print('primal_hist', runhist['primobj'])
print('x_hist', x_hist)

plt.plot(x_hist)
plt.show()
