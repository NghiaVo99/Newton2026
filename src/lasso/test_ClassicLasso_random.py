import numpy as np
from numpy.linalg import norm
from src.lasso.Classic_Lasso_SSNAL import classic_lasso_ssnal

def test_ClassicLasso_random_wrapper(A,b,x0, alpha, max_iter, step_size, beta, newton_stepsize, tol, approx_sol):

    # 1) Set random seed and generate input data
    # np.random.seed(0)
    # m = 50
    n = A.shape[1]

    # # Create a sparse true solution xx with first 10% entries = xstar
    # k = int(0.1 * n)
    # xx = np.zeros(n)
    # xstar = 1.2 * np.sqrt(2 * np.log(n))
    # xx[:k] = xstar

    # # Generate A and b
    # A = np.random.randn(m, n)
    # b = A.dot(xx)

    # 2) Define Amap and ATmap
    Amap  = lambda x: A.dot(x)
    ATmap = lambda x: A.T.dot(x)

    # 3) Compute lambda_max = ||A' * b||_∞
    lambdamax = norm(ATmap(b), np.inf)

    # 4) Reuse the benchmark Lipschitz estimate.  comparison.py already pays
    # for ||A||_2 when it constructs step_size, so avoid a second full SVD here.
    Lip = 1.0 / float(step_size)

    # print("\n" + "-" * 60)
    # print(f" Problem: n = {n},  m = {m},   lambda(max) = {lambdamax:.4e}")
    # print(f" Lip = {Lip:.4e}")
    # print("-" * 60)

    # 5) Set stopping tolerance
    stoptol = tol if tol is not None else 1e-6

    # 6) Loop over one value of crho = 1e-3 (as in MATLAB)
    for crho in [1e-3]:
        #lam = crho * lambdamax
        lam = alpha
        # Build options dict
        opts = {
            'stoptol': stoptol,
            'Lip':     Lip,
            'maxiter': max_iter,
            'stoptol': tol,
            'approx_sol': approx_sol,
            'printyes': 0,
            'record_history': True,
            # You can add other options here if needed, e.g. 'maxiter', 'printyes', etc.
        }

        # Build Ainput struct to pass into the solver
        Ainput = {
            'A':     A,
            'Amap':  lambda x_vec: Amap(x_vec),
            'ATmap': lambda y_vec: ATmap(y_vec)
        }

        # Call the SSNAL solver from the same primal initial point as the
        # other methods.  y is initialized in the feasible infinity ball.
        x_init = np.asarray(x0, dtype=float).reshape(-1).copy()
        xi_init = np.zeros(A.shape[0])
        y_init = np.sign(x_init) * np.minimum(np.abs(x_init), lam)
        obj, y, xi, x, info, runhist, x_history, time_history = classic_lasso_ssnal(
            Ainput, b, n, lam, opts, y=y_init, xi=xi_init, x=x_init)
        x_history = np.array(x_history, dtype=float)
        if x_history.ndim == 2 and x_history.shape[0] > 0:
            x_history[0, :] = x_init
        time_history = np.asarray(time_history, dtype=float)
        if time_history.size + 1 == x_history.shape[0]:
            time_history = np.concatenate(([0.0], time_history))

        # Print a brief summary of results
        if opts.get('printyes', 0):
            print("\n--- SSNAL output ---")
            print(f"  nonzeros in x (0.999 cutoff): {info['nnz']}")
            print(f"  relgap = {info['relgap']:.2e}, iter = {info['iter']}, time = {info['time']:.2f} s")
            print(f"  min(x) = {info['minx']:.2e}, max(x) = {info['maxx']:.2e}")
            print(f"  primobj = {info['obj'][0]:.8e}, dualobj = {info['obj'][1]:.8e}")
            print(f"  runhist (iter, totaltime) = ({runhist['iter']}, {runhist['totaltime']:.4f})")
        #print('primal_hist', runhist['primobj'])
        return x, runhist['primobj'], x_history, time_history

