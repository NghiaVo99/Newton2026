#!/usr/bin/env python3
import numpy as np
import os
from src.lasso.newton_lasso import (
    ISTA,
    FISTA1,
    Algo_Newton_Ista,
    Algo_Newton_BT_Ista,
    Algo_Newton_Fista,
    Algo_Newton_BT_Fista
)
from src.lasso.lasso_GDFBE_LM import lasso_GDFBE_LM
from src.lasso.test_ClassicLasso_random import test_ClassicLasso_random_wrapper

# -------------------------
# Benchmark configuration
# -------------------------
n_list               = [128, 256, 512]
m_ratio_list         = [0.1, 0.15, 0.2]
sparsity_ratio_list  = [0.01, 0.05, 0.1]
num_instances        = 4
#alpha                = 1.0
noise_level          = 0.01
max_iter             = 1000
tol                  = 1e-6
beta                 = 0.5
newton_stepsize      = 1.0

solver_list = [
    ("Newton_Ista",      Algo_Newton_Ista),
    ("Newton_BT_Ista",   Algo_Newton_BT_Ista),
    ("Newton_Fista",     Algo_Newton_Fista),
    ("ISTA",             ISTA),
    ("FISTA",            FISTA1),
    ("Newton_BT_Fista",  Algo_Newton_BT_Fista),
    ("GDFBE_LM",         lasso_GDFBE_LM),
    ("SSNAL",            test_ClassicLasso_random_wrapper),
]
solver_names = [name for name, _ in solver_list]

# Ensure output directory
os.makedirs('results', exist_ok=True)

# -------------------------
# Reference solver for f* (Newton-BT-Fista)
# -------------------------
def solve_lasso_reference(A, b, alpha):
    """
    Compute reference solution x_star and optimal cost f_star using
    Newton-BT-Fista with double the iteration budget.
    """
    n = A.shape[1]
    x0 = np.zeros(n)
    step_size = 1.0 / np.linalg.norm(A, 2)**2
    out = Algo_Newton_BT_Fista(
        A, b, x0, alpha,
        2 * max_iter, step_size,
        beta, newton_stepsize,
        1e-3 * tol, x0
    )
    assert out is not None, "Reference solver returned None"
    cost_list = out[0]
    f_star = cost_list[-1]
    return out[1], f_star

# -------------------------
# Run benchmarks
# -------------------------
results = []        # list of final objective values, one row per problem
f_star_list = []    # list of reference optimal costs
problem_info = []   # list of dicts with (n, m, s, instance)

for n in n_list:
    for m_ratio in m_ratio_list:
        m = max(1, int(m_ratio * n))
        for s_ratio in sparsity_ratio_list:
            s = max(1, int(s_ratio * n))
            for inst in range(num_instances):
                # Generate sparse ground truth and data
                z_true = np.zeros(n)
                idx = np.random.choice(n, s, replace=False)
                z_true[idx] = np.random.randn(s)
                A = np.random.randn(m, n)
                b = A.dot(z_true) + noise_level * np.random.randn(m)
                alpha = 0.001* np.linalg.norm(A.T @ b,np.inf)
                # Compute reference optimum
                try:
                    x_star, f_star = solve_lasso_reference(A, b, alpha)
                except Exception as e:
                    print(f"Reference failed [n={n},m={m},s={s},inst={inst}]: {e}")
                    results.append([np.inf] * len(solver_list))
                    f_star_list.append(np.inf)
                    problem_info.append({'n': n, 'm': m, 's': s, 'instance': inst})
                    continue

                # Run each solver
                f_vals = []
                step_size = 1.0 / np.linalg.norm(A, 2)**2
                x0 = np.zeros(n)

                for name, solver in solver_list:
                    try:
                        # Call with correct signature
                        if name in ["Newton_Ista", "Newton_BT_Ista", "Newton_Fista", "Newton_BT_Fista"]:
                            out = solver(
                                A, b, x0, alpha,
                                max_iter, step_size,
                                beta, newton_stepsize,
                                tol, x_star
                            )
                            assert out is not None, f"Solver {name} returned None"
                            cost_list = out[0]

                        elif name in ["ISTA", "FISTA"]:
                            out = solver(
                                A, b, x0, alpha,
                                max_iter, step_size,
                                tol, x_star
                            )
                            assert out is not None, f"Solver {name} returned None"
                            cost_list = out[0]

                        elif name == "GDFBE_LM":
                            out = solver(
                                A, b, alpha,
                                x_star, tol
                            )
                            assert out is not None, f"Solver {name} returned None"
                            cost_list = out[1]

                        elif name == "SSNAL":
                            out = solver(
                                A, b, x0, alpha,
                                max_iter, step_size,
                                beta, newton_stepsize,
                                tol, x_star
                            )
                            assert out is not None, f"Solver {name} returned None"
                            cost_list = out[1]

                        else:
                            raise ValueError(f"Unknown solver {name}")

                        # Store final cost
                        f_vals.append(cost_list[-1])
                    except Exception as e:
                        print(f"Solver {name} failed [n={n},m={m},s={s},inst={inst}]: {e}")
                        f_vals.append(np.inf)

                # Save problem results
                results.append(f_vals)
                f_star_list.append(f_star)
                problem_info.append({'n': n, 'm': m, 's': s, 'instance': inst})

# Save all results to disk
F_array = np.array(results)      # shape (P, S)
f_star_arr = np.array(f_star_list)  # shape (P,)
np.savez(
    f'results/lasso_benchmark_noise={noise_level}.npz',
    f_vals=F_array,
    f_star=f_star_arr,
    solver_names=solver_names,
    problem_info=problem_info
)
print("Benchmarking complete. Results saved to 'results/lasso_benchmark.npz'.")
