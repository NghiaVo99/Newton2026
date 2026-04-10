#!/usr/bin/env python3
"""
Converted from the original MATLAB script to Python.

Assumes:
  - You have SciPy and NumPy installed.
  - The .mat files live in a directory named "UCIdata" under the current working directory.
  - A Python version of `Classic_Lasso_SSNAL` is available and accepts the same signature:
      Classic_Lasso_SSNAL(Ainput, b, n, rho, opts)
    and returns:
      obj, y, xi, x, info, runhist
"""

import os
import sys
import numpy as np
from scipy.io import loadmat
from scipy.sparse.linalg import eigsh

# For reproducibility, match rng('default') in MATLAB
np.random.seed(0)

def main():
    HOME = os.getcwd()
    datadir = os.path.join(HOME, "UCIdata")

    # List of filenames (MATLAB used 1-based indexing). We'll keep the same order.
    fname_list = [
        "E2006.train",
        "log1p.E2006.train",
        "E2006.test",
        "log1p.E2006.test",
        "pyrim_scale_expanded5",
        "triazines_scale_expanded4",
        "abalone_scale_expanded7",
        "bodyfat_scale_expanded7",
        "housing_scale_expanded7",
        "mpg_scale_expanded7",
        "space_ga_scale_expanded9",
    ]

    # In the original MATLAB: for i = [5]
    # MATLAB’s fname{5} is "pyrim_scale_expanded5".
    # To replicate exactly, we’ll process only that one file.
    target_index = 4  # zero-based index for "pyrim_scale_expanded5"

    dataset_name = fname_list[target_index]
    probname = os.path.join(datadir, dataset_name)
    print(f"\nProblem name: {dataset_name}")

    mat_path = probname + ".mat"
    if not os.path.isfile(mat_path):
        print("Cannot find the .mat file in UCIdata:")
        print(f"  Expected at: {mat_path}")
        sys.exit(1)

    # Load A and b from the .mat file
    data = loadmat(mat_path)
    if "A" not in data or "b" not in data:
        print("The .mat file does not contain variables 'A' and 'b'.")
        sys.exit(1)

    A = data["A"]
    b = data["b"].ravel()  # ensure b is a 1D array

    m, n = A.shape

    # Define Amap and ATmap exactly as in MATLAB (no mexMatvec assumed)
    def Amap(x):
        return A.dot(x)

    def ATmap(y):
        return A.T.dot(y)

    # Define AATmap(x) = A * (A^T * x)
    def AATmap(x):
        return Amap(ATmap(x))

    # Compute the Lipschitz constant Lip = largest eigenvalue of (A * A^T)
    # In MATLAB: Lip = eigs(AATmap, length(b), 1, 'LA', eigsopt)
    # In Python, we can form AAT = A @ A^T and call eigsh(..., which='LA')
    AAT = A.dot(A.T)
    # Since AAT is symmetric, eigsh is appropriate.
    # We want the largest algebraic eigenvalue => which="LA"
    eigvals, _ = eigsh(AAT, k=1, which="LA")
    Lip = float(eigvals[0])

    print(f"Lipschitz constant (Lip) = {Lip:.2e}, norm(b) = {np.linalg.norm(b):.2e}")

    # Loop over crho values [3, 4] (MATLAB's intended loop).
    for crho in [4]:
        c = 10.0 ** (-crho)
        rho = c * np.max(np.abs(ATmap(b)))

        print("\n" + "-" * 60)
        print(f"    rho = {c:g} * ||A^T b||_∞")
        print(f"  => rho = {rho:g}")
        print("-" * 60)
        print(f"  Problem dimensions: n = {n}, m = {m}")
        print("-" * 60)

        stoptol = 1e-6

        # Set up options dictionary (mirroring opts in MATLAB)
        opts = {
            "stoptol": stoptol,
            "Lip": Lip,
            "Ascale": 1,
        }

        # Ainput structure with A, Amap, and ATmap
        Ainput = {
            "A": A,
            "Amap": Amap,
            "ATmap": ATmap,
        }

        # Call the SSNAL solver (assumes you have a Python implementation available)
        # [obj, y, xi, x, info, runhist] = Classic_Lasso_SSNAL(Ainput, b, n, rho, opts)
        try:
            from src.lasso.Classic_Lasso_SSNAL import classic_lasso_ssnal
        except ImportError:
            print(
                "ERROR: Could not import Classic_Lasso_SSNAL. "
                "Please ensure you have a Python version of the solver named "
                "'Classic_Lasso_SSNAL_module.py' with function Classic_Lasso_SSNAL."
            )
            sys.exit(1)

        obj, y, xi, x, info, runhist = classic_lasso_ssnal(Ainput, b, n, rho, opts)
        Snal_res = info

        # Print summary of the results
        # (You can expand this to display whatever fields info/runhist contains.)
        print(f"  Solver returned info: {Snal_res}")
        print(f"  Final objective value: {obj[-1] if isinstance(obj, (list, np.ndarray)) else obj:.6e}")

    print("\nFinished all crho loops.")


if __name__ == "__main__":
    main()
