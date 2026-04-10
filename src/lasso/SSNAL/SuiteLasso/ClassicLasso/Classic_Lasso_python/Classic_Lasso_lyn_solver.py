import numpy as np
from scipy import linalg
from scipy.sparse.linalg import cg, LinearOperator

def classic_lasso_linsys_solver(Ainput, rhs, par):
    """
    Python version of the MATLAB Classic_Lasso_linsys_solver, with fixes applied.

    Solves either
        (I + sigma * A_pp * A_pp^T) xi = rhs
    or the “projected” variant
        xi = rhs - A_pp * (M^{-1} (A_pp^T rhs)),
    depending on the size parameters and active columns.

    Parameters
    ----------
    Ainput : dict
        Must contain:
            'A'    : NumPy array of shape (m, n), the data matrix.
            'Amap' : callable that computes A @ x (shape (m,))
            'ATmap': callable that computes A^T @ x (shape (n,))
    rhs : array_like, shape (m,)
        Right‐hand side vector.
    par : dict
        Must contain:
            'rr'    : Boolean mask of length n; True for columns to exclude.
            'sigma' : Positive float.
            'n'     : Integer, number of columns in A.
            'precond': 0 or 1, whether to use diagonal preconditioning.
    Returns
    -------
    xi : ndarray, shape (m,)
        The computed solution vector.
    resnrm : list of float
        Residual norm(s) from the CG solver (one-element list if direct solve).
    solve_ok : bool
        True if CG converged (info == 0), or True if direct‐solve branch was used.
    """
    # Ensure rhs is a flat 1D array
    rhs = np.asarray(rhs).ravel()
    m = len(rhs)

    # Boolean mask for “active” columns in A: pp[i] = True means column i is included
    pp = ~par['rr']
    Ayes = ('A' in Ainput)  # Fix: check key in dict, not hasattr
    solver = 'd_pcg'

    dn = 10000
    sp = int(np.sum(pp))  # number of active columns

    # ————— Solver‐selection logic (match MATLAB) —————
    if (m <= dn) and Ayes:
        if m <= 1000:
            solver = 'd_direct'
        elif sp <= max(0.01 * par['n'], dn):
            solver = 'd_direct'

    if (sp <= 0.7 * m) and Ayes and (sp <= dn):
        solver = 'p_direct'

    if ((m > 5e3 and sp >= 200) or
        (m > 2000 and sp > 800) or
        (m > 100  and sp > 1e4)):
        solver = 'd_pcg'
    # ——————————————————————————————————————————————————

    # CASE 1: “d_pcg” branch
    if solver == 'd_pcg':
        if Ayes:
            AP = Ainput['A'][:, pp]  # pick the active columns
            sigma = par['sigma']

            # If preconditioning requested, build the diagonal preconditioner
            if par.get('precond', 0) == 1:
                # Same as MATLAB: invdiagM = 1 ./ (1 + sigma * sum(AP.^2,2))
                par['invdiagM'] = 1.0 / (1.0 + sigma * np.sum(AP * AP, axis=1))

            # Define the LinearOperator for (I + sigma * AP * AP^T)
            def matvec(x):
                return x + sigma * (AP @ (AP.T @ x))

            M_op = LinearOperator((m, m), matvec=matvec)

            # Solve via CG: (I + σ A A^T) xi = rhs
            xi, info = cg(M_op, rhs)
            solve_ok = (info == 0)

            # Wrap final residual in a list so that len(resnrm)-1 = 0
            final_res = np.linalg.norm(matvec(xi) - rhs)
            resnrm = [final_res]

        else:
            # MATLAB’s Amap‐only variant is not implemented here
            raise NotImplementedError("d_pcg with Amap‐only is not implemented.")

    # CASE 2: “d_direct” branch
    elif solver == 'd_direct':
        AP = Ainput['A'][:, pp]
        sigma = par['sigma']

        # Compute σ * (AP AP^T)
        sigAPAt = sigma * (AP @ AP.T)

        if m <= 1500:
            # Direct dense solve
            M = np.eye(m) + sigAPAt
            xi = np.linalg.solve(M, rhs)
        else:
            # Cholesky factorization for larger m
            M = np.eye(m) + sigAPAt
            L = np.linalg.cholesky(M)  # L @ L^T = M
            y = linalg.solve_triangular(L, rhs, lower=True)
            xi = linalg.solve_triangular(L.T, y, lower=False)

        resnrm = [0.0]
        solve_ok = True

    # CASE 3: “p_direct” branch
    elif solver == 'p_direct':
        AP = Ainput['A'][:, pp]
        sigma = par['sigma']
        APT = AP.T
        rhstmp = APT @ rhs
        PAtAP = APT @ AP
        sp = PAtAP.shape[0]

        if sp <= 1500:
            M = np.eye(sp) / sigma + PAtAP
            tmp = np.linalg.solve(M, rhstmp)
        else:
            M = np.eye(sp) / sigma + PAtAP
            L = np.linalg.cholesky(M)
            y = linalg.solve_triangular(L, rhstmp, lower=True)
            tmp = linalg.solve_triangular(L.T, y, lower=False)

        xi = rhs - AP @ tmp
        resnrm = [0.0]
        solve_ok = True

    else:
        raise ValueError(f"Unrecognized solver string: {solver}")

    return xi, resnrm, solve_ok