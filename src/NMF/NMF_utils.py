from re import S
import numpy as np
from numpy.linalg import norm
from scipy import sparse
import gurobipy as gp
from gurobipy import GRB

def frob(A): return float(norm(A, 'fro'))

def nmf_objective(A,W, H): return 0.5 * frob(A - W @ H) ** 2

def grad_f(A,W,H):
  R = W@H - A
  gradW = R@H.T
  gradH = W.T@R
  return gradW, gradH


def proj_plus(X): return np.maximum(X, 0)

def backtracking_linesearch(A, b, x, grad, alpha, beta=1.5):
    """ Backtracking line search for step size selection """
    L = 1
    while True:
        x_new = proj_plus(x - (1/L) * grad)
        lhs = 0.5 * np.linalg.norm(A @ x_new - b)**2
        rhs = 0.5 * np.linalg.norm(A @ x - b)**2 - np.dot(grad, x - x_new) + (0.5 * L) * np.linalg.norm(x - x_new)**2
        if lhs <= rhs:
            break
        L = L*beta
    return 1/L



# ---------- helpers ----------
def vecF(X):  # column-major vectorization
    return np.reshape(X, (-1,), order='F')

def unvecF(v, shape):
    return np.reshape(v, shape, order='F')

def selection_matrix(mask):
    # mask is bool of shape (m,r) or (r,n). Returns S \in R^{(mn) x p} and idx.
    m, n = mask.shape
    idx = np.flatnonzero(mask.reshape(-1, order='F'))
    p = idx.size
    if p == 0:
        return sparse.csr_matrix((m*n, 0)), idx
    data = np.ones(p, dtype=float)
    rows = idx
    cols = np.arange(p)
    S = sparse.csr_matrix((data, (rows, cols)), shape=(m*n, p))
    return S, idx

def commutation_matrix(p, q):
    # K_{pq} such that vec(A^T) = K_{pq} vec(A) for A \in R^{p x q}
    i = np.tile(np.arange(p), q)
    j = np.repeat(np.arange(q), p)
    row = j + q * i
    col = i + p * j
    data = np.ones(p * q)
    return sparse.coo_matrix((data, (row, col)), shape=(p*q, p*q)).tocsr()


