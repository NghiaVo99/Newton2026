import numpy as np
import scipy.sparse as sp

def mat_D1D(n):
    """
    Returns the difference operator D, which is used to compute the discrete gradient.
    The operator is a sparse matrix that computes the difference between adjacent elements.
    """
    D = np.zeros((n-1, n))
    D[:, 1:n] = np.eye(n-1)
    D[:, 0:n-1] -= np.eye(n-1)
    
    return D

def mat_D2D(H, W):
    """
    Build sparse matrices D_x, D_y for 2D finite differences:
      - D_x: horizontal differences, shape = (H*(W-1), H*W)
      - D_y: vertical   differences, shape = ((H-1)*W, H*W)
    and the full TV operator D = [D_x; D_y].
    """
    N = H * W

    # Horizontal differences: for each pixel except the last column,
    # diff = X[i, j+1] - X[i, j]
    rows, cols, data = [], [], []
    row = 0
    for i in range(H):
        for j in range(W - 1):
            idx = i * W + j
            # X[i,j+1]  → +1
            rows.append(row); cols.append(idx + 1); data.append(1)
            # X[i,j]    → -1
            rows.append(row); cols.append(idx    ); data.append(-1)
            row += 1
    D_x = sp.csr_matrix((data, (rows, cols)), shape=(row, N))

    # Vertical differences: for each pixel except the last row,
    # diff = X[i+1, j] - X[i, j]
    rows, cols, data = [], [], []
    row = 0
    for i in range(H - 1):
        for j in range(W):
            idx = i * W + j
            # X[i+1,j]  → +1
            rows.append(row); cols.append(idx + W); data.append(1)
            # X[i,  j]  → -1
            rows.append(row); cols.append(idx    ); data.append(-1)
            row += 1
    D_y = sp.csr_matrix((data, (rows, cols)), shape=(row, N))

    # Stack into one operator
    D = sp.vstack([D_x, D_y], format='csr')
    return D_x, D_y, D

def conv_matrix(kernel, image_shape):
    """
    Build sparse matrix A so that:
        vec(convolve2d(X, kernel, mode='same')) == A @ vec(X)
    kernel: 2D array of shape (k,k), assume k odd
    image_shape: (H, W)
    """
    H, W = image_shape
    k, _ = kernel.shape
    pad = k // 2

    rows, cols, data = [], [], []
    for i in range(H):
        for j in range(W):
            out_idx = i * W + j
            # loop over kernel support
            for u in range(k):
                for v in range(k):
                    ii = i + u - pad
                    jj = j + v - pad
                    if 0 <= ii < H and 0 <= jj < W:
                        in_idx = ii * W + jj
                        weight = kernel[u, v]
                        rows.append(out_idx)
                        cols.append(in_idx)
                        data.append(weight)

    A = sp.csr_matrix((data, (rows, cols)), shape=(H*W, H*W))
    return A

def gaussian_kernel(k=9, sigma=1.5):
    ax = np.arange(-k//2+1, k//2+1, dtype=float)
    xx, yy = np.meshgrid(ax, ax)
    K = np.exp(-(xx**2+yy**2)/(2*sigma**2))
    return K / K.sum()

def cost(A,x,b,alpha, Dx):
  #D = mat_D(len(x))
  return 0.5*np.linalg.norm(A@x-b)**2 + alpha*np.linalg.norm(Dx, 1)

def augmented_cost(A, x, y, z, b, alpha, rho, D):
   """
    Computes the augmented cost function for the optimization problem.
    The function combines the least squares term and the L1 norm term.
    """
   #D = mat_D(len(x))
   Dx = D @ x
   fx = 0.5 * np.linalg.norm(A @ x - b)**2 + np.dot(z, Dx - y) + 0.5* rho * np.linalg.norm(Dx - y)**2
   gx = np.linalg.norm(y, 1)
   return fx + alpha * gx

def f_function(A, x, y, z, b, rho, D):
    Dx = D @ x
    print((Dx - y).shape)
    print(z.shape)

    fx = 0.5 * np.linalg.norm(A @ x - b)**2 + np.dot(z, Dx - y) + 0.5 * rho * np.linalg.norm(Dx - y)**2
    return fx

def grad_f(A,x,y,z,b, rho, D):
  #D = mat_D(len(x))
  partial_x = A.T@(A@x - b) + D.T @ z + rho * D.T @ (D @ x - y)
  partial_y = -z - rho * (D @ x - y)

  return partial_x, partial_y

def hessian_f(A, x, y, z, rho, D):
    n = x.size
    m = y.size
    
    Hxx = A.T @ A + rho * (D.T @ D)   # (n,n)
    Hyy = rho * np.eye(m)             # (m,m)
    
    Hxy = - rho * D.T                 # (n,m)
    Hyx = - rho * D                   # (m,n)
    
    H = np.block([[Hxx, Hxy],
                  [Hyx, Hyy]])        # (n+m, n+m)
    return H


def prox(x,lamda):
  return np.sign(x)*np.maximum(np.abs(x) - lamda,0)

def backtracking_linesearch(A,x,y,z, b, rho, D, grad, alpha, beta=1.5):
    """ Backtracking line search for step size selection """
    L = 1
    fx = f_function
    while True:
        x_new = prox(x - (1/L) * grad,alpha * 1/L)
        lhs = 0.5 * fx(A, x_new, y, z, b, rho, D)**2
        rhs = 0.5 * fx(A, x, y, z, b, rho, D)**2 - np.dot(grad, x - x_new) + (0.5 * L) * np.linalg.norm(x - x_new)**2
        if lhs <= rhs:
            break
        L = L*beta
    return 1/L


   