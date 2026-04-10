import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import scipy.sparse as sp
import scipy as spy
from src.lasso.ultils_TV import mat_D1D, mat_D2D, gaussian_kernel, conv_matrix

H, W = 256, 256  # Image dimensions
K = gaussian_kernel(9, 1.5)
A = conv_matrix(K, (H, W))


img = cv2.imread('cameraman.pgm', cv2.IMREAD_GRAYSCALE)
#blurred_img1 = spy.signal.convolve2d(img, K, mode='same')
blurred_img = A.dot(img.ravel()).reshape(H, W)        
fig, axs = plt.subplots(1, 2, figsize=(8, 8))

axs[0].imshow(img, cmap='gray', interpolation='nearest')
axs[0].set_title('Original Image')
axs[1].imshow(blurred_img, cmap='gray', interpolation='nearest')

plt.show()

# Example usage:
# D_x, D_y, D = mat_D2D(4, 4)
# X = np.random.randint(16,size=16).reshape(4, 4)
# print("X\n", X)
# print("D_shape:", D.shape)
# print("Dx\n", D@X.flatten())
