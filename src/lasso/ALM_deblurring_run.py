import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.lasso.ALM import *
from src.lasso.ultils_TV import mat_D1D, mat_D2D, gaussian_kernel, conv_matrix
from skimage.metrics import peak_signal_noise_ratio as psnr

H, W = 256, 256  # Image dimensions

K = gaussian_kernel(9, 1.5)
A = conv_matrix(K, (H, W))
D_x, D_y, D = mat_D2D(H, W) #for image deblurring

img = cv2.imread('cameraman.pgm', cv2.IMREAD_GRAYSCALE)
vectorize_img = img.ravel()
blurred_img = A.dot(vectorize_img) 
b = blurred_img
outer_max_iter = 200
inner_max_iter = 1
x0 = b #np.zeros_like(vectorize_img)
y0, z0 = np.zeros(len(D@vectorize_img)), np.zeros(len(D@vectorize_img))
alpha = 0.05
rho = 1.5
#step_size = 1.0 / np.linalg.norm((A.T@A + rho*D.T@D).toarray(), 2)
step_size = 1e-2

#cvx_sol, value_cvx = solve_TV_gurobi(A, D, b, alpha)
#x1, y1, z1, i1, j1, cost_list1, time_list1 = Augmented_Lag_method(x0,y0,z0, A, D, b, alpha, rho, step_size, outer_max_iter = 200, inner_max_iter = 100, tol=1e-6)
#x11, y11, z11, i11, j11, cost_list11, time_list11 = Augmented_Lag_method(x0,y0,z0, A, D, b, alpha, rho, step_size, outer_max_iter = 100, inner_max_iter = 50, tol=1e-6)
x, y, z, i, j, cost_list, time_list = Augmented_Lag_Newt_method(x0,y0,z0, A, D, b, alpha, rho, step_size, outer_max_iter = 100, inner_max_iter = 1, tol=1e-6)

#Calculate PSNR
data_range = img.max() - img.min()
psnr_value_ALM_Newt = psnr(img, x.reshape(H, W), data_range=data_range)
psnr_value_ALM = psnr(img, x11.reshape(H, W), data_range=data_range)
psnr_value_blurred = psnr(img, blurred_img.reshape(H, W), data_range=data_range)

# Function to display images with PSNR values
def show_with_psnr(ax, img, title, psnr=None):
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_title(title, pad=10)
    # Hide ticks and spines, but keep the xlabel
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if psnr is not None:
        ax.set_xlabel(f"PSNR: {psnr:.2f} dB", labelpad=12)

fig, axs = plt.subplots(1, 4, figsize=(12, 6))
show_with_psnr(axs[0], img,                 "Original Image",     psnr=None)
show_with_psnr(axs[1], blurred_img.reshape(H, W), "Blurred Image",    psnr_value_blurred)
show_with_psnr(axs[2], x.reshape(H, W),     "Recovered Image (ALM_Newt)", psnr_value_ALM_Newt)
show_with_psnr(axs[3], x11.reshape(H, W),  "Recovered Image (ALM)", psnr_value_ALM)

#plt.tight_layout()


plt.figure(figsize=(12, 6))
plt.plot(time_list, abs(np.array(cost_list) - cost_list1[-1]), color='red', label='ALM_Newt')
plt.plot(time_list1, abs(np.array(cost_list11) - cost_list1[-1]), color='blue', label='ALM')
#plt.xscale('log')
plt.grid()
plt.xlabel("Time (s)")
plt.legend()
plt.yscale('log')
plt.ylabel(r'$|f(x_k) - f(x^*)|$')
plt.show()