import numpy as np
import matplotlib.pyplot as plt
from src.lasso.ALM import *
from src.lasso.ultils_TV import mat_D1D, mat_D2D, gaussian_kernel, conv_matrix

# Generate a piecewise constant 1D signal
def generate_piecewise_constant_signal(length=150, num_segments=10):
    segment_length = length // num_segments
    signal = np.zeros(length)
    for i in range(num_segments):
        constant_value = np.random.uniform(-1.5, 1.5)  # Random constant value
        signal[i * segment_length:(i + 1) * segment_length] = constant_value
    
    return signal

signal = generate_piecewise_constant_signal(length=100, num_segments=20)
noise = np.random.normal(0, 0.03, signal.shape)  # Add some noise
noisy_signal = signal + noise
b = noisy_signal
outer_max_iter = 100
inner_max_iter = 1
x0 = b #np.zeros_like(signal)
y0, z0 = np.zeros(len(signal)-1), np.zeros(len(signal)-1)
alpha = 0.1
rho = 2

A = np.eye(len(signal))  # Identity for denoising
mat_D = mat_D1D #for denoising
#mat_D = mat_D2D #for image deblurring
D = mat_D(len(signal))
step_size = 1.0 / np.linalg.norm(A.T@A + rho*D.T@D, 2)**2

x, y, z, i, j, cost_list, time_list, inner_accumulator = Augmented_Lag_method(x0,y0,z0, A, D, b, alpha, rho, step_size, outer_max_iter = 700, inner_max_iter = 100, tol=1e-6)
x1, y1, z1, i1, j1, cost_list1, time_list1, inner_accumulator1 = Augmented_Lag_method(x0,y0,z0, A, D, b, alpha, rho, step_size, outer_max_iter = 500, inner_max_iter = 50, tol=1e-6)
#x11, y11, z11, i11, j11, cost_list11, time_list11, inner_accumulator11 = Augmented_Lag_Newt_method(x0,y0,z0, A, D, b, alpha, rho, step_size,outer_max_iter = 500, inner_max_iter = 1, tol=1e-6)

# Plot the signal
plt.figure(figsize=(12, 6))
plt.plot(signal, color='blue', label='Original Signal')
plt.plot(noisy_signal, color='orange', label='Noisy Signal')
plt.plot(x, color='green', label='Recovered Signal (ALM)')
plt.title("Piecewise Constant 1D Signal")
plt.legend()
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid()

# plt.figure(figsize=(12, 6))
# plt.plot(time_list1, abs(np.array(cost_list1) - cost_list[-1]), color='blue', label='ALM')
# plt.plot(time_list11, abs(np.array(cost_list11) - cost_list[-1]), color='red', label='Newt_ALM')
# plt.legend()
# plt.grid()
# plt.xlabel("Time (s)")
# plt.ylabel(r'$|f(x_k) - f(x^*)|$')
# #plt.xscale('log')
# plt.yscale('log')

# plt.figure(figsize=(12, 6))
# plt.plot(np.arange(len(inner_accumulator1)),inner_accumulator1, color='blue', label='ALM')
# #plt.plot(np.arange(len(inner_accumulator11)),inner_accumulator11, color='red', label='Newt_ALM')
# plt.legend()
# plt.grid()
# plt.xlabel("Iteration")
# plt.ylabel("Total inner iterations")
plt.show()