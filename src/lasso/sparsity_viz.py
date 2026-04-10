import numpy as np
import matplotlib.pyplot as plt

def sparsity_plot(sol_list, label_list):

    solutions = sol_list
    labels = label_list

    # Collect coordinates of nonzero entries (support pattern)
    rows, cols = [], []
    for i, x in enumerate(solutions):
        x[np.abs(x) < 1e-6] = 0
        support = np.nonzero(x)[0]
        rows.extend([i] * len(support))   # Y-axis: row index = algorithm label
        cols.extend(support)              # X-axis: nonzero index in vector

    # Create scatter plot of support pattern
    plt.figure(figsize=(14, 4))
    plt.scatter(cols, rows, color='black', s=20, marker='o')

    # Customize axes
    plt.yticks(ticks=np.arange(len(solutions)), labels=labels)
    plt.xlabel('Coefficient Index')
    plt.ylabel('Solution Vector')
    plt.title('Support Pattern of Sparse Solutions (Scatter Plot View)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
