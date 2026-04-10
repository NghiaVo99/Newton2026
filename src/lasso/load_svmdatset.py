# from libsvmdata import fetch_libsvm as fetch_dataset
# from sklearn.datasets import dump_svmlight_file
# from scipy import sparse
# import numpy as np

# dataset_name = "abalone"     # <-- which dataset to load
# out_stem     = dataset_name 
# X, y = fetch_dataset(dataset_name)
# print(X.shape)
# print(y.shape)   # change if you want another dataset


# dump_svmlight_file(X, y, f"{out_stem}.txt")
# sparse.save_npz(f"{out_stem}_X.npz", X)
# np.save(f"{out_stem}_y.npy", y)

# print(f"Saved as {out_stem}.svm, {out_stem}_X.npz, {out_stem}_y.npy")

from scipy.io import loadmat

# Basic load
mat = loadmat("space_ga_scale_expanded9.mat")  # e.g., contains A, b
A = mat["A"]               # numpy array or scipy.sparse
b = mat["b"]
print(A.shape)
print(b.shape)