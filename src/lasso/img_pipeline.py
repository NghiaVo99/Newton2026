import numpy as np
import imageio.v3 as iio
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.io import loadmat
import scipy
import pathlib
from src.lasso.newton_poisson import *
from src.lasso.Poisson_utils import *
import imageio.v2 as imageio

FOLDER = pathlib.Path("sequence")
files = sorted([
    *FOLDER.glob("*.tif"), *FOLDER.glob("*.tiff"),
    *FOLDER.glob("*.png"), *FOLDER.glob("*.jpg"), *FOLDER.glob("*.jpeg")
])

scale = 4                      # upsampling
lr_px_nm = 100
hr_px_nm = lr_px_nm / scale    # 25 nm/pixel

psf_hr = gaussian_psf_hr_cel0(fwhm_nm=258.2, hr_px_nm=hr_px_nm, size=33)
max_iter = 200
beta, newton_stepsize = 0.5, 1.0
tol = 1e-8
newt_tol = 0.1

b_map = estimate_background_from_stack(files, pattern="*.tif", method="median")
b_map = np.maximum(np.asarray(b_map, np.float64), 1e-12)
b_vec = b_map.ravel()
        


# ────────────────────────────────────────────────────────────
# Batch reconstruction over the whole sequence (no input changes)
# ────────────────────────────────────────────────────────────
OUTPUT_DIR = pathlib.Path("reconstructed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def read_image_as_is(p):
    """Read image without modifying dtype or channels."""
    p = pathlib.Path(p)
    if p.suffix.lower() in [".tif", ".tiff"]:
        # Match your single-image code: use first page only
        arr = iio.imread(p, index=0)
    else:
        arr = iio.imread(p)
    return arr  # return exactly as stored

approx_sol = 0  # keep as in your single-image script

for idx, f in enumerate(files, start=1):
    print(f"\n[{idx}/{len(files)}] Processing: {f.name}")
    z_lr = read_image_as_is(f)

    # Ensure we don't change the input: only accept 2D arrays; otherwise, skip
    if z_lr.ndim != 2:
        print(f"⚠️  Skipping {f.name}: not single-channel 2D (shape={z_lr.shape}).")
        continue

    # Build problem for this LR image (no modifications to z_lr)
    ops = make_problem(
        psf_hr,
        scale=scale,
        z_lr_2d=np.asarray(z_lr, dtype=np.float64),
        b=b_vec,                       # pass the vectorized map
        x0_mode="backproj"
    )

    x0      = ops["x0"]
    A, AT   = ops["A"], ops["AT"]
    noisy_z = ops["z"]   # as provided by make_problem
    b_cur   = ops["b"]
    prox = ops['prox_g']

    lam_max = lambda_max(ops["A"], ops["AT"], ops["z"], ops["b"])
    print("lambda_max =", lam_max)
    alpha = 0.5*lam_max

    subproblem_solver = sub_problem_of_poisson
    cost = cost_poisson


    # Run your Newton-BT-FISTA (same as single-image call)
    cost_val_newton_bt_fista, x_rec_vec, i6, x_k6, time_k6 = Algo_Newton_BT_Fista_new(A,AT,b_vec,x0, noisy_z,
                            alpha,max_iter, beta, newton_stepsize, tol, cost,
                            prox, subproblem_solver, newt_tol = newt_tol, approx_sol = approx_sol)

    # Reshape vector solution to HR image and save
    x_hr = x_rec_vec.reshape(ops["hr_shape"])
    out_path = OUTPUT_DIR / f"{f.stem}_recon.tif"
    imageio.imwrite(out_path.as_posix(), x_hr.astype(np.float32))
    print(f"Saved: {out_path}")

print("\nDone processing all images.")