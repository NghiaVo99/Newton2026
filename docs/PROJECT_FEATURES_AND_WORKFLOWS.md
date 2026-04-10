# Project Features and Workflows

## Scan Summary
- Repository type: optimization research/benchmark sandbox (sparse regularization + imaging + NMF).
- Top-level modules: `Benchmarking_Free_FISTA`, `Gen_lasso`, `Group_Lasso`, `NMF`, `OSCAR`, `ell_inf`, `im_super_resolve`, `lasso`.
- Approximate size: 1368 files total, 85 Python files.
- Project style: mostly script-driven experiments (not a packaged library).

## Feature Inventory by Module

### 1) `Benchmarking_Free_FISTA`
Core feature set:
- Generic first-order optimization toolkit for composite objectives `f+h`.
- Implementations in `algorithms.py`:
  - `ForwardBackward`, `FISTA`, `VFISTA`
  - Restart variants: fixed restart, automatic restart
  - Backtracking variants: `FISTA_BT`, `Free_FISTA`
  - Hessian-damped inertial variant: `FISTA_Hessian`
- Convergence visualization helpers in `visualizer.py` (`To_Plot`, `Plot`).
- Scenario notebooks:
  - `Benchmark_Inpainting_BT.ipynb`
  - `Benchmark_Poisson_BT.ipynb`
  - `Benchmark_RegLog_BT.ipynb`
- Image browsing helper: `ISBI_visualize.py`.

Primary workflow:
1. Define objective pieces (`f`, `h`, gradients, prox).
2. Run multiple solver variants from `algorithms.py`.
3. Track objective/time/extra metrics.
4. Compare methods via `visualizer.Plot` or notebook plots.

---

### 2) `Gen_lasso`
Core feature set:
- Generalized Lasso solvers:
  - ISTA/FISTA + backtracking
  - Newton-hybrid variants (`Algo_Newton_*`)
  - Optional Fast-ADMM routine
- Utilities in `Gen_Lasso_utils.py`:
  - Forward-difference operator build
  - Objective/gradient/Hessian helpers
  - Subproblem solver for Newton step
  - Reference solvers via Gurobi/CVXPY
- Synthetic data generator: `test_prob_gpt.py`.

Primary workflow (`Gen_Lasso_run.py`):
1. Build synthetic generalized-lasso instance (`A, b, D`).
2. Compute reference solution with Gurobi.
3. Run all algorithm variants.
4. Plot objective gap, iterate distance, and runtime bars.

---

### 3) `Group_Lasso`
Core feature set:
- Group-Lasso objective and grouped proximal operator (`proxL1_L2`).
- Same algorithm family pattern as Lasso/Gen-Lasso:
  - ISTA/FISTA (+ backtracking)
  - Newton hybrid + backtracking Newton hybrid
- Utilities include group index builders and reference solvers (CVXPY/Gurobi).

Primary workflow (`comparison.py`):
1. Generate grouped sparse synthetic data.
2. Run solver family.
3. Compare objective gap, distance-to-reference, runtime.

---

### 4) `NMF`
Core feature set:
- Nonnegative Matrix Factorization objective + gradients.
- Projected-gradient variants:
  - Jacobi (simultaneous updates)
  - Gauss-Seidel (alternating updates)
- Newton-triggered enhancement when rank pattern stabilizes.
- QP subproblem solvers with CVXPY and Gurobi (`sub_problem_*`).

Primary workflow (`NMF_main.py`):
1. Generate synthetic nonnegative matrix `A = W_true H_true + noise`.
2. Run baseline PG and Newton-augmented variants.
3. Plot objective-vs-iteration and objective-vs-time.
4. Summarize final metrics in a DataFrame.

---

### 5) `OSCAR`
Core feature set:
- OSCAR regularization utilities (`prox_oscar`, objective, problem builders).
- Solver family:
  - ISTA/FISTA (+ backtracking)
  - Newton-hybrid variants
- SSNAL/Newton-ALM OSCAR solver in `SSNAL_OSCAR.py` (`NewtALM_OSCAR`).
- Exact/reference OSCAR solves via Gurobi utilities.

Primary workflows:
- `OSCAR_run.py`: full method benchmark (Newton/ISTA/FISTA + SSNAL).
- `SSNAL_OSCAR_run.py`: SSNAL-focused run and convergence plotting.

---

### 6) `ell_inf`
Core feature set:
- Infinity-norm regularized regression experiments.
- Utilities for objective/prox/line-search + CVXPY/Gurobi references.
- Solver variants include ISTA/FISTA/backtracking and Newton hybrids.

Primary workflow (`newton_infinity.py`):
1. Create synthetic sparse data.
2. Solve reference problem.
3. Run first-order and Newton-hybrid methods.
4. Plot objective gap and convergence traces.

---

### 7) `lasso` (largest module)
Core feature set:
- Base Lasso toolchain:
  - `utils_lasso.py` (cost, gradient, prox, line-search, scaling/preprocess)
  - `newton_lasso.py` (ISTA/FISTA, BT variants, Newton-hybrid variants)
- Additional solver families:
  - `BaGSS.py`: BasGSS/GSSN semismooth direction framework
  - `lasso_GDFBE_LM.py`, `lasso_GDNM.py`: alternative Newton-like methods
  - `Classic_Lasso_SSNAL*.py`: SSNAL + SSNCG implementation stack
  - `Classic_Lasso_lyn_solver.py`: linear-system backend
- TV/ALM imaging:
  - `ultils_TV.py`, `ALM.py`
  - Denoising/deblurring drivers: `ALM_denosing_run.py`, `ALM_deblurring_run.py`
- Poisson imaging/super-resolution:
  - `Poisson_utils.py`, `newton_poisson.py`
  - Drivers: `Poisson_run.py`, `img_pipeline.py`
- Benchmark and analysis scripts:
  - `comparison.py`, `comparison_real_data.py`
  - `Benchmark_Lasso.py` + `performace_profile.py`
- Visualization/post-processing:
  - `ISBI_viz.py`, `images_visualization.py`, `tube_reconstruct.py`, `zoom_img.py`, `sparsity_viz.py`
- Misc demos/tests:
  - `PDHG_newtv1.py`, `PDHG_ADMM_newt_v2.py`
  - `toy_test.py`, `test_ClassicLasso_random.py`, `test_ClassicLasso_UCI.py`, `test_mexscale.py`

Major workflows inside `lasso`:
1. **Synthetic Lasso benchmark** (`run.py`, `comparison.py`)
   - Generate synthetic sparse truth.
   - Obtain reference (`CVXPY`/`Gurobi`/SSNAL helper).
   - Run multiple methods and compare objective gap/distance/runtime.

2. **Real-data benchmark** (`comparison_real_data.py`)
   - Load UCI/libsvm-style `.mat` data from `lasso/UCIdata`.
   - Preprocess matrix scaling.
   - Run selected methods (currently Newton BT variants active).

3. **Batch benchmark export + profile**
   - `Benchmark_Lasso.py`: sweeps `(n, m/n, sparsity)` grids, saves `.npz` in `lasso/results`.
   - `performace_profile.py`: consumes saved `.npz`, plots performance profile curves.

4. **TV denoising/deblurring (ALM)**
   - `ALM_denosing_run.py` for 1D piecewise-constant denoising.
   - `ALM_deblurring_run.py` for image deblurring (`cameraman.pgm`) with PSNR comparison.

5. **Poisson image reconstruction / SR workflow**
   - `Poisson_run.py`: single-image reconstruction and method comparison.
   - `img_pipeline.py`: batch reconstruction from `sequence/*.tif` to `reconstructed/*.tif`.
   - `tube_reconstruct.py`: aggregate stacks (mean/sum) and save montage in `outputs/`.
   - `zoom_img.py` / `ISBI_viz.py`: ROI and gallery visualization.

6. **Classic SSNAL Lasso path**
   - `test_ClassicLasso_UCI.py`: run SSNAL on UCI `.mat` dataset.
   - `toy_test.py`: tiny sanity-check for SSNAL stack.

7. **Hybrid switching demos**
   - `PDHG_newtv1.py`: PDHG then Newton when support stabilizes.
   - `PDHG_ADMM_newt_v2.py`: FISTA/PDHG/ADMM and hybrid comparisons.

---

### 8) `im_super_resolve`
Core feature set:
- Literature/reference PDFs only (no executable code).
- Used as research background for image super-resolution experiments.

## Shared Design Pattern Across Solvers
Many modules follow the same pattern:
1. Build objective + prox + line-search utilities.
2. Implement ISTA/FISTA baselines.
3. Add backtracking variants.
4. Add Newton-triggered or semismooth step once close to active manifold/support.
5. Compare against reference solve (CVXPY/Gurobi/SSNAL) and plot:
   - objective gap
   - distance to reference
   - runtime

## Data and Artifact Flow
Input sources:
- Synthetic generators (many scripts).
- UCI `.mat` files under `lasso/UCIdata`.
- Imaging stacks under `lasso/sequence`.

Generated artifacts:
- `lasso/reconstructed/*.tif` (batch reconstruction outputs).
- `lasso/outputs/*.png` (aggregate visualizations).
- `lasso/results/*.npz` (benchmark result bundles).

## Dependencies Observed in Code
Primary:
- `numpy`, `scipy`, `matplotlib`
- `cvxpy`, `gurobipy`

Frequent extras:
- `pyproximal`, `imageio`, `opencv-python (cv2)`, `tifffile`
- `scikit-image`, `scikit-learn`, `pandas`
- optional: `tick` (used in one TV prox example script)

## Notable Practical Notes
- Most scripts are designed to be run directly and contain hard-coded experiment parameters.
- Several scripts are experimental/scratch-like (e.g., temporary files, partially commented method blocks).
- `lasso/SSNAL/SuiteLasso` is a bundled external codebase snapshot (MATLAB/C/mex + Python mirror) used as a reference/legacy implementation source.
