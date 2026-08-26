# Nonsmooth Newton Methods with Effective Subspaces for Polyhedral Regularization

This repository contains the experimental code for the paper
**"Nonsmooth Newton methods with effective subspaces for polyhedral
regularization"**.  The code implements and benchmarks hybrid first-order /
Newton methods for polyhedral regularized optimization problems, including
Lasso, generalized Lasso / TV-1D, infinity-norm regularization, group Lasso,
OSCAR, and related imaging examples.

Paper link: [https://arxiv.org/pdf/2511.16514](https://arxiv.org/pdf/2511.16514)

The central idea in the numerical experiments is to run a robust first-order
method at the beginning, identify an effective subspace, and then apply a damped Newton correction on that subspace.
The Newton step is protected by objective-decrease safeguards and, in the
Benchopt benchmarks, compared against standard solvers from the corresponding
benchmark suites.

## Repository Layout

- `src/lasso/`: Lasso utilities and algorithms, including ISTA/FISTA,
  Newton-ISTA/FISTA variants, backtracking variants, GSSN/BaGSS experiments,
  SSNAL-related code, and imaging demos.
- `src/Gen_lasso/`: generalized Lasso and TV-style Newton methods.  This
  includes the shared switching and damped Newton routines used by TV-1D
  benchmark wrappers.
- `src/ell_inf/`: infinity-norm regularized regression experiments.
- `src/OSCAR/`: OSCAR regularization solvers, utilities, Newton variants, and
  SSNAL/Newton-ALM comparison code.
- `src/Benchmarking_Free_FISTA/`: first-order method experiments and notebooks
  for composite optimization problems.
- `benchmarks/benchmark_lasso/`: Benchopt benchmark for dense Lasso with
  baseline solvers and custom Newton variants.
- `benchmarks/benchmark_oscar/`: Benchopt benchmark for OSCAR, using the
  OSCAR/SLOPE equivalence to compare against standard SLOPE solvers.
- `benchmarks/benchmark_tv_1d/`: Benchopt benchmark for TV-1D regression with
  analysis/synthesis baselines and custom Newton variants.
- `docs/`: notes about repository structure, workflows, and benchmark setup.

## Dependencies

The core scripts use:

- Python 3.12.7 (the version reported in the manuscript)
- `numpy`
- `scipy`
- `matplotlib`
- `scikit-learn`
- `pandas`
- `numba`

Several comparison solvers are optional:

- `benchopt==1.9.0` for the benchmark folders
- `cvxpy` for convex optimization reference solves
- `gurobipy` for Gurobi reference solves and some legacy subproblem solvers
- `celer` for Lasso / TV baseline solvers
- `skglm` for sparse generalized linear model baselines
- `sortedl1` for OSCAR/SLOPE baselines
- `modopt` for some FISTA baselines

The paper-facing environment is provided in `environment-paper.yml` and pins
the reported Python version:

```bash
conda env create -f environment-paper.yml
conda activate newton2026-paper
```

`environment-benchopt.yml` remains available for development of the optional
qpOASES experiments that are not part of Figures 1-9 or Tables 1-3.

Optional benchmark extras can then be installed as needed:

```bash
pip install cvxpy gurobipy modopt celer sortedl1 skglm
```

If `prox-tv` is unavailable, the TV-1D benchmark falls back to the local
Condat TV-1D proximal implementation.

## Running Benchopt Experiments

Run commands from the repository root.

The exact paper configurations are versioned and should be used instead of
ad-hoc command-line selections. Lasso (Table 1 and Figure 5):

```bash
benchopt run benchmarks/benchmark_lasso \
  --config benchmarks/benchmark_lasso/paper_config.yml \
  --output paper_table1_figure5
```

OSCAR (Table 2 and Figure 6):

```bash
benchopt run benchmarks/benchmark_oscar \
  --config benchmarks/benchmark_oscar/paper_config.yml \
  --output paper_table2_figure6
```

TV-1D (Table 3 and Figure 7):

```bash
benchopt run benchmarks/benchmark_tv_1d \
  --config benchmarks/benchmark_tv_1d/paper_config.yml \
  --output paper_table3_figure7
```

Benchopt writes interactive HTML and parquet outputs under each benchmark's
`outputs/` directory.

## Running Script-Based Experiments

The paper-facing standalone figures use the common settings in
`benchmarks/paper_settings.py`:

```bash
python src/lasso/comparison.py
python src/ell_inf/newton_infinity.py
python src/OSCAR/OSCAR_run.py
python src/Gen_lasso/Gen_Lasso_run.py
python src/lasso/ISBI_viz.py
python src/lasso/img_pipeline.py
python src/lasso/tube_reconstruct.py
```

The complete command-to-table/figure mapping and the explicit no-rerun archive
status are recorded in `reproducibility/paper_run_manifest.json`.

## Representative Results

### Lasso

![Lasso benchmark result](src/lasso_final.png)

### Infinity-Norm Regularization

![Infinity-norm benchmark result](src/infinity_final.png)

### OSCAR

![OSCAR benchmark result](src/oscar_final.png)

### TV-1D

![TV-1D benchmark result](src/TV_v1.png)

## Notes

- The benchmark wrappers live under `benchmarks/`, while the algorithmic source
  implementations live under `src/`.
- Benchopt uses cached runs by default.  Use `-f <solver_name>` to force a
  fresh run for a solver if timings look stale.
- Several directories contain exploratory or legacy scripts from the paper's
  development process; the Benchopt folders provide the most reproducible entry
  points for the final numerical comparisons.
