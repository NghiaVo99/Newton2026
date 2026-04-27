# Benchopt Dense Lasso Benchmark

This benchmark is the first Benchopt integration for this repository. It keeps
the comparison close to the public `benchopt/benchmark_lasso` benchmark by
using the official baseline solver definitions where possible, and adds this
repo's Newton variants as custom solvers.

- objective: official-style Lasso objective with `fit_intercept=False`
- datasets: official-style parameterized simulated datasets from Benchopt's synthetic data helper
- official baseline solvers in this benchmark: `cd`, `sklearn`, `skglm`, `ModOpt-FISTA`, `Celer`, `glmnet`
- custom repo solvers: `newton_ista`, `newton_fista`, `newton_bt_ista`, `newton_bt_fista`
- optional extra solvers: `cvxpy` and `gurobi`

## Setup

Use a fresh conda environment instead of the current repo environment. The
existing environment has a broken `benchopt` import caused by the
`pyarrow/libutf8proc` shared-library issue.

```bash
conda env create -f environment-benchopt.yml
conda activate benchopt-lasso
```

Optional solvers:

```bash
pip install cvxpy
pip install gurobipy
pip install modopt
pip install git+https://github.com/scikit-learn-contrib/skglm.git
pip install celer
# glmnet solver needs R + rpy2 + glmnet packages:
# conda install -c conda-forge r-base rpy2 r-glmnet r-matrix
```

## Run from the repo root

Smoke test:

```bash
benchopt test benchmarks/benchmark_lasso
```

Single solver on one dataset:

```bash
benchopt run benchmarks/benchmark_lasso -d Simulated -s sklearn
```

Example with one official parameter setting:

```bash
benchopt run benchmarks/benchmark_lasso \
  -d "Simulated[n_samples=500,n_features=600,rho=0.6]" \
  -s sklearn
```

Official baselines plus your Newton solvers:

```bash
benchopt run benchmarks/benchmark_lasso \
  -s cd -s sklearn -s skglm -s ModOpt-FISTA -s Celer -s glmnet \
  -s newton_ista -s newton_fista -s newton_bt_ista -s newton_bt_fista
```

## Notes

- This benchmark is for local use inside the repo first.
- `src/lasso` remains the source of truth for the Newton algorithm implementations.
- The Newton Benchopt wrappers use a NumPy dense subproblem helper so the core
  benchmark stays open-source by default.
- The simulated dataset now follows the same single-file parameterized pattern
  as the public `benchopt/benchmark_lasso` benchmark.
- There are no standalone official `ista.py` or `fista.py` solvers in the
  current public `benchopt/benchmark_lasso` repo, so this benchmark now uses
  the official baselines that are currently published there.
- UCI datasets, sparse data, imaging workflows, OSCAR, Group Lasso, and NMF
  are intentionally out of scope for this first milestone.
