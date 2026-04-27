# Benchopt Lasso Setup

This repo now includes a first Benchopt benchmark at
`benchmarks/benchmark_lasso`.

## Why use a fresh environment

Use a fresh conda environment for Benchopt instead of the current local Python
environment. In the existing environment, `benchopt` fails to import because
`pyarrow` cannot load `libutf8proc.2.dylib`.

## Core environment

```bash
conda env create -f environment-benchopt.yml
conda activate benchopt-lasso
```

Core stack in that environment:

- `benchopt==1.9.0`
- `numpy`
- `scipy`
- `matplotlib`
- `scikit-learn`
- `numba`

## Optional solver extras

```bash
pip install cvxpy
pip install gurobipy
pip install modopt
pip install git+https://github.com/scikit-learn-contrib/skglm.git
pip install celer
# glmnet solver needs R + rpy2 + glmnet packages:
# conda install -c conda-forge r-base rpy2 r-glmnet r-matrix
```

The benchmark skips those solvers automatically when the package or Gurobi
license is unavailable.

## OSCAR benchmark extras

For `benchmarks/benchmark_oscar`, install:

```bash
pip install sortedl1
pip install skglm
```

## TV-1D benchmark extras

For `benchmarks/benchmark_tv_1d`, install:

```bash
pip install celer
pip install git+https://github.com/scikit-learn-contrib/skglm.git
```

`prox-tv` is optional. If it is unavailable, the benchmark uses the local
Condat TV-1D prox implementation for analysis PGD and the Newton solvers.

## Run commands

From the repo root:

```bash
benchopt test benchmarks/benchmark_lasso
```

```bash
benchopt run benchmarks/benchmark_lasso -d Simulated -s sklearn
```

```bash
benchopt run benchmarks/benchmark_lasso \
  -d "Simulated[n_samples=500,n_features=600,rho=0.6]" \
  -s sklearn
```

```bash
benchopt run benchmarks/benchmark_lasso \
  -s cd -s sklearn -s skglm -s ModOpt-FISTA -s Celer -s glmnet \
  -s newton_ista -s newton_fista -s newton_bt_ista -s newton_bt_fista
```

## Current scope

- dense simulated Lasso only
- no intercept
- the dataset layout now mirrors the public `benchmark_lasso/datasets/simulated.py`
- official public Benchopt baseline solvers are used where available
- `src/lasso` remains the source implementation for the Newton variants
- optional `cvxpy` and `gurobipy` wrappers are isolated from the open-source core

## OSCAR benchmark run

```bash
benchopt test benchmarks/benchmark_oscar
```

```bash
benchopt run benchmarks/benchmark_oscar \
  -d "Simulated[n_samples=500,n_features=200,n_signals=20,X_density=1.0,rho=0.8]" \
  -o "OSCAR Regression[w1=1e-3,w2=1e-4,fit_intercept=False]" \
  -s ADMM -s PGD -s skglm -s sortedl1 -s Newt-ALM \
  -s newton_ista -s newton_fista -s newton_bt_ista -s newton_bt_fista
```

## TV-1D benchmark run

```bash
benchopt test benchmarks/benchmark_tv_1d
```

```bash
benchopt run benchmarks/benchmark_tv_1d \
  -d "Simulated[n_samples=500,n_features=600,type_A=random,type_x=block,type_n=gaussian]" \
  -o "TV1D[data_fit=quad,delta=0,reg=0.5]" \
  -s "ADMM analysis" -s "Celer synthesis" -s "CondatVu analysis" \
  -s "Primal PGD analysis" -s "Primal PGD synthesis" -s "skglm synthesis" \
  -s newton_ista -s newton_fista
```
