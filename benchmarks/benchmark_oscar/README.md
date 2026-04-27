# BenchOpt OSCAR Benchmark

This benchmark compares OSCAR Newton solvers from this repo against official
BenchOpt SLOPE baseline solvers, leveraging the OSCAR/SLOPE equivalence.

- objective: OSCAR-native parameters `w1`, `w2`, converted to SLOPE weights
  `alphas_i = w1 + w2 * (p - 1 - i)`
- dataset: simulated correlated design (mirrors `benchopt/benchmark_slope` style)
- official mirrored baseline solvers in this benchmark:
  - `ADMM` (`admm.py`)
  - `PGD` (`python_pgd.py`)
  - `skglm` (`skglm.py`)
  - `sortedl1` (`sortedl1.py`)
  - `Newt-ALM` (`newt_alm.py`)
- custom repo solvers:
  - `newton_ista`, `newton_fista`, `newton_bt_ista`, `newton_bt_fista`

## Setup

Use your BenchOpt env:

```bash
conda activate benchopt-lasso
```

Optional extras for this benchmark:

```bash
pip install sortedl1
pip install skglm
```

## Run

Smoke test:

```bash
benchopt test benchmarks/benchmark_oscar
```

Single solver:

```bash
benchopt run benchmarks/benchmark_oscar -d Simulated -s PGD
```

Single dataset setting:

```bash
benchopt run benchmarks/benchmark_oscar \
  -d "Simulated[n_samples=500,n_features=200,n_signals=20,X_density=1.0,rho=0.8]" \
  -o "OSCAR Regression[w1=1e-3,w2=1e-4,fit_intercept=False]" \
  -s PGD
```

Full comparison:

```bash
benchopt run benchmarks/benchmark_oscar \
  -d "Simulated[n_samples=500,n_features=200,n_signals=20,X_density=1.0,rho=0.8]" \
  -o "OSCAR Regression[w1=1e-3,w2=1e-4,fit_intercept=False]" \
  -s ADMM -s PGD -s skglm -s sortedl1 -s Newt-ALM \
  -s newton_ista -s newton_fista -s newton_bt_ista -s newton_bt_fista
```
