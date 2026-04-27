# BenchOpt TV-1D Benchmark

This benchmark mirrors the public `benchopt/benchmark_tv_1d` benchmark and
adds this repo's Newton variants for the quadratic 1D TV regression problem.

- objective: `TV1D`, with upstream-style quadratic and Huber data fits
- dataset: upstream-style simulated signals with `identity`, `random`, and
  `conv` forward operators
- official mirrored baseline solvers:
  - `ADMM analysis`
  - `Celer synthesis`
  - `Chambolle-Pock PD-split analysis`
  - `CondatVu analysis`
  - `Dual PGD analysis`
  - `FP synthesis`
  - `Primal PGD analysis`
  - `Primal PGD synthesis`
  - `skglm synthesis`
- custom repo solvers:
  - `newton_ista`
  - `newton_fista`

The Newton solvers currently skip Huber loss and run only on `data_fit=quad`.

## Setup

Use your BenchOpt env:

```bash
conda activate benchopt-lasso
```

Optional solver extras:

```bash
pip install celer
pip install git+https://github.com/scikit-learn-contrib/skglm.git
```

`prox-tv` is optional. `Primal PGD analysis`, `newton_ista`, and
`newton_fista` use `prox_tv` when available, otherwise they fall back to the
local exact Condat TV-1D prox implementation.

## Run

Smoke test:

```bash
benchopt test benchmarks/benchmark_tv_1d
```

Main comparison:

```bash
benchopt run benchmarks/benchmark_tv_1d \
  -d "Simulated[n_samples=500,n_features=600,type_A=random,type_x=block,type_n=gaussian]" \
  -o "TV1D[data_fit=quad,delta=0,reg=0.5]" \
  -s "ADMM analysis" -s "Celer synthesis" -s "CondatVu analysis" \
  -s "Primal PGD analysis" -s "Primal PGD synthesis" -s "skglm synthesis" \
  -s newton_ista -s newton_fista
```
