# Paper benchmark reproducibility audit

This document covers every numerical item in the manuscript: Figures 1-9 and
Tables 1-3. It records the historical evidence, the corrected configuration,
and the exact command associated with each item.

## Audit conclusion

No reachable or dangling Git commit contains the exact configuration stated in
the manuscript. The historical Lasso and OSCAR benchmark snapshots set the
Newton trigger count to 2. The Lasso snapshot also sets the Gurobi Newton
subproblem flag to `False`. The result files record `benchmark-git-tag=None`,
so they cannot establish a different uncommitted runtime configuration.

The closest historical snapshots are:

- `010d9ad7ac482999cc29583b07eb8da064db6447`, which first archived the April
  Lasso and OSCAR result files;
- `a9b6e7d5bcb50aae65f81b07a06a74fc6748808a`, which first archived the final
  April Lasso and TV-1D result files; and
- `7647e6c2a3972c557737cacb3b231e1ed3cff6a8`, which first archived the final
  assembled Lasso and TV plot images.

There is a second historical mismatch: the right-hand OSCAR case in the
archived results uses `rho=0.8`, while Table 2 and Figure 6 label it `rho=0.6`.
The old timings therefore must not be presented as results of the corrected
paper configuration.

Exact paths and SHA-256 checksums for the historical Parquet files are stored
in `reproducibility/legacy_results.json`.

## Corrected common settings

The paper-facing benchmark wrappers import `benchmarks/paper_settings.py`.
The common settings are:

- initial iterate: zero (implemented by each Newton wrapper);
- relative KKT tolerance: `1e-8`;
- maximum iterations: `10000`;
- stability test: `norm(x[k+1] - x[k]) < 1e-2` for 3 consecutive iterations;
- initial Newton step: `1`;
- backtracking shrinkage: `0.5`;
- maximum backtracking trials: `25`;
- rejection cooldown: `8` iterations after the first rejected Newton step; and
- Newton subproblem solver: Gurobi, with no dense first-choice path in the
  paper-facing Lasso, OSCAR, or TV-1D wrappers.

All paper-facing Newton wrappers use the same `1e-2` stability tolerance.

## Verification

From the repository root, verify settings, legacy checksums, and the Gurobi
license:

```bash
python reproducibility/verify_manifest.py --check-gurobi
```

To also verify the exact reviewed manuscript file:

```bash
python reproducibility/verify_manifest.py --check-gurobi \
  --manuscript "/path/to/Newton_s_method__Copy_ (4).pdf"
```

The machine-readable mapping is in
`reproducibility/paper_run_manifest.json`; the reproducible environment is in
`environment-paper.yml` and pins Python 3.12.7 and Benchopt 1.9.0.

## Exact commands for Figures 1-4

```bash
python src/lasso/comparison.py
python src/ell_inf/newton_infinity.py
python src/OSCAR/OSCAR_run.py
python src/Gen_lasso/Gen_Lasso_run.py
```

These drivers import the common safeguards and their problem-specific
dimensions, regularization parameters, correlation, and noise variances from
`benchmarks/paper_settings.py`. In particular, the scripts now use the square
root of the stated variance as the Gaussian standard deviation.

## Exact commands for the revised results

Use Benchopt 1.9.0 in the `newton2026-paper` environment. Each command performs
one repetition with seed 0 and writes one named Parquet result. The HTML output
from the same run supplies the corresponding figure.

Table 1 and Figure 5:

```bash
benchopt run benchmarks/benchmark_lasso \
  --config benchmarks/benchmark_lasso/paper_config.yml \
  --output paper_table1_figure5
```

Table 2 and Figure 6:

```bash
benchopt run benchmarks/benchmark_oscar \
  --config benchmarks/benchmark_oscar/paper_config.yml \
  --output paper_table2_figure6
```

Table 3 and Figure 7:

```bash
benchopt run benchmarks/benchmark_tv_1d \
  --config benchmarks/benchmark_tv_1d/paper_config.yml \
  --output paper_table3_figure7
```

Tables use the first measured time whose objective is within `1e-8` of the
best objective reached by any included solver for that configuration. Figures
plot objective suboptimality against iteration from those same Parquet files.

## Exact commands for Figures 8-9

Figure 8 displays frames from the 361-image SMLM ISBI 2013 stack:

```bash
python src/lasso/ISBI_viz.py
```

Figure 9 uses all 361 checked-in frames, the 4x upsampling factor, 258.2 nm
FWHM PSF, the paper initialization `H.T @ M.T @ y`, and
`lambda = 0.5 * ||max(grad f(0), 0)||_inf`:

```bash
python src/lasso/img_pipeline.py
python src/lasso/tube_reconstruct.py
```

## Scope of the final-review tag

Do not tag the historical commits as if they matched the manuscript. The
annotated tag `paper-reproducibility-v1` identifies the corrected and frozen
code, configuration, input data, environment, and command manifest.

At the authors' explicit request, the numerical experiments were not rerun for
this final-review archive. Consequently, the existing Parquet, HTML, PNG, and
TIFF outputs remain historical artifacts; the tag does not assert that they
were regenerated from the corrected configuration. This limitation is encoded
in `reproducibility/paper_run_manifest.json` so it cannot be lost when the
repository is archived.
