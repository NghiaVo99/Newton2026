# Paper benchmark reproducibility audit

This document covers Tables 1-3 and Figures 5-7 in Section 5.3 of the
manuscript. It records both the historical evidence and the corrected commands
that must be used for the revised paper.

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

## Exact commands for the revised results

Use Benchopt 1.9.0 in the `benchopt-lasso` environment. Each command performs
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

## Tagging and archiving rule

Do not tag the historical commits as if they matched the manuscript. After the
three corrected commands finish:

1. commit the three named Parquet files, their generated HTML files, the exact
   environment lock, and any manuscript tables or plots derived from them;
2. run the verifier again on a clean checkout;
3. create an annotated tag on that result commit; and
4. push both the commit and the tag, then cite the tag URL and full commit SHA
   in the manuscript and response letter.

Suggested tag name: `paper-reproducibility-v1`. A release archive or Zenodo
deposit should be created from that tag so the commit, configuration, result
files, and environment remain immutable.

## Remaining paper figures

Figures 1-4 and 8-9 are driven by standalone scripts rather than the Benchopt
configs above. Their current scripts do not all match the dimensions printed
in the manuscript, so they must not be claimed as reproduced by the eventual
tag until each is converted to a parameterized, noninteractive driver and its
input data and output checksum are added to the manifest.
