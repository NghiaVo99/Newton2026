#!/usr/bin/env python3
"""Verify the paper configuration and archived legacy result checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import paper_settings


EXPECTED_SETTINGS = {
    "KKT_TOL": 1e-8,
    "MAX_ITER": 10_000,
    "NEWTON_STABILITY_TOL": 1e-2,
    "NEWTON_TRIGGER_STEPS": 3,
    "NEWTON_INITIAL_STEP": 1.0,
    "NEWTON_BACKTRACK_SHRINK": 0.5,
    "NEWTON_MAX_BACKTRACKS": 25,
    "NEWTON_REJECT_STREAK_TRIGGER": 1,
    "NEWTON_REJECT_COOLDOWN": 8,
    "NEWTON_SUBPROBLEM_SOLVER": "gurobi",
}

CONFIGS = (
    "benchmarks/benchmark_lasso/paper_config.yml",
    "benchmarks/benchmark_oscar/paper_config.yml",
    "benchmarks/benchmark_tv_1d/paper_config.yml",
)

EXPECTED_EXPERIMENTS = {
    "FIGURE_1_LASSO": {
        "n_samples": 48, "n_features": 128, "n_nonzero": 8,
        "lambda_c": 0.1, "noise_variance": 0.001,
    },
    "FIGURE_2_INFINITY": {
        "n_samples": 63, "n_features": 64, "n_maximum_entries": 8,
        "lambda_c": 0.1, "noise_variance": 0.001,
    },
    "FIGURE_3_OSCAR": {
        "n_samples": 300, "n_features": 300, "rho": 0.7,
        "noise_variance": 0.01, "lambda_c": 1e-6,
    },
    "FIGURE_4_TV": {
        "n_samples": 20, "n_features": 90,
        "block_values": (0.5, -0.3, 0.8), "block_size": 30,
        "lambda_c": 0.3, "noise_variance": 0.001,
    },
    "FIGURES_8_9_POISSON": {
        "dataset": "SMLM ISBI 2013", "n_frames": 361, "n_tubes": 8,
        "low_resolution_shape": (64, 64),
        "high_resolution_shape": (256, 256),
        "upsampling_factor": 4, "low_resolution_pixel_nm": 100.0,
        "high_resolution_pixel_nm": 25.0, "psf_fwhm_nm": 258.2,
        "lambda_fraction": 0.5, "initialization": "H.T @ M.T @ y",
    },
}

PAPER_DRIVERS = {
    "src/lasso/comparison.py": "FIGURE_1_LASSO",
    "src/ell_inf/newton_infinity.py": "FIGURE_2_INFINITY",
    "src/OSCAR/OSCAR_run.py": "FIGURE_3_OSCAR",
    "src/Gen_lasso/Gen_Lasso_run.py": "FIGURE_4_TV",
    "src/lasso/ISBI_viz.py": "FIGURES_8_9_POISSON",
    "src/lasso/img_pipeline.py": "FIGURES_8_9_POISSON",
    "src/lasso/tube_reconstruct.py": "FIGURES_8_9_POISSON",
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_gurobi() -> None:
    import gurobipy as gp

    model = gp.Model("paper_reproducibility_license_check")
    model.Params.OutputFlag = 0
    variable = model.addVar(lb=-gp.GRB.INFINITY)
    model.setObjective((variable - 1.0) * (variable - 1.0))
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi license check ended with status {model.Status}.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-gurobi",
        action="store_true",
        help="also verify that a Gurobi model can be solved",
    )
    parser.add_argument(
        "--manuscript",
        type=pathlib.Path,
        help="optionally verify the manuscript PDF against the recorded SHA-256",
    )
    args = parser.parse_args()

    failures = []
    for name, expected in EXPECTED_SETTINGS.items():
        actual = getattr(paper_settings, name)
        if actual != expected:
            failures.append(f"{name}: expected {expected!r}, found {actual!r}")

    for name, expected in EXPECTED_EXPERIMENTS.items():
        actual = getattr(paper_settings, name)
        if actual != expected:
            failures.append(f"{name}: expected {expected!r}, found {actual!r}")

    for relative in CONFIGS:
        path = REPO_ROOT / relative
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{relative}: invalid JSON-compatible YAML: {exc}")
            continue
        if config.get("seed") != 0:
            failures.append(f"{relative}: seed must be 0")
        if config.get("n-repetitions") != 1:
            failures.append(f"{relative}: n-repetitions must be 1")

    manifest_path = REPO_ROOT / "reproducibility/legacy_results.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["legacy_artifacts"]:
        path = REPO_ROOT / item["path"]
        if not path.is_file():
            failures.append(f"missing legacy artifact: {item['path']}")
            continue
        actual = sha256(path)
        if actual != item["sha256"]:
            failures.append(
                f"checksum mismatch for {item['path']}: {actual}"
            )

    run_manifest_path = REPO_ROOT / "reproducibility/paper_run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    scope = run_manifest["archive_scope"]
    if scope.get("numerical_experiments_rerun_for_this_tag") is not False:
        failures.append("paper run manifest must record the requested no-rerun status")
    if scope.get("legacy_outputs_are_new_results") is not False:
        failures.append("legacy outputs must not be represented as corrected results")
    if len(run_manifest.get("runs", [])) != 9:
        failures.append("paper run manifest must map all Figures 1-9 and Tables 1-3")

    for relative, settings_key in PAPER_DRIVERS.items():
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        if "paper_settings" not in source or settings_key not in source:
            failures.append(
                f"{relative}: not tied to paper_settings.{settings_key}"
            )

    environment = (REPO_ROOT / "environment-paper.yml").read_text(encoding="utf-8")
    if "python=3.12.7" not in environment or "benchopt==1.9.0" not in environment:
        failures.append("environment-paper.yml does not pin the reported Python/Benchopt versions")

    raw_frames = sorted((REPO_ROOT / "src/lasso/sequence").glob("*.tif"))
    recon_frames = sorted((REPO_ROOT / "src/lasso/reconstructed").glob("*.tif"))
    if len(raw_frames) != 361:
        failures.append(f"expected 361 raw image frames, found {len(raw_frames)}")
    if len(recon_frames) != 361:
        failures.append(f"expected 361 reconstructed frames, found {len(recon_frames)}")

    if args.manuscript:
        expected_pdf_hash = run_manifest["manuscript"]["sha256"]
        if not args.manuscript.is_file():
            failures.append(f"manuscript not found: {args.manuscript}")
        elif sha256(args.manuscript) != expected_pdf_hash:
            failures.append("manuscript SHA-256 does not match paper_run_manifest.json")

    if args.check_gurobi:
        try:
            check_gurobi()
        except Exception as exc:
            failures.append(f"Gurobi check failed: {exc}")

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    print(f"HEAD={commit}")
    print(f"verified {len(EXPECTED_SETTINGS)} manuscript settings")
    print(f"verified {len(EXPECTED_EXPERIMENTS)} experiment specifications")
    print(f"verified {len(CONFIGS)} paper run configurations")
    print(f"verified {len(PAPER_DRIVERS)} paper-facing standalone drivers")
    print(f"verified {len(manifest['legacy_artifacts'])} legacy artifacts")
    if args.check_gurobi:
        print("verified Gurobi availability")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
