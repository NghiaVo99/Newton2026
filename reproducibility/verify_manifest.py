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
    args = parser.parse_args()

    failures = []
    for name, expected in EXPECTED_SETTINGS.items():
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
    print(f"verified {len(CONFIGS)} paper run configurations")
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
