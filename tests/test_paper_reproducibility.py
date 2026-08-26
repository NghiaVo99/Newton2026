import inspect
import json
from pathlib import Path

import numpy as np

from benchmarks import paper_settings
from src.Gen_lasso import Gen_Lasso_algo
from src.OSCAR import OSCAR_algo
from src.lasso import newton_lasso
from src.lasso.untils_infinity import ProxL_infinity


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_all_newton_variants_use_paper_switching_defaults():
    modules = (newton_lasso, OSCAR_algo, Gen_Lasso_algo)
    names = (
        "Algo_Newton_Ista",
        "Algo_Newton_BT_Ista",
        "Algo_Newton_Fista_new",
        "Algo_Newton_BT_Fista_new",
    )
    for module in modules:
        for name in names:
            defaults = inspect.signature(getattr(module, name)).parameters
            assert defaults["newton_trigger_steps"].default == 3
            assert defaults["newton_reject_cooldown"].default == 8
            assert defaults["max_newton_backtracks"].default == 25
            if "newton_reject_streak_trigger" in defaults:
                assert defaults["newton_reject_streak_trigger"].default == 1


def test_infinity_prox_zeroes_points_inside_projection_ball():
    x = np.array([0.2, -0.3, 0.1])
    np.testing.assert_array_equal(ProxL_infinity(x, 1.0), np.zeros_like(x))
    np.testing.assert_array_equal(ProxL_infinity(x, 0.0), x)


def test_manifest_covers_all_figures_and_tables_and_records_no_rerun():
    path = REPO_ROOT / "reproducibility/paper_run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    covered = {
        item
        for run in manifest["runs"]
        for item in run["paper_items"]
    }
    assert covered == {
        *(f"Figure {index}" for index in range(1, 10)),
        "Table 1",
        "Table 2",
        "Table 3",
    }
    assert manifest["archive_scope"]["configuration_aligned"] is True
    assert (
        manifest["archive_scope"]["numerical_experiments_rerun_for_this_tag"]
        is False
    )
    assert manifest["archive_scope"]["legacy_outputs_are_new_results"] is False


def test_reported_image_inputs_are_present():
    raw = list((REPO_ROOT / "src/lasso/sequence").glob("*.tif"))
    reconstructed = list(
        (REPO_ROOT / "src/lasso/reconstructed").glob("*.tif")
    )
    assert len(raw) == paper_settings.FIGURES_8_9_POISSON["n_frames"]
    assert len(reconstructed) == paper_settings.FIGURES_8_9_POISSON["n_frames"]
