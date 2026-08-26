"""Settings stated in Section 5.1 of the manuscript.

Keep paper-facing benchmark wrappers tied to these constants.  Development
benchmarks may define separate settings, but must not silently change these.
"""

KKT_TOL = 1e-8
MAX_ITER = 10_000
NEWTON_STABILITY_TOL = 1e-2
NEWTON_TRIGGER_STEPS = 3
NEWTON_INITIAL_STEP = 1.0
NEWTON_BACKTRACK_SHRINK = 0.5
NEWTON_MAX_BACKTRACKS = 25
NEWTON_REJECT_STREAK_TRIGGER = 1
NEWTON_REJECT_COOLDOWN = 8
NEWTON_SUBPROBLEM_SOLVER = "gurobi"

# Section 5.2 standalone experiments (Figures 1-4).  Seeds were not stated in
# the manuscript, so they are fixed here to make the tagged implementation
# deterministic.  NOISE_VARIANCE is a variance; drivers must use its square
# root as the Gaussian scale.
SYNTHETIC_SEED = 42

FIGURE_1_LASSO = {
    "n_samples": 48,
    "n_features": 128,
    "n_nonzero": 8,
    "lambda_c": 0.1,
    "noise_variance": 0.001,
}

FIGURE_2_INFINITY = {
    "n_samples": 63,
    "n_features": 64,
    "n_maximum_entries": 8,
    "lambda_c": 0.1,
    "noise_variance": 0.001,
}

FIGURE_3_OSCAR = {
    "n_samples": 300,
    "n_features": 300,
    "rho": 0.7,
    "noise_variance": 0.01,
    "lambda_c": 1e-6,
}

FIGURE_4_TV = {
    "n_samples": 20,
    "n_features": 90,
    "block_values": (0.5, -0.3, 0.8),
    "block_size": 30,
    "lambda_c": 0.3,
    "noise_variance": 0.001,
}

# Section 5.4 image experiment (Figures 8-9).
FIGURES_8_9_POISSON = {
    "dataset": "SMLM ISBI 2013",
    "n_frames": 361,
    "n_tubes": 8,
    "low_resolution_shape": (64, 64),
    "high_resolution_shape": (256, 256),
    "upsampling_factor": 4,
    "low_resolution_pixel_nm": 100.0,
    "high_resolution_pixel_nm": 25.0,
    "psf_fwhm_nm": 258.2,
    "lambda_fraction": 0.5,
    "initialization": "H.T @ M.T @ y",
}
