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
