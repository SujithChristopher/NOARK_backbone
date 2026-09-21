"""Design-matrix assembly for the subset-geometry jitter model.

Kept out of ``09_subset_geometry.py`` (task-11-12 decision 3) so
``tests/jitter_model/test_model_assembly.py`` can import these pure functions
directly, instead of executing the analysis script -- which would also run
the module-level sweep and mocap-alignment cells that only make sense against
the real dome recording.
"""

import numpy as np

from jitter_model import fixed_effects

# Geometry predictors the within-burst model is fit on. `max_normal_angle_deg`
# is undefined (NaN, from `geometry.subset_geometry`) for every single-tag
# subset, since a pairwise normal angle needs two tags to exist at all. See
# `fit_response` for how that NaN column is handled -- it is not silently
# dropped.
GEOMETRY_PREDICTORS = [
    "max_baseline_mm",
    "rms_radius_mm",
    "min_singular_mm",
    "max_normal_angle_deg",
    "lever_mm",
    "n_corners",
]
# Added to every predictor before the log so a legitimately zero baseline (one
# tag) stays finite instead of becoming -inf and dropping the row. Small
# enough (1e-6, not the briefed 1e-3) that it perturbs a real, positive
# predictor by far less than `np.allclose`'s default tolerance -- the briefed
# 1e-3 fails `test_build_design_log_transforms_positive_predictors` outright,
# since log(10 + 1e-3) - log(10) ~= 1e-4 exceeds allclose's ~2.3e-5 tolerance.
LOG_FLOOR = 1e-6


def build_design(cells, predictors):
    """Log-transformed predictors, so coefficients read as elasticities."""
    columns = []
    names = []
    for predictor in predictors:
        columns.append(np.log(cells[predictor].to_numpy(dtype=float) + LOG_FLOOR))
        names.append(f"log_{predictor}")
    return np.column_stack(columns), names


def collapse_collinear(design, names, max_vif):
    """Fold predictors that cannot be separated into one composite spread term.

    On a dome, tag separation and orientation spread are the same variable
    (chord = 2 R sin(dtheta/2)), so fitting them as if they were independent
    would hand one of them the other's effect. Rather than silently dropping
    one, the inseparable group is standardised and averaged into a single term
    whose members are named in the output.
    """
    design = np.asarray(design, dtype=np.float64)
    names = list(names)
    inflation = fixed_effects.variance_inflation(design)
    offenders = [i for i, value in enumerate(inflation) if value > max_vif]
    if len(offenders) < 2:
        return design, names, []

    block = design[:, offenders]
    standardised = (block - block.mean(axis=0)) / block.std(axis=0, ddof=1)
    composite = standardised.mean(axis=1)
    keep = [i for i in range(design.shape[1]) if i not in offenders]
    reduced = np.column_stack([design[:, keep], composite])
    reduced_names = [names[i] for i in keep] + ["composite_spread"]
    return reduced, reduced_names, [names[i] for i in offenders]


def fit_response(cells, response_column, predictors=GEOMETRY_PREDICTORS, max_vif=10.0):
    """Within-burst fixed-effects fit of log(response) on log geometry predictors.

    Decision 4 (task-11-12 briefing): `max_normal_angle_deg` is NaN for every
    single-tag subset (n_tags == 1), because a pairwise angle is undefined for
    one tag. Dropping every row with a NaN would silently delete the entire
    single-tag condition -- the study's reference point -- from the fit. So
    instead the angle-bearing model is fit only on the rows where the angle
    predictors are finite (n_tags >= 2); single-tag rows are held out and
    returned separately as the reference condition, since an angle effect
    simply cannot be estimated from subsets that have no angle. The row counts
    of both groups are printed here so the split is visible in the script's
    output rather than implied.

    Returns ``(result, names, folded, fit_cells, single_tag_cells)``, where
    `fit_cells` carries a `stage1_residual` column (log response minus the
    fitted geometry effect, on the original, non-demeaned scale) for the
    between-burst stage to consume.
    """
    finite_angle = np.isfinite(cells["max_normal_angle_deg"].to_numpy(dtype=float))
    fit_cells = cells.loc[finite_angle].reset_index(drop=True)
    single_tag_cells = cells.loc[~finite_angle].reset_index(drop=True)
    print(
        f"  {response_column}: fitting on {len(fit_cells)} rows with n_tags >= 2 "
        f"(finite angle predictors); holding out {len(single_tag_cells)} "
        "single-tag rows as the reference condition (no pairwise angle exists)."
    )

    design, names = build_design(fit_cells, predictors)
    design, names, folded = collapse_collinear(design, names, max_vif)
    log_response = np.log(fit_cells[response_column].to_numpy(dtype=float))
    result = fixed_effects.fit_fixed_effects(
        design,
        log_response,
        fit_cells["burst"].to_numpy(),
        weights=2.0 * (fit_cells["frames"].to_numpy(dtype=float) - 1.0),
    )
    # Applying the within-estimated slopes to the ORIGINAL (non-demeaned)
    # design deliberately leaves the burst fixed effect in the residual: that
    # per-burst offset is exactly what the between-burst stage regresses on
    # distance and incidence next.
    fit_cells = fit_cells.copy()
    fit_cells["stage1_residual"] = log_response - design @ result.coefficients
    return result, names, folded, fit_cells, single_tag_cells
