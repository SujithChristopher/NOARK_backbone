"""`build_design` and `collapse_collinear` are pure and don't need the recording.

Task-11-12 decision 3: these live in `jitter_model/model_assembly.py`, not in
the `09_subset_geometry.py` script, so this test imports the module directly.
"""

import numpy as np
import pandas as pd

from jitter_model import model_assembly


def test_build_design_log_transforms_positive_predictors():
    cells = pd.DataFrame({"max_baseline_mm": [10.0, 100.0], "lever_mm": [1.0, 10.0]})
    design, names = model_assembly.build_design(cells, ["max_baseline_mm", "lever_mm"])
    assert names == ["log_max_baseline_mm", "log_lever_mm"]
    assert np.allclose(design[:, 0], np.log([10.0, 100.0]))


def test_build_design_is_finite_when_a_predictor_is_zero():
    cells = pd.DataFrame({"max_baseline_mm": [0.0, 50.0]})
    design, _names = model_assembly.build_design(cells, ["max_baseline_mm"])
    assert np.isfinite(design).all()


def test_collapse_collinear_folds_a_redundant_pair():
    rng = np.random.default_rng(9)
    x1 = rng.normal(0.0, 1.0, 400)
    design = np.column_stack(
        [x1, x1 + rng.normal(0.0, 0.01, 400), rng.normal(0.0, 1.0, 400)]
    )
    reduced, names, folded = model_assembly.collapse_collinear(
        design, ["a", "b", "c"], max_vif=10.0
    )
    assert "composite_spread" in names
    assert set(folded) == {"a", "b"}
    assert reduced.shape[1] == 2


def test_collapse_collinear_leaves_independent_predictors_alone():
    rng = np.random.default_rng(10)
    design = rng.normal(0.0, 1.0, (400, 3))
    reduced, names, folded = model_assembly.collapse_collinear(
        design, ["a", "b", "c"], max_vif=10.0
    )
    assert folded == []
    assert names == ["a", "b", "c"]
    assert reduced.shape == design.shape
