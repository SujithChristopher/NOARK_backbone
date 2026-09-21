"""Within-group least squares with cluster-robust errors.

A per-burst intercept absorbs pose, distance and lighting exactly, so every
geometry coefficient is identified from subsets compared against each other
inside the same burst rather than across bursts that differ in other ways.
Implemented by demeaning rather than by dummy columns, which keeps it to one
`lstsq` and adds no dependency.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class FitResult:
    coefficients: np.ndarray
    standard_errors: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    vif: np.ndarray
    r_squared: float
    n_groups: int
    n_observations: int


def within_transform(values, groups, weights=None):
    """Subtract each group's (optionally weighted) mean."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    weights = np.ones(len(groups)) if weights is None else np.asarray(weights, float)
    out = np.array(values, dtype=np.float64, copy=True)
    for group in np.unique(groups):
        mask = groups == group
        mean = np.average(values[mask], axis=0, weights=weights[mask])
        out[mask] = values[mask] - mean
    return out


def variance_inflation(design):
    """VIF per column: how much collinearity inflates that coefficient's variance."""
    design = np.asarray(design, dtype=np.float64)
    inflation = np.empty(design.shape[1])
    for column in range(design.shape[1]):
        others = np.delete(design, column, axis=1)
        others = np.column_stack([others, np.ones(len(others))])
        target = design[:, column]
        fitted = others @ np.linalg.lstsq(others, target, rcond=None)[0]
        residual_ss = float(np.sum((target - fitted) ** 2))
        total_ss = float(np.sum((target - target.mean()) ** 2))
        r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else 0.0
        inflation[column] = np.inf if r_squared >= 1.0 else 1.0 / (1.0 - r_squared)
    return inflation


def fit_fixed_effects(design, response, groups, weights=None, confidence=0.95):
    """Weighted within-group least squares with errors clustered by group."""
    design = np.asarray(design, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    groups = np.asarray(groups)
    weights = np.ones(len(response)) if weights is None else np.asarray(weights, float)

    x = within_transform(design, groups, weights)
    y = within_transform(response, groups, weights)
    root = np.sqrt(weights)[:, None]
    coefficients = np.linalg.lstsq(x * root, y * root.ravel(), rcond=None)[0]

    residual = y - x @ coefficients
    bread = np.linalg.pinv((x * weights[:, None]).T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]))
    unique_groups = np.unique(groups)
    for group in unique_groups:
        mask = groups == group
        score = (x[mask] * weights[mask, None]).T @ residual[mask]
        meat += np.outer(score, score)
    n_groups = len(unique_groups)
    # Small-cluster correction, the usual one for a finite number of clusters.
    scale = n_groups / max(n_groups - 1, 1)
    covariance = scale * bread @ meat @ bread
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    critical = stats.t.ppf(0.5 + confidence / 2.0, max(n_groups - 1, 1))
    total_ss = float(np.sum(weights * (y - np.average(y, weights=weights)) ** 2))
    residual_ss = float(np.sum(weights * residual**2))
    return FitResult(
        coefficients=coefficients,
        standard_errors=standard_errors,
        ci_low=coefficients - critical * standard_errors,
        ci_high=coefficients + critical * standard_errors,
        vif=variance_inflation(design),
        r_squared=1.0 - residual_ss / total_ss if total_ss > 0 else float("nan"),
        n_groups=n_groups,
        n_observations=len(response),
    )
