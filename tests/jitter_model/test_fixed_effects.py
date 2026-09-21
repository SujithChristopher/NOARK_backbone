import numpy as np

from jitter_model import fixed_effects


def test_within_transform_removes_group_means():
    values = np.array([1.0, 3.0, 10.0, 14.0])
    groups = np.array([0, 0, 1, 1])
    demeaned = fixed_effects.within_transform(values, groups)
    assert np.allclose(demeaned, [-1.0, 1.0, -2.0, 2.0])


def test_fit_recovers_known_coefficients_despite_group_offsets():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(40), 12)
    offsets = rng.normal(0.0, 5.0, 40)[groups]
    x1 = rng.normal(0.0, 1.0, groups.size)
    x2 = rng.normal(0.0, 1.0, groups.size)
    y = offsets + 0.7 * x1 - 0.3 * x2 + rng.normal(0.0, 0.05, groups.size)
    result = fixed_effects.fit_fixed_effects(np.column_stack([x1, x2]), y, groups)
    assert np.allclose(result.coefficients, [0.7, -0.3], atol=0.02)
    assert result.n_groups == 40


def test_confidence_intervals_bracket_the_truth():
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(30), 15)
    x = rng.normal(0.0, 1.0, groups.size)
    y = rng.normal(0.0, 3.0, 30)[groups] + 0.5 * x + rng.normal(0.0, 0.2, groups.size)
    result = fixed_effects.fit_fixed_effects(x.reshape(-1, 1), y, groups)
    assert result.ci_low[0] < 0.5 < result.ci_high[0]


def test_clustered_errors_exceed_naive_errors_under_group_correlation():
    """Residuals correlated inside a group must widen the interval, not narrow it."""
    rng = np.random.default_rng(6)
    groups = np.repeat(np.arange(25), 20)
    x = rng.normal(0.0, 1.0, groups.size)
    shock = rng.normal(0.0, 1.0, 25)[groups]
    y = 0.4 * x + shock * x + rng.normal(0.0, 0.05, groups.size)
    clustered = fixed_effects.fit_fixed_effects(x.reshape(-1, 1), y, groups)
    residual = y - fixed_effects.within_transform(x, groups) * clustered.coefficients[0]
    naive = np.sqrt(
        np.var(residual, ddof=1)
        / np.sum(fixed_effects.within_transform(x, groups) ** 2)
    )
    assert clustered.standard_errors[0] > naive


def test_variance_inflation_flags_a_collinear_pair():
    rng = np.random.default_rng(7)
    x1 = rng.normal(0.0, 1.0, 300)
    x2 = x1 + rng.normal(0.0, 0.02, 300)
    x3 = rng.normal(0.0, 1.0, 300)
    vif = fixed_effects.variance_inflation(np.column_stack([x1, x2, x3]))
    assert vif[0] > 50.0
    assert vif[1] > 50.0
    assert vif[2] < 2.0


def test_weights_favour_the_precise_observations():
    rng = np.random.default_rng(8)
    groups = np.repeat(np.arange(20), 10)
    x = rng.normal(0.0, 1.0, groups.size)
    y = 1.0 * x
    y[::2] += 5.0  # noisy half
    weights = np.where(np.arange(groups.size) % 2 == 0, 0.01, 100.0)
    result = fixed_effects.fit_fixed_effects(
        x.reshape(-1, 1), y, groups, weights=weights
    )
    assert np.isclose(result.coefficients[0], 1.0, atol=0.1)
