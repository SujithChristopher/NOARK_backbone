import cv2
import numpy as np

from jitter_model import jitter_stats


def test_fixed_point_positions_applies_the_board_offset():
    rvecs = np.zeros((3, 3))
    tvecs = np.tile([0.0, 0.0, 0.5], (3, 1))
    positions = jitter_stats.fixed_point_positions(
        rvecs, tvecs, np.array([0.1, 0.0, 0.0])
    )
    assert np.allclose(positions, [[0.1, 0.0, 0.5]] * 3)


def test_fixed_point_positions_rotates_the_offset():
    rvec = np.array([0.0, 0.0, np.pi / 2])
    positions = jitter_stats.fixed_point_positions(
        rvec.reshape(1, 3), np.zeros((1, 3)), np.array([0.1, 0.0, 0.0])
    )
    assert np.allclose(positions[0], [0.0, 0.1, 0.0], atol=1e-9)


def test_position_jitter_is_the_norm_of_the_per_axis_std():
    rng = np.random.default_rng(1)
    positions = rng.normal(0.0, 0.001, (500, 3))
    result = jitter_stats.position_jitter_mm(positions)
    assert np.isclose(result["pos_jitter_mm"], np.sqrt(3.0), rtol=0.15)
    assert np.isclose(
        result["pos_jitter_mm"],
        np.linalg.norm([result[f"pos_jitter_{a}_mm"] for a in "xyz"]),
    )


def test_position_jitter_is_zero_for_a_still_estimate():
    result = jitter_stats.position_jitter_mm(np.tile([0.1, 0.2, 0.3], (10, 1)))
    assert result["pos_jitter_mm"] == 0.0


def test_chordal_mean_of_identical_rotations_is_that_rotation():
    rotation = cv2.Rodrigues(np.array([0.1, -0.2, 0.3]))[0]
    mean = jitter_stats.chordal_mean_rotation(np.tile(rotation, (5, 1, 1)))
    assert np.allclose(mean, rotation, atol=1e-9)


def test_rotation_jitter_is_zero_for_a_still_estimate():
    rvecs = np.tile([0.1, -0.2, 0.3], (8, 1))
    assert jitter_stats.rotation_jitter_mdeg(rvecs)["rot_jitter_mdeg"] == 0.0


def test_rotation_jitter_recovers_a_known_spread():
    rng = np.random.default_rng(2)
    base = np.array([0.4, -0.3, 1.2])
    sigma_rad = np.radians(0.01)
    rvecs = []
    for _ in range(800):
        perturb = cv2.Rodrigues(rng.normal(0.0, sigma_rad, 3))[0]
        rvecs.append(cv2.Rodrigues(cv2.Rodrigues(base)[0] @ perturb)[0].ravel())
    result = jitter_stats.rotation_jitter_mdeg(np.asarray(rvecs))
    expected_mdeg = 1000.0 * np.degrees(sigma_rad) * np.sqrt(3.0)
    assert np.isclose(result["rot_jitter_mdeg"], expected_mdeg, rtol=0.15)


def test_rotation_jitter_is_immune_to_rotvec_wrap():
    """Near pi the raw rvec can flip sign; the chordal residual must not care."""
    axis = np.array([0.0, 0.0, 1.0])
    rvecs = np.array(
        [(np.pi - 1e-6) * axis, -(np.pi - 1e-6) * axis, (np.pi - 1e-6) * axis]
    )
    result = jitter_stats.rotation_jitter_mdeg(rvecs)
    assert result["rot_jitter_mdeg"] < 1.0
