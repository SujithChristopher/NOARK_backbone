import numpy as np

from jitter_model import common, geometry
from tests.jitter_model.test_common_loading import _rigidbody_dict


def _rig(spec):
    return common.build_tag_rig(_rigidbody_dict(spec))


def test_single_tag_has_no_baseline_and_no_thickness():
    spec = {1: {"R": np.eye(3), "t": np.zeros(3)}}
    result = geometry.subset_geometry(_rig(spec), (1,), np.zeros(3))
    assert result["max_baseline_mm"] == 0.0
    assert result["min_singular_mm"] < 1e-9
    assert result["n_corners"] == 4
    assert np.isnan(result["max_normal_angle_deg"])
    assert np.isnan(result["mean_normal_angle_deg"])


def test_two_parallel_tags_have_zero_angle():
    spec = {
        1: {"R": np.eye(3), "t": np.zeros(3)},
        2: {"R": np.eye(3), "t": np.array([0.10, 0.0, 0.0])},
    }
    result = geometry.subset_geometry(_rig(spec), (1, 2), np.zeros(3))
    assert np.isclose(result["max_normal_angle_deg"], 0.0)
    assert np.isclose(result["mean_normal_angle_deg"], 0.0)


def test_baseline_is_the_largest_pairwise_centre_distance():
    spec = {
        1: {"R": np.eye(3), "t": np.zeros(3)},
        2: {"R": np.eye(3), "t": np.array([0.10, 0.0, 0.0])},
        3: {"R": np.eye(3), "t": np.array([0.04, 0.0, 0.0])},
    }
    result = geometry.subset_geometry(_rig(spec), (1, 2, 3), np.zeros(3))
    assert np.isclose(result["max_baseline_mm"], 100.0)


def test_coplanar_tags_have_near_zero_min_singular():
    spec = {
        1: {"R": np.eye(3), "t": np.zeros(3)},
        2: {"R": np.eye(3), "t": np.array([0.10, 0.0, 0.0])},
    }
    result = geometry.subset_geometry(_rig(spec), (1, 2), np.zeros(3))
    assert result["min_singular_mm"] < 1e-6


def test_tilted_tags_have_a_real_min_singular(synthetic_rig_spec):
    result = geometry.subset_geometry(_rig(synthetic_rig_spec), (1, 2), np.zeros(3))
    assert result["min_singular_mm"] > 1.0


def test_normal_angle_matches_the_facet_tilt(synthetic_rig_spec):
    result = geometry.subset_geometry(_rig(synthetic_rig_spec), (1, 2), np.zeros(3))
    assert np.isclose(result["max_normal_angle_deg"], 30.0, atol=1e-6)


def test_lever_is_the_centroid_to_fixed_point_distance():
    spec = {
        1: {"R": np.eye(3), "t": np.zeros(3)},
        2: {"R": np.eye(3), "t": np.array([0.10, 0.0, 0.0])},
    }
    result = geometry.subset_geometry(_rig(spec), (2,), np.zeros(3))
    assert np.isclose(result["lever_mm"], 100.0)


def test_incidence_is_zero_for_a_tag_facing_the_camera():
    spec = {1: {"R": np.eye(3), "t": np.zeros(3)}}
    rig = _rig(spec)
    detections = {1: np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])}
    result = geometry.pose_geometry(
        rig, (1,), np.zeros(3), np.array([0.0, 0.0, 0.5]), detections
    )
    assert np.isclose(result["distance_m"], 0.5)
    assert result["mean_incidence_deg"] < 1e-6
    assert np.isclose(result["mean_apparent_size_px"], 10.0)
