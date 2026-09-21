import numpy as np

from tests.jitter_model.conftest import SPHERE_RADIUS_M


def test_facet_tags_sit_on_the_sphere(synthetic_rig_spec):
    """Validate tag placement: distinct positions, radial normals, pole at origin."""
    centre = np.array([0.0, 0.0, -SPHERE_RADIUS_M])

    # Tag 1 (pole) sits at origin with identity rotation
    pole = synthetic_rig_spec[1]
    assert np.allclose(pole["R"], np.eye(3), atol=1e-9)
    assert np.allclose(pole["t"], np.zeros(3), atol=1e-9)

    # Four facet tags have distinct centres (no two within 1e-9)
    facet_positions = [synthetic_rig_spec[i]["t"] for i in (2, 3, 4, 5)]
    for i, pos_i in enumerate(facet_positions):
        for j, pos_j in enumerate(facet_positions):
            if i < j:
                assert np.linalg.norm(pos_i - pos_j) > 1e-9

    # Each facet's outward normal R[:,2] equals unit vector from sphere centre to tag
    for marker_id in (2, 3, 4, 5):
        tag = synthetic_rig_spec[marker_id]
        normal = tag["R"][:, 2]
        # Expected normal: unit vector from sphere centre to tag position
        expected_normal = (tag["t"] - centre) / np.linalg.norm(tag["t"] - centre)
        assert np.allclose(normal, expected_normal, atol=1e-9)


def test_facet_normals_are_tilted_thirty_degrees(synthetic_rig_spec):
    pole_normal = synthetic_rig_spec[1]["R"][:, 2]
    for marker_id in (2, 3, 4, 5):
        normal = synthetic_rig_spec[marker_id]["R"][:, 2]
        angle = np.degrees(np.arccos(np.clip(pole_normal @ normal, -1.0, 1.0)))
        assert np.isclose(angle, 30.0, atol=1e-6)


def test_projection_round_trips_a_known_point(project_corners, synthetic_camera):
    """Validate projection: on-axis projects to principal point, off-axis uses focal length."""
    # On-axis point [0,0,1] projects to principal point regardless of focal length
    image = project_corners([[0.0, 0.0, 1.0]], np.zeros(3), np.zeros(3))
    assert np.allclose(image[0], [320.0, 240.0])

    # Off-axis point: with zero distortion, fisheye projects [X,Y,Z] to radius
    # r = fx * arctan(sqrt(X^2+Y^2)/Z) from principal point (cx, cy)
    K = synthetic_camera["K"]
    fx = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]

    # Test point: [1, 0, 2] — off-axis but symmetric in x
    object_point = np.array([[1.0, 0.0, 2.0]])
    image_point = project_corners(object_point, np.zeros(3), np.zeros(3))

    # Expected: radius from principal point = fx * arctan(1/2)
    theta = np.arctan(np.sqrt(1.0**2 + 0.0**2) / 2.0)
    expected_radius = fx * theta
    expected_x = cx + expected_radius
    expected_y = cy

    actual_point = image_point[0]
    assert np.allclose(actual_point, [expected_x, expected_y], atol=1e-6)
