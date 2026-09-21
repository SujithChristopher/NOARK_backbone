import numpy as np

from tests.jitter_model.conftest import SPHERE_RADIUS_M


def test_facet_tags_sit_on_the_sphere(synthetic_rig_spec):
    centre = np.array([0.0, 0.0, -SPHERE_RADIUS_M])
    for tag in synthetic_rig_spec.values():
        assert np.isclose(np.linalg.norm(tag["t"] - centre), SPHERE_RADIUS_M)


def test_facet_normals_are_tilted_thirty_degrees(synthetic_rig_spec):
    pole_normal = synthetic_rig_spec[1]["R"][:, 2]
    for marker_id in (2, 3, 4, 5):
        normal = synthetic_rig_spec[marker_id]["R"][:, 2]
        angle = np.degrees(np.arccos(np.clip(pole_normal @ normal, -1.0, 1.0)))
        assert np.isclose(angle, 30.0, atol=1e-6)


def test_projection_round_trips_a_known_point(project_corners):
    image = project_corners([[0.0, 0.0, 1.0]], np.zeros(3), np.zeros(3))
    assert np.allclose(image[0], [320.0, 240.0])
