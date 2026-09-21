import numpy as np
import pytest

from jitter_model import common


def _rigidbody_dict(spec, tag_size_m=0.05):
    return {
        "meta": {
            "tag_size_m": tag_size_m,
            "reference_id": 1,
            "marker_ids": sorted(spec),
        },
        "markers": {
            str(marker_id): {
                "rotation_marker_to_reference": tag["R"].tolist(),
                "translation_marker_to_reference_m": tag["t"].tolist(),
            }
            for marker_id, tag in spec.items()
        },
    }


def test_tag_corners_local_is_a_centred_square():
    corners = common.tag_corners_local(0.05)
    assert corners.shape == (4, 3)
    assert np.allclose(corners.mean(axis=0), 0.0)
    assert np.allclose(corners[:, 2], 0.0)
    side = np.linalg.norm(corners[1] - corners[0])
    assert np.isclose(side, 0.05)


def test_build_tag_rig_places_the_reference_tag_at_the_origin(synthetic_rig_spec):
    rig = common.build_tag_rig(_rigidbody_dict(synthetic_rig_spec))
    assert rig.reference_id == 1
    assert np.allclose(rig.centers()[1], 0.0)
    assert np.allclose(rig.corners_reference[1], common.tag_corners_local(0.05))


def test_tag_rig_normals_match_the_facet_tilt(synthetic_rig_spec):
    rig = common.build_tag_rig(_rigidbody_dict(synthetic_rig_spec))
    normals = rig.normals()
    angle = np.degrees(np.arccos(np.clip(normals[1] @ normals[2], -1.0, 1.0)))
    assert np.isclose(angle, 30.0, atol=1e-6)


def test_build_camera_models_reshapes_distortion(synthetic_camera):
    stereo = {
        "cam0": {
            "camera_matrix": synthetic_camera["K"].tolist(),
            "dist_coeffs": [0.1, 0.01, 0.0, 0.0],
            "resolution": [640, 480],
        }
    }
    models = common.build_camera_models(stereo, ["cam0"])
    assert models["cam0"].D.shape == (4, 1)
    assert models["cam0"].resolution == (640, 480)


def test_stereo_extrinsic_prefers_the_refined_block():
    identity = np.eye(3).tolist()
    stereo = {"stereo": {"R": identity, "T": [70.0, 0.0, 0.0]}}
    rigidbody = {
        "stereo_refined": {
            "rotation_cam0_to_cam1": identity,
            "translation_cam0_to_cam1_m": [0.0775, 0.0, 0.0],
        }
    }
    _R, T, _rvec = common.stereo_extrinsic(stereo, rigidbody)
    assert np.isclose(T[0], 0.0775)


def test_stereo_extrinsic_falls_back_and_converts_mm():
    stereo = {"stereo": {"R": np.eye(3).tolist(), "T": [70.0, 0.0, 0.0]}}
    with pytest.warns(UserWarning):
        _R, T, _rvec = common.stereo_extrinsic(stereo, {})
    assert np.isclose(T[0], 0.070)
