"""Round-trip tests: project a known pose, solve it back, expect it recovered.

With zero distortion the projection is exact, so any error here is the
estimator's, not the lens model's.
"""

import cv2
import numpy as np
import pytest

from jitter_model import common
from tests.jitter_model.test_common_loading import _rigidbody_dict

TRUE_RVEC = np.array([0.05, -0.10, 0.02])
TRUE_TVEC = np.array([0.01, -0.02, 0.60])


@pytest.fixture
def solver(synthetic_rig_spec, synthetic_camera):
    rig = common.build_tag_rig(_rigidbody_dict(synthetic_rig_spec))
    cameras = {
        name: common.CameraModel(
            name=name,
            K=synthetic_camera["K"],
            D=synthetic_camera["D"],
            resolution=synthetic_camera["resolution"],
        )
        for name in ("cam0", "cam1")
    }
    stereo_rotation = np.eye(3)
    stereo_translation = np.array([0.0775, 0.0, 0.0])
    return common.PoseSolver(rig, cameras, stereo_rotation, stereo_translation)


def _detections(solver, marker_ids, project_corners, rvec=TRUE_RVEC, tvec=TRUE_TVEC):
    return {
        marker_id: project_corners(solver.rig.corners_reference[marker_id], rvec, tvec)
        for marker_id in marker_ids
    }


def test_single_tag_pose_recovers_the_board_pose(solver, project_corners):
    detections = _detections(solver, [1], project_corners)
    pose = solver.single_tag_pose(detections, 1, "cam0")
    assert np.allclose(pose["tvec"], TRUE_TVEC, atol=1e-4)
    assert np.allclose(pose["rvec"], TRUE_RVEC, atol=1e-3)
    assert pose["rmse_px"] < 1e-3


def test_single_tag_pose_of_an_offset_tag_still_reports_the_board(
    solver, project_corners
):
    detections = _detections(solver, [3], project_corners)
    pose = solver.single_tag_pose(detections, 3, "cam0")
    assert np.allclose(pose["tvec"], TRUE_TVEC, atol=1e-3)


def test_single_tag_pose_returns_none_when_the_tag_is_absent(solver):
    assert solver.single_tag_pose({}, 1, "cam0") is None


def test_mono_board_pose_recovers_the_pose_from_four_tags(solver, project_corners):
    detections = _detections(solver, [1, 2, 3, 4], project_corners)
    pose = solver.mono_board_pose(detections, (1, 2, 3, 4), "cam0")
    assert np.allclose(pose["tvec"], TRUE_TVEC, atol=1e-5)
    assert np.allclose(pose["rvec"], TRUE_RVEC, atol=1e-5)


def test_mono_board_pose_requires_every_requested_tag(solver, project_corners):
    detections = _detections(solver, [1, 2], project_corners)
    assert solver.mono_board_pose(detections, (1, 2, 3), "cam0") is None


def test_stack_correspondences_orders_object_and_image_points_together(
    solver, project_corners
):
    detections = _detections(solver, [1, 2], project_corners)
    object_points, image_points = solver.stack_correspondences(detections, (1, 2))
    assert object_points.shape == (8, 3)
    assert image_points.shape == (8, 2)
    assert np.allclose(object_points[:4], solver.rig.corners_reference[1])


def test_stereo_board_pose_recovers_the_pose(solver, project_corners):
    frame0 = _detections(solver, [1, 2, 3], project_corners)
    rvec1, tvec1 = cv2.composeRT(
        TRUE_RVEC.reshape(3, 1),
        TRUE_TVEC.reshape(3, 1),
        cv2.Rodrigues(np.eye(3))[0],
        np.array([0.0775, 0.0, 0.0]).reshape(3, 1),
    )[:2]
    frame1 = {
        marker_id: project_corners(
            solver.rig.corners_reference[marker_id], rvec1, tvec1
        )
        for marker_id in (1, 2, 3)
    }
    pose = solver.stereo_board_pose(frame0, frame1, (1, 2, 3))
    assert np.allclose(pose["tvec"], TRUE_TVEC, atol=1e-4)
    assert np.allclose(pose["rvec"], TRUE_RVEC, atol=1e-4)


def test_single_tag_pose_scores_rmse_against_raw_points_not_undistorted_ones(
    synthetic_rig_spec, distorted_camera, project_corners
):
    """Pins the raw-vs-undistorted split inside `single_tag_pose`.

    `single_tag_pose` must solve PnP on undistorted points but score
    `rmse_px` against the RAW (distorted) points it was actually handed.
    `synthetic_camera`'s zero D makes undistortion the identity everywhere
    else in this file, so raw and undistorted points are indistinguishable
    there and a swap of the two at the `raw_reprojection_rmse` call site
    would still pass every other test. With `distorted_camera`'s real
    distortion the two point sets differ, so if that call site were ever
    swapped to pass the undistorted points instead of the raw ones,
    `raw_reprojection_rmse` would reproject through the full distortion
    model and compare against points that are already in the undistorted
    space -- a large, wrong RMSE instead of the near-zero one asserted here.
    """
    rig = common.build_tag_rig(_rigidbody_dict(synthetic_rig_spec))
    camera = common.CameraModel(
        name="cam0",
        K=distorted_camera["K"],
        D=distorted_camera["D"],
        resolution=distorted_camera["resolution"],
    )
    solver = common.PoseSolver(rig, {"cam0": camera})
    detections = {
        1: project_corners(
            rig.corners_reference[1], TRUE_RVEC, TRUE_TVEC, camera=distorted_camera
        )
    }
    pose = solver.single_tag_pose(detections, 1, "cam0")
    assert np.allclose(pose["tvec"], TRUE_TVEC, atol=1e-3)
    assert np.allclose(pose["rvec"], TRUE_RVEC, atol=1e-3)
    assert pose["rmse_px"] < 1e-3


def test_more_tags_reduce_jitter_under_corner_noise(solver, project_corners):
    rng = np.random.default_rng(0)

    def spread(marker_ids):
        positions = []
        clean = _detections(solver, marker_ids, project_corners)
        for _ in range(60):
            noisy = {
                marker_id: corners + rng.normal(0.0, 0.3, corners.shape)
                for marker_id, corners in clean.items()
            }
            pose = solver.mono_board_pose(noisy, tuple(marker_ids), "cam0")
            positions.append(pose["tvec"])
        return np.linalg.norm(np.std(np.asarray(positions), axis=0, ddof=1))

    assert spread([1, 2, 3, 4]) < spread([1])
