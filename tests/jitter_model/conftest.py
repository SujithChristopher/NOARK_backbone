"""Synthetic camera and tag rig, so the estimators can be tested without data.

The rig mimics the dome: one tag at the pole and four tilted onto facets of a
sphere of radius 115 mm, which is what the real dome measures.
"""

import cv2
import numpy as np
import pytest

SPHERE_RADIUS_M = 0.115
TAG_SIZE_M = 0.05


@pytest.fixture
def synthetic_camera():
    return {
        "K": np.array(
            [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        # Zero distortion keeps the projection invertible to machine precision,
        # so a round-trip test failure means the estimator is wrong, not the lens.
        "D": np.zeros((4, 1), dtype=np.float64),
        "resolution": (640, 480),
    }


@pytest.fixture
def distorted_camera():
    """Same intrinsics as `synthetic_camera`, but with real fisheye distortion.

    `synthetic_camera`'s zero D makes undistortion the identity, so nothing
    built on it alone can tell raw image points from undistorted ones apart.
    This fixture exists for tests that need that distinction to be real.
    """
    return {
        "K": np.array(
            [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "D": np.array([[-0.02], [0.004], [-0.001], [0.0002]], dtype=np.float64),
        "resolution": (640, 480),
    }


def _facet(azimuth_deg, tilt_deg):
    """One tag rotated onto a sphere facet, returned as board-frame R and t."""
    azimuth = np.radians(azimuth_deg)
    tilt = np.radians(tilt_deg)
    axis = np.array([np.cos(azimuth), np.sin(azimuth), 0.0]) * tilt
    rotation = cv2.Rodrigues(axis)[0]
    # The pole sits at the board origin; a facet centre is the pole swung
    # through `tilt` about the sphere centre, which lies at -z.
    centre = np.array([0.0, 0.0, -SPHERE_RADIUS_M])
    translation = centre + rotation @ np.array([0.0, 0.0, SPHERE_RADIUS_M])
    return rotation, translation


@pytest.fixture
def synthetic_rig_spec():
    spec = {1: {"R": np.eye(3), "t": np.zeros(3)}}
    for index, azimuth in enumerate((0.0, 90.0, 180.0, 270.0), start=2):
        rotation, translation = _facet(azimuth, 30.0)
        spec[index] = {"R": rotation, "t": translation}
    return spec


@pytest.fixture
def project_corners(synthetic_camera):
    def _project(object_points, rvec, tvec, camera=None):
        camera = camera or synthetic_camera
        projected, _ = cv2.fisheye.projectPoints(
            np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3),
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            camera["K"],
            camera["D"],
        )
        return projected.reshape(-1, 2)

    return _project
