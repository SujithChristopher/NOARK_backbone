"""Shared calibration, rig, and pose machinery for the jitter analyses.

`05_static_jitter.py` and `06_movement_error.py` each carried their own copy of
this code. A third copy in `08_subset_geometry.py` is where copies start to
drift, so it lives here once, with its dependencies passed in rather than read
from module globals — which is also what lets worker processes and unit tests
use it.
"""

import warnings
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraModel:
    name: str
    K: np.ndarray
    D: np.ndarray
    resolution: tuple


@dataclass(frozen=True)
class TagRig:
    tag_size_m: float
    reference_id: int
    marker_ids: tuple
    corners_reference: dict
    marker_to_reference: dict

    def centers(self):
        """Each tag's centre in the reference tag's frame."""
        return {
            marker_id: corners.mean(axis=0)
            for marker_id, corners in self.corners_reference.items()
        }

    def normals(self):
        """Each tag's outward normal in the reference tag's frame."""
        return {
            marker_id: rotation[:, 2]
            for marker_id, (rotation, _t) in self.marker_to_reference.items()
        }


def tag_corners_local(tag_size_m):
    """One tag's four corners in its own frame, counter-clockwise from top-left."""
    half = tag_size_m / 2.0
    return np.asarray(
        [
            [-half, +half, 0.0],
            [+half, +half, 0.0],
            [+half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def build_tag_rig(rigidbody):
    """Every tag's corners in the reference tag's frame, as one rigid point cloud.

    Expressing all tags in one frame is what lets any subset be solved by a
    single PnP instead of averaging independent per-tag poses.
    """
    tag_size_m = float(rigidbody["meta"]["tag_size_m"])
    local = tag_corners_local(tag_size_m)
    corners_reference = {}
    marker_to_reference = {}
    for marker_id_text, marker_data in rigidbody["markers"].items():
        rotation = np.asarray(
            marker_data["rotation_marker_to_reference"], dtype=np.float64
        )
        translation = np.asarray(
            marker_data["translation_marker_to_reference_m"], dtype=np.float64
        )
        marker_id = int(marker_id_text)
        corners_reference[marker_id] = local @ rotation.T + translation
        marker_to_reference[marker_id] = (rotation, translation)
    return TagRig(
        tag_size_m=tag_size_m,
        reference_id=int(rigidbody["meta"]["reference_id"]),
        marker_ids=tuple(int(x) for x in rigidbody["meta"]["marker_ids"]),
        corners_reference=corners_reference,
        marker_to_reference=marker_to_reference,
    )


def build_camera_models(stereo, names):
    models = {}
    for name in names:
        camera_data = stereo[name]
        models[name] = CameraModel(
            name=name,
            K=np.asarray(camera_data["camera_matrix"], dtype=np.float64),
            D=np.asarray(camera_data["dist_coeffs"], dtype=np.float64).reshape(-1, 1),
            resolution=tuple(camera_data["resolution"]),
        )
    return models


def stereo_extrinsic(stereo, rigidbody):
    """Rotation and translation cam0 -> cam1, preferring notebook 02's refit.

    The self-calibrated extrinsic was solved against this very rig, so it beats
    the checkerboard one; the calibration TOML is the fallback and states its
    translation in millimetres.
    """
    if "stereo_refined" in rigidbody:
        rotation = np.asarray(
            rigidbody["stereo_refined"]["rotation_cam0_to_cam1"], dtype=np.float64
        )
        translation = np.asarray(
            rigidbody["stereo_refined"]["translation_cam0_to_cam1_m"], dtype=np.float64
        )
    else:
        warnings.warn("No refined stereo extrinsic; falling back to calibration TOML")
        rotation = np.asarray(stereo["stereo"]["R"], dtype=np.float64)
        translation = np.asarray(stereo["stereo"]["T"], dtype=np.float64) / 1000.0
    return rotation, translation, cv2.Rodrigues(rotation)[0]
