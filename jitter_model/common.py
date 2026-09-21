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
from scipy.optimize import least_squares


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


class PoseSolver:
    """Joint board-PnP estimators over any subset of the rig's tags.

    One solve over every visible corner, never an average of per-tag `tvec`s:
    averaging independent poses throws away the constraint that the tags are
    one rigid body, which is most of what the extra tags are worth.
    """

    def __init__(self, rig, cameras, stereo_rotation=None, stereo_translation=None):
        self.rig = rig
        self.cameras = cameras
        self._local_corners = tag_corners_local(rig.tag_size_m)
        self._stereo_translation = (
            None
            if stereo_translation is None
            else np.asarray(stereo_translation, dtype=np.float64).reshape(3, 1)
        )
        self._stereo_rvec = (
            None if stereo_rotation is None else cv2.Rodrigues(stereo_rotation)[0]
        )

    def stack_correspondences(self, frame_detections, marker_ids):
        """Every requested tag's corners as one board, or None if any is missing.

        Requiring the whole set keeps a condition's geometry fixed: a 4-tag
        number is always four tags, never whichever two happened to be visible.
        """
        if not all(marker_id in frame_detections for marker_id in marker_ids):
            return None
        return (
            np.concatenate([self.rig.corners_reference[m] for m in marker_ids]),
            np.concatenate([frame_detections[m] for m in marker_ids]),
        )

    def raw_reprojection_rmse(
        self, object_points, image_points, rvec, tvec, camera_name
    ):
        camera = self.cameras[camera_name]
        projected, _ = cv2.fisheye.projectPoints(
            object_points.reshape(-1, 1, 3), rvec, tvec, camera.K, camera.D
        )
        residual = projected.reshape(-1, 2) - image_points
        return float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))

    def _undistort(self, image_points_raw, camera):
        return cv2.fisheye.undistortPoints(
            image_points_raw.reshape(-1, 1, 2), camera.K, camera.D, P=camera.K
        ).reshape(-1, 2)

    def single_tag_pose(self, frame_detections, marker_id, camera_name):
        """Board pose from one tag, solved on that tag's own square then composed.

        IPPE_SQUARE wants a planar square centred on the origin, which only the
        reference tag's corners satisfy in board coordinates. So the solve runs in
        the tag's own frame and the result is carried onto the board through the
        rigid-body transform, letting any tag stand in as the single-tag estimator.
        """
        if marker_id not in frame_detections:
            return None
        model = self.cameras[camera_name]
        image_points_raw = frame_detections[marker_id]
        image_points = self._undistort(image_points_raw, model)
        solutions = cv2.solvePnPGeneric(
            self._local_corners,
            image_points,
            model.K,
            None,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not solutions[0]:
            return None

        rotation_marker, translation_marker = self.rig.marker_to_reference[marker_id]
        best = None
        for rvec_tag, tvec_tag in zip(solutions[1], solutions[2]):
            if float(tvec_tag.reshape(3)[2]) <= 0:
                continue
            # p_cam = R_tag p_local + t_tag and p_ref = R_m2r p_local + t_m2r, so
            # the board pose is R_tag R_m2r' with the origin shifted accordingly.
            rotation_board = cv2.Rodrigues(rvec_tag)[0] @ rotation_marker.T
            translation_board = (
                tvec_tag.reshape(3) - rotation_board @ translation_marker
            )
            rvec = cv2.Rodrigues(rotation_board)[0]
            tvec = translation_board.reshape(3, 1)
            error = self.raw_reprojection_rmse(
                self.rig.corners_reference[marker_id],
                image_points_raw,
                rvec,
                tvec,
                camera_name,
            )
            if best is None or error < best[0]:
                best = (error, rvec, tvec)
        if best is None:
            return None
        return {
            "rvec": np.asarray(best[1]).reshape(3),
            "tvec": np.asarray(best[2]).reshape(3),
            "rmse_px": best[0],
        }

    def seed_pose(self, frame_detections, marker_ids, camera_name):
        """The best single-tag board pose among ``marker_ids``, to start PnP from."""
        candidates = [
            pose
            for pose in (
                self.single_tag_pose(frame_detections, m, camera_name)
                for m in marker_ids
            )
            if pose is not None
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda pose: pose["rmse_px"])

    def mono_board_pose(self, frame_detections, marker_ids, camera_name="cam0"):
        """One joint PnP over every visible corner of the requested tags.

        A joint solve, never an average of per-tag `tvec`s: averaging independent
        poses throws away the constraint that the tags are one rigid body, which is
        most of what the extra tags are worth.
        """
        if len(marker_ids) == 1:
            return self.single_tag_pose(frame_detections, marker_ids[0], camera_name)

        correspondences = self.stack_correspondences(frame_detections, marker_ids)
        if correspondences is None:
            return None
        object_points, image_points_raw = correspondences
        model = self.cameras[camera_name]
        image_points = self._undistort(image_points_raw, model)

        # A small tag subset is nearly planar and ITERATIVE can otherwise land on
        # its mirrored local solution. Seed it with the best single-tag pose from
        # this same image: geometric initialization, not temporal filtering.
        initial = self.seed_pose(frame_detections, marker_ids, camera_name)
        if initial is None:
            return None
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            model.K,
            None,
            initial["rvec"].reshape(3, 1).copy(),
            initial["tvec"].reshape(3, 1).copy(),
            True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or float(tvec.reshape(3)[2]) <= 0:
            return None
        return {
            "rvec": np.asarray(rvec).reshape(3),
            "tvec": np.asarray(tvec).reshape(3),
            "rmse_px": self.raw_reprojection_rmse(
                object_points, image_points_raw, rvec, tvec, camera_name
            ),
        }

    def stereo_board_pose(self, frame0, frame1, marker_ids):
        """One pose minimizing corner reprojection in both fisheye cameras at once.

        The cross-baseline constraint is what tightens depth, so the two images are
        fitted together rather than triangulating two independent single-camera
        poses.
        """
        corr0 = self.stack_correspondences(frame0, marker_ids)
        corr1 = self.stack_correspondences(frame1, marker_ids)
        if corr0 is None or corr1 is None:
            return None
        object_points, image0 = corr0
        _object_points1, image1 = corr1
        initial = self.mono_board_pose(frame0, marker_ids, "cam0")
        if initial is None:
            return None

        def residual(parameters):
            rvec0 = parameters[:3].reshape(3, 1)
            tvec0 = parameters[3:].reshape(3, 1)
            projected0, _ = cv2.fisheye.projectPoints(
                object_points.reshape(-1, 1, 3),
                rvec0,
                tvec0,
                self.cameras["cam0"].K,
                self.cameras["cam0"].D,
            )
            rvec1, tvec1 = cv2.composeRT(
                rvec0, tvec0, self._stereo_rvec, self._stereo_translation
            )[:2]
            projected1, _ = cv2.fisheye.projectPoints(
                object_points.reshape(-1, 1, 3),
                rvec1,
                tvec1,
                self.cameras["cam1"].K,
                self.cameras["cam1"].D,
            )
            return np.concatenate(
                [
                    (projected0.reshape(-1, 2) - image0).ravel(),
                    (projected1.reshape(-1, 2) - image1).ravel(),
                ]
            )

        result = least_squares(
            residual,
            np.concatenate([initial["rvec"], initial["tvec"]]),
            # Matches the validated stereo PnP in
            # trunkpose/dual_notebooks/verification_dual_camera.py. The cam0-only
            # seed can have a large cam1 residual, for which a robust loss would
            # suppress the very measurements needed to refine depth across the
            # baseline.
            method="lm",
            max_nfev=100,
        )
        if result.x[5] <= 0:
            return None
        point_residuals = residual(result.x).reshape(-1, 2)
        return {
            "rvec": result.x[:3],
            "tvec": result.x[3:],
            "rmse_px": float(np.sqrt(np.mean(np.sum(point_residuals**2, axis=1)))),
        }
