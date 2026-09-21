"""Geometry descriptors for one tag subset.

Everything here is a candidate predictor in the jitter model, so each is
defined once and computed the same way for measured and simulated layouts.
"""

import itertools

import cv2
import numpy as np


def subset_geometry(rig, marker_ids, fixed_point):
    """Exact board-frame geometry of a subset, independent of any camera pose.

    For single-tag subsets: lengths are genuinely zero (baseline, radius, thickness),
    but angles are NaN because they are undefined for a single tag.
    """
    marker_ids = tuple(marker_ids)
    centers = rig.centers()
    normals = rig.normals()
    positions = np.asarray([centers[m] for m in marker_ids])
    corners = np.concatenate([rig.corners_reference[m] for m in marker_ids])

    if len(marker_ids) > 1:
        baselines = [
            np.linalg.norm(centers[a] - centers[b])
            for a, b in itertools.combinations(marker_ids, 2)
        ]
        angles = [
            np.degrees(np.arccos(np.clip(normals[a] @ normals[b], -1.0, 1.0)))
            for a, b in itertools.combinations(marker_ids, 2)
        ]
        max_angle = float(np.max(angles))
        mean_angle = float(np.mean(angles))
    else:
        baselines = [0.0]
        max_angle = float("nan")
        mean_angle = float("nan")

    centroid = positions.mean(axis=0)
    # The smallest singular value of the mean-centred corner cloud is the
    # cloud's thickness out of its best-fit plane: near zero means the subset
    # sits on the planar-ambiguity ridge where PnP is worst conditioned.
    singular = np.linalg.svd(corners - corners.mean(axis=0), compute_uv=False)

    return {
        "n_tags": len(marker_ids),
        "n_corners": 4 * len(marker_ids),
        "max_baseline_mm": 1000.0 * float(np.max(baselines)),
        "rms_radius_mm": 1000.0
        * float(np.sqrt(np.mean(np.sum((positions - centroid) ** 2, axis=1)))),
        "min_singular_mm": 1000.0 * float(singular[-1]),
        "max_normal_angle_deg": max_angle,
        "mean_normal_angle_deg": mean_angle,
        "lever_mm": 1000.0 * float(np.linalg.norm(centroid - np.asarray(fixed_point))),
    }


def pose_geometry(rig, marker_ids, rvec, tvec, frame_detections):
    """Pose-dependent descriptors: how far away and how obliquely it was seen."""
    marker_ids = tuple(marker_ids)
    rotation = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    centers = rig.centers()
    normals = rig.normals()

    incidences = []
    sizes = []
    for marker_id in marker_ids:
        centre_cam = rotation @ centers[marker_id] + translation
        normal_cam = rotation @ normals[marker_id]
        viewing = centre_cam / np.linalg.norm(centre_cam)
        # The tag may be defined with either facing, so the obtuse case is
        # folded back: what matters is the tilt away from face-on, not the sign.
        cosine = abs(float(normal_cam @ viewing))
        incidences.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        corners = np.asarray(frame_detections[marker_id], dtype=np.float64)
        sides = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
        sizes.append(float(np.mean(sides)))

    centroid_cam = (
        rotation @ np.asarray([centers[m] for m in marker_ids]).mean(axis=0)
        + translation
    )
    return {
        "distance_m": float(np.linalg.norm(centroid_cam)),
        "mean_incidence_deg": float(np.mean(incidences)),
        "min_incidence_deg": float(np.min(incidences)),
        "mean_apparent_size_px": float(np.mean(sizes)),
    }
