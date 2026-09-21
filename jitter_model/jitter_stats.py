"""Jitter of a repeated pose estimate, in position and in rotation.

The dome is still inside a burst, so the spread of the estimate across the
burst is the estimator's jitter with nothing else mixed in — no successive
differencing and no mocap residual, unlike the movement takes.
"""

import cv2
import numpy as np


def fixed_point_positions(rvecs, tvecs, fixed_point):
    """Where one fixed board point lands in camera coordinates, per frame.

    Every subset is reported at the same physical point, so jitter numbers
    stay comparable instead of each subset being quoted at its own origin.
    """
    rvecs = np.asarray(rvecs, dtype=np.float64).reshape(-1, 3)
    tvecs = np.asarray(tvecs, dtype=np.float64).reshape(-1, 3)
    fixed_point = np.asarray(fixed_point, dtype=np.float64).reshape(3)
    positions = np.empty_like(tvecs)
    for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        positions[index] = cv2.Rodrigues(rvec.reshape(3, 1))[0] @ fixed_point + tvec
    return positions


def position_jitter_mm(positions):
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if len(positions) < 2:
        raise ValueError("position jitter needs at least two samples")
    per_axis = 1000.0 * np.std(positions, axis=0, ddof=1)
    return {
        "pos_jitter_mm": float(np.linalg.norm(per_axis)),
        "pos_jitter_x_mm": float(per_axis[0]),
        "pos_jitter_y_mm": float(per_axis[1]),
        "pos_jitter_z_mm": float(per_axis[2]),
    }


def chordal_mean_rotation(rotations):
    """The rotation closest to a set of rotations in the Frobenius sense.

    Averaging rotation vectors component-wise is wrong near the pi wrap; this
    projects the arithmetic mean matrix back onto SO(3), which is not.
    """
    rotations = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    u, _s, vt = np.linalg.svd(rotations.mean(axis=0))
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u @ vt))
    return u @ correction @ vt


def rotation_jitter_mdeg(rvecs):
    """Spread of the rotation estimate, as residuals about the chordal mean.

    Working in the residual keeps every angle small, so nothing wraps and the
    standard deviation means what it looks like it means.
    """
    rvecs = np.asarray(rvecs, dtype=np.float64).reshape(-1, 3)
    if len(rvecs) < 2:
        raise ValueError("rotation jitter needs at least two samples")
    rotations = np.asarray([cv2.Rodrigues(r.reshape(3, 1))[0] for r in rvecs])
    mean = chordal_mean_rotation(rotations)
    residuals = np.asarray(
        [cv2.Rodrigues(mean.T @ rotation)[0].ravel() for rotation in rotations]
    )
    per_axis = 1000.0 * np.degrees(np.std(residuals, axis=0, ddof=1))
    return {
        "rot_jitter_mdeg": float(np.linalg.norm(per_axis)),
        "rot_jitter_x_mdeg": float(per_axis[0]),
        "rot_jitter_y_mdeg": float(per_axis[1]),
        "rot_jitter_z_mdeg": float(per_axis[2]),
    }
