# %% [markdown]
# # First frame, camera coordinates: tag 1, ring 30 and tags 11/12 vs CAD
#
# No rigid body. Tags 1-5, 11 and 12 are posed independently in one frame straight from the
# cached corner detections (IPPE on the undistorted corners), all in camera
# coordinates. The hardcoded CAD offsets and orientations, which live in tag 1's
# frame, are moved into the camera with tag 1's measured pose:
#
#     t_pred = t_1 + R_1 @ cad_offset        R_pred = R_1 @ R_cad
#
# so position and orientation are compared with nothing fitted or aligned.
#
# ```powershell
# uv run python jitter_model/first_frame_ring_compare.py
# ```

# %% Imports
from pathlib import Path
import pickle

import cv2
import numpy as np
import polars as pl
from scipy.spatial.transform import Rotation
import toml

# %% Settings
try:
    NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:  # running cell-by-cell in an interactive kernel
    NOTEBOOK_DIR = Path.cwd()
    if NOTEBOOK_DIR.name != "jitter_model":
        NOTEBOOK_DIR = NOTEBOOK_DIR / "jitter_model"
PROJECT_ROOT = NOTEBOOK_DIR.parent
RECORDING_DIR = PROJECT_ROOT / "data" / "dome" / "sep18_26" / "dome_rb_def"
DETECTION_CACHE = RECORDING_DIR / "jitter_detections.pkl"
CALIBRATION_TOML = (
    PROJECT_ROOT
    / "data"
    / "calibration"
    / "dual_160"
    / "calib_radxa_sep18_26"
    / "stereo_calibration.toml"
)

CAMERA = "cam0"
FRAME = 0
TAG_SIZE_M = 0.05
REFERENCE_ID = 1
EULER_ORDER = "xyz"
MM = 1000.0

# CAD, in tag 1's frame: tag centre offset (x, y, z) in mm, and orientation as
# xyz Euler angles in degrees.
# Ring 30 (2-5) sits 56.61 mm out and 15.17 mm down; each tag is tag 1 tilted 30
# degrees about the axis at right angles to its azimuth, with no in-plane roll.
# Tag 12 is on ring 60 at 180 degrees (98.058 mm out, 56.61 down, tilt 60). Tag
# 11 is the 225 degree diagonal: 64.146 mm each in x and y, tilt 67.792.
# CAD only fixes each facet normal, not the in-plane roll. The ring 60 tags are
# rolled in-plane on the dome, so their roll (90 and 135 degrees about the tag's
# own z) is taken from the solved rigid body, not from CAD. `normal_err`
# compares the facet normal alone and does not depend on it.
CAD_OFFSET_MM = {
    2: (56.61, 0.0, -15.17),
    5: (0.0, 56.61, -15.17),
    3: (-56.61, 0.0, -15.17),
    4: (0.0, -56.61, -15.17),
    12: (-98.058, 0.0, -56.61),
    11: (-64.146, -64.146, -56.61),
}
CAD_EULER_DEG = {
    2: (0.0, 30.0, 0.0),
    5: (-30.0, 0.0, 0.0),
    3: (0.0, -30.0, 0.0),
    4: (30.0, 0.0, 0.0),
    12: (-60.0, 0.0, 90.0),
    11: (-67.792, 0.0, 135.0),
}
RING_IDS = (2, 3, 4, 5, 12, 11)

TAG_CORNERS = np.asarray(
    [
        [-TAG_SIZE_M / 2, +TAG_SIZE_M / 2, 0.0],
        [+TAG_SIZE_M / 2, +TAG_SIZE_M / 2, 0.0],
        [+TAG_SIZE_M / 2, -TAG_SIZE_M / 2, 0.0],
        [-TAG_SIZE_M / 2, -TAG_SIZE_M / 2, 0.0],
    ],
    dtype=np.float64,
)


# %% Measure
def estimate_marker_pose(corners, K, D):
    """Lowest-reprojection-error positive-depth IPPE pose of one square tag."""
    undistorted = cv2.fisheye.undistortPoints(
        np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2), K, D, P=K
    ).reshape(4, 2)
    ok, rvecs, tvecs, _err = cv2.solvePnPGeneric(
        TAG_CORNERS, undistorted, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    best = None
    for rvec, tvec in zip(rvecs, tvecs):
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        if tvec[2] <= 0:
            continue
        projected, _ = cv2.projectPoints(TAG_CORNERS, rvec, tvec, K, None)
        rmse = float(
            np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - undistorted) ** 2, axis=1)))
        )
        if best is None or rmse < best[2]:
            best = (cv2.Rodrigues(rvec)[0], tvec, rmse)
    return best


calibration = toml.load(CALIBRATION_TOML)[CAMERA]
K = np.asarray(calibration["camera_matrix"], dtype=np.float64)
D = np.asarray(calibration["dist_coeffs"], dtype=np.float64).reshape(-1, 1)

with DETECTION_CACHE.open("rb") as stream:
    detections = pickle.load(stream)["cameras"][CAMERA]["detections"][FRAME]  # noqa: S301

poses = {
    marker_id: estimate_marker_pose(detections[marker_id], K, D)
    for marker_id in (REFERENCE_ID, *RING_IDS)
}
R_ref, t_ref, _ = poses[REFERENCE_ID]

# %% Tables
pl.Config.set_tbl_formatting("ASCII_FULL_CONDENSED")
pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(200)
pl.Config.set_float_precision(3)

reference_euler = Rotation.from_matrix(R_ref).as_euler(EULER_ORDER, degrees=True)
print(f"{CAMERA} frame {FRAME}, tags {sorted(poses)}, camera coordinates")
print(
    pl.DataFrame(
        {
            "tag": [marker_id for marker_id in poses],
            "x_mm": [MM * poses[m][1][0] for m in poses],
            "y_mm": [MM * poses[m][1][1] for m in poses],
            "z_mm": [MM * poses[m][1][2] for m in poses],
            "reproj_px": [poses[m][2] for m in poses],
        }
    )
)
print(f"tag 1 orientation in camera, xyz Euler (deg): {np.round(reference_euler, 3)}")

position_rows = []
orientation_rows = []
for marker_id in RING_IDS:
    R_meas, t_meas, _ = poses[marker_id]

    cad_offset = np.asarray(CAD_OFFSET_MM[marker_id]) / MM
    cad_vector_cam = R_ref @ cad_offset
    measured_vector_cam = t_meas - t_ref
    position_error = measured_vector_cam - cad_vector_cam
    position_rows.append(
        {
            "tag": marker_id,
            "meas_x": MM * measured_vector_cam[0],
            "meas_y": MM * measured_vector_cam[1],
            "meas_z": MM * measured_vector_cam[2],
            "cad_x": MM * cad_vector_cam[0],
            "cad_y": MM * cad_vector_cam[1],
            "cad_z": MM * cad_vector_cam[2],
            "err_x": MM * position_error[0],
            "err_y": MM * position_error[1],
            "err_z": MM * position_error[2],
            "err_norm": MM * float(np.linalg.norm(position_error)),
            "dist_err": MM
            * float(np.linalg.norm(measured_vector_cam) - np.linalg.norm(cad_vector_cam)),
        }
    )

    R_cad = Rotation.from_euler(EULER_ORDER, CAD_EULER_DEG[marker_id], degrees=True)
    R_pred = R_ref @ R_cad.as_matrix()
    measured_rel = Rotation.from_matrix(R_ref.T @ R_meas)
    error = Rotation.from_matrix(R_pred.T @ R_meas)
    meas_e = measured_rel.as_euler(EULER_ORDER, degrees=True)
    err_e = error.as_euler(EULER_ORDER, degrees=True)
    orientation_rows.append(
        {
            "tag": marker_id,
            "meas_rx": meas_e[0],
            "meas_ry": meas_e[1],
            "meas_rz": meas_e[2],
            "cad_rx": CAD_EULER_DEG[marker_id][0],
            "cad_ry": CAD_EULER_DEG[marker_id][1],
            "cad_rz": CAD_EULER_DEG[marker_id][2],
            "err_rx": err_e[0],
            "err_ry": err_e[1],
            "err_rz": err_e[2],
            "err_angle": float(np.degrees(error.magnitude())),
            "normal_err": float(
                np.degrees(
                    np.arccos(np.clip(R_meas[:, 2] @ R_pred[:, 2], -1.0, 1.0))
                )
            ),
        }
    )

print("\nPosition, tag 1 -> tag N in camera coordinates (mm): measured vs CAD")
print(pl.DataFrame(position_rows))
print("\nOrientation, tag N in tag 1's frame, xyz Euler (deg): measured vs CAD")
print(pl.DataFrame(orientation_rows))

# %%
