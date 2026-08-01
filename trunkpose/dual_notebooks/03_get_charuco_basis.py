# %% Imports

import os
import sys
from pathlib import Path

import cv2
import msgpack as mp
import msgpack_numpy as mpn
import numpy as np
import toml
from cv2 import aruco
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm

try:
    NB_DIR = Path(__file__).resolve().parent
except NameError:  # running cell-by-cell in an interactive/Jupyter kernel
    NB_DIR = Path.cwd()
    if NB_DIR.name != "dual_notebooks":
        NB_DIR = NB_DIR / "trunkpose" / "dual_notebooks"
sys.path.insert(0, str(NB_DIR))
from ar_support import calculate_rotmat  # noqa: E402
from pd_support import get_marker_name, read_rigid_body_csv  # noqa: E402

# %% Board dimensions

# (squaresX, squaresY) -- confirmed by sweeping (3,4) vs (4,3): (4,3) gives sub-pixel
# reprojection error (~1px) on real frames, (3,4) gives ~27-490px (wrong board model,
# solvePnP still "succeeds" but silently fits garbage). Don't swap this back without
# re-checking reprojection error against real frames.
chessboard_shape = (4, 3)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_size = 0.027
checker_size = 0.037
MIN_MARKERS = 2  # need >=2 markers (8 pts) for a stable solvePnP
SUBPIX_WIN = (5, 5)
SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)

board = aruco.CharucoBoard(chessboard_shape, checker_size, aruco_size, aruco_dict)
marker_detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())

# CharucoDetector.detectBoard() silently fails to interpolate chessboard corners on
# this board under OpenCV 5's charuco pipeline (board.matchImagePoints() also rejects
# raw marker corners since CharucoBoard overrides it for interpolated corners only).
# Instead we solvePnP directly against each detected marker's own 4 corners, using the
# board's per-marker object points (board.getObjPoints() / board.getIds()) -- this only
# needs plain ArUco marker detection, which works reliably on every frame.
BOARD_IDS = board.getIds().ravel()
ID_TO_OBJPTS = {
    int(i): np.asarray(o, dtype=np.float64) for i, o in zip(BOARD_IDS, board.getObjPoints())
}

# %% Paths
# NOTE: PROJECT_ROOT is 2 levels up from this file (trunkpose/dual_notebooks/../..).
# Other scripts in this folder (02_dual_calibration.py etc.) use parents[3], which
# now resolves outside the repo after the trunkpose/ nesting change -- don't copy that.
PROJECT_ROOT = NB_DIR.parents[1]

RECORDING_NAME = "july_27_calib"
RECORDING_DIR = PROJECT_ROOT / "data" / "july27" / RECORDING_NAME
STEREO_TOML = (
    PROJECT_ROOT
    / "data" / "calibration" / "dual_160"
    / "calib_cz30_dual_v2" / "stereo_calibration.toml"
)
OUT_TOML = RECORDING_DIR / "charuco_basis.toml"

# markers glued on top of the charuco board (optitrack rigid body "tframe")
ORIGIN_MARKER = 4
XDIR_MARKER = 1
ZDIR_MARKER = 3

# %% Load stereo calibration (fisheye K, D per camera)
calib = toml.load(STEREO_TOML)
K0 = np.array(calib["cam0"]["camera_matrix"])
D0 = np.array(calib["cam0"]["dist_coeffs"][0]).reshape(-1, 1)
K1 = np.array(calib["cam1"]["camera_matrix"])
D1 = np.array(calib["cam1"]["dist_coeffs"][0]).reshape(-1, 1)
R_st = np.array(calib["stereo"]["R"])
T_st = np.array(calib["stereo"]["T"]).reshape(3, 1) / 1000.0  # mm -> m


# %% Frame helpers
def iter_frames(path):
    with open(path, "rb") as f:
        yield from mp.Unpacker(f, object_hook=mpn.decode)


# %% Charuco board pose per frame (fisheye-aware)
def detect_board_pose(frame, K, D):
    """Detect the board's markers in one frame -> (R, t) in that camera's frame, or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
    corners, ids, _rejected = marker_detector.detectMarkers(gray)
    if ids is None:
        return None

    objp, imgp = [], []
    for c, i in zip(corners, ids.ravel()):
        pts3d = ID_TO_OBJPTS.get(int(i))
        if pts3d is not None:
            cc = c.reshape(-1, 1, 2).astype(np.float32)
            cv2.cornerSubPix(gray, cc, SUBPIX_WIN, (-1, -1), SUBPIX_CRIT)
            objp.append(pts3d)
            imgp.append(cc.reshape(4, 2))
    if len(objp) < MIN_MARKERS:
        return None
    objp = np.concatenate(objp, axis=0)
    imgp = np.concatenate(imgp, axis=0)

    und = cv2.fisheye.undistortPoints(
        imgp.reshape(-1, 1, 2).astype(np.float64), K, D, P=K
    )
    ok, rvec, tvec = cv2.solvePnP(objp, und, K, np.zeros(4))
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel()


def camera_board_basis(frame_path, K, D, label):
    poses = [
        detect_board_pose(frame, K, D)
        for frame in tqdm(iter_frames(frame_path), desc=label)
    ]
    poses = [p for p in poses if p is not None]
    if not poses:
        raise RuntimeError(f"{label}: charuco board never detected")

    Rs = np.array([p[0] for p in poses])
    ts = np.array([p[1] for p in poses])
    R_mean = Rotation.from_matrix(Rs).mean().as_matrix()
    t_med = np.median(ts, axis=0)
    print(f"  {label}: {len(poses)} frames used, t={t_med.round(4)}")
    return R_mean, t_med, len(poses)


# %% Mocap marker basis (origin=m4, x=m2, z=m3)
def mocap_board_basis(csv_path):
    mocap_df, _st_time = read_rigid_body_csv(str(csv_path))
    cols_o = get_marker_name(ORIGIN_MARKER)
    cols_x = get_marker_name(XDIR_MARKER)
    cols_z = get_marker_name(ZDIR_MARKER)

    org = mocap_df[[cols_o["x"], cols_o["y"], cols_o["z"]]].to_numpy(float)
    xdir = mocap_df[[cols_x["x"], cols_x["y"], cols_x["z"]]].to_numpy(float)
    zdir = mocap_df[[cols_z["x"], cols_z["y"], cols_z["z"]]].to_numpy(float)

    valid = np.isfinite(org).all(1) & np.isfinite(xdir).all(1) & np.isfinite(zdir).all(1)
    org, xdir, zdir = org[valid], xdir[valid], zdir[valid]
    if len(org) == 0:
        raise RuntimeError("mocap: no frames with all 3 basis markers visible")

    Rs = np.array(
        [
            calculate_rotmat(x.reshape(3, 1), z.reshape(3, 1), o.reshape(3, 1))
            for x, z, o in zip(xdir, zdir, org)
        ]
    )
    R_mean = Rotation.from_matrix(Rs).mean().as_matrix()
    t_med = np.median(org, axis=0)
    print(f"  mocap: {len(org)} frames used, origin={t_med.round(4)}")
    return R_mean, t_med, len(org)


# %% Save
def save_basis_toml(path, cam0, cam1, mocap):
    R0, t0, n0 = cam0
    R1, t1, n1 = cam1
    Rm, tm, nm = mocap

    data = {
        "meta": {
            "recording": RECORDING_NAME,
            "board_dict": "DICT_4X4_50",
            "chessboard_shape": list(chessboard_shape),
            "aruco_size": aruco_size,
            "checker_size": checker_size,
        },
        "cam0": {"origin": t0.tolist(), "rotation": R0.tolist(), "n_frames": n0},
        "cam1": {"origin": t1.tolist(), "rotation": R1.tolist(), "n_frames": n1},
        "mocap": {
            "origin": tm.tolist(),
            "rotation": Rm.tolist(),
            "n_frames": nm,
            "markers": {
                "origin": f"m{ORIGIN_MARKER}",
                "x_dir": f"m{XDIR_MARKER}",
                "z_dir": f"m{ZDIR_MARKER}",
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        toml.dump(data, f)
    print(f"Saved -> {path}")


# %% Main
if __name__ == "__main__":
    print(f"Recording: {RECORDING_DIR}")

    print("\n[cam0] detecting charuco board...")
    R0, t0, n0 = camera_board_basis(RECORDING_DIR / "cam0_frame.msgpack", K0, D0, "cam0")

    print("\n[cam1] detecting charuco board...")
    R1, t1, n1 = camera_board_basis(RECORDING_DIR / "cam1_frame.msgpack", K1, D1, "cam1")

    print("\n[mocap] computing marker basis...")
    Rm, tm, nm = mocap_board_basis(RECORDING_DIR / f"{RECORDING_NAME}.csv")

    # sanity check: cam1 basis predicted from cam0 basis via the stereo extrinsics
    # should roughly match the independently-detected cam1 basis
    t1_pred = (R_st @ t0.reshape(3, 1) + T_st).ravel()
    R1_pred = R_st @ R0
    t_err = np.linalg.norm(t1_pred - t1) * 1000  # mm
    r_err = np.degrees(
        np.arccos(np.clip((np.trace(R1_pred.T @ R1) - 1) / 2, -1, 1))
    )
    print(f"\n[check] cam0->cam1 via stereo extrinsics vs direct detection:"
          f"  t_err={t_err:.2f} mm  R_err={r_err:.3f} deg")

    save_basis_toml(OUT_TOML, (R0, t0, n0), (R1, t1, n1), (Rm, tm, nm))
    print("\nDone.")
