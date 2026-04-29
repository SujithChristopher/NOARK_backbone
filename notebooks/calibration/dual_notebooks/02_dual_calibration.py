"""Dual-camera fisheye stereo calibration.

Strategy
--------
cv2.fisheye.stereoCalibrate has an unconditional CV_Assert(abs_max < 1e10) inside
CalibrateExtrinsics that fires on many real-world planar datasets regardless of flags.
Instead we:
  1. Calibrate each camera individually on its own frames → K, D, per-view R/t
  2. For paired frames, run individual calibrate again to get per-view extrinsics
     with the fixed K/D
  3. Compute stereo R, T per paired view: R = R1 @ R0.T, T = t1 - R @ t0
  4. Aggregate with rotation mean + translation median (robust to outlier frames)

This is equivalent to what stereoCalibrate does internally before its bundle
adjustment step, and is sufficient for practical stereo rectification.
"""

import pickle
import toml
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parents[3]
CALIB_DATA = (
    PROJECT_ROOT
    / "data" / "calibration" / "dual_160"
    / "dual_cam_calibration_checker_sz_30mm"
)
corners_imx219_pth = CALIB_DATA / "chessb_corners_cam0_imx219.pkl"
corners_ov9281_pth = CALIB_DATA / "chessb_corners_cam1_ov9281.pkl"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def _load(pth):
    with open(pth, "rb") as f:
        return pickle.load(f)

d0 = _load(corners_imx219_pth)   # cam0 IMX219
d1 = _load(corners_ov9281_pth)   # cam1 OV9281
print(f"cam0: {len(d0['corners'])} frames   cam1: {len(d1['corners'])} frames")

# ---------------------------------------------------------------------------
# Image sizes  (cv2 wants W, H)
# img_size_cam0 is numpy (H=1232, W=1640, C=3)  →  WH = (1640, 1232)
# img_size_cam1 is stored as (W=1280, H=800)     →  WH = (1280, 800)
# ---------------------------------------------------------------------------
WH0 = (1640, 1232)
WH1 = (1280, 800)

# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------
PATTERN = (8, 12)
SQ_MM   = 30

def _board_pts(pattern, sq):
    X = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    X[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * sq
    return X

BOARD = _board_pts(PATTERN, SQ_MM)           # (96, 3) float32
BOARD64 = BOARD.reshape(-1, 1, 3).astype(np.float64)

# ---------------------------------------------------------------------------
# Calibration flags / criteria
# ---------------------------------------------------------------------------
CALIB_FLAGS = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    | cv2.fisheye.CALIB_FIX_SKEW
    | cv2.fisheye.CALIB_CHECK_COND
)
CALIB_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1e-9)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_ip(corners_list, indices):
    return [
        np.asarray(corners_list[i], dtype=np.float64).reshape(-1, 1, 2)
        for i in indices
    ]

def _obj_list(n):
    return [BOARD64.copy() for _ in range(n)]


def _calibrate(ip, wh):
    """Individual fisheye calibration. Returns (rms, K, D, rvecs, tvecs)."""
    op = _obj_list(len(ip))
    return cv2.fisheye.calibrate(
        op, ip, wh, None, None,
        flags=CALIB_FLAGS, criteria=CALIB_CRIT,
    )


def _perview_extrinsics(ip_list, K, D):
    """Per-view pose via fisheye undistort + solvePnP (truly fixed K, D)."""
    rvecs, tvecs = [], []
    board_f64 = BOARD.astype(np.float64)
    K_id  = np.eye(3, dtype=np.float64)
    D_zero = np.zeros(4, dtype=np.float64)
    for pts in ip_list:
        # undistortPoints gives normalized camera coords (K divided out, distortion removed)
        u = cv2.fisheye.undistortPoints(pts, K, D).reshape(-1, 1, 2).astype(np.float64)
        ok, rv, tv = cv2.solvePnP(board_f64, u, K_id, D_zero,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        rvecs.append(rv if ok else None)
        tvecs.append(tv if ok else None)
    return rvecs, tvecs


def _paired_indices(n=150, seed=0):
    """Return (idx0_list, idx1_list) for n common frame indices."""
    rng = np.random.default_rng(seed)
    idx0 = {int(f): i for i, f in enumerate(d0["frame_idx"])}
    idx1 = {int(f): i for i, f in enumerate(d1["frame_idx"])}
    common = sorted(set(idx0) & set(idx1))
    print(f"Common paired frames: {len(common)}")
    chosen = rng.choice(len(common), size=min(n, len(common)), replace=False)
    frames = [common[i] for i in chosen]
    return [idx0[f] for f in frames], [idx1[f] for f in frames]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(n_individual=300, n_paired=150, seed=0):
    rng = np.random.default_rng(seed)

    # --- Step 1: individual calibration on unpaired frames for best K, D ---
    idx0_all = rng.choice(len(d0["corners"]), size=min(n_individual, len(d0["corners"])), replace=False)
    idx1_all = rng.choice(len(d1["corners"]), size=min(n_individual, len(d1["corners"])), replace=False)

    ip0_all = _to_ip(d0["corners"], idx0_all)
    ip1_all = _to_ip(d1["corners"], idx1_all)

    rms0, K0, D0, _, _ = _calibrate(ip0_all, WH0)
    rms1, K1, D1, _, _ = _calibrate(ip1_all, WH1)
    print(f"Individual  cam0 RMS={rms0:.4f}  cam1 RMS={rms1:.4f}")
    print(f"K0 fx={K0[0,0]:.1f} fy={K0[1,1]:.1f}  D0={D0.ravel().round(4)}")
    print(f"K1 fx={K1[0,0]:.1f} fy={K1[1,1]:.1f}  D1={D1.ravel().round(4)}")

    # --- Step 2: per-view extrinsics for paired frames ---
    pidx0, pidx1 = _paired_indices(n=n_paired, seed=seed)
    ip0_paired = _to_ip(d0["corners"], pidx0)
    ip1_paired = _to_ip(d1["corners"], pidx1)

    rvecs0, tvecs0 = _perview_extrinsics(ip0_paired, K0, D0)
    rvecs1, tvecs1 = _perview_extrinsics(ip1_paired, K1, D1)

    # --- Step 3: compute stereo R, T per view ---
    R_list, T_list = [], []
    for rv0, tv0, rv1, tv1 in zip(rvecs0, tvecs0, rvecs1, tvecs1):
        if rv0 is None or rv1 is None:
            continue
        R0i, _ = cv2.Rodrigues(rv0)
        R1i, _ = cv2.Rodrigues(rv1)
        Ri = R1i @ R0i.T
        Ti = tv1 - Ri @ tv0
        R_list.append(Ri)
        T_list.append(Ti.ravel())

    T_arr = np.array(T_list)   # (N, 3) mm

    # MAD-based outlier rejection on T
    T_med = np.median(T_arr, axis=0)
    mad   = np.median(np.abs(T_arr - T_med), axis=0)
    inlier_mask = np.all(np.abs(T_arr - T_med) < 3 * (mad + 1e-6), axis=1)
    T_inliers = T_arr[inlier_mask]
    R_inliers = [R for R, m in zip(R_list, inlier_mask) if m]
    print(f"Paired views: {len(T_arr)} total, {inlier_mask.sum()} inliers after MAD filter")

    # --- Step 4: aggregate inliers ---
    R_stereo = Rotation.from_matrix(np.array(R_inliers)).mean().as_matrix()
    T_stereo = np.median(T_inliers, axis=0)
    T_std    = T_inliers.std(axis=0)

    print(f"\nStereo R (cam0->cam1):\n{R_stereo.round(6)}")
    print(f"Stereo T (mm): {T_stereo.round(2)}  std {T_std.round(2)}")
    euler = Rotation.from_matrix(R_stereo).as_euler("xyz", degrees=True)
    print(f"Rotation euler xyz (deg): {euler.round(3)}")

    return {
        "K0": K0, "D0": D0, "rms0": rms0,
        "K1": K1, "D1": D1, "rms1": rms1,
        "R":  R_stereo,
        "T":  T_stereo,
    }


def save_toml(result, path: Path):
    data = {
        "cam0": {
            "camera_matrix": result["K0"].tolist(),
            "dist_coeffs":   [result["D0"].ravel().tolist()],  # [[k1, k2, k3, k4]]
            "resolution":    list(WH0),
            "rms":           float(result["rms0"]),
        },
        "cam1": {
            "camera_matrix": result["K1"].tolist(),
            "dist_coeffs":   [result["D1"].ravel().tolist()],
            "resolution":    list(WH1),
            "rms":           float(result["rms1"]),
        },
        "stereo": {
            "R": result["R"].tolist(),           # 3x3 rotation cam0->cam1
            "T": result["T"].tolist(),           # [Tx, Ty, Tz] in mm
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        toml.dump(data, f)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    result = run(n_individual=300, n_paired=150, seed=42)
    out_path = Path(__file__).parent / "stereo_calibration.toml"
    save_toml(result, out_path)
    print("\nDone.")
