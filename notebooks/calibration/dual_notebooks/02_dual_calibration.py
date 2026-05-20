"""Dual-camera fisheye stereo calibration — dual OV9281.

Strategy
--------
cv2.fisheye.stereoCalibrate has an unconditional CV_Assert(abs_max < 1e10) inside
CalibrateExtrinsics that fires on many real-world planar datasets regardless of flags.
Instead we:
  1. Calibrate each camera individually on its own frames → K, D, per-view R/t
  2. Filter frames by per-frame reprojection error (keep best %)
  3. Re-calibrate on filtered frames for tighter K, D
  4. For paired frames, run individual calibrate again to get per-view extrinsics
     with the fixed K/D
  5. Compute stereo R, T per paired view: R = R1 @ R0.T, T = t1 - R @ t0
  6. Aggregate with rotation mean + translation median (robust to outlier frames)
  7. Validate with stereo epipolar error

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
    / "dual_ov9281_calibration_checker_sz_30mm"
)
corners_cam0_pth = CALIB_DATA / "chessb_corners_cam0_frame.pkl"
corners_cam1_pth = CALIB_DATA / "chessb_corners_cam1_frame.pkl"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def _load(pth):
    with open(pth, "rb") as f:
        return pickle.load(f)

d0 = _load(corners_cam0_pth)   # cam0 OV9281
d1 = _load(corners_cam1_pth)   # cam1 OV9281
print(f"cam0: {len(d0['corners'])} frames   cam1: {len(d1['corners'])} frames")

# ---------------------------------------------------------------------------
# Image sizes  (cv2 wants W, H) — both OV9281 at 1280×800
# ---------------------------------------------------------------------------
WH0 = (1280, 800)
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

BOARD   = _board_pts(PATTERN, SQ_MM)           # (96, 3) float32
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


def _per_frame_rms(ip_list, K, D, rvecs, tvecs):
    """Per-frame reprojection RMS for a calibrated camera."""
    errors = []
    for img_pts, rv, tv in zip(ip_list, rvecs, tvecs):
        proj, _ = cv2.fisheye.projectPoints(
            BOARD64, rv, tv, K, D
        )
        err = float(np.sqrt(np.mean((img_pts - proj) ** 2)))
        errors.append(err)
    return np.array(errors)


def _calibrate_filtered(ip_all, wh, rms_percentile=80):
    """
    Two-pass calibration: first pass selects frames below rms_percentile,
    second pass re-calibrates on the filtered set.
    Returns (rms, K, D, rvecs, tvecs, kept_indices).
    """
    rms, K, D, rvecs, tvecs = _calibrate(ip_all, wh)
    errs = _per_frame_rms(ip_all, K, D, rvecs, tvecs)
    thresh = np.percentile(errs, rms_percentile)
    keep = np.where(errs <= thresh)[0]
    ip_filt = [ip_all[i] for i in keep]
    rms2, K2, D2, rvecs2, tvecs2 = _calibrate(ip_filt, wh)
    errs2 = _per_frame_rms(ip_filt, K2, D2, rvecs2, tvecs2)
    print(f"    pass1 rms={rms:.4f}  pass2 rms={rms2:.4f}"
          f"  frames {len(ip_all)}->{len(ip_filt)}"
          f"  per-frame p50={np.median(errs2):.4f} p95={np.percentile(errs2,95):.4f}")
    return rms2, K2, D2, rvecs2, tvecs2, keep


def _perview_extrinsics(ip_list, K, D):
    """Per-view pose via fisheye undistort + solvePnP (truly fixed K, D)."""
    rvecs, tvecs = [], []
    board_f64 = BOARD.astype(np.float64)
    K_id   = np.eye(3, dtype=np.float64)
    D_zero = np.zeros(4, dtype=np.float64)
    for pts in ip_list:
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


def _stereo_epipolar_error(ip0_list, ip1_list, K0, D0, K1, D1, R, T):
    """
    After stereo calibration, corresponding undistorted points should satisfy
    the epipolar constraint. We compute the mean squared y-distance after
    virtual rectification (approximate: use F matrix from R, T, K).

    Returns mean epipolar error in pixels.
    """
    # Essential matrix E = t× R, Fundamental F = K1^{-T} E K0^{-1}
    tx, ty, tz = T.ravel()
    T_cross = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]])
    E = T_cross @ R
    F = np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)

    errors = []
    for pts0, pts1 in zip(ip0_list, ip1_list):
        p0 = cv2.fisheye.undistortPoints(pts0, K0, D0, P=K0).reshape(-1, 2)
        p1 = cv2.fisheye.undistortPoints(pts1, K1, D1, P=K1).reshape(-1, 2)
        for (x0, y0), (x1, y1) in zip(p0, p1):
            lx, ly, lw = F @ np.array([x0, y0, 1.0])
            dist = abs(lx * x1 + ly * y1 + lw) / np.sqrt(lx**2 + ly**2 + 1e-12)
            errors.append(dist)
    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(n_individual=300, n_paired=150, seed=0):
    rng = np.random.default_rng(seed)

    # --- Step 1: individual calibration with per-frame filtering ---
    print("\n[cam0] individual calibration:")
    idx0_all = rng.choice(len(d0["corners"]), size=min(n_individual, len(d0["corners"])), replace=False)
    ip0_all  = _to_ip(d0["corners"], idx0_all)
    rms0, K0, D0, _, _, _ = _calibrate_filtered(ip0_all, WH0)

    print("\n[cam1] individual calibration:")
    idx1_all = rng.choice(len(d1["corners"]), size=min(n_individual, len(d1["corners"])), replace=False)
    ip1_all  = _to_ip(d1["corners"], idx1_all)
    rms1, K1, D1, _, _, _ = _calibrate_filtered(ip1_all, WH1)

    print(f"\nK0 fx={K0[0,0]:.2f} fy={K0[1,1]:.2f} cx={K0[0,2]:.2f} cy={K0[1,2]:.2f}  D0={D0.ravel().round(5)}")
    print(f"K1 fx={K1[0,0]:.2f} fy={K1[1,1]:.2f} cx={K1[0,2]:.2f} cy={K1[1,2]:.2f}  D1={D1.ravel().round(5)}")

    # --- Step 2: per-view extrinsics for paired frames ---
    print("\n[stereo] computing per-view extrinsics...")
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

    T_arr = np.array(T_list)

    # MAD-based outlier rejection on T
    T_med = np.median(T_arr, axis=0)
    mad   = np.median(np.abs(T_arr - T_med), axis=0)
    inlier_mask = np.all(np.abs(T_arr - T_med) < 3 * (mad + 1e-6), axis=1)
    T_inliers = T_arr[inlier_mask]
    R_inliers = [R for R, m in zip(R_list, inlier_mask) if m]
    print(f"Paired views: {len(T_arr)} total, {inlier_mask.sum()} inliers after MAD filter")

    # --- Step 4: aggregate ---
    R_stereo = Rotation.from_matrix(np.array(R_inliers)).mean().as_matrix()
    T_stereo = np.median(T_inliers, axis=0)
    T_std    = T_inliers.std(axis=0)

    euler = Rotation.from_matrix(R_stereo).as_euler("xyz", degrees=True)
    print(f"\nStereo R euler xyz (deg): {euler.round(3)}")
    print(f"Stereo T (mm): {T_stereo.round(2)}  std {T_std.round(2)}")
    baseline_mm = np.linalg.norm(T_stereo)
    print(f"Baseline: {baseline_mm:.2f} mm")

    # --- Step 5: stereo validation ---
    # use only inlier paired frames for validation
    inlier_idx = np.where(inlier_mask)[0]
    ip0_val = [ip0_paired[i] for i in inlier_idx]
    ip1_val = [ip1_paired[i] for i in inlier_idx]
    epi_err = _stereo_epipolar_error(ip0_val, ip1_val, K0, D0, K1, D1, R_stereo, T_stereo.reshape(3, 1))
    print(f"Stereo epipolar error (mean px): {epi_err:.4f}")

    return {
        "K0": K0, "D0": D0, "rms0": rms0,
        "K1": K1, "D1": D1, "rms1": rms1,
        "R":  R_stereo,
        "T":  T_stereo,
        "epipolar_err": epi_err,
    }


def save_toml(result, path: Path):
    data = {
        "cam0": {
            "camera_matrix": result["K0"].tolist(),
            "dist_coeffs":   [result["D0"].ravel().tolist()],
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
            "R":             result["R"].tolist(),
            "T":             result["T"].tolist(),
            "epipolar_err":  result["epipolar_err"],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        toml.dump(data, f)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    result = run(n_individual=300, n_paired=150, seed=42)
    out_path = CALIB_DATA / "stereo_calibration.toml"
    save_toml(result, out_path)
    print("\nDone.")
