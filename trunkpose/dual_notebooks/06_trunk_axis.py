# %% 06_trunk_axis.py
# Trunk axis + 3 trunk angles from dual-camera data, rendered as a 2x2 video.
#
# Per frame:
#   * MediaPipe (both cams) -> triangulate upper-limb joints + shoulders (board frame)
#   * dense fisheye stereo (SGBM) -> full-scene point cloud (colored by intensity)
#   * YOLO-seg torso -> torso-masked subset of the cloud -> RANSAC plane -> anterior
#     normal (EMA-smoothed)
#   * trunk frame R_t = [lateral | up | anterior]:
#         lateral = L_shoulder - R_shoulder,  anterior = plane normal,  up = a x lateral
#   * 3 trunk angles vs a NEUTRAL frame (mean trunk frame over first N frames)
#
# 2x2 layout:
#   TL cam0 + axis overlay | TR 3D perspective (cloud + skeleton + axis + plane + mocap)
#   BL 3D top-down (cloud + plane + axis) | BR realtime 3-angle plots
#
# Both the geometry and render passes are split across a process pool (env TRUNK_WORKERS,
# default 4; set 1 for a sequential/debuggable run). EMA + neutral stay sequential.

import os
import pickle
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import mediapipe as mp
import msgpack
import msgpack_numpy as mpn
import numpy as np
import pandas as pd
import toml
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pd_support import read_rigid_body_csv  # noqa: E402

# %% Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

RECORDING_NAME = "dual_160_trunk_ragav"
RECORDING_DIR = PROJECT_ROOT / "data" / "trunk_july1_2026" / RECORDING_NAME
CAM0_FRAMES = RECORDING_DIR / "cam0_frame.msgpack"
CAM1_FRAMES = RECORDING_DIR / "cam1_frame.msgpack"
STEREO_TOML = (
    PROJECT_ROOT / "data" / "calibration" / "dual_160"
    / "calib_cz30_dual_v2" / "stereo_calibration.toml"
)
CHARUCO_TOML = (
    PROJECT_ROOT / "data" / "trunk_july1_2026"
    / "dual_160_tframe_july1" / "charuco_basis.toml"
)
SEG_WEIGHTS = PROJECT_ROOT / "trunkpose" / "runs" / "trunk_seg-2" / "weights" / "best.pt"
POSE_MODEL = SCRIPT_DIR / "pose_landmarker_full.task"
OUT_VIDEO = RECORDING_DIR / "trunk_axis_charuco.mp4"

# %% Config
CAM_SIZE = (1280, 800)
VISIBILITY_THRESH = 0.5
DEPTH_MIN_M, DEPTH_MAX_M = 0.2, 3.0     # torso-fit depth gate
SCENE_DEPTH_MAX_M = 4.0                  # depth-heatmap far clip
MASK_ERODE_FRAC = 0.06                   # erode torso mask by ~this * sqrt(area) px
FRONT_PCTL = 5.0                         # frontmost-depth percentile for the front shell
SHELL_M = 0.10                           # keep points within this depth of the front cap
Z_MAX_M = 5.0
N_NEUTRAL = 15
SAVGOL_WINDOW = 11       # camera-angle smoothing window (frames, odd); mocap left raw
SAVGOL_POLY = 3
MAX_SYNC_LAG_S = 15.0    # search +/- this many seconds for the camera<->mocap lag
ANTERIOR_EMA = 0.3
AXIS_LEN_M = 0.15
PLANE_HALF_M = 0.075
# Drop the trunk frame + plane this far DOWN the torso (along -up) so they sit below
# the reflective-marker object mounted on the chest, which would otherwise corrupt the
# plane fit. The plane is refit on the lower torso band; the drawn frame moves with it.
TRUNK_DROP_M = float(os.environ.get("TRUNK_DROP_M", "0.12"))
# Downsampled torso front-shell cloud cached per frame (board frame, float32) so
# rigid-registration experiments (07_icp_trunk.py) can run without re-doing stereo.
CLOUD_VOXEL_M = 0.01
CLOUD_MAX_PTS = 2000
DEPTH_W, DEPTH_H = 512, 320              # stored depth-map resolution (colorized at render)
DEPTH_CMAP = cv2.COLORMAP_TURBO
AXIS_PAD_M = 0.3
QUAD_W, QUAD_H = 800, 600                # each of the 2x2 panels -> final 1600x1200

RANSAC_ITERS = 200
RANSAC_THRESH_M = 0.012
FIT_MAX_PTS = 8000
SGBM_MIN_DISP, SGBM_NUM_DISP, SGBM_BLOCK = 0, 160, 5
SEG_CONF = 0.25

N_WORKERS = int(os.environ.get("TRUNK_WORKERS", "4"))

UPPER_LIMB = {
    "L_shoulder": 11, "R_shoulder": 12,
    "L_elbow": 13, "R_elbow": 14,
    "L_wrist": 15, "R_wrist": 16,
}
BONES = [
    ("L_shoulder", "R_shoulder"),
    ("L_shoulder", "L_elbow"), ("L_elbow", "L_wrist"),
    ("R_shoulder", "R_elbow"), ("R_elbow", "R_wrist"),
]
MOCAP_MARKERS = {
    "L_shoulder": "mls", "R_shoulder": "mrs",
    "L_elbow": "mle", "R_elbow": "mre",
    "L_wrist": "mlw", "R_wrist": "mrw",
}
CAM_JOINT_COLORS = {
    "L_shoulder": "limegreen", "R_shoulder": "dodgerblue",
    "L_elbow": "green", "R_elbow": "royalblue",
    "L_wrist": "darkgreen", "R_wrist": "navy",
}
AXIS_COLORS_MPL = {"lateral": "red", "up": "lime", "anterior": "deepskyblue"}
AXIS_COLORS_BGR = {"lateral": (0, 0, 255), "up": (0, 255, 0), "anterior": (255, 128, 0)}
ANGLE_COLORS = {"flexion": "tomato", "lateral": "gold", "axial": "deepskyblue"}


# %% Calibration / basis
def load_stereo(path):
    d = toml.load(path)
    K0 = np.array(d["cam0"]["camera_matrix"])
    D0 = np.array(d["cam0"]["dist_coeffs"][0], dtype=np.float64).reshape(4, 1)
    K1 = np.array(d["cam1"]["camera_matrix"])
    D1 = np.array(d["cam1"]["dist_coeffs"][0], dtype=np.float64).reshape(4, 1)
    R = np.array(d["stereo"]["R"])
    T_mm = np.array(d["stereo"]["T"], dtype=np.float64).reshape(3, 1)
    return K0, D0, K1, D1, R, T_mm


def load_charuco_basis(path):
    d = toml.load(path)
    return (np.array(d["cam0"]["rotation"]), np.array(d["cam0"]["origin"]),
            np.array(d["mocap"]["rotation"]), np.array(d["mocap"]["origin"]))


def to_board(p, R0, t0):
    return R0.T @ (p - t0)


def to_board_bulk(pts, R0, t0):
    return (pts - t0) @ R0


def dir_to_board(v, R0):
    return R0.T @ v


def board_to_cam0(p, R0, t0):
    return R0 @ p + t0


def mocap_to_board(p, Rm, tm):
    return Rm.T @ (p - tm)


# %% Frame / timestamps
def iter_frame_pairs(path0, path1):
    with open(path0, "rb") as f0, open(path1, "rb") as f1:
        u0 = msgpack.Unpacker(f0, object_hook=mpn.decode)
        u1 = msgpack.Unpacker(f1, object_hook=mpn.decode)
        yield from zip(u0, u1)


def iter_frames(path0):
    with open(path0, "rb") as f0:
        yield from msgpack.Unpacker(f0, object_hook=mpn.decode)


def load_frame_times(path):
    with open(path, "rb") as f:
        recs = list(msgpack.Unpacker(f, object_hook=mpn.decode))
    return np.array([datetime.fromisoformat(r[1]) for r in recs], dtype="datetime64[us]")


def load_sync_flags(path):
    """GPIO sync bit per frame (column 0 of the timestamp msgpack). High while the
    mocap system is recording, so the first rising edge marks mocap t=0."""
    with open(path, "rb") as f:
        recs = list(msgpack.Unpacker(f, object_hook=mpn.decode))
    return np.array([int(r[0]) for r in recs])


def nearest_index(times, t):
    j = int(np.searchsorted(times, t))
    cands = [k for k in (j - 1, j) if 0 <= k < len(times)]
    return min(cands, key=lambda k: abs(times[k] - t))


# %% Stereo helpers
def build_rectification(K0, D0, K1, D1, R, T_mm, size):
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K0, D0, K1, D1, size, R, T_mm,
        flags=cv2.CALIB_ZERO_DISPARITY, balance=0.0, fov_scale=1.0)
    m0 = cv2.fisheye.initUndistortRectifyMap(K0, D0, R1, P1, size, cv2.CV_16SC2)
    m1 = cv2.fisheye.initUndistortRectifyMap(K1, D1, R2, P2, size, cv2.CV_16SC2)
    return R1, Q, m0, m1


def make_sgbm():
    return cv2.StereoSGBM_create(
        minDisparity=SGBM_MIN_DISP, numDisparities=SGBM_NUM_DISP, blockSize=SGBM_BLOCK,
        P1=8 * SGBM_BLOCK ** 2, P2=32 * SGBM_BLOCK ** 2, disp12MaxDiff=1,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def triangulate(p0_px, p1_px, K0, D0, K1, D1, R, T_m):
    n0 = cv2.fisheye.undistortPoints(
        np.array([[p0_px]], dtype=np.float64), K0, D0).reshape(-1, 2).T
    n1 = cv2.fisheye.undistortPoints(
        np.array([[p1_px]], dtype=np.float64), K1, D1).reshape(-1, 2).T
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    h = cv2.triangulatePoints(P0, np.hstack([R, T_m]), n0, n1)
    if abs(h[3, 0]) < 1e-9:
        return None
    p = (h[:3] / h[3]).ravel()
    return p if 0 < p[2] < Z_MAX_M else None


def stereo_step(gray0, gray1, mask0, sgbm, maps0, maps1, R1, Q):
    """Return (torso_pts_cam0, depth_u8) where depth_u8 is a small heatmap-ready
    depth image (0 = no data, 1..255 = near..far over [DEPTH_MIN_M, SCENE_DEPTH_MAX_M])."""
    rect0 = cv2.remap(gray0, *maps0, cv2.INTER_LINEAR)
    rect1 = cv2.remap(gray1, *maps1, cv2.INTER_LINEAR)
    disp = sgbm.compute(rect0, rect1).astype(np.float32) / 16.0
    pts = cv2.reprojectImageTo3D(disp, Q) / 1000.0  # meters, rectified-cam0 frame
    z = pts[..., 2]
    valid = (disp > SGBM_MIN_DISP) & np.isfinite(pts).all(axis=2)

    depth_ok = valid & (z > DEPTH_MIN_M) & (z < SCENE_DEPTH_MAX_M)
    norm = np.clip((z - DEPTH_MIN_M) / (SCENE_DEPTH_MAX_M - DEPTH_MIN_M), 0, 1)
    depth_full = np.zeros(z.shape, dtype=np.uint8)
    depth_full[depth_ok] = 1 + (253 * norm[depth_ok]).astype(np.uint8)
    depth_u8 = cv2.resize(depth_full, (DEPTH_W, DEPTH_H), interpolation=cv2.INTER_NEAREST)

    # front-shell torso: erode the mask to drop the curved flanks, then keep only the
    # frontmost depth layer (convex torso -> front cap ~ the tangent plane we want).
    mrect = cv2.remap(mask0, *maps0, cv2.INTER_NEAREST)
    area = int((mrect > 0).sum())
    if area > 0:
        ksz = max(3, int(round(np.sqrt(area) * MASK_ERODE_FRAC))) | 1
        mrect = cv2.erode(mrect, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)))
    torso_ok = valid & (mrect > 0) & (z > DEPTH_MIN_M) & (z < DEPTH_MAX_M)
    tp = pts[torso_ok]
    if len(tp):
        zc = np.percentile(tp[:, 2], FRONT_PCTL)
        tp = tp[tp[:, 2] <= zc + SHELL_M]
    return tp @ R1, depth_u8


def voxel_downsample(pts, voxel=CLOUD_VOXEL_M, max_pts=CLOUD_MAX_PTS, rng=None):
    """Mean point per occupied voxel; random-subsample to max_pts if still too many."""
    if len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(cnt), 3))
    np.add.at(sums, inv, pts)
    ds = sums / cnt[:, None]
    if len(ds) > max_pts:
        idx = (rng or np.random.default_rng(0)).choice(len(ds), max_pts, replace=False)
        ds = ds[idx]
    return ds


def torso_mask(seg_model, gray, size):
    W, H = size
    res = seg_model.predict(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
                            conf=SEG_CONF, verbose=False)[0]
    mask = np.zeros((H, W), dtype=np.uint8)
    if res.masks is not None:
        for m in res.masks.data.cpu().numpy():
            mask[cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5] = 255
    return mask


def fit_plane(pts, rng):
    if len(pts) < 50:
        return None
    if len(pts) > FIT_MAX_PTS:
        pts = pts[rng.choice(len(pts), FIT_MAX_PTS, replace=False)]
    best_inl, best_cnt = None, 0
    for _ in range(RANSAC_ITERS):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        nrm = np.cross(b - a, c - a)
        nn = np.linalg.norm(nrm)
        if nn < 1e-9:
            continue
        nrm /= nn
        inl = np.abs((pts - a) @ nrm) < RANSAC_THRESH_M
        if inl.sum() > best_cnt:
            best_cnt, best_inl = int(inl.sum()), inl
    inliers = pts[best_inl]
    centroid = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vt[2] / np.linalg.norm(vt[2])
    if normal @ centroid > 0:
        normal = -normal
    return centroid, normal


# %% MediaPipe (IMAGE mode -> stateless -> chunk-parallel-safe)
def make_pose_landmarker():
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=mp_vision.RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    return mp_vision.PoseLandmarker.create_from_options(opts)


def landmark_px(lms, idx, W, H):
    if lms is None:
        return None
    lm = lms[idx]
    if lm.visibility < VISIBILITY_THRESH:
        return None
    return (float(np.clip(lm.x * W, 0, W - 1)), float(np.clip(lm.y * H, 0, H - 1)))


# %% Trunk frame + angles
def trunk_frame(l_sh, r_sh, anterior):
    x = l_sh - r_sh
    nx = np.linalg.norm(x)
    if nx < 1e-6:
        return None
    x /= nx
    a = anterior - (anterior @ x) * x
    na = np.linalg.norm(a)
    if na < 1e-6:
        return None
    a /= na
    u = np.cross(a, x)
    u /= np.linalg.norm(u)
    a = np.cross(x, u)
    return np.column_stack([x, u, a])


def trunk_origin(l_sh, r_sh, R_t, drop=TRUNK_DROP_M):
    """Shoulder midpoint dropped `drop` metres down the trunk up-axis (R_t[:,1])."""
    return (l_sh + r_sh) / 2 - R_t[:, 1] * drop


def trunk_angles(R_t, R_neu):
    x0, u0, a0 = R_neu[:, 0], R_neu[:, 1], R_neu[:, 2]
    x, u = R_t[:, 0], R_t[:, 1]
    return (np.degrees(np.arctan2(u @ a0, u @ u0)),
            np.degrees(np.arctan2(u @ x0, u @ u0)),
            np.degrees(np.arctan2(x @ a0, x @ x0)))


def plane_grid(centroid, x_ax, a_ax, half=PLANE_HALF_M, n=10):
    s = np.linspace(-half, half, n)
    aa, bb = np.meshgrid(s, s)
    pts = centroid + aa[..., None] * x_ax + bb[..., None] * a_ax
    return pts[..., 0], pts[..., 1], pts[..., 2]


def smooth_series(arr, window=SAVGOL_WINDOW, poly=SAVGOL_POLY):
    """Savitzky-Golay smoothing; interpolates over NaN gaps first."""
    a = arr.astype(float).copy()
    valid = np.isfinite(a)
    if valid.sum() < poly + 2:
        return a
    idx = np.arange(len(a))
    a[~valid] = np.interp(idx[~valid], idx[valid], a[valid])
    w = min(window, len(a))
    if w % 2 == 0:
        w -= 1
    if w <= poly:
        return a
    return savgol_filter(a, w, poly)


# %% Mocap trunk rigid body (trunk:Marker1 origin, Marker4 xvec, Marker2 zvec)
def load_trunk_rb(path):
    """Return (m1, m4, m2) each (N_mocap, 3), the trunk rigid-body markers in mocap
    world coords, row-aligned with read_rigid_body_csv output."""
    raw = pd.read_csv(path, skiprows=2, header=None, dtype=str)
    types, names, poslab, xyz = raw.iloc[0], raw.iloc[1], raw.iloc[3], raw.iloc[4]
    data = raw.iloc[5:].reset_index(drop=True)

    def vec(marker):
        cols = {}
        for j in raw.columns:
            if (str(types[j]) == "Rigid Body Marker"
                    and str(names[j]).lower() == f"trunk:marker{marker}"
                    and str(poslab[j]) == "Position"):
                cols[str(xyz[j]).strip().upper()] = j
        return np.stack([pd.to_numeric(data[cols[a]], errors="coerce").to_numpy()
                         for a in ("X", "Y", "Z")], axis=1)

    return vec(1), vec(4), vec(2)


def mocap_trunk_frame(o, m4, m2):
    """Trunk frame from rigid-body markers, columns [lateral, up, anterior] to match
    the camera convention (lateral = toward Marker4, anterior = toward Marker2)."""
    if not (np.isfinite(o).all() and np.isfinite(m4).all() and np.isfinite(m2).all()):
        return None
    x = m4 - o
    nx = np.linalg.norm(x)
    if nx < 1e-6:
        return None
    x /= nx
    a = (m2 - o)
    a = a - (a @ x) * x
    na = np.linalg.norm(a)
    if na < 1e-6:
        return None
    a /= na
    u = np.cross(a, x)
    u /= np.linalg.norm(u)
    a = np.cross(x, u)
    return np.column_stack([x, u, a])


def series_speed(pos, smooth=15):
    """Smoothed speed |d/dt| of an (N,3) position series (NaN gaps interpolated).
    Clean sign/axis-invariant sync signal that spikes whenever the point moves."""
    pos = np.asarray(pos, float).copy()
    idx = np.arange(len(pos))
    for c in range(3):
        col = pos[:, c]
        v = np.isfinite(col)
        if 1 < v.sum() < len(col):
            col[~v] = np.interp(idx[~v], idx[v], col[v])
            pos[:, c] = col
    spd = np.linalg.norm(np.gradient(pos, axis=0), axis=1)
    if smooth > 1:
        spd = np.convolve(spd, np.ones(smooth) / smooth, mode="same")
    return spd


def angle_activity(f, l, a, smooth=15):
    """Sign/axis-invariant motion-energy envelope: total angular speed across the
    three angles, smoothed. Robust sync signal -- both systems spike when the subject
    moves fast, regardless of how each labels flexion/lateral/axial."""
    parts = []
    for x in (f, l, a):
        x = np.asarray(x, float).copy()
        v = np.isfinite(x)
        if v.sum() < 2:
            parts.append(np.zeros(len(x)))
            continue
        idx = np.arange(len(x))
        x[~v] = np.interp(idx[~v], idx[v], x[v])
        parts.append(np.abs(np.gradient(x)))
    act = parts[0] + parts[1] + parts[2]
    if smooth > 1:
        act = np.convolve(act, np.ones(smooth) / smooth, mode="same")
    return act


def best_lag(a, b, max_lag):
    """Integer lag L in [-max_lag, max_lag] maximizing Pearson corr of a[i] vs b[i+L],
    over finite overlap. L>0 means b lags a (b's features occur L frames later)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = len(a)
    max_lag = min(max_lag, n - 1)
    best_c, best_L = -np.inf, 0
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            aa, bb = a[:n - L], b[L:]
        else:
            aa, bb = a[-L:], b[:n + L]
        m = np.isfinite(aa) & np.isfinite(bb)
        if m.sum() < 50:
            continue
        x = aa[m] - aa[m].mean(); y = bb[m] - bb[m].mean()
        denom = np.sqrt((x * x).sum() * (y * y).sum())
        if denom < 1e-9:
            continue
        c = float((x * y).sum() / denom)
        if c > best_c:
            best_c, best_L = c, L
    return best_c, best_L


def mocap_trunk_angles(m1, m4, m2, ts0, mocap_time, n):
    """Per-camera-frame mocap trunk angles (deg) vs the mocap neutral (first N)."""
    R_list = []
    for i in range(n):
        mi = nearest_index(mocap_time, ts0[i])
        R_list.append(mocap_trunk_frame(m1[mi], m4[mi], m2[mi]))
    valid = [R for R in R_list if R is not None]
    flex = np.full(n, np.nan); lat = np.full(n, np.nan); axi = np.full(n, np.nan)
    if len(valid) < 3:
        return flex, lat, axi
    R_neu = Rotation.from_matrix(np.array(valid[:N_NEUTRAL])).mean().as_matrix()
    for i, R_m in enumerate(R_list):
        if R_m is not None:
            flex[i], lat[i], axi[i] = trunk_angles(R_m, R_neu)
    return flex, lat, axi


# %% ---- Geometry pass (parallel) ----
_G = None  # per-worker model/calib bundle


def _init_geom():
    global _G
    K0, D0, K1, D1, R, T_mm = load_stereo(STEREO_TOML)
    R0_c, t0_c, Rm_c, tm_c = load_charuco_basis(CHARUCO_TOML)
    _G = dict(
        K0=K0, D0=D0, K1=K1, D1=D1, R=R, T_m=T_mm / 1000.0,
        R0_c=R0_c, t0_c=t0_c,
        seg=YOLO(str(SEG_WEIGHTS)),
        lmk0=make_pose_landmarker(), lmk1=make_pose_landmarker(),
        rect=build_rectification(K0, D0, K1, D1, R, T_mm, CAM_SIZE),
        sgbm=make_sgbm(), rng=np.random.default_rng(0))


def _geom_chunk(rng_tuple):
    start, end = rng_tuple
    g = _G
    R1, Q, maps0, maps1 = g["rect"]
    W, H = CAM_SIZE
    out = []
    for i, (f0, f1) in enumerate(iter_frame_pairs(CAM0_FRAMES, CAM1_FRAMES)):
        if i < start:
            continue
        if i >= end:
            break
        g0 = f0 if f0.ndim == 2 else cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY)
        g1 = f1 if f1.ndim == 2 else cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        img0 = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(g0, cv2.COLOR_GRAY2RGB))
        img1 = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(g1, cv2.COLOR_GRAY2RGB))
        r0, r1 = g["lmk0"].detect(img0), g["lmk1"].detect(img1)
        lms0 = r0.pose_landmarks[0] if r0.pose_landmarks else None
        lms1 = r1.pose_landmarks[0] if r1.pose_landmarks else None

        cam_pts, sh_px = {}, {}
        for name, idx in UPPER_LIMB.items():
            p0 = landmark_px(lms0, idx, W, H)
            p1 = landmark_px(lms1, idx, W, H)
            xyz = None
            if p0 and p1:
                pc = triangulate(p0, p1, g["K0"], g["D0"], g["K1"], g["D1"],
                                 g["R"], g["T_m"])
                if pc is not None:
                    xyz = to_board(pc, g["R0_c"], g["t0_c"])
            cam_pts[name] = xyz
            if name in ("L_shoulder", "R_shoulder"):
                sh_px[name] = p0

        mask0 = torso_mask(g["seg"], g0, CAM_SIZE)
        tp, depth_u8 = stereo_step(g0, g1, mask0, g["sgbm"], maps0, maps1, R1, Q)
        centroid_b = normal_b = None
        fit = fit_plane(tp, g["rng"])
        # Refit on the lower torso band (below the chest marker-object) so the plane
        # normal is not corrupted by it. "Down" = trunk up-axis from shoulders + the
        # preliminary normal, mapped into the cam0 frame that tp lives in.
        ls, rs = cam_pts["L_shoulder"], cam_pts["R_shoulder"]
        if fit is not None and TRUNK_DROP_M > 0 and ls is not None and rs is not None:
            Rtf = trunk_frame(ls, rs, dir_to_board(fit[1], g["R0_c"]))
            if Rtf is not None:
                up_c = g["R0_c"] @ Rtf[:, 1]
                sh_mid_c = board_to_cam0((ls + rs) / 2, g["R0_c"], g["t0_c"])
                below = (tp - sh_mid_c) @ up_c < -TRUNK_DROP_M
                if below.sum() >= 50:
                    fit = fit_plane(tp[below], g["rng"]) or fit
        if fit is not None:
            centroid_b = to_board(fit[0], g["R0_c"], g["t0_c"])
            normal_b = dir_to_board(fit[1], g["R0_c"])

        # full front-shell cloud (incl. chest object -- it is rigid with the trunk),
        # board frame, downsampled: input for ICP experiments (07_icp_trunk.py)
        cloud_b = None
        if len(tp) >= 50:
            ds = voxel_downsample(tp, rng=g["rng"])
            cloud_b = to_board_bulk(ds, g["R0_c"], g["t0_c"]).astype(np.float32)

        out.append(dict(cam_pts=cam_pts, sh_px=sh_px, centroid=centroid_b,
                        normal=normal_b, depth=depth_u8, cloud=cloud_b))
    return start, out


def run_geometry(n):
    ranges = [(int(r[0]), int(r[-1]) + 1) for r in np.array_split(np.arange(n), N_WORKERS)]
    ranges = [r for r in ranges if r[1] > r[0]]
    if N_WORKERS <= 1:
        _init_geom()
        chunks = [_geom_chunk(r) for r in tqdm(ranges, desc="geometry")]
    else:
        with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_geom) as ex:
            chunks = list(tqdm(ex.map(_geom_chunk, ranges), total=len(ranges),
                               desc="geometry(chunks)"))
    chunks.sort(key=lambda c: c[0])
    frames = []
    for _, lst in chunks:
        frames.extend(lst)
    return frames


# %% Assemble trunk frames + angles (sequential: EMA + neutral)
def assemble(frames):
    n = len(frames)
    R_t_list, all_pts = [], []
    ant_ema = None
    for fr in frames:
        for xyz in fr["cam_pts"].values():
            if xyz is not None:
                all_pts.append(xyz)
        anterior = None
        if fr["normal"] is not None:
            raw = fr["normal"]
            ant_ema = raw if ant_ema is None else (
                ANTERIOR_EMA * raw + (1 - ANTERIOR_EMA) * ant_ema)
            ant_ema = ant_ema / np.linalg.norm(ant_ema)
            anterior = ant_ema
        fr["anterior"] = anterior
        ls, rs = fr["cam_pts"]["L_shoulder"], fr["cam_pts"]["R_shoulder"]
        R_t = trunk_frame(ls, rs, anterior) if (
            ls is not None and rs is not None and anterior is not None) else None
        R_t_list.append(R_t)

    valid = [R for R in R_t_list if R is not None]
    if len(valid) < 3:
        raise SystemExit("Too few valid trunk frames for a neutral pose.")
    R_neu = Rotation.from_matrix(np.array(valid[:N_NEUTRAL])).mean().as_matrix()
    flex = np.full(n, np.nan); lat = np.full(n, np.nan); axi = np.full(n, np.nan)
    for i, R_t in enumerate(R_t_list):
        if R_t is not None:
            flex[i], lat[i], axi[i] = trunk_angles(R_t, R_neu)
    return R_t_list, R_neu, flex, lat, axi, all_pts


# %% ---- Render pass (parallel) ----
_R = None  # per-worker render bundle


def _init_render(K0, D0, R0_c, t0_c, flex, lat, axi, mflex, mlat, maxi, t, limits, fps):
    global _R
    _R = dict(K0=K0, D0=D0, R0_c=R0_c, t0_c=t0_c,
              flex=flex, lat=lat, axi=axi, mflex=mflex, mlat=mlat, maxi=maxi,
              t=t, limits=limits, fps=fps)


def _axes3d_style(ax, limits):
    ax.set_facecolor("#111111")
    ax.set_xlim(*limits["x"]); ax.set_ylim(*limits["y"]); ax.set_zlim(*limits["z"])
    ax.tick_params(colors="gray", labelsize=6)
    for lab, f in zip("XYZ", (ax.set_xlabel, ax.set_ylabel, ax.set_zlabel)):
        f(lab, color="gray", fontsize=7)


def _draw_scene(ax, fr, limits, view, show_skeleton, show_mocap):
    ax.cla()
    _axes3d_style(ax, limits)
    ax.view_init(elev=view[0], azim=view[1])
    R_t = fr["R_t"]
    ls, rs = fr["cam_pts"].get("L_shoulder"), fr["cam_pts"].get("R_shoulder")
    if show_skeleton:
        for a, b in BONES:
            pa, pb = fr["cam_pts"].get(a), fr["cam_pts"].get(b)
            if pa is not None and pb is not None:
                ax.plot(*np.stack([pa, pb]).T, color="white", lw=2)
        for name, xyz in fr["cam_pts"].items():
            if xyz is not None:
                ax.scatter(*xyz, c=CAM_JOINT_COLORS[name], s=25, depthshade=False)
    if show_mocap:
        mp_ = fr["mocap_pts"]
        for a, b in BONES:
            ma, mb = mp_.get(a), mp_.get(b)
            if ma is not None and mb is not None:
                ax.plot(*np.stack([ma, mb]).T, color="orange", lw=1.2, ls="--")
        for xyz in mp_.values():
            if xyz is not None:
                ax.scatter(*xyz, c="orange", marker="x", s=20, depthshade=False)
    if R_t is not None and ls is not None and rs is not None:
        origin = trunk_origin(ls, rs, R_t)
        for col, name in zip((0, 1, 2), ("lateral", "up", "anterior")):
            ax.quiver(*origin, *(R_t[:, col] * AXIS_LEN_M),
                      color=AXIS_COLORS_MPL[name], lw=2)
        if fr["centroid"] is not None:
            gx, gy, gz = plane_grid(fr["centroid"], R_t[:, 0], R_t[:, 2])
            ax.plot_surface(gx, gy, gz, color="orange", alpha=0.3, shade=False)
    ax.scatter(0, 0, 0, c="red", marker="^", s=35)


def _fig_to_bgr(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
    return cv2.cvtColor(buf[..., 1:], cv2.COLOR_RGB2BGR)


def _panel_3d(fig, ax, fr, limits, view, show_skeleton, show_mocap):
    fig.patch.set_facecolor("#111111")
    _draw_scene(ax, fr, limits, view, show_skeleton, show_mocap)
    img = _fig_to_bgr(fig)
    return cv2.resize(img, (QUAD_W, QUAD_H))


def _panel_angles(fig, axes, i):
    r = _R
    cam = {"flexion": r["flex"], "lateral": r["lat"], "axial": r["axi"]}
    moc = {"flexion": r["mflex"], "lateral": r["mlat"], "axial": r["maxi"]}
    t = r["t"]
    fig.patch.set_facecolor("#111111")
    for k, (ax, name) in enumerate(zip(axes, cam)):
        arr, marr = cam[name], moc[name]
        ax.cla()
        ax.plot(t, marr, color="white", lw=0.8, ls="--", alpha=0.7, label="mocap")
        ax.plot(t, arr, color=ANGLE_COLORS[name], lw=1.3, label="camera")
        ax.axvline(t[i], color="white", lw=1)
        ax.axhline(0, color="gray", lw=0.5)
        cur, mcur = arr[i], marr[i]
        cs = f"{cur:+.0f}" if np.isfinite(cur) else "--"
        ms = f"{mcur:+.0f}" if np.isfinite(mcur) else "--"
        ax.set_ylabel(f"{name}\ncam {cs}  moc {ms}", color=ANGLE_COLORS[name], fontsize=8)
        ax.set_facecolor("#111111")
        ax.tick_params(colors="gray", labelsize=6)
        ax.set_xlim(t[0], t[-1])
        if k == 0:
            ax.legend(loc="upper right", fontsize=6, facecolor="#222222",
                      labelcolor="white", framealpha=0.6)
    axes[-1].set_xlabel("time (s)", color="gray", fontsize=8)
    return cv2.resize(_fig_to_bgr(fig), (QUAD_W, QUAD_H))


def _panel_cam0(g0, fr):
    r = _R
    img = cv2.cvtColor(g0, cv2.COLOR_GRAY2BGR)
    for px in fr["sh_px"].values():
        if px is not None:
            cv2.circle(img, (int(px[0]), int(px[1])), 6, (0, 255, 0), -1)
    R_t = fr["R_t"]
    ls, rs = fr["cam_pts"].get("L_shoulder"), fr["cam_pts"].get("R_shoulder")
    if R_t is not None and ls is not None and rs is not None:
        origin_b = trunk_origin(ls, rs, R_t)
        ends = [origin_b] + [origin_b + R_t[:, c] * AXIS_LEN_M for c in (0, 1, 2)]
        pc = np.array([board_to_cam0(p, r["R0_c"], r["t0_c"]) for p in ends])
        ip, _ = cv2.fisheye.projectPoints(
            pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), r["K0"], r["D0"])
        ip = ip.reshape(-1, 2).astype(int)
        for c, name in zip((1, 2, 3), ("lateral", "up", "anterior")):
            cv2.line(img, tuple(ip[0]), tuple(ip[c]), AXIS_COLORS_BGR[name], 3)
    return cv2.resize(img, (QUAD_W, QUAD_H))


def _panel_depth(depth_u8):
    """Colorize the small depth map (0 = no data -> black) to a QUAD-size heatmap."""
    d = cv2.resize(depth_u8, (QUAD_W, QUAD_H), interpolation=cv2.INTER_NEAREST)
    heat = cv2.applyColorMap(d, DEPTH_CMAP)
    heat[d == 0] = (0, 0, 0)
    cv2.putText(heat, "depth (near->far)", (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return heat


def _render_chunk(args):
    start, end, tmp_path, frames = args
    fig_tr = plt.figure(figsize=(QUAD_W / 100, QUAD_H / 100), dpi=100)
    ax_tr = fig_tr.add_subplot(111, projection="3d")
    fig_br, axes_br = plt.subplots(3, 1, figsize=(QUAD_W / 100, QUAD_H / 100), dpi=100)
    lim = _R["limits"]

    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             _R["fps"], (QUAD_W * 2, QUAD_H * 2))
    fi = 0
    for i, g0 in enumerate(iter_frames(CAM0_FRAMES)):
        if i < start:
            continue
        if i >= end:
            break
        g0 = g0 if g0.ndim == 2 else cv2.cvtColor(g0, cv2.COLOR_RGB2GRAY)
        fr = frames[fi]; fi += 1
        tl = _panel_cam0(g0, fr)
        tr = _panel_3d(fig_tr, ax_tr, fr, lim, (18, -70), True, True)
        bl = _panel_depth(fr["depth"])
        br = _panel_angles(fig_br, axes_br, i)
        top = cv2.hconcat([tl, tr])
        bot = cv2.hconcat([bl, br])
        writer.write(cv2.vconcat([top, bot]))
    writer.release()
    for f in (fig_tr, fig_br):
        plt.close(f)
    return start, tmp_path


def run_render(frames, flex, lat, axi, mflex, mlat, maxi, t, limits, fps,
               K0, D0, R0_c, t0_c, n):
    ranges = [(int(r[0]), int(r[-1]) + 1) for r in np.array_split(np.arange(n), N_WORKERS)]
    ranges = [r for r in ranges if r[1] > r[0]]
    tmpdir = tempfile.mkdtemp(prefix="trunk_render_")
    tasks = []
    for k, (s, e) in enumerate(ranges):
        tmp = str(Path(tmpdir) / f"seg_{k:03d}.mp4")
        tasks.append((s, e, tmp, frames[s:e]))
    initargs = (K0, D0, R0_c, t0_c, flex, lat, axi, mflex, mlat, maxi, t, limits, fps)

    if N_WORKERS <= 1:
        _init_render(*initargs)
        segs = [_render_chunk(t_) for t_ in tqdm(tasks, desc="render")]
    else:
        with ProcessPoolExecutor(max_workers=N_WORKERS,
                                 initializer=_init_render, initargs=initargs) as ex:
            segs = list(tqdm(ex.map(_render_chunk, tasks), total=len(tasks),
                             desc="render(chunks)"))
    segs.sort(key=lambda s: s[0])

    writer = cv2.VideoWriter(str(OUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (QUAD_W * 2, QUAD_H * 2))
    for _, path in segs:
        cap = cv2.VideoCapture(path)
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            writer.write(fr)
        cap.release()
        os.remove(path)
    writer.release()
    os.rmdir(tmpdir)


# %% Main
def compute_limits(all_pts, pad=AXIS_PAD_M):
    pts = np.array(all_pts)
    lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
    return {"x": (lo[0] - pad, hi[0] + pad),
            "y": (lo[1] - pad, hi[1] + pad),
            "z": (lo[2] - pad, hi[2] + pad)}


def main():
    print(f"Workers: {N_WORKERS}")
    print("Loading basis + mocap + timestamps...")
    R0_c, t0_c, Rm_c, tm_c = load_charuco_basis(CHARUCO_TOML)
    K0, D0 = load_stereo(STEREO_TOML)[:2]
    mocap_df, st_time = read_rigid_body_csv(str(RECORDING_DIR / f"{RECORDING_NAME}.csv"))
    mocap_time = np.datetime64(st_time) + (
        mocap_df["seconds"].to_numpy() * 1e6).astype("timedelta64[us]")
    ts0 = load_frame_times(RECORDING_DIR / "cam0_timestamp.msgpack")
    ts1 = load_frame_times(RECORDING_DIR / "cam1_timestamp.msgpack")
    n = min(len(ts0), len(ts1))
    max_frames = int(os.environ.get("TRUNK_MAX_FRAMES", "0"))
    if max_frames:
        n = min(n, max_frames)
    ts0 = ts0[:n]

    # optional geometry cache (TRUNK_GEOM_CACHE=path): lets mocap/sync/mapping tweaks
    # skip the expensive geometry pass. Only used for full runs. Delete/change env to
    # force a recompute after any geometry-affecting change (plane, seg, triangulation).
    cache = os.environ.get("TRUNK_GEOM_CACHE", "")
    if cache and Path(cache).exists() and not max_frames:
        print(f"Loading geometry cache {cache} ...")
        with open(cache, "rb") as f:
            frames = pickle.load(f)
        assert len(frames) == n, f"cache has {len(frames)} frames, expected {n}"
    else:
        print(f"Pass 1/2: geometry over {n} frames...")
        frames = run_geometry(n)
        if cache and not max_frames:
            with open(cache, "wb") as f:
                pickle.dump(frames, f)
            print(f"Cached geometry -> {cache}")

    # mocap per frame (cheap, sequential)
    for i, fr in enumerate(frames):
        row = mocap_df.iloc[nearest_index(mocap_time, ts0[i])]
        mpts = {}
        for name, pref in MOCAP_MARKERS.items():
            pw = np.array([row[f"{pref}_x"], row[f"{pref}_y"], row[f"{pref}_z"]], float)
            mpts[name] = mocap_to_board(pw, Rm_c, tm_c) if np.isfinite(pw).all() else None
        fr["mocap_pts"] = mpts

    R_t_list, _R_neu, flex, lat, axi, all_pts = assemble(frames)
    for fr, R_t in zip(frames, R_t_list):
        fr["R_t"] = R_t
    for fr in frames:
        for name, xyz in fr["mocap_pts"].items():
            if xyz is not None:
                all_pts.append(xyz)

    # smooth the camera angles (Savitzky-Golay); mocap stays raw
    flex_s, lat_s, axi_s = (smooth_series(flex), smooth_series(lat), smooth_series(axi))

    # mocap trunk rigid-body angles (Marker1 origin, Marker4 xvec, Marker2 zvec)
    m1, m4, m2 = load_trunk_rb(str(RECORDING_DIR / f"{RECORDING_NAME}.csv"))

    # --- time-sync camera & mocap ---
    # Wall clocks are only coarsely aligned; cross-correlate camera vs mocap flexion
    # to recover the true residual lag, then shift the mocap timeline by it. Positive
    # lag means mocap features occur later (camera leads) -> pull mocap earlier.
    period = float(np.median(np.diff(ts0) / np.timedelta64(1, "s")))
    # Sync on clean landmark motion (MediaPipe shoulder-mid vs mocap trunk origin),
    # not the noisy trunk angles -- both spike together when the subject moves.
    cam_pos = np.array([
        (fr["cam_pts"]["L_shoulder"] + fr["cam_pts"]["R_shoulder"]) / 2
        if fr["cam_pts"]["L_shoulder"] is not None
        and fr["cam_pts"]["R_shoulder"] is not None else [np.nan, np.nan, np.nan]
        for fr in frames])
    moc_pos = np.array([m1[nearest_index(mocap_time, ts0[i])] for i in range(n)])
    corr, lagL = best_lag(series_speed(cam_pos), series_speed(moc_pos),
                          max_lag=int(round(MAX_SYNC_LAG_S / period)))
    lag_s = lagL * period
    mocap_time = mocap_time - np.timedelta64(int(round(lag_s * 1e6)), "us")
    mflex, mlat, maxi = mocap_trunk_angles(m1, m4, m2, ts0, mocap_time, n)

    # mocap exists only within its (shifted) recording window; blank it elsewhere
    in_win = (ts0 >= mocap_time[0]) & (ts0 <= mocap_time[-1])
    mflex[~in_win] = np.nan
    mlat[~in_win] = np.nan
    maxi[~in_win] = np.nan

    i0 = int(np.argmax(in_win))                       # first in-window camera frame
    i1 = n - 1 - int(np.argmax(in_win[::-1]))         # last in-window camera frame
    t = ((ts0 - ts0[i0]) / np.timedelta64(1, "s")).astype(np.float64)  # t=0 at overlap

    zref = slice(i0, min(i0 + N_NEUTRAL, n))          # zero every trace here
    for arr in (flex_s, lat_s, axi_s, mflex, mlat, maxi):
        off = np.nanmean(arr[zref])
        if np.isfinite(off):
            arr -= off

    limits = compute_limits(all_pts)
    dt = np.diff(t)
    fps = float(1.0 / np.median(dt[dt > 0])) if (dt > 0).any() else 15.0
    print(f"cam valid {int(np.isfinite(flex_s).sum())}/{n}  "
          f"xcorr lag {lag_s:+.2f}s (r={corr:.2f})  overlap frames [{i0},{i1}]  "
          f"fps~{fps:.2f}")

    print("Pass 2/2: rendering 2x2 video...")
    run_render(frames, flex_s, lat_s, axi_s, mflex, mlat, maxi, t, limits, fps,
               K0, D0, R0_c, t0_c, n)
    print(f"Done -> {OUT_VIDEO}")


if __name__ == "__main__":
    main()
