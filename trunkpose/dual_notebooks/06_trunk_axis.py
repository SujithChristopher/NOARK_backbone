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

import hashlib
import itertools
import json
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

RECORDING_NAME = os.environ.get("TRUNK_RECORDING_NAME", "july_27_sujith_no_comp")
RECORDING_DIR = Path(os.environ.get(
    "TRUNK_RECORDING_DIR",
    str(PROJECT_ROOT / "data" / "july27" / RECORDING_NAME),
))

# %% Session config
# Everything that changes between capture sessions -- which stereo calibration, how
# the rig was mounted, how the Motive rigid body is named -- lives in a session.toml
# next to the recordings (this dir or its parent), so a new session is a data change,
# not a code edit. The july-27 defaults below apply when no file is found.
def load_session(rec_dir):
    """Nearest session.toml at or above the recording, searching up to the repo root.

    A protocol take keeps it one level up (data/sep23_26/session.toml serves
    data/sep23_26/gokul_wc). A game recording sits five levels down inside the
    recorder's own tree, so the search walks the parents instead of checking one.
    """
    d = Path(rec_dir).resolve()
    for cand in (d, *d.parents):
        q = cand / "session.toml"
        if q.exists():
            return toml.load(q), q
        if cand == PROJECT_ROOT:
            break
    return {}, None


SESSION, SESSION_TOML = load_session(RECORDING_DIR)
_cam_cfg = SESSION.get("camera", {})
_cal_cfg = SESSION.get("calibration", {})
_moc_cfg = SESSION.get("mocap", {})


def _session_path(key, cfg, default):
    """Path from env > session.toml (repo-relative) > hard default. '' / 'none' -> None."""
    raw = os.environ.get(f"TRUNK_{key.upper()}", cfg.get(key))
    if raw is None:
        return default
    if str(raw).strip().lower() in ("", "none"):
        return None
    q = Path(raw)
    return q if q.is_absolute() else PROJECT_ROOT / q


# Sensor resolution of THIS recording. The stereo calibration may have been shot at a
# different resolution; load_stereo rescales the intrinsics to match.
CAM_SIZE = tuple(int(v) for v in _cam_cfg.get("size", (1280, 800)))
# The pair is recorded cam0/cam1, but the calibration's stereo T tells you which one is
# physically on the left, and SGBM needs the left image first. When the rig is plugged
# in the other way round (sep23: T_x > 0 => cam1 is left) set swap = true and the whole
# pipeline runs with cam1 as the reference camera.
SWAP_CAMS = bool(_cam_cfg.get("swap", False))
# Camera mounted upside down: geometry is unaffected (the calibration was shot the same
# way), but YOLO and MediaPipe both fail on an inverted person, so inference runs on a
# 180-rotated copy and the results are mapped back to sensor pixels.
ROT180 = bool(_cam_cfg.get("rot180", False))

_c0, _c1 = ("cam1", "cam0") if SWAP_CAMS else ("cam0", "cam1")


def stream_paths(stem):
    """The files making up one recorded stream, in play order.

    The trunk-protocol recorder writes a take as a single <stem>.msgpack. The NOARK
    game recorder splits a long take into 900-frame chunks, <stem>_0000.msgpack,
    _0001, ... (chunks.json lists them; the zero-padded names sort into the same
    order). Returning a list either way keeps every reader below layout-agnostic.
    """
    one = RECORDING_DIR / f"{stem}.msgpack"
    if one.exists():
        return [one]
    shards = sorted(RECORDING_DIR.glob(f"{stem}_[0-9][0-9][0-9][0-9].msgpack"))
    # no shards either: keep the single-file path so the failure names the file the
    # caller expected rather than an empty list.
    return shards or [one]


CAM0_FRAMES = stream_paths(f"{_c0}_frame")
CAM1_FRAMES = stream_paths(f"{_c1}_frame")
CAM0_TIMES = stream_paths(f"{_c0}_timestamp")
CAM1_TIMES = stream_paths(f"{_c1}_timestamp")

# Use every Nth recorded frame. The validation takes are short and run at full rate;
# the game takes are ~6 min at 30 Hz, where the trunk moves slowly enough that a
# stride buys back the geometry pass and the cache it writes. It is applied inside the
# readers, so frame index i means raw frame i*FRAME_STRIDE for timestamps, the GPIO
# bit and the geometry pass alike -- nothing downstream has to know.
FRAME_STRIDE = max(1, int(os.environ.get("TRUNK_FRAME_STRIDE", "1")))
# First frame of the neutral window (in strided indices). A protocol take opens with
# the subject sitting neutral, so 0 is right. A game recording rolls for a minute
# before Motive starts, so the driver moves this to the GPIO rise -- mocap t=0 -- and
# both systems are then zeroed on the same instant.
NEUTRAL_START = int(os.environ.get("TRUNK_NEUTRAL_START", "0"))
# Motive's export for this take. Protocol takes keep it beside the frames as
# <recording>.csv; the game session keeps one mocap folder for every subject, so the
# path is given explicitly.
MOCAP_CSV = Path(os.environ.get("TRUNK_MOCAP_CSV",
                                str(RECORDING_DIR / f"{RECORDING_NAME}.csv")))

# No default: a stereo calibration belongs to a rig, and picking the wrong rig's is
# silent (cz30_dual_v2 has T_x < 0, i.e. the cameras the other way round, so the depth
# comes out inverted rather than absent). Every session states its own -- see
# data/july27/session.toml and data/sep23_26/session.toml.
STEREO_TOML = _session_path("stereo_toml", _cal_cfg, None)
if STEREO_TOML is None:
    raise SystemExit(
        f"No stereo calibration for {RECORDING_DIR}. Add a session.toml next to the "
        "recordings (or in its parent) with [calibration] stereo_toml = ..., or set "
        "TRUNK_STEREO_TOML.")
if not Path(STEREO_TOML).exists():
    raise SystemExit(f"Stereo calibration not found: {STEREO_TOML}")
# Optional: with no ChArUco basis the "board frame" is just the reference-camera frame.
# That is fine for the trunk angles -- they are read off R_neu's own columns, so a
# constant frame rotation cancels -- and the mocap side fits its residual rotation S
# anyway. It only costs the shared world frame used for rendering mocap with the cloud.
CHARUCO_TOML = _session_path("charuco_toml", _cal_cfg, None)
# NOTE: default still points at the old single-class model. After retraining on the
# multi-class (torso/arm/head) dataset, either edit the default or set env TRUNK_SEG_WEIGHTS
# (ultralytics names each run trunk_segN, so the new path isn't known ahead of time).
SEG_WEIGHTS = _session_path("seg_weights", _cal_cfg, None) or Path(os.environ.get(
    "TRUNK_SEG_WEIGHTS",
    str(PROJECT_ROOT / "trunkpose" / "runs" / "trunk_seg-2" / "weights" / "best.pt")))
POSE_MODEL = SCRIPT_DIR / "pose_landmarker_full.task"
OUT_VIDEO = RECORDING_DIR / "trunk_axis_charuco.mp4"

# %% Config
VISIBILITY_THRESH = 0.5
DEPTH_MIN_M, DEPTH_MAX_M = 0.2, 3.0     # torso-fit depth gate
SCENE_DEPTH_MAX_M = 4.0                  # depth-heatmap far clip
# Cloud-extraction knobs. A session with a different framing (sep23: half-res sensor,
# subject ~1.1 m away, so the torso is ~7 k px instead of ~40 k) can widen the shell
# from its session.toml [geometry] table, or per run via TRUNK_<KEY>. More of the
# torso flanks means more curvature, which is what makes an axial rotation observable
# at all -- a perfectly cylindrical front patch is degenerate under it.
def _geom_cfg(key, default, cast=float):
    return cast(os.environ.get(f"TRUNK_{key.upper()}",
                               SESSION.get("geometry", {}).get(key, default)))


MASK_ERODE_FRAC = _geom_cfg("mask_erode_frac", 0.06)  # erode mask by this * sqrt(area) px
ARM_EXCLUDE_DILATE_PX = _geom_cfg("arm_exclude_dilate_px", 9, int)
FRONT_PCTL = _geom_cfg("front_pctl", 5.0)   # frontmost-depth percentile for the shell
SHELL_M = _geom_cfg("shell_m", 0.10)        # keep points within this depth of the cap
MAX_POSES = _geom_cfg("max_poses", 3, int)  # people MediaPipe may detect per frame
# Where YOLO runs: "auto" takes the GPU when torch reports one. Each geometry worker is
# a separate process and so holds its own CUDA context (~0.5 GB), so keep TRUNK_WORKERS
# modest on a GPU run -- MediaPipe stays on the CPU either way (no Windows GPU delegate),
# which is what the remaining workers are for.
SEG_DEVICE = str(_geom_cfg("seg_device", "auto", str))
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
# Edge-preserving smoothing on raw disparity before reprojection (cuts SGBM
# speckle/quantization noise without blurring across depth discontinuities).
# cv2.ximgproc's WLS disparity filter would be the sharper tool here, but this
# build's opencv-contrib-python doesn't compile the ximgproc contrib module in.
BILATERAL_D = 7
BILATERAL_SIGMA_COLOR = 8.0   # disparity units (px); ~ expected noise, keeps edges
BILATERAL_SIGMA_SPACE = 7.0
SEG_CONF = 0.25

N_WORKERS = int(os.environ.get("TRUNK_WORKERS", "4"))
# Motive's solved rigid body can re-lock at a ghost pose after a brief occlusion
# (this take: 326 mm across one 10 ms dropped frame). Marker distances stay exactly
# constant because the output is solved, so teleports are detected as position jumps
# across NaN gaps. Rotations either side carry different constant offsets, so the
# camera<->mocap comparison is restricted to the longest teleport-free segment.
SEG_JUMP_M = 0.10

# July-27 `trunk` rigid body mounting:
#   Marker3 = frame origin, Marker1 = +X direction, Marker4 = +Z direction.
# These are intentionally separate from the calibration T-frame mapping
# (calibration: origin=4, +X=1, +Z=3).
TRUNK_RB_NAME = str(_moc_cfg.get("rigid_body", "trunk"))
TRUNK_ORIGIN_MARKER = int(_moc_cfg.get("origin_marker", 3))
TRUNK_X_MARKER = int(_moc_cfg.get("x_marker", 1))
TRUNK_Z_MARKER = int(_moc_cfg.get("z_marker", 4))

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
MOCAP_BONES = [
    ("trunk_origin", "trunk_x"),
    ("trunk_origin", "trunk_z"),
]
CAM_JOINT_COLORS = {
    "L_shoulder": "limegreen", "R_shoulder": "dodgerblue",
    "L_elbow": "green", "R_elbow": "royalblue",
    "L_wrist": "darkgreen", "R_wrist": "navy",
}
AXIS_COLORS_MPL = {"lateral": "red", "up": "lime", "anterior": "deepskyblue"}
AXIS_COLORS_BGR = {"lateral": (0, 0, 255), "up": (0, 255, 0), "anterior": (255, 128, 0)}
ANGLE_COLORS = {"flexion": "tomato", "lateral": "gold", "axial": "deepskyblue"}


# %% Geometry cache
# The cached clouds are a function of the calibration, the segmentation weights and the
# extraction knobs, none of which the cache path records -- so a calibration swap used
# to be invisible and the stale clouds were silently reused. The key below travels with
# the pickle as a sidecar (the pickle itself stays a plain list of frames, which the
# 08-12 figure scripts load directly).
def _fingerprint(path):
    """Short identity of a file: sha1 for small ones, size+mtime for model weights."""
    q = Path(path) if path else None
    if q is None or not q.exists():
        return "none"
    st = q.stat()
    if st.st_size < 4_000_000:
        return f"{q.name}:{hashlib.sha1(q.read_bytes()).hexdigest()[:12]}"
    return f"{q.name}:{st.st_size}:{int(st.st_mtime)}"


def geom_cache_key():
    """Everything that changes the cached clouds. Mocap/sync/ICP settings stay out."""
    key = {
        "stereo": _fingerprint(STEREO_TOML),
        "charuco": _fingerprint(CHARUCO_TOML),
        "seg_weights": _fingerprint(SEG_WEIGHTS),
        "cam": [list(CAM_SIZE), SWAP_CAMS, ROT180],
        "mask": [MASK_ERODE_FRAC, ARM_EXCLUDE_DILATE_PX, MAX_POSES],
        "shell": [FRONT_PCTL, SHELL_M, DEPTH_MIN_M, DEPTH_MAX_M, TRUNK_DROP_M],
        "cloud": [CLOUD_VOXEL_M, CLOUD_MAX_PTS],
        "sgbm": [SGBM_MIN_DISP, SGBM_NUM_DISP, SGBM_BLOCK],
    }
    if FRAME_STRIDE != 1:
        # only added when it is in play, so the full-rate caches written before the
        # stride existed still match their key files.
        key["stride"] = FRAME_STRIDE
    return key


def _key_path(cache):
    return Path(str(cache) + ".key.json")


def load_geom_cache(cache, n):
    """Cached frames, or None with the reason printed (missing / stale / mismatched)."""
    if not cache or not Path(cache).exists():
        return None
    want = geom_cache_key()
    kp = _key_path(cache)
    if not kp.exists():
        print(f"Cache {Path(cache).name} predates the settings key -> recomputing "
              "geometry (delete it to silence this)")
        return None
    got = json.loads(kp.read_text(encoding="utf-8"))
    diff = [k for k in want if got.get(k) != want[k]]
    if diff:
        print(f"Cache {Path(cache).name} was built under different settings "
              f"({', '.join(diff)}) -> recomputing geometry")
        for k in diff:
            print(f"    {k}: cached {got.get(k)} -> now {want[k]}")
        return None
    with open(cache, "rb") as f:
        frames = pickle.load(f)
    if len(frames) != n or not frames or "cloud" not in frames[0]:
        print(f"Cache {Path(cache).name} has {len(frames)} frames (expected {n}) or no "
              "clouds -> recomputing geometry")
        return None
    print(f"Loaded geometry cache {cache}")
    return frames


def save_geom_cache(cache, frames):
    if not cache:
        return
    with open(cache, "wb") as f:
        pickle.dump(frames, f)
    _key_path(cache).write_text(json.dumps(geom_cache_key(), indent=1), encoding="utf-8")
    print(f"Cached geometry -> {cache}")


# %% Calibration / basis
def load_stereo(path, size=None, swap=None):
    """Fisheye intrinsics + stereo extrinsics, rescaled to `size` and optionally swapped.

    A calibration shot at 1280x800 stays valid for a 640x400 binned recording after
    scaling fx/fy/cx/cy -- the fisheye distortion coefficients are in normalized
    coordinates, so they carry over untouched. `swap` re-expresses the pair with cam1
    as the reference camera (R -> R^T, T -> -R^T T)."""
    size = CAM_SIZE if size is None else size
    swap = SWAP_CAMS if swap is None else swap
    d = toml.load(path)
    K0 = np.array(d["cam0"]["camera_matrix"])
    D0 = np.array(d["cam0"]["dist_coeffs"][0], dtype=np.float64).reshape(4, 1)
    K1 = np.array(d["cam1"]["camera_matrix"])
    D1 = np.array(d["cam1"]["dist_coeffs"][0], dtype=np.float64).reshape(4, 1)
    R = np.array(d["stereo"]["R"])
    T_mm = np.array(d["stereo"]["T"], dtype=np.float64).reshape(3, 1)
    res = d["cam0"].get("resolution")
    if size is not None and res is not None and tuple(res) != tuple(size):
        sx, sy = size[0] / res[0], size[1] / res[1]
        if abs(sx - sy) > 1e-6:
            raise SystemExit(f"Calibration {tuple(res)} and recording {tuple(size)} "
                             "differ in aspect ratio -- not a pure rescale.")
        for K in (K0, K1):
            K[:2] *= sx
    if swap:
        K0, K1, D0, D1 = K1, K0, D1, D0
        R, T_mm = R.T, -R.T @ T_mm
    return K0, D0, K1, D1, R, T_mm


def load_charuco_basis(path):
    """(R0, t0, Rm, tm): reference-cam and mocap poses in the ChArUco board frame.

    `None` -> identity, i.e. the board frame collapses onto the reference-camera frame
    and mocap stays in Motive world. Trunk angles are unaffected (see CHARUCO_TOML)."""
    if path is None:
        return np.eye(3), np.zeros(3), np.eye(3), np.zeros(3)
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
# Every reader here takes either one file or the chunk list from stream_paths(), and
# every one applies FRAME_STRIDE, so the frame axis is the same in all of them.
def _records(path):
    """Records of one stream, concatenated across chunks, strided."""
    def raw():
        for q in ([path] if isinstance(path, (str, Path)) else path):
            with open(q, "rb") as f:
                yield from msgpack.Unpacker(f, object_hook=mpn.decode)
    return itertools.islice(raw(), 0, None, FRAME_STRIDE)


def iter_frame_pairs(path0, path1):
    # strided separately, then zipped: both streams have the same frame count, and
    # striding after the zip would need the pairs materialised.
    yield from zip(_records(path0), _records(path1))


def iter_frames(path0):
    yield from _records(path0)


def load_frame_times(path):
    return np.array([datetime.fromisoformat(r[1]) for r in _records(path)],
                    dtype="datetime64[us]")


def load_sync_flags(path):
    """GPIO sync bit per frame (column 0 of the timestamp msgpack). High while the
    mocap system is recording, so the first rising edge marks mocap t=0."""
    return np.array([int(r[0]) for r in _records(path)])


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
    disp_raw = sgbm.compute(rect0, rect1).astype(np.float32) / 16.0
    # smooth for the 3D values, but gate validity on the UNsmoothed disparity so the
    # invalid-pixel boundary (SGBM's own confidence) doesn't get blurred away
    disp = cv2.bilateralFilter(disp_raw, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
    pts = cv2.reprojectImageTo3D(disp, Q) / 1000.0  # meters, rectified-cam0 frame
    z = pts[..., 2]
    valid = (disp_raw > SGBM_MIN_DISP) & np.isfinite(pts).all(axis=2)

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


def for_inference(gray):
    """Sensor image as the upright image the networks were trained on."""
    return cv2.rotate(gray, cv2.ROTATE_180) if ROT180 else gray


def from_inference_mask(mask):
    return cv2.rotate(mask, cv2.ROTATE_180) if ROT180 else mask


def from_inference_px(p, size):
    """Inference-image pixel -> sensor pixel."""
    if p is None or not ROT180:
        return p
    W, H = size
    return (W - 1 - p[0], H - 1 - p[1])


def load_seg_model(weights=None, device=SEG_DEVICE):
    """YOLO segmentation model on the requested device ("auto" -> GPU if present)."""
    from ultralytics.utils.torch_utils import select_device

    model = YOLO(str(weights or SEG_WEIGHTS))
    dev = select_device("" if device == "auto" else device, verbose=False)
    return model.to(dev)


def torso_mask(seg_model, gray, size):
    """Torso mask of the SUBJECT: the centre-most torso instance, arms removed.

    A second person in frame (assistant, bystander) gets its own torso instance. The
    subject is the one seated in front of the rig, so instances are ranked by how far
    their centroid sits from the image centre and only the closest one is kept -- ORing
    them together would merge two bodies into one cloud and wreck the registration.
    Arm masks are all subtracted (after a small dilation) so an arm crossing the chest
    does not bleed into the torso surface.
    """
    W, H = size
    res = seg_model.predict(cv2.cvtColor(for_inference(gray), cv2.COLOR_GRAY2RGB),
                            conf=SEG_CONF, verbose=False)[0]
    torso = np.zeros((H, W), dtype=np.uint8)
    arm = np.zeros((H, W), dtype=np.uint8)
    if res.masks is not None:
        names = res.names
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        best, best_d = None, None
        for m, cid in zip(res.masks.data.cpu().numpy(), cls_ids):
            resized = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
            if not resized.any():
                continue
            if names[cid] in ("torso", "chest"):
                ys, xs = np.nonzero(resized)
                d = float(np.hypot(xs.mean() - W / 2, ys.mean() - H / 2))
                if best_d is None or d < best_d:
                    best, best_d = resized, d
            elif names[cid] == "arm":
                arm[resized] = 255
        if best is not None:
            torso[best] = 255
    if arm.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ARM_EXCLUDE_DILATE_PX,) * 2)
        torso[cv2.dilate(arm, k) > 0] = 0
    return from_inference_mask(torso)


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
        # several poses, because a bystander in frame can outrank the subject; the
        # caller keeps whichever pose is closest to the image centre
        running_mode=mp_vision.RunningMode.IMAGE, num_poses=MAX_POSES,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    return mp_vision.PoseLandmarker.create_from_options(opts)


def centre_pose(result):
    """The detected pose whose torso sits closest to the image centre (or None).

    Landmarks are normalized, so "centre" is (0.5, 0.5) and no image size is needed.
    Ranking on the shoulder/hip quad rather than the whole skeleton keeps an
    outstretched arm from dragging a bystander's score below the subject's."""
    poses = getattr(result, "pose_landmarks", None)
    if not poses:
        return None
    TORSO = (11, 12, 23, 24)   # shoulders + hips

    def dist(lms):
        pts = [lms[i] for i in TORSO if i < len(lms)]
        if not pts:
            return np.inf
        return float(np.hypot(np.mean([p.x for p in pts]) - 0.5,
                              np.mean([p.y for p in pts]) - 0.5))

    return min(poses, key=dist)


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


# %% Mocap trunk rigid body
def load_trunk_rb(
        path,
        origin_marker=TRUNK_ORIGIN_MARKER,
        x_marker=TRUNK_X_MARKER,
        z_marker=TRUNK_Z_MARKER,
        rb_name=TRUNK_RB_NAME):
    """Return (origin, x-marker, z-marker), each (N_mocap, 3), in mocap world.

    Marker indices describe the physical rigid-body mounting and are independent of
    the T-frame marker indices used to create ``charuco_basis.toml``.
    """
    raw = pd.read_csv(path, skiprows=2, header=None, dtype=str)
    types, names, poslab, xyz = raw.iloc[0], raw.iloc[1], raw.iloc[3], raw.iloc[4]
    data = raw.iloc[5:].reset_index(drop=True)

    def vec(marker):
        cols = {}
        for j in raw.columns:
            if (str(types[j]) == "Rigid Body Marker"
                    and str(names[j]).lower() == f"{rb_name.lower()}:marker{marker}"
                    and str(poslab[j]) == "Position"):
                cols[str(xyz[j]).strip().upper()] = j
        return np.stack([pd.to_numeric(data[cols[a]], errors="coerce").to_numpy()
                         for a in ("X", "Y", "Z")], axis=1)

    return vec(origin_marker), vec(x_marker), vec(z_marker)


def mocap_trunk_frame(o, mx, mz):
    """Trunk frame from rigid-body markers, columns [lateral, up, anterior] to match
    the camera convention (lateral = origin->X marker, anterior = origin->Z marker)."""
    if not (np.isfinite(o).all() and np.isfinite(mx).all() and np.isfinite(mz).all()):
        return None
    x = mx - o
    nx = np.linalg.norm(x)
    if nx < 1e-6:
        return None
    x /= nx
    a = (mz - o)
    a = a - (a @ x) * x
    na = np.linalg.norm(a)
    if na < 1e-6:
        return None
    a /= na
    u = np.cross(a, x)
    u /= np.linalg.norm(u)
    a = np.cross(x, u)
    return np.column_stack([x, u, a])


def kabsch(A, B):
    """Rigid (R, t) mapping points A onto points B, no scaling."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cb - R @ ca


def gpio_mocap_time(ts0, seconds, sync):
    """Mocap timeline on the CAMERA clock. The GPIO pin-17 bit is HIGH while Motive
    records, so its first rising edge IS mocap seconds=0 -- the two systems share a
    wire. Motive's own 'Capture Start Time' header is seconds off (this take: -2.38 s),
    and best_lag() xcorr is worse still (+3.03 s at r=0.20, i.e. 5.4 s / 162 frames from
    truth), so neither is used for sync. best_lag is kept only for 08_basis_check."""
    if not (sync == 1).any():
        raise SystemExit("No GPIO sync pulse in cam0_timestamp.msgpack.")
    rise = int(np.argmax(sync == 1))
    return ts0[rise] + (seconds * 1e6).astype("timedelta64[us]"), rise


def teleport_free_window(m1, mocap_time, ts0, in_win):
    """Restrict in_win to the longest span of mocap rows free of solver re-lock jumps."""
    ok = np.isfinite(m1).all(axis=1)
    vidx = np.where(ok)[0]
    if len(vidx) < 2:
        return in_win
    cut = np.where(np.linalg.norm(np.diff(m1[vidx], axis=0), axis=1) > SEG_JUMP_M)[0]
    bounds = np.concatenate(([0], cut + 1, [len(vidx)]))
    segs = [(vidx[a], vidx[b - 1]) for a, b in zip(bounds[:-1], bounds[1:])]
    s0, s1 = max(segs, key=lambda ab: mocap_time[ab[1]] - mocap_time[ab[0]])
    span = [(mocap_time[a] - mocap_time[0]) / np.timedelta64(1, "s") for a, _ in segs]
    print(f"  mocap teleport-split into {len(segs)} segment(s) starting at "
          + ", ".join(f"{s:.1f}s" for s in span)
          + f" -> using {(mocap_time[s0] - mocap_time[0]) / np.timedelta64(1, 's'):.1f}"
            f"-{(mocap_time[s1] - mocap_time[0]) / np.timedelta64(1, 's'):.1f}s")
    return in_win & (ts0 >= mocap_time[s0]) & (ts0 <= mocap_time[s1])


def fit_marker_alignment(frames, in_win):
    """Rigid transform putting mocap marker positions in the board frame, FITTED from
    paired joints (fit on first half of the overlap, scored on the held-out second).

    charuco_basis.toml's [mocap] block cannot do this: 03_get_charuco_basis.py assumes
    the 4 'tframe' markers lie on the board face, but they sit on the stand's base --
    their plane normal is Motive +Y while the board normal is near-horizontal, ~90 deg
    apart, and the marker-to-pattern offset was never measured. Straight through that
    basis the mocap skeleton lands 671 mm / 160 deg off. This fit is a visualisation
    aid; it must never be written back into the TOML."""
    idx = [i for i in range(len(frames)) if in_win[i] and any(
        frames[i]["cam_pts"][k] is not None and frames[i]["mocap_world"][k] is not None
        for k in UPPER_LIMB)]
    if len(idx) < 20:
        raise SystemExit("Too few paired camera/mocap frames to fit a marker alignment.")
    split = idx[len(idx) // 2]

    def pairs(lo, hi):
        A, B = [], []
        for i in range(lo, hi):
            for k in UPPER_LIMB:
                c, m = frames[i]["cam_pts"][k], frames[i]["mocap_world"][k]
                if c is not None and m is not None:
                    A.append(m)
                    B.append(c)
        return np.array(A), np.array(B)

    A_fit, B_fit = pairs(idx[0], split)
    R, t = kabsch(A_fit, B_fit)
    A_val, B_val = pairs(split, idx[-1] + 1)
    val = np.linalg.norm((A_val @ R.T + t) - B_val, axis=1)
    print(f"  marker alignment: {np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))):.1f} deg, "
          f"{np.linalg.norm(t) * 1000:.0f} mm; HELD-OUT n={len(val)} "
          f"mean {val.mean() * 1000:.0f} mm median {np.median(val) * 1000:.0f} mm")
    return R, t


def mocap_angles_in_camera_axes(R_moc, R_t_list, R_neu, in_win, n):
    """Mocap trunk angles expressed in the CAMERA anatomical axes, per 07_icp_trunk.

    Both systems are reduced to rotation-from-zero (R(t) @ Rz^T), so each one's absolute
    frame convention drops out -- that is what makes this immune to the broken [mocap]
    basis. What does NOT drop out is how each labels its axes: mocap_trunk_frame assumes
    trunk:Marker4 is lateral and Marker2 anterior, which is a mounting fact nobody
    verified. So the residual frame rotation S is fitted (Kabsch on rotation vectors of
    well-excited frames), first half only, and scored on the held-out second half."""
    both = [i for i in range(n) if in_win[i]
            and R_moc[i] is not None and R_t_list[i] is not None]
    if len(both) < N_NEUTRAL:
        raise SystemExit("Too few frames with both camera and mocap trunk frames.")
    zwin = both[:N_NEUTRAL]
    Rz_c = Rotation.from_matrix(np.array([R_t_list[i] for i in zwin])).mean().as_matrix()
    Rz_m = Rotation.from_matrix(np.array([R_moc[i] for i in zwin])).mean().as_matrix()

    def cam_angles():
        f = np.full(n, np.nan); l = np.full(n, np.nan); a = np.full(n, np.nan)
        for i, R_t in enumerate(R_t_list):
            if R_t is None:
                continue
            f[i], l[i], a[i] = trunk_angles((R_t @ Rz_c.T) @ R_neu, R_neu)
        return f, l, a

    def moc_angles(S):
        f = np.full(n, np.nan); l = np.full(n, np.nan); a = np.full(n, np.nan)
        for i, R_f in enumerate(R_moc):
            if R_f is None:
                continue
            Rc = S @ (R_f @ Rz_m.T) @ S.T
            f[i], l[i], a[i] = trunk_angles(Rc @ R_neu, R_neu)
        return f, l, a

    pairs = []
    for i in both:
        ra = Rotation.from_matrix(R_t_list[i] @ Rz_c.T).as_rotvec()
        rm = Rotation.from_matrix(R_moc[i] @ Rz_m.T).as_rotvec()
        if np.degrees(np.linalg.norm(ra)) > 5 and np.degrees(np.linalg.norm(rm)) > 5:
            pairs.append((i, ra, rm))
    if len(pairs) < 50:
        raise SystemExit("Too few well-excited rotation pairs to estimate S.")
    half = pairs[len(pairs) // 2][0]
    S_half, _ = kabsch(np.array([p[2] for p in pairs if p[0] < half]),
                       np.array([p[1] for p in pairs if p[0] < half]))
    S, _ = kabsch(np.array([p[2] for p in pairs]), np.array([p[1] for p in pairs]))
    s_deg = np.degrees(np.linalg.norm(Rotation.from_matrix(S).as_rotvec()))
    dh = np.degrees(np.linalg.norm(Rotation.from_matrix(S @ S_half.T).as_rotvec()))
    print(f"  axis rotation S={s_deg:.1f} deg ({len(pairs)} excited pairs; "
          f"half-fit differs {dh:.1f} deg)")

    flex, lat, axi = cam_angles()
    mfh, mlh, mah = moc_angles(S_half)
    held = np.arange(n) >= half
    print("  held-out (S fit on 1st half, scored on 2nd):")
    for lab, c, mo in (("flexion", flex, mfh), ("lateral", lat, mlh),
                       ("axial", axi, mah)):
        cs = smooth_series(c)
        m = np.isfinite(cs) & np.isfinite(mo) & held
        if m.sum() < 10:
            print(f"    {lab:8s}: too few held-out frames")
            continue
        r = float(np.corrcoef(cs[m], mo[m])[0, 1])
        e = float(np.sqrt(np.mean((cs[m] - mo[m]) ** 2)))
        print(f"    {lab:8s}: r={r:+.2f} RMSE={e:4.1f} deg")

    mflex, mlat, maxi = moc_angles(S)
    return flex, lat, axi, mflex, mlat, maxi


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
        seg=load_seg_model(),
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
                        data=cv2.cvtColor(for_inference(g0), cv2.COLOR_GRAY2RGB))
        img1 = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(for_inference(g1), cv2.COLOR_GRAY2RGB))
        lms0 = centre_pose(g["lmk0"].detect(img0))
        lms1 = centre_pose(g["lmk1"].detect(img1))

        cam_pts, sh_px = {}, {}
        for name, idx in UPPER_LIMB.items():
            p0 = from_inference_px(landmark_px(lms0, idx, W, H), CAM_SIZE)
            p1 = from_inference_px(landmark_px(lms1, idx, W, H), CAM_SIZE)
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

    valid = [R for R in R_t_list[NEUTRAL_START:] if R is not None]
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
        for a, b in MOCAP_BONES:
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
    mocap_df, st_time = read_rigid_body_csv(str(MOCAP_CSV))
    ts0 = load_frame_times(stream_paths("cam0_timestamp"))
    ts1 = load_frame_times(stream_paths("cam1_timestamp"))
    n = min(len(ts0), len(ts1))
    max_frames = int(os.environ.get("TRUNK_MAX_FRAMES", "0"))
    if max_frames:
        n = min(n, max_frames)
    ts0 = ts0[:n]

    sync = load_sync_flags(stream_paths("cam0_timestamp"))[:n]
    mocap_time, rise = gpio_mocap_time(ts0, mocap_df["seconds"].to_numpy(), sync)
    in_win = (ts0 >= mocap_time[0]) & (ts0 <= mocap_time[-1])
    print(f"GPIO sync: mocap t=0 at camera frame {rise}; Motive header clock "
          f"{(np.datetime64(st_time) - mocap_time[0]) / np.timedelta64(1, 's'):+.2f} s off; "
          f"mocap covers {int(in_win.sum())}/{n} frames")

    # Movement take contains only the trunk rigid body:
    # Marker3=origin, Marker1=+X, Marker4=+Z.
    mo, mx, mz = load_trunk_rb(str(MOCAP_CSV))
    in_win = teleport_free_window(mo, mocap_time, ts0, in_win)
    print(f"  usable overlap after teleport split: {int(in_win.sum())}/{n} frames")

    # optional geometry cache (TRUNK_GEOM_CACHE=path): lets mocap/sync/mapping tweaks
    # skip the expensive geometry pass. Only used for full runs; it is rebuilt by itself
    # whenever the calibration, the weights or an extraction knob changes.
    cache = "" if max_frames else os.environ.get("TRUNK_GEOM_CACHE", "")
    frames = load_geom_cache(cache, n)
    if frames is None:
        print(f"Pass 1/2: geometry over {n} frames...")
        frames = run_geometry(n)
        save_geom_cache(cache, frames)

    # Transform the trunk L-frame directly from Motive world into the ChArUco board
    # frame using the dedicated July-27 T-frame calibration.
    mob, mxb, mzb = ((m - tm_c) @ Rm_c for m in (mo, mx, mz))
    midx = np.array([nearest_index(mocap_time, ts0[i]) for i in range(n)])
    for i, fr in enumerate(frames):
        mpts = dict.fromkeys(("trunk_origin", "trunk_x", "trunk_z"))
        if in_win[i]:
            j = midx[i]
            for name, p in zip(mpts, (mob[j], mxb[j], mzb[j])):
                mpts[name] = p if np.isfinite(p).all() else None
        fr["mocap_pts"] = mpts

    R_t_list, R_neu, flex, lat, axi, all_pts = assemble(frames)
    for fr, R_t in zip(frames, R_t_list):
        fr["R_t"] = R_t

    for fr in frames:
        for xyz in fr["mocap_pts"].values():
            if xyz is not None:
                all_pts.append(xyz)

    # mocap trunk frames per camera frame -> angles in the camera's anatomical axes
    R_moc = [mocap_trunk_frame(mob[j], mxb[j], mzb[j]) if in_win[i] else None
             for i, j in enumerate(midx)]
    print("Mocap angles in camera axes...")
    flex, lat, axi, mflex, mlat, maxi = mocap_angles_in_camera_axes(
        R_moc, R_t_list, R_neu, in_win, n)

    # both systems are already zeroed at the common neutral by rotation-from-zero, so
    # no extra offset subtraction here. Camera smoothed (Savitzky-Golay), mocap raw.
    flex_s, lat_s, axi_s = smooth_series(flex), smooth_series(lat), smooth_series(axi)

    i0 = int(np.argmax(in_win))                       # first in-window camera frame
    i1 = n - 1 - int(np.argmax(in_win[::-1]))         # last in-window camera frame
    t = ((ts0 - ts0[i0]) / np.timedelta64(1, "s")).astype(np.float64)  # t=0 at overlap

    limits = compute_limits(all_pts)
    dt = np.diff(t)
    fps = float(1.0 / np.median(dt[dt > 0])) if (dt > 0).any() else 15.0
    print(f"cam valid {int(np.isfinite(flex_s).sum())}/{n}  "
          f"overlap frames [{i0},{i1}]  fps~{fps:.2f}")

    print("Pass 2/2: rendering 2x2 video...")
    run_render(frames, flex_s, lat_s, axi_s, mflex, mlat, maxi, t, limits, fps,
               K0, D0, R0_c, t0_c, n)
    print(f"Done -> {OUT_VIDEO}")


if __name__ == "__main__":
    main()
