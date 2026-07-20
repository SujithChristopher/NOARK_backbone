# %% 05_create_plane.py
# Estimate the torso as a 3D plane from the first N dual-camera frames.
#
# Per frame pair: YOLO-seg the torso in cam0, dense-rectify both fisheye views,
# StereoSGBM disparity -> 3D point cloud (cam0-rectified frame), keep only points
# inside the torso mask at plausible depth. Pool the cloud over the first N frames,
# RANSAC-fit a plane, and draw a 10x10 cm patch of that plane together with the
# MediaPipe upper-limb stick figure -- all in the charuco board frame.

import sys
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
import toml
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision
from tqdm.auto import tqdm
from ultralytics import YOLO

# %% Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

RECORDING_NAME = "dual_160_trunk_ragav"
RECORDING_DIR = PROJECT_ROOT / "data" / "trunk_july1_2026" / RECORDING_NAME
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
OUT_PNG = RECORDING_DIR / "torso_plane_charuco.png"

# %% Config
CAM_SIZE = (1280, 800)  # (W, H) -- both OV9281
N_FRAMES = 15           # pool the torso cloud over the first N frame pairs
SEG_CONF = 0.25
DEPTH_MIN_M, DEPTH_MAX_M = 0.2, 3.0  # reject implausible triangulated depth
PLANE_HALF_M = 0.075    # 15x15 cm patch -> half-side 7.5 cm
RANSAC_ITERS = 800
RANSAC_THRESH_M = 0.012  # inlier band, +/-1.2 cm
VISIBILITY_THRESH = 0.5

# SGBM (tuned for a ~78 mm baseline torso at ~0.5-1.5 m)
SGBM_MIN_DISP = 0
SGBM_NUM_DISP = 160  # must be divisible by 16
SGBM_BLOCK = 5

# BlazePose landmark indices (33-point)
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


# %% Calibration / basis loaders (same layout as 04_plot_points.py)
def load_stereo(path):
    d = toml.load(path)
    K0 = np.array(d["cam0"]["camera_matrix"])
    D0 = np.array(d["cam0"]["dist_coeffs"][0], dtype=np.float64).reshape(4, 1)
    K1 = np.array(d["cam1"]["camera_matrix"])
    D1 = np.array(d["cam1"]["dist_coeffs"][0], dtype=np.float64).reshape(4, 1)
    R = np.array(d["stereo"]["R"])
    T = np.array(d["stereo"]["T"], dtype=np.float64).reshape(3, 1)  # mm
    return K0, D0, K1, D1, R, T


def load_charuco_basis(path):
    d = toml.load(path)
    R0 = np.array(d["cam0"]["rotation"])
    t0 = np.array(d["cam0"]["origin"])
    return R0, t0


def to_board(p_cam0, R0, t0):
    """cam0-frame point(s) (m) -> charuco board frame. Accepts (3,) or (N,3)."""
    return (p_cam0 - t0) @ R0  # == (R0.T @ (p - t0)) row-wise


# %% Frame / timestamp loading
def iter_frame_pairs(path0, path1):
    with open(path0, "rb") as f0, open(path1, "rb") as f1:
        u0 = msgpack.Unpacker(f0, object_hook=mpn.decode)
        u1 = msgpack.Unpacker(f1, object_hook=mpn.decode)
        yield from zip(u0, u1)


def load_frame_times(path):
    with open(path, "rb") as f:
        recs = list(msgpack.Unpacker(f, object_hook=mpn.decode))
    return np.array(
        [datetime.fromisoformat(r[1]) for r in recs], dtype="datetime64[us]"
    )


# %% Stereo rectification (fisheye)
def build_rectification(K0, D0, K1, D1, R, T, size):
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K0, D0, K1, D1, size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, balance=0.0, fov_scale=1.0,
    )
    m0x, m0y = cv2.fisheye.initUndistortRectifyMap(K0, D0, R1, P1, size, cv2.CV_16SC2)
    m1x, m1y = cv2.fisheye.initUndistortRectifyMap(K1, D1, R2, P2, size, cv2.CV_16SC2)
    return R1, Q, (m0x, m0y), (m1x, m1y)


def make_sgbm():
    return cv2.StereoSGBM_create(
        minDisparity=SGBM_MIN_DISP,
        numDisparities=SGBM_NUM_DISP,
        blockSize=SGBM_BLOCK,
        P1=8 * SGBM_BLOCK ** 2,
        P2=32 * SGBM_BLOCK ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def torso_cloud(gray0, gray1, mask0, sgbm, maps0, maps1, R1, Q):
    """Return (M,3) torso points in the cam0 (unrectified) frame, meters."""
    rect0 = cv2.remap(gray0, *maps0, cv2.INTER_LINEAR)
    rect1 = cv2.remap(gray1, *maps1, cv2.INTER_LINEAR)
    mrect = cv2.remap(mask0, *maps0, cv2.INTER_NEAREST)

    disp = sgbm.compute(rect0, rect1).astype(np.float32) / 16.0
    pts_rect = cv2.reprojectImageTo3D(disp, Q)  # (H,W,3) in mm (Q built from mm T)

    valid = (
        (disp > SGBM_MIN_DISP)
        & (mrect > 0)
        & np.isfinite(pts_rect).all(axis=2)
    )
    p = pts_rect[valid] / 1000.0  # mm -> m, in rectified-cam0 frame
    z = p[:, 2]
    p = p[(z > DEPTH_MIN_M) & (z < DEPTH_MAX_M)]
    if len(p) == 0:
        return p
    return p @ R1  # rectified -> cam0 frame  (X_cam0 = R1.T @ X_rect)


# %% Torso segmentation (cam0)
def torso_mask(seg_model, gray, size):
    W, H = size
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) if gray.ndim == 2 else gray
    res = seg_model.predict(rgb, conf=SEG_CONF, verbose=False)[0]
    mask = np.zeros((H, W), dtype=np.uint8)
    if res.masks is None:
        return mask
    for m in res.masks.data.cpu().numpy():
        mr = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        mask[mr > 0.5] = 255
    return mask


# %% MediaPipe upper-limb joints (reused approach from 04)
def make_pose_landmarker():
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def landmark_px(lms, idx, W, H):
    if lms is None:
        return None
    lm = lms[idx]
    if lm.visibility < VISIBILITY_THRESH:
        return None
    return (float(np.clip(lm.x * W, 0, W - 1)), float(np.clip(lm.y * H, 0, H - 1)))


def triangulate(p0_px, p1_px, K0, D0, K1, D1, R, T):
    n0 = cv2.fisheye.undistortPoints(
        np.array([[p0_px]], dtype=np.float64), K0, D0
    ).reshape(-1, 2).T
    n1 = cv2.fisheye.undistortPoints(
        np.array([[p1_px]], dtype=np.float64), K1, D1
    ).reshape(-1, 2).T
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = np.hstack([R, T / 1000.0])  # T mm -> m so the joints come out in meters
    h = cv2.triangulatePoints(P0, P1, n0, n1)
    if abs(h[3, 0]) < 1e-9:
        return None
    p = (h[:3] / h[3]).ravel()
    if not (DEPTH_MIN_M < p[2] < DEPTH_MAX_M):
        return None
    return p


# %% Plane fit
def fit_plane_ransac(pts, iters=RANSAC_ITERS, thresh=RANSAC_THRESH_M, rng=None):
    """RANSAC plane. Returns (centroid, unit_normal) refined by SVD on inliers."""
    rng = rng or np.random.default_rng(0)
    n = len(pts)
    best_inliers = None
    best_count = 0
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        a, b, c = pts[idx]
        nrm = np.cross(b - a, c - a)
        norm = np.linalg.norm(nrm)
        if norm < 1e-9:
            continue
        nrm = nrm / norm
        d = np.abs((pts - a) @ nrm)
        inl = d < thresh
        if inl.sum() > best_count:
            best_count = int(inl.sum())
            best_inliers = inl
    inliers = pts[best_inliers]
    centroid = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vt[2] / np.linalg.norm(vt[2])
    return centroid, normal, best_inliers


def plane_axes(normal):
    """Two orthonormal in-plane axes (u, v) spanning the plane with given normal."""
    ref = np.array([1.0, 0.0, 0.0])
    if abs(normal @ ref) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def plane_square(centroid, normal, half=PLANE_HALF_M):
    """4 corners of a `2*half` square lying in the plane, centered at centroid."""
    u, v = plane_axes(normal)
    return np.array([
        centroid + half * (u + v),
        centroid + half * (u - v),
        centroid + half * (-u - v),
        centroid + half * (-u + v),
    ])


def plane_grid(centroid, normal, half=PLANE_HALF_M, n=12):
    """Filled (n x n) mesh of the plane patch for a solid, gap-free surface."""
    u, v = plane_axes(normal)
    s = np.linspace(-half, half, n)
    a, b = np.meshgrid(s, s)
    pts = centroid + a[..., None] * u + b[..., None] * v  # (n, n, 3)
    return pts[..., 0], pts[..., 1], pts[..., 2]


# %% Main
def main():
    print("Loading calibration + basis + models...")
    K0, D0, K1, D1, R_st, T_st = load_stereo(STEREO_TOML)
    R0_c, t0_c = load_charuco_basis(CHARUCO_TOML)
    seg_model = YOLO(str(SEG_WEIGHTS))
    R1, Q, maps0, maps1 = build_rectification(K0, D0, K1, D1, R_st, T_st, CAM_SIZE)
    sgbm = make_sgbm()

    ts0 = load_frame_times(RECORDING_DIR / "cam0_timestamp.msgpack")
    t_ms = ((ts0 - ts0[0]) / np.timedelta64(1, "ms")).astype(np.int64)

    lmk0 = make_pose_landmarker()
    lmk1 = make_pose_landmarker()
    W, H = CAM_SIZE

    cloud = []            # torso points, cam0 frame (m)
    joint_samples = {k: [] for k in UPPER_LIMB}  # cam0-frame joint samples

    pairs = iter_frame_pairs(
        RECORDING_DIR / "cam0_frame.msgpack", RECORDING_DIR / "cam1_frame.msgpack"
    )
    print(f"Pass: torso cloud + joints over first {N_FRAMES} frames...")
    for i, (f0, f1) in enumerate(tqdm(pairs, total=N_FRAMES)):
        if i >= N_FRAMES:
            break
        g0 = f0 if f0.ndim == 2 else cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY)
        g1 = f1 if f1.ndim == 2 else cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)

        m0 = torso_mask(seg_model, g0, CAM_SIZE)
        if m0.any():
            pc = torso_cloud(g0, g1, m0, sgbm, maps0, maps1, R1, Q)
            if len(pc):
                cloud.append(pc)

        r0 = lmk0.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(g0, cv2.COLOR_GRAY2RGB)), int(t_ms[i]))
        r1 = lmk1.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(g1, cv2.COLOR_GRAY2RGB)), int(t_ms[i]))
        lms0 = r0.pose_landmarks[0] if r0.pose_landmarks else None
        lms1 = r1.pose_landmarks[0] if r1.pose_landmarks else None
        for name, idx in UPPER_LIMB.items():
            p0 = landmark_px(lms0, idx, W, H)
            p1 = landmark_px(lms1, idx, W, H)
            if p0 and p1:
                p = triangulate(p0, p1, K0, D0, K1, D1, R_st, T_st)
                if p is not None:
                    joint_samples[name].append(p)

    lmk0.__exit__(None, None, None)
    lmk1.__exit__(None, None, None)

    if not cloud:
        raise SystemExit("No torso points recovered -- check seg masks / disparity.")
    cloud = np.vstack(cloud)
    print(f"Pooled torso points: {len(cloud)}")

    # cam0 frame -> board frame
    cloud_b = to_board(cloud, R0_c, t0_c)
    centroid_b, normal_b, inliers = fit_plane_ransac(cloud_b)
    print(f"Plane inliers: {int(inliers.sum())}/{len(cloud_b)}  "
          f"normal={np.round(normal_b, 3)}")
    square_b = plane_square(centroid_b, normal_b)

    joints_b = {}
    for name, samples in joint_samples.items():
        if samples:
            med = np.median(np.array(samples), axis=0)
            joints_b[name] = to_board(med, R0_c, t0_c)

    render(cloud_b, inliers, centroid_b, normal_b, square_b, joints_b)
    print(f"Done -> {OUT_PNG}")


DISPLAY_MAX_PTS = 6000  # subsample cloud for display so the plane shows through


def _subsample(mask, rng):
    idx = np.flatnonzero(mask)
    if len(idx) > DISPLAY_MAX_PTS:
        idx = rng.choice(idx, DISPLAY_MAX_PTS, replace=False)
    return idx


def render(cloud_b, inliers, centroid, normal, square, joints):
    rng = np.random.default_rng(0)
    out_idx = _subsample(~inliers, rng)
    in_idx = _subsample(inliers, rng)
    fig = plt.figure(figsize=(14, 7))
    for si, (elev, azim) in enumerate([(18, -60), (18, 30)]):
        ax = fig.add_subplot(1, 2, si + 1, projection="3d")
        ax.scatter(*cloud_b[out_idx].T, s=3, c="dimgray", alpha=0.25, label="torso cloud")
        ax.scatter(*cloud_b[in_idx].T, s=3, c="deepskyblue", alpha=0.35, label="plane inliers")

        gx, gy, gz = plane_grid(centroid, normal)
        ax.plot_surface(gx, gy, gz, color="orange", alpha=1.0, shade=False,
                        edgecolor="none", zorder=10)
        edge = np.vstack([square, square[0]])
        ax.plot(*edge.T, color="black", lw=2, zorder=11)
        ax.quiver(*centroid, *(normal * 0.10), color="red", lw=2.5, zorder=12)

        for a, b in BONES:
            if a in joints and b in joints:
                p = np.stack([joints[a], joints[b]])
                ax.plot(*p.T, c="white", lw=2.5)
        for name, xyz in joints.items():
            ax.scatter(*xyz, c="lime", s=45, depthshade=False)

        ax.scatter(0, 0, 0, c="red", marker="^", s=60)
        ax.text(0, 0, 0, "board", color="red", fontsize=7)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.set_title(f"view {si + 1}")
        if si == 0:
            ax.legend(loc="upper right", fontsize=7)
        ax.view_init(elev=elev, azim=azim)
        set_equal_aspect(ax, cloud_b, square, joints)

    fig.suptitle("Torso plane (RANSAC) + upper-limb joints — charuco board frame")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)


def set_equal_aspect(ax, cloud_b, square, joints):
    pts = [cloud_b, square]
    if joints:
        pts.append(np.array(list(joints.values())))
    allp = np.vstack(pts)
    c = allp.mean(axis=0)
    r = np.abs(allp - c).max() * 1.1
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


if __name__ == "__main__":
    main()
