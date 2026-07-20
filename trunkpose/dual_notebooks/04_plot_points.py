# %% Imports

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pd_support import read_rigid_body_csv  # noqa: E402

# %% Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

RECORDING_NAME = "dual_160_trunk_ragav"
RECORDING_DIR = PROJECT_ROOT / "data" / "trunk_july1_2026" / RECORDING_NAME
STEREO_TOML = (
    PROJECT_ROOT
    / "data" / "calibration" / "dual_160"
    / "calib_cz30_dual_v2" / "stereo_calibration.toml"
)
CHARUCO_TOML = (
    PROJECT_ROOT
    / "data" / "trunk_july1_2026"
    / "dual_160_tframe_july1" / "charuco_basis.toml"
)
POSE_MODEL = SCRIPT_DIR / "pose_landmarker_full.task"
OUT_VIDEO = RECORDING_DIR / "upperlimb_3d_charuco.mp4"

# %% Config
CAM_SIZE = (1280, 800)  # (W, H) -- both OV9281
VISIBILITY_THRESH = 0.5
PANEL = 720  # square video panel, px
AXIS_PAD_M = 0.3

# BlazePose landmark indices (MediaPipe PoseLandmarker, 33-point)
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
# optitrack "body:xx" free markers -> matching UPPER_LIMB joint (see pd_support
# .read_rigid_body_csv: "body:re" becomes column prefix "mre", etc)
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
CAM_BONE_COLOR = "white"
MOCAP_COLOR = "orange"

FALLBACK_LIMITS = {"x": (-1, 1), "y": (-1, 1), "z": (-1, 1)}


# %% Calibration / reference frame
def load_stereo(path):
    d = toml.load(path)
    K0 = np.array(d["cam0"]["camera_matrix"])
    D0 = np.array(d["cam0"]["dist_coeffs"][0]).reshape(-1, 1)
    K1 = np.array(d["cam1"]["camera_matrix"])
    D1 = np.array(d["cam1"]["dist_coeffs"][0]).reshape(-1, 1)
    R = np.array(d["stereo"]["R"])
    T = np.array(d["stereo"]["T"]).reshape(3, 1) / 1000.0  # mm -> m
    return K0, D0, K1, D1, R, T


def load_charuco_basis(path):
    d = toml.load(path)
    R0 = np.array(d["cam0"]["rotation"])
    t0 = np.array(d["cam0"]["origin"])
    Rm = np.array(d["mocap"]["rotation"])
    tm = np.array(d["mocap"]["origin"])
    return R0, t0, Rm, tm


def to_board(p_cam0, R0, t0):
    """cam0-frame point (m) -> charuco board frame."""
    return R0.T @ (p_cam0 - t0)


def mocap_to_board(p_world, Rm, tm):
    """optitrack-world point (m) -> charuco board frame."""
    return Rm.T @ (p_world - tm)


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


def nearest_index(times, t):
    j = int(np.searchsorted(times, t))
    cands = [k for k in (j - 1, j) if 0 <= k < len(times)]
    return min(cands, key=lambda k: abs(times[k] - t))


# %% Stereo triangulation (fisheye, meters, cam0 frame)
def triangulate(p0_px, p1_px, K0, D0, K1, D1, R, T):
    n0 = cv2.fisheye.undistortPoints(
        np.array([[p0_px]], dtype=np.float64), K0, D0
    ).reshape(-1, 2).T
    n1 = cv2.fisheye.undistortPoints(
        np.array([[p1_px]], dtype=np.float64), K1, D1
    ).reshape(-1, 2).T
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = np.hstack([R, T])
    h = cv2.triangulatePoints(P0, P1, n0, n1)
    return (h[:3] / h[3]).ravel()


# %% MediaPipe PoseLandmarker
def make_pose_landmarker():
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def landmark_px(lms, idx, W, H):
    if lms is None:
        return None
    lm = lms[idx]
    if lm.visibility < VISIBILITY_THRESH:
        return None
    return (float(np.clip(lm.x * W, 0, W - 1)), float(np.clip(lm.y * H, 0, H - 1)))


# %% Rendering
def render_frame(fig, ax, cam_pts, mocap_pts, limits):
    ax.cla()
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")
    ax.set_xlim(*limits["x"])
    ax.set_ylim(*limits["y"])
    ax.set_zlim(*limits["z"])
    ax.set_xlabel("X (m)", color="gray", fontsize=8)
    ax.set_ylabel("Y (m)", color="gray", fontsize=8)
    ax.set_zlabel("Z (m)", color="gray", fontsize=8)
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")

    for a, b in BONES:
        pa, pb = cam_pts.get(a), cam_pts.get(b)
        if pa is not None and pb is not None:
            pts = np.stack([pa, pb])
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=CAM_BONE_COLOR, lw=2.5)
        ma, mb = mocap_pts.get(a), mocap_pts.get(b)
        if ma is not None and mb is not None:
            pts = np.stack([ma, mb])
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=MOCAP_COLOR, lw=1.5, ls="--")

    for name, xyz in cam_pts.items():
        if xyz is not None:
            ax.scatter(*xyz, c=CAM_JOINT_COLORS[name], s=50, zorder=5, depthshade=False)
    for name, xyz in mocap_pts.items():
        if xyz is not None:
            ax.scatter(*xyz, c=MOCAP_COLOR, marker="x", s=40, zorder=5, depthshade=False)

    # charuco board origin, for reference
    ax.scatter(0, 0, 0, c="red", marker="^", s=80, zorder=6)
    ax.text(0, 0, 0, "board", color="red", fontsize=7)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
    return cv2.cvtColor(buf[..., 1:], cv2.COLOR_RGB2BGR)  # ARGB -> RGB -> BGR


def compute_limits(all_pts, pad=AXIS_PAD_M):
    pts = np.array(all_pts)
    return {
        "x": (pts[:, 0].min() - pad, pts[:, 0].max() + pad),
        "y": (pts[:, 1].min() - pad, pts[:, 1].max() + pad),
        "z": (pts[:, 2].min() - pad, pts[:, 2].max() + pad),
    }


# %% Main
def main():
    print("Loading calibration + reference frame...")
    K0, D0, K1, D1, R_st, T_st = load_stereo(STEREO_TOML)
    R0_c, t0_c, Rm_c, tm_c = load_charuco_basis(CHARUCO_TOML)

    print("Loading mocap...")
    mocap_df, st_time = read_rigid_body_csv(str(RECORDING_DIR / f"{RECORDING_NAME}.csv"))
    mocap_time = np.datetime64(st_time) + (
        mocap_df["seconds"].to_numpy() * 1e6
    ).astype("timedelta64[us]")

    print("Loading camera timestamps...")
    ts0 = load_frame_times(RECORDING_DIR / "cam0_timestamp.msgpack")
    ts1 = load_frame_times(RECORDING_DIR / "cam1_timestamp.msgpack")
    n = min(len(ts0), len(ts1))
    t_ms = ((ts0[:n] - ts0[0]) / np.timedelta64(1, "ms")).astype(np.int64)

    print(f"Pass 1/2: pose detection + triangulation ({n} paired frames)...")
    lmk0 = make_pose_landmarker()
    lmk1 = make_pose_landmarker()
    W, H = CAM_SIZE

    cam_frames = []    # list[dict name -> xyz(board frame) or None]
    mocap_frames = []  # list[dict name -> xyz(board frame) or None]
    all_pts = []

    frame_pairs = iter_frame_pairs(
        RECORDING_DIR / "cam0_frame.msgpack", RECORDING_DIR / "cam1_frame.msgpack"
    )
    for i, (f0, f1) in enumerate(tqdm(frame_pairs, total=n, desc="detect")):
        f0_rgb = cv2.cvtColor(f0, cv2.COLOR_GRAY2RGB) if f0.ndim == 2 else f0
        f1_rgb = cv2.cvtColor(f1, cv2.COLOR_GRAY2RGB) if f1.ndim == 2 else f1
        res0 = lmk0.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=f0_rgb), int(t_ms[i])
        )
        res1 = lmk1.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=f1_rgb), int(t_ms[i])
        )
        lms0 = res0.pose_landmarks[0] if res0.pose_landmarks else None
        lms1 = res1.pose_landmarks[0] if res1.pose_landmarks else None

        cam_pts = {}
        for name, idx in UPPER_LIMB.items():
            p0 = landmark_px(lms0, idx, W, H)
            p1 = landmark_px(lms1, idx, W, H)
            xyz = None
            if p0 is not None and p1 is not None:
                p_cam0 = triangulate(p0, p1, K0, D0, K1, D1, R_st, T_st)
                xyz = to_board(p_cam0, R0_c, t0_c)
                all_pts.append(xyz)
            cam_pts[name] = xyz
        cam_frames.append(cam_pts)

        mi = nearest_index(mocap_time, ts0[i])
        row = mocap_df.iloc[mi]
        mocap_pts = {}
        for name, prefix in MOCAP_MARKERS.items():
            p_world = np.array(
                [row[f"{prefix}_x"], row[f"{prefix}_y"], row[f"{prefix}_z"]], dtype=float
            )
            xyz = None
            if np.isfinite(p_world).all():
                xyz = mocap_to_board(p_world, Rm_c, tm_c)
                all_pts.append(xyz)
            mocap_pts[name] = xyz
        mocap_frames.append(mocap_pts)

    lmk0.__exit__(None, None, None)
    lmk1.__exit__(None, None, None)

    limits = compute_limits(all_pts) if all_pts else FALLBACK_LIMITS
    print(f"Axis limits (board frame, m): {limits}")

    dt = np.diff(ts0[:n]).astype("timedelta64[us]").astype(np.float64) / 1e6
    fps = float(1.0 / np.median(dt[dt > 0])) if (dt > 0).any() else 15.0
    print(f"fps ~= {fps:.2f}")

    print("Pass 2/2: rendering...")
    fig = plt.figure(figsize=(PANEL / 100, PANEL / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_VIDEO), fourcc, fps, (PANEL, PANEL))

    for cam_pts, mocap_pts in tqdm(list(zip(cam_frames, mocap_frames)), desc="render"):
        frame = render_frame(fig, ax, cam_pts, mocap_pts, limits)
        if frame.shape[1::-1] != (PANEL, PANEL):
            frame = cv2.resize(frame, (PANEL, PANEL))
        writer.write(frame)

    writer.release()
    plt.close(fig)
    print(f"Done -> {OUT_VIDEO}")


if __name__ == "__main__":
    main()
