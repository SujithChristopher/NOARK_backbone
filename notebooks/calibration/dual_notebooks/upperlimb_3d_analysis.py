"""Upper limb 3D motion analysis — dual fisheye camera triangulation.

Pipeline
--------
1. Load msgpack frames from both cameras
2. Run MediaPipe Pose on each frame
3. Undistort 2D landmarks with fisheye model (fisheye.undistortPoints)
4. Triangulate to 3D using stereo calibration (R, T from stereo_calibration.toml)
5. Render 3D skeleton with matplotlib and write to MP4
"""

import cv2
import numpy as np
import msgpack
import msgpack_numpy as mpn
import toml
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
CALIB_TOML = SCRIPT_DIR / "stereo_calibration.toml"
DATA_DIR   = SCRIPT_DIR.parents[2] / "data" / "dual_data" / "dual_camera_trunk_test"
CAM0_VIDEO = DATA_DIR / "cam0_imx219.msgpack"
CAM1_VIDEO = DATA_DIR / "cam1_ov9281.msgpack"
CAM0_TIMESTAMP = DATA_DIR / "cam0_timestamp.msgpack"
CAM1_TIMESTAMP = DATA_DIR / "cam1_timestamp.msgpack"
OUT_VIDEO    = SCRIPT_DIR / "upperlimb_combined.mp4"
PANEL_HEIGHT = 480   # all three panels scaled to this height

# ---------------------------------------------------------------------------
# MediaPipe upper-body landmark indices
# ---------------------------------------------------------------------------
# 0=nose  11=L-shoulder  12=R-shoulder  13=L-elbow  14=R-elbow
# 15=L-wrist  16=R-wrist
# Hips are excluded here because they are often occluded and unstable in this view.
UPPER_JOINTS = [0, 11, 12, 13, 14, 15, 16]

BONES = [
    (11, 12, "gray"),   # shoulder bar
    (11, 13, "royalblue"), (13, 15, "royalblue"),   # left arm
    (12, 14, "tomato"),    (14, 16, "tomato"),       # right arm
]

# Same in BGR for cv2 drawing
BONES_BGR = [
    (11, 12, (180, 180, 180)),
    (11, 13, (205, 90,  65)),  (13, 15, (205, 90,  65)),   # left arm  (blue)
    (12, 14, (71,  99, 255)),  (14, 16, (71,  99, 255)),   # right arm (red)
]

VIS_THRESH    = 0.5
MODEL_PATH    = SCRIPT_DIR / "pose_landmarker_full.task"

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def load_calib(path):
    d  = toml.load(path)
    K0 = np.array(d["cam0"]["camera_matrix"])
    D0 = np.array(d["cam0"]["dist_coeffs"])   # (1, 4)
    K1 = np.array(d["cam1"]["camera_matrix"])
    D1 = np.array(d["cam1"]["dist_coeffs"])
    R  = np.array(d["stereo"]["R"])            # 3×3
    T  = np.array(d["stereo"]["T"]).reshape(3, 1)   # mm
    return K0, D0, K1, D1, R, T

# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------
def load_all_frames(path):
    frames = []
    with open(path, "rb") as f:
        for frame in msgpack.Unpacker(f, object_hook=mpn.decode):
            frames.append(np.array(frame))
    return frames

def load_timestamps(path):
    """Return a per-frame timestamp series in milliseconds from the msgpack log."""
    stamps = []
    with open(path, "rb") as f:
        for item in msgpack.Unpacker(f, object_hook=mpn.decode):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                raw = item[1]
            else:
                raw = item
            stamps.append(datetime.fromisoformat(str(raw)))
    if not stamps:
        return []
    t0 = stamps[0]
    return [int((ts - t0).total_seconds() * 1000) for ts in stamps]

def estimate_fps(timestamp_ms, fallback=15.0):
    if len(timestamp_ms) < 2:
        return fallback
    deltas = np.diff(np.asarray(timestamp_ms, dtype=np.float64))
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return fallback
    return float(1000.0 / np.median(deltas))

# ---------------------------------------------------------------------------
# Pose detection  (MediaPipe Tasks API, 0.10+)
# ---------------------------------------------------------------------------
def make_pose_detector():
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)

def detect(landmarker, bgr, timestamp_ms):
    """Return {idx: (px, py)} for visible upper-body landmarks."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect_for_video(mp_img, timestamp_ms)
    if not res.pose_landmarks:
        return {}
    h, w = bgr.shape[:2]
    lms = res.pose_landmarks[0]   # first (only) person
    out = {}
    for idx in UPPER_JOINTS:
        lm = lms[idx]
        if lm.visibility >= VIS_THRESH:
            out[idx] = (lm.x * w, lm.y * h)
    return out

# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------
def undistort_pts(pts_px, K, D):
    arr = np.array(pts_px, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.fisheye.undistortPoints(arr, K, D).reshape(-1, 2)

def triangulate(lms0, lms1, K0, D0, K1, D1, R, T):
    """Triangulate common visible landmarks. Returns {idx: [X,Y,Z] mm} in cam0 frame."""
    common = sorted(set(lms0) & set(lms1))
    if len(common) < 2:
        return {}

    pts0 = np.array([lms0[i] for i in common], dtype=np.float64)
    pts1 = np.array([lms1[i] for i in common], dtype=np.float64)

    u0 = undistort_pts(pts0, K0, D0)   # normalized cam0
    u1 = undistort_pts(pts1, K1, D1)   # normalized cam1

    # Projection matrices in normalized space (K already divided out by undistortPoints)
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = np.hstack([R, T])

    pts4d = cv2.triangulatePoints(P0, P1, u0.T, u1.T)  # (4, N)
    pts3d = (pts4d[:3] / pts4d[3]).T                    # (N, 3) mm

    return {idx: pts3d[i] for i, idx in enumerate(common)}

# ---------------------------------------------------------------------------
# 2D overlay drawing
# ---------------------------------------------------------------------------
def draw_2d_skeleton(bgr, lms, label=""):
    out = bgr.copy()
    for a, b, color in BONES_BGR:
        if a in lms and b in lms:
            p = (int(lms[a][0]), int(lms[a][1]))
            q = (int(lms[b][0]), int(lms[b][1]))
            cv2.line(out, p, q, color, 3, cv2.LINE_AA)
    for idx, (px, py) in lms.items():
        cv2.circle(out, (int(px), int(py)), 6, (0, 255, 255), -1, cv2.LINE_AA)
    if label:
        cv2.putText(out, label, (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out

def scale_to_height(img, h):
    """Resize img to height h, keeping aspect ratio."""
    oh, ow = img.shape[:2]
    w = int(ow * h / oh)
    return cv2.resize(img, (w, h))

def make_panel(img, h, label=""):
    panel = scale_to_height(img, h)
    if label:
        cv2.putText(panel, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return panel

# ---------------------------------------------------------------------------
# 3D rendering
# ---------------------------------------------------------------------------
def render_frame(ax, pts3d, frame_num, total):
    ax.cla()
    ax.set_facecolor("black")

    if pts3d:
        xs = np.array([pts3d[i][0] for i in pts3d])
        ys = np.array([pts3d[i][1] for i in pts3d])
        zs = np.array([pts3d[i][2] for i in pts3d])

        # Centre view on skeleton midpoint each frame
        cx, cy, cz = xs.mean(), ys.mean(), zs.mean()
        r = 600  # mm half-range

        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_zlim(cz - r, cz + r)

        # Joints
        ax.scatter(xs, ys, zs, c="cyan", s=50, depthshade=True, zorder=5)

        # Bones
        for a, b, color in BONES:
            if a in pts3d and b in pts3d:
                p, q = pts3d[a], pts3d[b]
                ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
                        color=color, lw=2.5)

        # Invert Y so "up" in image maps to "up" visually
        ax.invert_yaxis()
    else:
        ax.set_xlim(-600, 600)
        ax.set_ylim(-600, 600)
        ax.set_zlim(-600, 600)

    ax.set_xlabel("X (mm)", color="white", labelpad=6)
    ax.set_ylabel("Y (mm)", color="white", labelpad=6)
    ax.set_zlabel("Z (mm)", color="white", labelpad=6)
    ax.tick_params(colors="white")
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
    ax.grid(True, color="gray", alpha=0.3)
    ax.set_title(f"Upper Limb 3D  [{frame_num}/{total}]", color="white", pad=10)
    ax.view_init(elev=15, azim=-60)

def fig_to_bgr(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading calibration...")
    K0, D0, K1, D1, R, T = load_calib(CALIB_TOML)
    print(f"  T={T.ravel().round(1)} mm")

    print("Loading frames...")
    frames0 = load_all_frames(CAM0_VIDEO)
    frames1 = load_all_frames(CAM1_VIDEO)
    n = min(len(frames0), len(frames1))
    print(f"  {n} paired frames")

    ts0_ms = load_timestamps(CAM0_TIMESTAMP)
    ts1_ms = load_timestamps(CAM1_TIMESTAMP)
    if len(ts0_ms) != n or len(ts1_ms) != n:
        print("  Timestamp logs missing or mismatched; falling back to synthetic timing.")
        ts0_ms = [int(i * 1000 / 15) for i in range(n)]
        ts1_ms = [int(i * 1000 / 15) for i in range(n)]

    writer_fps = estimate_fps(ts0_ms)
    print(f"  analysis fps ~{writer_fps:.2f}")

    pose0 = make_pose_detector()
    pose1 = make_pose_detector()

    # matplotlib 3D panel (square)
    fig = plt.figure(figsize=(5, 5), facecolor="black")
    ax  = fig.add_subplot(111, projection="3d", facecolor="black")

    # Determine combined video width from a sample render
    render_frame(ax, {}, 0, n)
    plot_bgr  = fig_to_bgr(fig)
    plot_panel = scale_to_height(plot_bgr, PANEL_HEIGHT)

    # Dummy cam frames to measure panel widths
    dummy0 = scale_to_height(np.zeros((frames0[0].shape[0], frames0[0].shape[1], 3), np.uint8), PANEL_HEIGHT)
    dummy1 = scale_to_height(np.zeros((frames1[0].shape[0], frames1[0].shape[1] if frames1[0].ndim == 3 else frames1[0].shape[1], 3), np.uint8), PANEL_HEIGHT)
    total_w = dummy0.shape[1] + dummy1.shape[1] + plot_panel.shape[1]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_VIDEO), fourcc, writer_fps, (total_w, PANEL_HEIGHT))

    print(f"Processing {n} frames -> {OUT_VIDEO}  ({total_w}x{PANEL_HEIGHT})")
    for i in range(n):
        f0 = frames0[i]
        f1 = frames1[i]
        ts0 = ts0_ms[i]
        ts1 = ts1_ms[i]

        f0_bgr = cv2.cvtColor(f0, cv2.COLOR_RGB2BGR)
        f1_bgr = cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR) if f1.ndim == 2 else f1

        lms0  = detect(pose0, f0_bgr, ts0)
        lms1  = detect(pose1, f1_bgr, ts1)
        pts3d = triangulate(lms0, lms1, K0, D0, K1, D1, R, T)

        # 2D overlays
        ov0 = draw_2d_skeleton(f0_bgr, lms0)
        ov1 = draw_2d_skeleton(f1_bgr, lms1)

        # Scale panels to common height
        p0   = make_panel(ov0, PANEL_HEIGHT, "Cam0 IMX219")
        p1   = make_panel(ov1, PANEL_HEIGHT, "Cam1 OV9281")

        # 3D plot panel
        render_frame(ax, pts3d, i + 1, n)
        p3d = scale_to_height(fig_to_bgr(fig), PANEL_HEIGHT)

        combined = np.concatenate([p0, p1, p3d], axis=1)
        writer.write(combined)

        if i % 50 == 0:
            print(f"  {i}/{n}  3D joints: {len(pts3d)}")

    writer.release()
    pose0.__exit__(None, None, None)
    pose1.__exit__(None, None, None)
    plt.close(fig)
    print(f"Done -> {OUT_VIDEO}")


if __name__ == "__main__":
    main()
