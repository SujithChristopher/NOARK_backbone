"""Upper limb 3D kinematics — dual OV9281 stereo + MediaPipe PoseLandmarker.

Pipeline
--------
1. Load, rectify, compute WLS+EMA depth map  (reuses upperlimb_3d_analysis)
2. MediaPipe PoseLandmarker on rectified cam0
3. For each upper-limb landmark: sample median 3D point from xyz_map patch
4. Smooth per-joint positions with EMA
5. Render: [cam0 skeleton | 3D stick figure] → MP4
6. Write per-frame joint XYZ to CSV
"""

import matplotlib
matplotlib.use("Agg")

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions
from pathlib import Path

from upperlimb_3d_analysis import (
    CALIB_TOML, DATA_DIR, CAM0_VIDEO, CAM1_VIDEO,
    CAM0_TIMESTAMP, CAM1_TIMESTAMP, CAM_SIZE, PANEL_HEIGHT,
    SCRIPT_DIR,
    load_calib, build_rectify_maps, rectify,
    load_all_frames, load_timestamps, estimate_fps,
    TemporalDepthSmoother, compute_disparity,
)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
POSE_MODEL      = SCRIPT_DIR / "pose_landmarker_full.task"
OUT_VIDEO       = DATA_DIR / "upperlimb_3d_kinematics.mp4"
OUT_CSV         = DATA_DIR / "upperlimb_joints.csv"

PATCH_RADIUS     = 9     # px neighborhood around landmark for depth-map fallback
JOINT_EMA_ALPHA  = 0.4   # per-joint position smoother alpha
Z_MAX_MM         = 4000  # discard depth beyond this (likely noise)
WARMUP_FRAMES    = 10    # frames to accumulate before locking axis limits
AXIS_PAD_MM      = 150   # padding added to observed joint range for axis limits
VISIBILITY_THRESH = 0.5  # minimum MediaPipe visibility to trust a landmark

# MediaPipe upper-limb landmark indices
UPPER_LIMB = {
    "L_shoulder": 11, "R_shoulder": 12,
    "L_elbow":    13, "R_elbow":    14,
    "L_wrist":    15, "R_wrist":    16,
}
BONES = [
    ("L_shoulder", "R_shoulder"),
    ("L_shoulder", "L_elbow"), ("L_elbow", "L_wrist"),
    ("R_shoulder", "R_elbow"), ("R_elbow", "R_wrist"),
]

# BGR colours for 2D overlay
JOINT_COLORS_2D = {
    "L_shoulder": (0,  230,  80),
    "R_shoulder": (80, 160, 255),
    "L_elbow":    (0,  200,  60),
    "R_elbow":    (60, 120, 230),
    "L_wrist":    (0,  160,  40),
    "R_wrist":    (40,  80, 200),
}

# Matplotlib colours for 3D scatter
JOINT_COLORS_3D = {
    "L_shoulder": "limegreen",
    "R_shoulder": "dodgerblue",
    "L_elbow":    "green",
    "R_elbow":    "royalblue",
    "L_wrist":    "darkgreen",
    "R_wrist":    "navy",
}

BONE_COLOR_L = "#55ee55"
BONE_COLOR_R = "#5599ff"
BONE_COLOR_C = "#ffffff"

# ---------------------------------------------------------------------------
# PoseLandmarker
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 3D patch sampler
# ---------------------------------------------------------------------------
def sample_joint_3d(xyz_map, px, py, radius=PATCH_RADIUS):
    """Median of valid 3D points in a square patch around (px, py).
    Returns np.array([x, y, z]) in mm, or None if no valid points."""
    H, W = xyz_map.shape[:2]
    y0, y1 = max(0, py - radius), min(H, py + radius + 1)
    x0, x1 = max(0, px - radius), min(W, px + radius + 1)
    patch = xyz_map[y0:y1, x0:x1]
    z = patch[..., 2]
    valid = np.isfinite(z) & (z > 0) & (z < Z_MAX_MM)
    if not valid.any():
        return None
    return np.median(patch[valid], axis=0)


def triangulate_joint(Q, px0, py0, px1, py1):
    """Direct triangulation from matched rectified landmarks via Q matrix.

    Disparity convention matches compute_disparity: d = px0 - px1 > 0.
    Returns np.array([x, y, z]) mm, or None if disparity is invalid.
    """
    d = float(px0 - px1)
    if d <= 0:
        return None
    py_avg = (py0 + py1) / 2.0
    pt_h = Q @ np.array([float(px0), py_avg, d, 1.0], dtype=np.float64)
    if abs(pt_h[3]) < 1e-9:
        return None
    xyz = (pt_h[:3] / pt_h[3]).astype(np.float32)
    if not (0 < xyz[2] < Z_MAX_MM):
        return None
    return xyz


# ---------------------------------------------------------------------------
# Per-joint EMA smoother
# ---------------------------------------------------------------------------
class JointSmoother:
    def __init__(self, alpha=JOINT_EMA_ALPHA):
        self.alpha = alpha
        self._state = {}

    def update(self, name, xyz):
        if xyz is None:
            return self._state.get(name)   # hold last known position
        if name not in self._state:
            self._state[name] = xyz.copy()
            return xyz.copy()
        self._state[name] = self.alpha * xyz + (1 - self.alpha) * self._state[name]
        return self._state[name].copy()

# ---------------------------------------------------------------------------
# Back-projection: 3D → cam0 pixel via Q matrix
# ---------------------------------------------------------------------------
def make_projector(Q):
    """Return a function xyz → (u, v) using cam0 intrinsics extracted from Q."""
    f  = float(Q[2, 3])
    cx = float(-Q[0, 3])
    cy = float(-Q[1, 3])
    W, H = CAM_SIZE

    def project(xyz):
        if xyz is None:
            return None
        X, Y, Z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        if Z <= 0:
            return None
        u = int(round(f * X / Z + cx))
        v = int(round(f * Y / Z + cy))
        if 0 <= u < W and 0 <= v < H:
            return (u, v)
        return None

    return project


def reproj_errors(lm_pixels, joints_3d, project):
    """Return dict name -> pixel error between MediaPipe landmark and reprojected 3D point."""
    errs = {}
    for name, pt2d in lm_pixels.items():
        xyz = joints_3d.get(name)
        if pt2d and xyz is not None:
            rp = project(xyz)
            if rp:
                errs[name] = float(np.hypot(pt2d[0] - rp[0], pt2d[1] - rp[1]))
    return errs


def bone_lengths(joints_3d):
    """Return dict bone_label -> length_mm for all valid bones."""
    lengths = {}
    for a, b in BONES:
        xa, xb = joints_3d.get(a), joints_3d.get(b)
        if xa is not None and xb is not None:
            lengths[f"{a}-{b}"] = float(np.linalg.norm(xa - xb))
    return lengths


# ---------------------------------------------------------------------------
# 2D skeleton overlay
# ---------------------------------------------------------------------------
def draw_skeleton_2d(bgr, lm_pixels, joints_3d=None, project=None):
    """lm_pixels: dict name -> (px, py) or None.
    If joints_3d and project provided, draws back-projected 3D positions as crosses."""
    out = bgr.copy()
    for (a, b) in BONES:
        pa, pb = lm_pixels.get(a), lm_pixels.get(b)
        if pa and pb:
            cv2.line(out, pa, pb, (255, 255, 255), 2, cv2.LINE_AA)
    for name, pt in lm_pixels.items():
        if pt:
            cv2.circle(out, pt, 8, JOINT_COLORS_2D[name], -1, cv2.LINE_AA)
            cv2.circle(out, pt, 8, (255, 255, 255), 1, cv2.LINE_AA)
    # Back-projected 3D positions — yellow cross; should overlap circles if geometry is correct
    if joints_3d and project:
        for name, xyz in joints_3d.items():
            rp = project(xyz)
            if rp:
                s = 6
                cv2.line(out, (rp[0]-s, rp[1]), (rp[0]+s, rp[1]), (0, 255, 255), 2, cv2.LINE_AA)
                cv2.line(out, (rp[0], rp[1]-s), (rp[0], rp[1]+s), (0, 255, 255), 2, cv2.LINE_AA)
    return out

# ---------------------------------------------------------------------------
# 3D stick figure renderer
# ---------------------------------------------------------------------------
def _bone_color(a, b):
    if a.startswith("L") and b.startswith("L"):
        return BONE_COLOR_L
    if a.startswith("R") and b.startswith("R"):
        return BONE_COLOR_R
    return BONE_COLOR_C

def render_stick3d(fig, ax, joints_3d, axis_limits, target_h=PANEL_HEIGHT):
    ax.cla()
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")
    ax.set_xlim(*axis_limits["x"])
    ax.set_ylim(*axis_limits["y"])
    ax.set_zlim(*axis_limits["z"])
    ax.set_xlabel("X mm", color="gray", fontsize=7)
    ax.set_ylabel("Y mm", color="gray", fontsize=7)
    ax.set_zlabel("Z mm", color="gray", fontsize=7)
    ax.tick_params(colors="gray", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")

    for a, b in BONES:
        pa, pb = joints_3d.get(a), joints_3d.get(b)
        if pa is not None and pb is not None:
            pts = np.stack([pa, pb])
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=_bone_color(a, b), lw=2.5)

    for name, xyz in joints_3d.items():
        if xyz is not None:
            ax.scatter(xyz[0], xyz[1], xyz[2],
                       c=JOINT_COLORS_3D[name], s=60, zorder=5, depthshade=False)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
    bgr = cv2.cvtColor(buf[..., 1:], cv2.COLOR_RGB2BGR)  # ARGB → RGB → BGR
    if h != target_h:
        bgr = cv2.resize(bgr, (target_h, target_h))
    return bgr

# ---------------------------------------------------------------------------
# Axis limits from accumulated joint positions
# ---------------------------------------------------------------------------
def compute_axis_limits(all_xyz, pad=AXIS_PAD_MM):
    pts = np.array(all_xyz)
    return {
        "x": (pts[:, 0].min() - pad, pts[:, 0].max() + pad),
        "y": (pts[:, 1].min() - pad, pts[:, 1].max() + pad),
        "z": (pts[:, 2].min() - pad, pts[:, 2].max() + pad),
    }

FALLBACK_LIMITS = {"x": (-500, 500), "y": (-600, 300), "z": (200, 3500)}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading calibration...")
    K0, D0, K1, D1, R, T = load_calib(CALIB_TOML)
    print(f"  T={T.ravel().round(1)} mm")

    maps0, maps1, Q = build_rectify_maps(K0, D0, K1, D1, R, T, CAM_SIZE)
    project = make_projector(Q)
    print("  Rectification maps built.")

    print("Loading frames...")
    frames0 = load_all_frames(CAM0_VIDEO)
    frames1 = load_all_frames(CAM1_VIDEO)
    n = min(len(frames0), len(frames1))
    print(f"  {n} paired frames")

    ts0_ms = load_timestamps(CAM0_TIMESTAMP)
    ts1_ms = load_timestamps(CAM1_TIMESTAMP)
    if len(ts0_ms) != n or len(ts1_ms) != n:
        print("  Timestamp mismatch — using synthetic 15 fps.")
        ts0_ms = [int(i * 1000 / 15) for i in range(n)]
        ts1_ms = list(ts0_ms)

    writer_fps = estimate_fps(ts0_ms)
    print(f"  fps ~{writer_fps:.2f}")

    depth_smoother = TemporalDepthSmoother()
    joint_smoother = JointSmoother()
    lmk0           = make_pose_landmarker()   # cam0
    lmk1           = make_pose_landmarker()   # cam1

    W, H = CAM_SIZE
    stick_w     = PANEL_HEIGHT              # square 3D panel
    cam_panel_w = int(W * PANEL_HEIGHT / H)
    total_w     = cam_panel_w + stick_w

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_VIDEO), fourcc, writer_fps, (total_w, PANEL_HEIGHT))
    print(f"Output: {OUT_VIDEO}  ({total_w}x{PANEL_HEIGHT})")

    fig = plt.figure(figsize=(4.8, 4.8), dpi=100)
    ax  = fig.add_subplot(111, projection="3d")

    axis_limits    = None
    warmup_xyz_acc = []
    csv_rows       = []

    for i in range(n):
        f0 = frames0[i]
        f1 = frames1[i]
        f0_bgr = cv2.cvtColor(f0, cv2.COLOR_GRAY2BGR) if f0.ndim == 2 else f0
        f1_bgr = cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR) if f1.ndim == 2 else f1

        # Rectify
        r0      = rectify(f0_bgr, maps0)
        r1      = rectify(f1_bgr, maps1)
        r0_gray = cv2.cvtColor(r0, cv2.COLOR_BGR2GRAY)
        r1_gray = cv2.cvtColor(r1, cv2.COLOR_BGR2GRAY)

        # Depth
        disp_raw = compute_disparity(r0_gray, r1_gray)
        disp     = depth_smoother.update(disp_raw)
        xyz_map  = cv2.reprojectImageTo3D(disp, Q)   # (H, W, 3) mm

        # Pose landmarks on both rectified frames
        r0_rgb  = cv2.cvtColor(r0, cv2.COLOR_BGR2RGB)
        r1_rgb  = cv2.cvtColor(r1, cv2.COLOR_BGR2RGB)
        res0    = lmk0.detect_for_video(
                      mp.Image(image_format=mp.ImageFormat.SRGB, data=r0_rgb), ts0_ms[i])
        res1    = lmk1.detect_for_video(
                      mp.Image(image_format=mp.ImageFormat.SRGB, data=r1_rgb), ts1_ms[i])

        lms0 = res0.pose_landmarks[0] if res0.pose_landmarks else None
        lms1 = res1.pose_landmarks[0] if res1.pose_landmarks else None

        lm_pixels = {}   # name -> (px, py) from cam0 — for 2D overlay
        joints_3d = {}   # name -> smoothed xyz mm

        for name, idx in UPPER_LIMB.items():
            # Extract per-camera detections + visibility
            p0 = p1 = None
            if lms0:
                lm = lms0[idx]
                if lm.visibility >= VISIBILITY_THRESH:
                    p0 = (int(np.clip(lm.x * W, 0, W - 1)),
                          int(np.clip(lm.y * H, 0, H - 1)))
                    lm_pixels[name] = p0   # always use cam0 for 2D overlay
            if lms1:
                lm = lms1[idx]
                if lm.visibility >= VISIBILITY_THRESH:
                    p1 = (int(np.clip(lm.x * W, 0, W - 1)),
                          int(np.clip(lm.y * H, 0, H - 1)))

            # 3D: triangulate if both cameras see the joint, else depth-map fallback
            if p0 and p1:
                xyz_raw = triangulate_joint(Q, p0[0], p0[1], p1[0], p1[1])
                if xyz_raw is None:   # bad disparity — try depth map
                    xyz_raw = sample_joint_3d(xyz_map, p0[0], p0[1])
            elif p0:
                xyz_raw = sample_joint_3d(xyz_map, p0[0], p0[1])
            elif p1:
                xyz_raw = sample_joint_3d(xyz_map, p1[0], p1[1])
            else:
                xyz_raw = None

            xyz_smooth      = joint_smoother.update(name, xyz_raw)
            joints_3d[name] = xyz_smooth
            if xyz_smooth is not None:
                warmup_xyz_acc.append(xyz_smooth)
                csv_rows.append((i, ts0_ms[i], name,
                                 xyz_smooth[0], xyz_smooth[1], xyz_smooth[2]))

        # Lock axis limits after warmup
        if axis_limits is None and i >= WARMUP_FRAMES - 1:
            if warmup_xyz_acc:
                axis_limits = compute_axis_limits(warmup_xyz_acc)
                print(f"  Axis limits set from {len(warmup_xyz_acc)} warmup points: {axis_limits}")
            else:
                axis_limits = FALLBACK_LIMITS
                print("  No valid joints in warmup — using fallback axis limits.")

        limits = axis_limits if axis_limits is not None else FALLBACK_LIMITS

        # Build panels
        cam_panel = draw_skeleton_2d(r0, lm_pixels, joints_3d, project)
        cam_panel = cv2.resize(cam_panel, (cam_panel_w, PANEL_HEIGHT))
        cv2.putText(cam_panel, "Cam0 rectified", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        stick_panel = render_stick3d(fig, ax, joints_3d, limits, PANEL_HEIGHT)
        cv2.putText(stick_panel, "3D kinematics (mm)", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        combined = np.concatenate([cam_panel, stick_panel], axis=1)
        writer.write(combined)

        if i % 50 == 0:
            valid_joints = sum(1 for v in joints_3d.values() if v is not None)
            errs = reproj_errors(lm_pixels, joints_3d, project)
            blens = bone_lengths(joints_3d)
            err_str  = "  ".join(f"{k.split('_')[1][:3]}={v:.1f}px" for k, v in errs.items())
            blen_str = "  ".join(f"{k.split('-')[0][-3:]}-{k.split('-')[1][-3:]}={v:.0f}mm"
                                 for k, v in blens.items())
            print(f"  {i:4d}/{n}  joints={valid_joints}/{len(UPPER_LIMB)}"
                  f"  reproj_err: {err_str}")
            print(f"         bone_len: {blen_str}")

    writer.release()
    plt.close(fig)
    lmk0.__exit__(None, None, None)
    lmk1.__exit__(None, None, None)

    # Write CSV
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "timestamp_ms", "joint", "x_mm", "y_mm", "z_mm"])
        for row in csv_rows:
            w.writerow([row[0], row[1], row[2],
                        f"{row[3]:.2f}", f"{row[4]:.2f}", f"{row[5]:.2f}"])

    print(f"Done -> {OUT_VIDEO}")
    print(f"CSV  -> {OUT_CSV}  ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
