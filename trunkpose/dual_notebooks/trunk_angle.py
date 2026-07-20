"""Realtime trunk angle — dual OV9281 stereo + MediaPipe, closed-form (no SMPL).

Objective
---------
Track trunk orientation (flexion/extension, lateral bend, axial rotation) at
5-10+ fps. The trunk is a rigid segment defined by 4 points: L/R shoulders and
L/R hips. We stereo-triangulate them, build an orthonormal torso frame, and read
off three angles relative to gravity (+ a captured neutral reference for the
axial zero). No mesh, no per-frame optimization, no torch.

Why not SMPL-X
--------------
A trunk angle is the orientation of ONE segment. SMPL-X spent ~99% of its compute
fitting a full-body mesh we then threw away. The model's only real contribution
was regularizing 4 noisy points into a consistent rigid body — which we get here
in closed form (orthonormal frame + temporal EMA).

Gravity
-------
Without an IMU the cameras don't know "down". Two modes (GRAVITY_MODE):
  "camera"    : assume a level mount; rectified camera up-axis (-Y) is vertical.
  "reference" : use the captured neutral torso long-axis as vertical.
Either way the first REF_FRAMES valid frames define the neutral reference that
zeroes all three angles (axial rotation has no absolute zero vs gravity).

Run
---
    uv run python notebooks/calibration/dual_notebooks/trunk_angle.py
"""

import csv
import cv2
import numpy as np
import mediapipe as mp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from upperlimb_3d_analysis import (
    CALIB_TOML, DATA_DIR, CAM0_VIDEO, CAM1_VIDEO,
    CAM0_TIMESTAMP, CAM1_TIMESTAMP, CAM_SIZE, PANEL_HEIGHT,
    load_calib, build_rectify_maps, rectify,
    load_all_frames, load_timestamps, estimate_fps,
)
from upperlimb_3d_kinematics import (
    make_pose_landmarker, triangulate_joint, make_projector,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUT_VIDEO = DATA_DIR / "trunk_angle.mp4"
OUT_CSV   = DATA_DIR / "trunk_angle.csv"

GRAVITY_MODE      = "camera"    # "camera" (level mount) or "reference" (upright capture)
REF_FRAMES        = 15          # neutral-reference window (subject upright/neutral at start)
VISIBILITY_THRESH = 0.5
ANGLE_EMA_ALPHA   = 0.3         # lower = smoother, more lag
Z_MAX_MM          = 4000

# Pelvis anchor — handles occluded hips (seated behind a desk). When both hips
# are visible the 3D hip midpoint is captured and slowly EMA-updated; when they
# are occluded the anchor stands in so the trunk angle keeps tracking.
USE_HIP_ANCHOR    = True
HIP_ANCHOR_ALPHA  = 0.05        # slow update -> tracks chair drift, ignores jitter

# MediaPipe torso landmark indices (BlazePose 33)
LSH, RSH, LHIP, RHIP = 11, 12, 23, 24

# Camera (rectified, OpenCV) axes
CAM_RIGHT = np.array([1.0, 0.0, 0.0])
CAM_DOWN  = np.array([0.0, 1.0, 0.0])
CAM_FWD   = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def landmark_px(lms, idx, W, H):
    if lms is None:
        return None
    lm = lms[idx]
    if lm.visibility < VISIBILITY_THRESH:
        return None
    return (int(np.clip(lm.x * W, 0, W - 1)), int(np.clip(lm.y * H, 0, H - 1)))


def triangulate_torso(lms0, lms1, Q, W, H):
    """Return dict of the 4 torso points in mm, only those visible in both cams."""
    pts = {}
    for name, idx in (("Lsh", LSH), ("Rsh", RSH), ("Lhip", LHIP), ("Rhip", RHIP)):
        p0 = landmark_px(lms0, idx, W, H)
        p1 = landmark_px(lms1, idx, W, H)
        if not (p0 and p1):
            continue
        xyz = triangulate_joint(Q, p0[0], p0[1], p1[0], p1[1])
        if xyz is not None:
            pts[name] = xyz.astype(np.float64)
    return pts


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def hip_mid_from(pts, hip_anchor):
    """Measured hip midpoint if both hips visible, else the pelvis anchor.

    Returns (hip_mid, estimated_bool) or (None, False) if neither available.
    """
    if "Lhip" in pts and "Rhip" in pts:
        return (pts["Lhip"] + pts["Rhip"]) / 2.0, False
    if hip_anchor is not None:
        return hip_anchor.copy(), True
    return None, False


def torso_frame(pts, hip_anchor=None):
    """Build an orthonormal torso rotation (cam coords).

    Needs both shoulders. Hip midpoint is measured when both hips are visible,
    otherwise falls back to `hip_anchor` (occluded-hip case, e.g. seated at desk).
    Columns: [mediolateral (L->R), longitudinal (up the spine), anteroposterior].
    Returns (R (3,3), sh_mid, hip_mid, hip_estimated) or None.
    """
    if "Lsh" not in pts or "Rsh" not in pts:
        return None
    sh_mid = (pts["Lsh"] + pts["Rsh"]) / 2.0
    hip_mid, hip_est = hip_mid_from(pts, hip_anchor)
    if hip_mid is None:
        return None

    u = _normalize(sh_mid - hip_mid)                    # trunk long axis (up)
    m = _normalize(pts["Rsh"] - pts["Lsh"])             # mediolateral (raw)
    if u is None or m is None:
        return None
    a = _normalize(np.cross(u, m))                      # anteroposterior (forward)
    if a is None:
        return None
    m = np.cross(a, u)                                  # re-orthogonalize
    R = np.column_stack([m, u, a])
    return R, sh_mid, hip_mid, hip_est


def signed_angle(v_from, v_to, axis):
    """Signed angle (deg) rotating v_from -> v_to about `axis` (right-hand)."""
    v_from = _normalize(v_from)
    v_to   = _normalize(v_to)
    if v_from is None or v_to is None:
        return 0.0
    c = np.clip(np.dot(v_from, v_to), -1.0, 1.0)
    s = np.dot(axis, np.cross(v_from, v_to))
    return float(np.degrees(np.arctan2(s, c)))


def trunk_angles(R, vup, m_ref_h):
    """Three trunk angles (deg) relative to gravity `vup` and axial reference.

    flexion+ = lean forward, lateral+ = lean to subject's right (cam right),
    axial+   = right-hand rotation of shoulders about vertical vs reference.
    """
    u = R[:, 1]   # trunk long axis
    m = R[:, 0]   # mediolateral (shoulder line)

    up_comp  = np.dot(u, vup)
    fwd_comp = np.dot(u, CAM_FWD)
    rgt_comp = np.dot(u, CAM_RIGHT)
    flexion = np.degrees(np.arctan2(fwd_comp, up_comp))
    lateral = np.degrees(np.arctan2(rgt_comp, up_comp))

    # Axial: yaw of shoulder line projected onto the horizontal (perp to vup)
    m_h = m - np.dot(m, vup) * vup
    axial = signed_angle(m_ref_h, m_h, vup) if m_ref_h is not None else 0.0
    return float(flexion), float(lateral), float(axial)


class EMA:
    def __init__(self, alpha):
        self.alpha = alpha
        self._s = None

    def update(self, x):
        x = np.asarray(x, np.float64)
        self._s = x if self._s is None else self.alpha * x + (1 - self.alpha) * self._s
        return self._s.copy()


# ---------------------------------------------------------------------------
# Visualization (cheap 2D overlay)
# ---------------------------------------------------------------------------
def torso_px(lms, W, H):
    """2D pixel coords of the 4 torso landmarks from one camera's detection."""
    return {k: landmark_px(lms, idx, W, H)
            for k, idx in (("Lsh", LSH), ("Rsh", RSH), ("Lhip", LHIP), ("Rhip", RHIP))}


def draw_torso(bgr, px, label, angles=None, seg3d=None, project=None, hip_est=False):
    """Draw torso landmarks (2D) + optionally the trunk segment from 3D points.

    seg3d=(sh_mid, hip_mid) reprojected via `project` draws the trunk segment even
    when the 2D hip is occluded; an estimated (anchored) hip is drawn red.
    """
    out = bgr.copy()
    for k in ("Lsh", "Rsh", "Lhip", "Rhip"):
        p = px.get(k)
        if p:
            cv2.circle(out, p, 6, (255, 255, 255), -1, cv2.LINE_AA)
    if px.get("Lsh") and px.get("Rsh"):
        cv2.line(out, px["Lsh"], px["Rsh"], (80, 200, 255), 2, cv2.LINE_AA)
    if px.get("Lhip") and px.get("Rhip"):
        cv2.line(out, px["Lhip"], px["Rhip"], (80, 200, 255), 2, cv2.LINE_AA)

    if seg3d is not None and project is not None:
        sh_uv  = project(seg3d[0])
        hip_uv = project(seg3d[1])
        if sh_uv and hip_uv:
            cv2.line(out, sh_uv, hip_uv, (0, 230, 120), 3, cv2.LINE_AA)  # trunk segment
            cv2.circle(out, hip_uv, 7, (0, 80, 255) if hip_est else (0, 230, 120),
                       -1, cv2.LINE_AA)

    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if hip_est:
        cv2.putText(out, "HIP ANCHORED", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2, cv2.LINE_AA)
    if angles is not None:
        f, lat, ax = angles
        y0 = 84 if hip_est else 64
        for j, (lbl, val) in enumerate((("Flex/Ext", f), ("Lateral", lat), ("Axial", ax))):
            cv2.putText(out, f"{lbl:>9}: {val:+6.1f}", (10, y0 + 28 * j),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 255, 160), 2, cv2.LINE_AA)
    return out


# Plot styling
PLOT_WINDOW = 150        # frames visible in the scrolling window
PLOT_YLIM   = (-90, 90)
_SERIES = [("Flex/Ext", "#46e05a"), ("Lateral", "#50b4ff"), ("Axial", "#ff7a46")]


class TrunkPlot:
    """Persistent matplotlib panel: scrolling line plot of the 3 trunk angles."""

    def __init__(self, w, h, window=PLOT_WINDOW):
        self.window = window
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        self.fig.patch.set_facecolor("#111111")
        self.ax.set_facecolor("#111111")
        self.ax.set_ylim(*PLOT_YLIM)
        self.ax.set_xlabel("frame", color="gray", fontsize=8)
        self.ax.set_ylabel("degrees", color="gray", fontsize=8)
        self.ax.tick_params(colors="gray", labelsize=7)
        for s in self.ax.spines.values():
            s.set_edgecolor("gray")
        self.ax.axhline(0, color="#444444", lw=0.8)
        self.lines = [self.ax.plot([], [], color=c, lw=2, label=n)[0] for n, c in _SERIES]
        self.ax.legend(loc="upper left", fontsize=8, facecolor="#222222",
                       edgecolor="gray", labelcolor="white")
        self.fig.tight_layout()
        self._t, self._y = [], [[], [], []]

    def update(self, t, angles):
        self._t.append(t)
        for k in range(3):
            self._y[k].append(angles[k])
        x0 = max(0, t - self.window)
        for k, ln in enumerate(self.lines):
            ln.set_data(self._t, self._y[k])
        self.ax.set_xlim(x0, max(self.window, t))
        # current values in the title
        self.ax.set_title(
            f"Flex {angles[0]:+.0f}   Lat {angles[1]:+.0f}   Axial {angles[2]:+.0f}",
            color="white", fontsize=10)
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        bgr = cv2.cvtColor(buf[..., 1:], cv2.COLOR_RGB2BGR)
        if (w, h) != (self.w, self.h):
            bgr = cv2.resize(bgr, (self.w, self.h))
        return bgr

    def close(self):
        plt.close(self.fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading calibration...")
    K0, D0, K1, D1, R, T = load_calib(CALIB_TOML)
    maps0, maps1, Q = build_rectify_maps(K0, D0, K1, D1, R, T, CAM_SIZE)
    project = make_projector(Q)   # 3D (cam0 frame) -> cam0 pixel, for overlay

    print("Loading frames...")
    frames0 = load_all_frames(CAM0_VIDEO)
    frames1 = load_all_frames(CAM1_VIDEO)
    n = min(len(frames0), len(frames1))
    print(f"  {n} paired frames")

    ts0_ms = load_timestamps(CAM0_TIMESTAMP)
    ts1_ms = load_timestamps(CAM1_TIMESTAMP)
    if len(ts0_ms) != n or len(ts1_ms) != n:
        ts0_ms = [int(i * 1000 / 15) for i in range(n)]
        ts1_ms = list(ts0_ms)
    writer_fps = estimate_fps(ts0_ms)

    lmk0 = make_pose_landmarker()
    lmk1 = make_pose_landmarker()
    W, H = CAM_SIZE

    cam_w   = int(W * PANEL_HEIGHT / H)
    plot_w  = cam_w
    total_w = cam_w * 2 + plot_w            # [cam0 | cam1 | plot]
    plot    = TrunkPlot(plot_w, PANEL_HEIGHT)
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(str(OUT_VIDEO), fourcc, writer_fps, (total_w, PANEL_HEIGHT))
    print(f"Output: {OUT_VIDEO}  ({total_w}x{PANEL_HEIGHT})")

    ema       = EMA(ANGLE_EMA_ALPHA)
    ref_ups   = []      # trunk long axes during reference window
    ref_mh    = []      # shoulder-line horizontal during reference window
    vup        = -CAM_DOWN           # default gravity = camera up
    m_ref_h    = None
    rows       = []
    last_ang   = (0.0, 0.0, 0.0)     # hold last value when detection drops
    hip_anchor = None                # pelvis 3D position, for occluded-hip fallback

    for i in range(n):
        f0, f1 = frames0[i], frames1[i]
        f0b = cv2.cvtColor(f0, cv2.COLOR_GRAY2BGR) if f0.ndim == 2 else f0
        f1b = cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR) if f1.ndim == 2 else f1
        r0  = rectify(f0b, maps0)
        r1  = rectify(f1b, maps1)

        res0 = lmk0.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(r0, cv2.COLOR_BGR2RGB)), ts0_ms[i])
        res1 = lmk1.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(r1, cv2.COLOR_BGR2RGB)), ts1_ms[i])
        lms0 = res0.pose_landmarks[0] if res0.pose_landmarks else None
        lms1 = res1.pose_landmarks[0] if res1.pose_landmarks else None

        pts = triangulate_torso(lms0, lms1, Q, W, H)

        # Update the pelvis anchor whenever both hips are actually measured
        if USE_HIP_ANCHOR and "Lhip" in pts and "Rhip" in pts:
            hm = (pts["Lhip"] + pts["Rhip"]) / 2.0
            hip_anchor = hm if hip_anchor is None else \
                (1 - HIP_ANCHOR_ALPHA) * hip_anchor + HIP_ANCHOR_ALPHA * hm

        frame   = torso_frame(pts, hip_anchor if USE_HIP_ANCHOR else None)
        seg3d   = None
        hip_est = False
        if frame is not None:
            R_torso, sh_mid, hip_mid, hip_est = frame
            seg3d = (sh_mid, hip_mid)
            u = R_torso[:, 1]
            m = R_torso[:, 0]

            # Build neutral reference from the first valid frames (hips must be real)
            if len(ref_ups) < REF_FRAMES and not hip_est:
                ref_ups.append(u)
                ref_mh.append(m - np.dot(m, vup) * vup)
                if len(ref_ups) == REF_FRAMES:
                    if GRAVITY_MODE == "reference":
                        vup = _normalize(np.mean(ref_ups, axis=0))
                    m_ref_h = _normalize(np.mean(ref_mh, axis=0))
                    print(f"  Reference locked at frame {i}: vup={vup.round(3)}")

            raw      = trunk_angles(R_torso, vup, m_ref_h)
            last_ang = tuple(ema.update(raw))
            rows.append((i, ts0_ms[i], *last_ang, int(hip_est)))

        # Three panels: cam0 overlay | cam1 overlay | scrolling plot
        p0 = draw_torso(r0, torso_px(lms0, W, H), "Cam0", last_ang,
                        seg3d=seg3d, project=project, hip_est=hip_est)
        p1 = draw_torso(r1, torso_px(lms1, W, H), "Cam1")
        p0 = cv2.resize(p0, (cam_w, PANEL_HEIGHT))
        p1 = cv2.resize(p1, (cam_w, PANEL_HEIGHT))
        pp = plot.update(i, last_ang)
        writer.write(np.concatenate([p0, p1, pp], axis=1))

        if i % 25 == 0:
            print(f"  {i:4d}/{n}  flex={last_ang[0]:+6.1f} lat={last_ang[1]:+6.1f} ax={last_ang[2]:+6.1f}")

    writer.release()
    plot.close()
    lmk0.__exit__(None, None, None)
    lmk1.__exit__(None, None, None)

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "timestamp_ms", "flexion_deg", "lateral_deg", "axial_deg",
                    "hip_estimated"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.2f}", f"{r[3]:.2f}", f"{r[4]:.2f}", r[5]])

    print(f"Done -> {OUT_VIDEO}")
    print(f"CSV  -> {OUT_CSV}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
