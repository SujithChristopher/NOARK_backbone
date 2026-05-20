"""Upper limb 3D point cloud — dual OV9281 fisheye stereo pipeline.

Pipeline
--------
1. Load grayscale msgpack frames from both OV9281 cameras
2. Stereo-rectify both frames using fisheye calibration (R, T)
3. Segment human region with MediaPipe ImageSegmenter on each rectified frame
4. Compute StereoSGBM disparity masked to intersection of both human masks
5. Back-project disparity to 3D point cloud via Q matrix from stereoRectify
6. Render: [cam0 seg | cam1 seg | 3D cloud] → MP4
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
PROJECT_ROOT = Path(__file__).parents[3]
SCRIPT_DIR   = Path(__file__).parent
CALIB_TOML   = (
    PROJECT_ROOT
    / "data" / "calibration" / "dual_160"
    / "dual_ov9281_calibration_checker_sz_30mm"
    / "stereo_calibration.toml"
)
DATA_DIR       = PROJECT_ROOT / "data" / "dual_data" / "dual_ov9281_trunk_test"
CAM0_VIDEO     = DATA_DIR / "cam0_frame.msgpack"
CAM1_VIDEO     = DATA_DIR / "cam1_frame.msgpack"
CAM0_TIMESTAMP = DATA_DIR / "cam0_timestamp.msgpack"
CAM1_TIMESTAMP = DATA_DIR / "cam1_timestamp.msgpack"
OUT_VIDEO      = DATA_DIR / "upperlimb_combined.mp4"

PANEL_HEIGHT    = 480
SEGMENTER_MODEL = SCRIPT_DIR / "selfie_segmenter.tflite"
SEG_THRESH      = 0.5

# StereoSGBM — tune numDisparities to camera separation
NUM_DISP   = 128   # must be multiple of 16
BLOCK_SIZE = 5

# Both OV9281 cameras share the same resolution
CAM_SIZE = (1280, 800)   # (W, H)

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def load_calib(path):
    d  = toml.load(path)
    K0 = np.array(d["cam0"]["camera_matrix"])
    D0 = np.array(d["cam0"]["dist_coeffs"])
    K1 = np.array(d["cam1"]["camera_matrix"])
    D1 = np.array(d["cam1"]["dist_coeffs"])
    R  = np.array(d["stereo"]["R"])
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
# Stereo rectification (fisheye → pinhole-rectified pair)
# ---------------------------------------------------------------------------
def build_rectify_maps(K0, D0, K1, D1, R, T, size):
    """size = (W, H). Returns (maps0, maps1, Q)."""
    R0, R1, P0, P1, Q = cv2.fisheye.stereoRectify(
        K0, D0, K1, D1, size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=size,
        balance=0.0,
        fov_scale=1.0,
    )
    m0x, m0y = cv2.fisheye.initUndistortRectifyMap(K0, D0, R0, P0, size, cv2.CV_32F)
    m1x, m1y = cv2.fisheye.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32F)
    return (m0x, m0y), (m1x, m1y), Q

def rectify(frame, maps):
    return cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)

# ---------------------------------------------------------------------------
# Segmentation (MediaPipe ImageSegmenter, selfie_segmenter.tflite)
# ---------------------------------------------------------------------------
def make_segmenter():
    opts = mp_vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
        running_mode=mp_vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    return mp_vision.ImageSegmenter.create_from_options(opts)

def segment_frame(segmenter, bgr, timestamp_ms):
    """Return float32 person-confidence mask (H, W) ∈ [0, 1]."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = segmenter.segment_for_video(mp_img, timestamp_ms)
    if not res.confidence_masks:
        return np.zeros(bgr.shape[:2], np.float32)
    # index 0 = person channel for selfie_segmenter (single-output model)
    return np.squeeze(res.confidence_masks[0].numpy_view()).copy()

# ---------------------------------------------------------------------------
# Disparity + point cloud
# ---------------------------------------------------------------------------
_stereo_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=NUM_DISP,
    blockSize=BLOCK_SIZE,
    P1=8  * 3 * BLOCK_SIZE ** 2,
    P2=32 * 3 * BLOCK_SIZE ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

def compute_disparity(gray0, gray1, human_mask=None):
    """StereoSGBM disparity. If human_mask given, zeroes pixels outside it."""
    disp = _stereo_matcher.compute(gray0, gray1).astype(np.float32) / 16.0
    if human_mask is not None:
        disp[human_mask == 0] = 0.0
    disp[disp <= 0] = 0.0
    return disp

def disp_to_pointcloud(disp, Q, max_pts=8000):
    """Back-project valid disparity pixels to 3D (mm). Returns (N, 3) float32."""
    valid = disp > 0
    pts = cv2.reprojectImageTo3D(disp, Q, handleMissingValues=False)
    cloud = pts[valid].astype(np.float32)
    # keep only reasonable depth range (100 mm – 5000 mm)
    cloud = cloud[(cloud[:, 2] > 100) & (cloud[:, 2] < 5000)]
    if len(cloud) > max_pts:
        idx = np.random.choice(len(cloud), max_pts, replace=False)
        cloud = cloud[idx]
    return cloud

# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def seg_overlay(bgr, mask_f32, color_bgr):
    """Semi-transparent colored overlay on segmented region."""
    out = bgr.copy()
    m = mask_f32 >= SEG_THRESH
    overlay = out.copy()
    overlay[m] = color_bgr
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
    return out

def scale_to_height(img, h):
    oh, ow = img.shape[:2]
    return cv2.resize(img, (int(ow * h / oh), h))

def make_panel(img, h, label=""):
    panel = scale_to_height(img, h)
    if label:
        cv2.putText(panel, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return panel

def fig_to_bgr(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)

def render_pointcloud(ax, cloud, frame_num, total):
    ax.cla()
    ax.set_facecolor("black")
    if len(cloud) > 0:
        ax.scatter(
            cloud[:, 0], cloud[:, 1], cloud[:, 2],
            c=cloud[:, 2], cmap="plasma", s=1, alpha=0.7, depthshade=False,
        )
        cx, cy, cz = cloud.mean(axis=0)
        r = 600
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_zlim(cz - r, cz + r)
        ax.invert_yaxis()
    else:
        for setter in [ax.set_xlim, ax.set_ylim, ax.set_zlim]:
            setter(-600, 600)
    ax.set_xlabel("X (mm)", color="white", labelpad=6)
    ax.set_ylabel("Y (mm)", color="white", labelpad=6)
    ax.set_zlabel("Z (mm)", color="white", labelpad=6)
    ax.tick_params(colors="white")
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
    ax.grid(True, color="gray", alpha=0.3)
    ax.set_title(f"Point Cloud  [{frame_num}/{total}]", color="white", pad=10)
    ax.view_init(elev=15, azim=-60)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading calibration...")
    K0, D0, K1, D1, R, T = load_calib(CALIB_TOML)
    print(f"  T={T.ravel().round(1)} mm")

    maps0, maps1, Q = build_rectify_maps(K0, D0, K1, D1, R, T, CAM_SIZE)
    print("  Rectification maps built.")

    print("Loading frames...")
    frames0 = load_all_frames(CAM0_VIDEO)
    frames1 = load_all_frames(CAM1_VIDEO)
    n = min(len(frames0), len(frames1))
    print(f"  {n} paired frames")

    ts0_ms = load_timestamps(CAM0_TIMESTAMP)
    ts1_ms = load_timestamps(CAM1_TIMESTAMP)
    if len(ts0_ms) != n or len(ts1_ms) != n:
        print("  Timestamp mismatch — using synthetic 15 fps timing.")
        ts0_ms = [int(i * 1000 / 15) for i in range(n)]
        ts1_ms = list(ts0_ms)

    writer_fps = estimate_fps(ts0_ms)
    print(f"  fps ~{writer_fps:.2f}")

    seg0 = make_segmenter()
    seg1 = make_segmenter()

    fig = plt.figure(figsize=(5, 5), facecolor="black")
    ax  = fig.add_subplot(111, projection="3d", facecolor="black")

    # Pre-compute output frame width
    render_pointcloud(ax, np.empty((0, 3)), 0, n)
    plot_w = scale_to_height(fig_to_bgr(fig), PANEL_HEIGHT).shape[1]
    cam_w  = int(CAM_SIZE[0] * PANEL_HEIGHT / CAM_SIZE[1])
    total_w = cam_w * 2 + plot_w

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_VIDEO), fourcc, writer_fps, (total_w, PANEL_HEIGHT))
    print(f"Processing {n} frames -> {OUT_VIDEO}  ({total_w}x{PANEL_HEIGHT})")

    for i in range(n):
        f0 = frames0[i]
        f1 = frames1[i]

        # OV9281 → grayscale BGR
        f0_bgr = cv2.cvtColor(f0, cv2.COLOR_GRAY2BGR) if f0.ndim == 2 else f0
        f1_bgr = cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR) if f1.ndim == 2 else f1

        # Segment on original (pre-rectification) frames — better for MediaPipe
        mask0_orig = segment_frame(seg0, f0_bgr, ts0_ms[i])
        mask1_orig = segment_frame(seg1, f1_bgr, ts1_ms[i])

        # Fisheye rectification (frames + masks warped together)
        r0 = rectify(f0_bgr, maps0)
        r1 = rectify(f1_bgr, maps1)
        mask0 = cv2.remap(mask0_orig, maps0[0], maps0[1], cv2.INTER_LINEAR)
        mask1 = cv2.remap(mask1_orig, maps1[0], maps1[1], cv2.INTER_LINEAR)

        # Dense disparity on full rectified image (mask warping clips too aggressively)
        r0_gray = cv2.cvtColor(r0, cv2.COLOR_BGR2GRAY)
        r1_gray = cv2.cvtColor(r1, cv2.COLOR_BGR2GRAY)
        disp = compute_disparity(r0_gray, r1_gray, human_mask=None)

        cloud = disp_to_pointcloud(disp, Q)

        # Panels — show rectified frames with warped mask overlay
        ov0 = seg_overlay(r0, mask0, color_bgr=(50, 220, 100))
        ov1 = seg_overlay(r1, mask1, color_bgr=(50, 180, 255))
        p0  = make_panel(ov0, PANEL_HEIGHT, "Cam0 OV9281")
        p1  = make_panel(ov1, PANEL_HEIGHT, "Cam1 OV9281")

        render_pointcloud(ax, cloud, i + 1, n)
        p3d = scale_to_height(fig_to_bgr(fig), PANEL_HEIGHT)

        combined = np.concatenate([p0, p1, p3d], axis=1)
        writer.write(combined)

        if i % 50 == 0:
            print(f"  {i}/{n}  cloud pts: {len(cloud)}")

    writer.release()
    seg0.__exit__(None, None, None)
    seg1.__exit__(None, None, None)
    plt.close(fig)
    print(f"Done -> {OUT_VIDEO}")


if __name__ == "__main__":
    main()
