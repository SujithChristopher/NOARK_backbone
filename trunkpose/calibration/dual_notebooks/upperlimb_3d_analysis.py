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
    / "dual_ov9281_parallel_calib_cz_30mm"
    / "stereo_calibration.toml"
)
DATA_DIR       = PROJECT_ROOT / "data" / "dual_data" / "dual_ov9281_parallel_trunk_test"
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

# WLS filter params — sigma controls spatial smoothness, lambda controls edge sensitivity
WLS_LAMBDA = 8000.0
WLS_SIGMA  = 1.5

# Temporal EMA alpha for depth smoothing — lower = smoother but more lag
DEPTH_EMA_ALPHA = 0.3

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
        balance=1.0,   # keep full fisheye content, black-fill unmapped borders
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
_left_matcher = cv2.StereoSGBM_create(
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
_right_matcher = cv2.ximgproc.createRightMatcher(_left_matcher)
_wls_filter    = cv2.ximgproc.createDisparityWLSFilter(_left_matcher)
_wls_filter.setLambda(WLS_LAMBDA)
_wls_filter.setSigmaColor(WLS_SIGMA)


class TemporalDepthSmoother:
    """Per-pixel EMA over depth frames; invalid pixels (<=0) excluded from blend."""

    def __init__(self, alpha=DEPTH_EMA_ALPHA):
        self.alpha = alpha
        self._prev = None

    def update(self, disp):
        if self._prev is None:
            self._prev = disp.copy()
            return disp.copy()
        valid_new  = disp > 0
        valid_prev = self._prev > 0
        out = self._prev.copy()
        # Both valid → EMA blend
        both = valid_new & valid_prev
        out[both] = self.alpha * disp[both] + (1 - self.alpha) * self._prev[both]
        # New pixel appeared → accept it directly
        appeared = valid_new & ~valid_prev
        out[appeared] = disp[appeared]
        # Pixel vanished → decay toward zero after 1/alpha frames
        vanished = ~valid_new & valid_prev
        out[vanished] = (1 - self.alpha) * self._prev[vanished]
        out[out < 0.5] = 0.0   # flush near-zero remnants
        self._prev = out.copy()
        return out


def compute_disparity(gray0, gray1, human_mask=None):
    """WLS-filtered StereoSGBM disparity. human_mask (uint8 0/1) restricts ROI."""
    disp_left  = _left_matcher.compute(gray0, gray1)
    disp_right = _right_matcher.compute(gray1, gray0)
    disp_wls   = _wls_filter.filter(disp_left, gray0, disparity_map_right=disp_right)
    disp = disp_wls.astype(np.float32) / 16.0
    if human_mask is not None:
        disp[human_mask == 0] = 0.0
    disp[disp <= 0] = 0.0
    return disp

def disp_to_heatmap(disp, target_size=None):
    """
    Convert float32 disparity map to a BGR heatmap image.
    Higher disparity (closer) = brighter. Invalid pixels = black.
    target_size: (W, H) to resize output, or None to keep original size.
    """
    valid = disp > 0
    vis = np.zeros(disp.shape, np.float32)
    if valid.any():
        lo, hi = disp[valid].min(), disp[valid].max()
        if hi > lo:
            vis[valid] = (disp[valid] - lo) / (hi - lo) * 255.0
    heatmap = cv2.applyColorMap(vis.astype(np.uint8), cv2.COLORMAP_INFERNO)
    heatmap[~valid] = 0   # black = no depth
    if target_size is not None:
        heatmap = cv2.resize(heatmap, target_size)
    return heatmap

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

    seg0          = make_segmenter()
    seg1          = make_segmenter()
    depth_smoother = TemporalDepthSmoother()

    # Output layout: [cam0 | cam1 | depth heatmap], all at PANEL_HEIGHT
    cam_w   = int(CAM_SIZE[0] * PANEL_HEIGHT / CAM_SIZE[1])
    total_w = cam_w * 3   # heatmap same size as each cam panel

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_VIDEO), fourcc, writer_fps, (total_w, PANEL_HEIGHT))
    print(f"Processing {n} frames -> {OUT_VIDEO}  ({total_w}x{PANEL_HEIGHT})")

    for i in range(n):
        f0 = frames0[i]
        f1 = frames1[i]

        # OV9281 → grayscale BGR
        f0_bgr = cv2.cvtColor(f0, cv2.COLOR_GRAY2BGR) if f0.ndim == 2 else f0
        f1_bgr = cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR) if f1.ndim == 2 else f1

        # Segment on original frames — better for MediaPipe on fisheye
        mask0_orig = segment_frame(seg0, f0_bgr, ts0_ms[i])
        mask1_orig = segment_frame(seg1, f1_bgr, ts1_ms[i])

        # Fisheye → rectified for disparity
        r0      = rectify(f0_bgr, maps0)
        r1      = rectify(f1_bgr, maps1)
        r0_gray = cv2.cvtColor(r0, cv2.COLOR_BGR2GRAY)
        r1_gray = cv2.cvtColor(r1, cv2.COLOR_BGR2GRAY)
        disp_raw = compute_disparity(r0_gray, r1_gray)
        disp     = depth_smoother.update(disp_raw)

        # Save first frame debug images to check disparity quality
        if i == 0:
            out_dir = DATA_DIR
            cv2.imwrite(str(out_dir / "dbg_r0.png"), r0)
            cv2.imwrite(str(out_dir / "dbg_r1.png"), r1)
            cv2.imwrite(str(out_dir / "dbg_disp_raw.png"), disp_to_heatmap(disp_raw))
            cv2.imwrite(str(out_dir / "dbg_disp_filtered.png"), disp_to_heatmap(disp))
            print(f"  [diag] raw  disp valid_px={(disp_raw>0).sum()} "
                  f"min={disp_raw[disp_raw>0].min() if (disp_raw>0).any() else 0:.1f} "
                  f"max={disp_raw.max():.1f}")
            print(f"  [diag] filt disp valid_px={(disp>0).sum()} "
                  f"min={disp[disp>0].min() if (disp>0).any() else 0:.1f} "
                  f"max={disp.max():.1f}")

        # Panels
        ov0  = seg_overlay(f0_bgr, mask0_orig, color_bgr=(50, 220, 100))
        ov1  = seg_overlay(f1_bgr, mask1_orig, color_bgr=(50, 180, 255))
        p0   = make_panel(ov0,  PANEL_HEIGHT, "Cam0 OV9281")
        p1   = make_panel(ov1,  PANEL_HEIGHT, "Cam1 OV9281")
        p_hm = disp_to_heatmap(disp, target_size=(cam_w, PANEL_HEIGHT))
        cv2.putText(p_hm, "Depth (brighter=closer)", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        combined = np.concatenate([p0, p1, p_hm], axis=1)
        writer.write(combined)

        if i % 50 == 0:
            print(f"  {i}/{n}  valid disp px: {(disp>0).sum()}")

    writer.release()
    seg0.__exit__(None, None, None)
    seg1.__exit__(None, None, None)
    print(f"Done -> {OUT_VIDEO}")


if __name__ == "__main__":
    main()
