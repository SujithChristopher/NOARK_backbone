# Trunk-Angle Estimation — Work Log

Goal: camera-based **trunk angle** (flexion / lateral / axial) for a **seated** person
(hips occluded below desk), validated against OptiTrack mocap.

Rig: dual OV9281 fisheye (160 FOV), 1280x800, baseline ~78 mm. Stereo calib:
`data/calibration/dual_160/calib_cz30_dual_v2/stereo_calibration.toml` (rms ~0.14/0.15,
epipolar err ~0.82). Board frame from ChArUco basis
(`data/trunk_july1_2026/dual_160_tframe_july1/charuco_basis.toml`).
Dataset: `data/trunk_july1_2026/dual_160_trunk_ragav` (1667 frames, ~30 fps) + Motive CSV
with trunk rigid body (`trunk:Marker1` origin, Marker4 x-vec, Marker2 z-vec).

**Status: pipeline works end-to-end, but camera angles do NOT yet agree with mocap.**
Best cross-correlation between camera and mocap signals so far: r ≈ 0.20–0.32 (weak).
Camera tracks mocap flexion for the first ~20 s, diverges after ~25 s.

---

## What was built (chronological)

### 1. Torso segmentation model (YOLO-seg)
- `trunkpose/segdataset.py` — DensePose COCO minival → YOLO-seg polygons.
  - **Bug found+fixed:** first version used `dp_masks` part index 2 = **Right Hand**
    (dp_masks is the 14-part COARSE scheme; 1 = Torso). Model was segmenting hands.
    Fixed to `CHEST_PARTS = {1}`.
  - 100 grayscale images, 80 train / 20 val, single class `chest`.
- `trunkpose/segtrain.py` — yolov8n-seg. Weights: `trunkpose/runs/trunk_seg-2/weights/best.pt`.
- Result: torso segmentation works well on our fisheye frames.

### 2. `05_create_plane.py` — dense-stereo torso plane (static check)
- Fisheye stereo rectify → SGBM disparity → torso point cloud (seg mask) → RANSAC plane
  → SVD refine, first ~15 frames pooled; renders cloud + 15x15 cm plane patch + MediaPipe joints.
- Fixes along the way: `svd(full_matrices=False)` (MemoryError on 211k pts), depth/percentile
  clipping for sane axes.
- Result: plane fit on torso looks plausible → promoted the idea into 06.

### 3. `06_trunk_axis.py` — full trunk-angle pipeline (MAIN)
Two parallel passes (4 workers, Windows spawn-safe), geometry cached to pkl.

- **Trunk frame:** lateral = L→R shoulder (MediaPipe triangulated), anterior = torso-plane
  normal (EMA α=0.3), up = orthogonalized cross. Angles = projected vs neutral pose
  (mean rotation of first 15 valid frames).
- **Per-frame geometry:** MediaPipe on both raw fisheye views → `cv2.fisheye.undistortPoints`
  → triangulation → board frame. Torso: seg mask → erode 6% → SGBM cloud → front-shell
  (frontmost 5th-percentile depth + 10 cm) → RANSAC plane.
- **Render:** 2x2 video (cam0 + axis overlay | 3D scene | depth heatmap | 3-angle plots),
  savgol (w=11, p=3) on camera angles, mocap raw.
- **Mocap overlay:** trunk rigid-body angles from Marker1/4/2, same projected-angle formula.
- Output: `data/trunk_july1_2026/dual_160_trunk_ragav/trunk_axis_charuco.mp4`.

### 4. Time synchronization attempts
- Wall-clock: cam0 starts 13:29:41.234, mocap 13:29:41.951 (100 Hz, 5090 frames) → 0.72 s
  offset applied via `timestamp.msgpack`. Residual misalignment remained.
- Cross-correlation of motion signals (tried flexion-vs-flexion, angle activity,
  landmark/marker position speed): best lag found but **r never exceeded ~0.32** →
  no reliable lock. Conclusion: not a sync problem — the camera signal itself diverges.

### 5. Chest marker-object mitigation (plane drop) — 2026-07-24
Physical confound: an object holding reflective markers is mounted on the chest, inside
the region the plane was fit on.
- Added `TRUNK_DROP_M` (env, default 0.12 m): plane **refit on lower-torso band**
  (points ≥ drop below shoulder line along trunk up-axis); drawn frame + plane origin
  moved down the same amount.
- Result: cam valid 1667/1667 (was 1623/1667) — band is more stable — but
  **xcorr still r=0.20 (lag +3.03 s)**. The chest object was NOT the dominant error.

---

## Current diagnosis

Camera 3D accuracy is the suspect, not the angle formula, not sync, not the chest object:
1. MediaPipe landmarks are detected on **raw fisheye** images (heavy distortion at 160 FOV)
   → shoulder positions may be biased; lateral axis noisy.
2. Plane normal is jittery frame-to-frame (EMA helps but can't fix bias).
3. Small baseline (78 mm) at ~1 m depth → depth noise amplifies angle noise.

## Next planned step (agreed, not yet built)

Three small validators in `trunkpose/dual_notebooks/`, reusing `06_trunk_axis.py` helpers
(`importlib.import_module("06_trunk_axis")`) + geometry cache:
1. `v_triangulation.py` — shoulder width + segment length stability over time (should be
   constant), joint reprojection error → is camera 3D trustworthy at all?
2. `v_undistort_mediapipe.py` — MediaPipe on raw fisheye vs undistorted frames (~5 frames,
   montage) → tests the top suspect.
3. `v_shoulder_only_angles.py` — lateral + axial from shoulder line only (no plane) vs
   mocap → separates "plane broken" from "everything broken".

## Knobs / repro

```powershell
$env:TRUNK_WORKERS="4"
$env:TRUNK_GEOM_CACHE="...\dual_160_trunk_ragav\geom_cache.pkl"   # delete after geometry-affecting edits!
$env:TRUNK_MAX_FRAMES="60"     # quick test runs (cache skipped)
$env:TRUNK_DROP_M="0.12"       # plane/frame drop below shoulders; 0 = old behavior
$env:GLOG_minloglevel="3"      # mute MediaPipe telemetry noise
uv run python trunkpose\dual_notebooks\06_trunk_axis.py
```

**Gotcha log:** stale `geom_cache.pkl` silently masks geometry changes — delete it whenever
`_geom_chunk`/`stereo_step`/`fit_plane` change. Windows spawn: worker initializers must be
module-level. `mp` name is reserved for mediapipe import.
