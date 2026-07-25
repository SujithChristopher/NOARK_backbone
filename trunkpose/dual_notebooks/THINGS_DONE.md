# Trunk-Angle Estimation — Work Log

Goal: camera-based **trunk angle** (flexion / lateral / axial) for a **seated** person
(hips occluded below desk), validated against OptiTrack mocap.

Rig: dual OV9281 fisheye (160 FOV), 1280x800, baseline ~78 mm. Stereo calib:
`data/calibration/dual_160/calib_cz30_dual_v2/stereo_calibration.toml` (rms ~0.14/0.15,
epipolar err ~0.82). Board frame from ChArUco basis
(`data/trunk_july1_2026/dual_160_tframe_july1/charuco_basis.toml`).
Dataset: `data/trunk_july1_2026/dual_160_trunk_ragav` (1667 frames, ~30 fps) + Motive CSV
with trunk rigid body (`trunk:Marker1` origin, Marker4 x-vec, Marker2 z-vec).

**Status: VALIDATED (2026-07-24).** ICP rigid-registration camera angles agree with
mocap on the clean data segment: held-out r = 0.81 / 0.96 / 0.95 (flex/lat/axial),
RMSE 2.6 / 4.8 / 1.3°. The earlier "camera doesn't match mocap" mystery was three
stacked data problems, not the method: wrong time sync (fixed by GPIO hardware sync),
wrong ChArUco mocap→board rotation, and a mocap CSV corrupted by two Motive re-lock
teleports (details in section 7).

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

### 6. `07_icp_trunk.py` — ICP rigid-registration trunk angles — 2026-07-24

Idea: skip landmarks + plane entirely; register each frame's torso front-shell cloud
(~1.5k pts, cached by 06 as `cloud`, board frame) to a pooled NEUTRAL cloud (first 15
frames) with trimmed point-to-point Kabsch ICP (annealed 6→3→2 cm, warm-started).
Body rotation-from-neutral = R_icp^T → same projected angles, camera anatomical axes.

**Two comparison bugs found and fixed while doing this:**

1. **ChArUco mocap→board basis is WRONG by ~136°** (det +1, pure rotation — looks like
   an axis-permutation-type error). This corrupted every prior camera↔mocap angle
   comparison (axes mixed/sign-flipped → the negative/near-zero r values). Fix inside
   07: convention-free comparison — mocap body-rotation-from-neutral only (marker
   labels cancel), then residual frame rotation S estimated from data via Kabsch on
   rotation-vector pairs (rotvec(Rb_icp) ≈ S·rotvec(Rb_moc), |angle|>5°, 913 pairs).
   → `charuco_basis.toml` [mocap] rotation needs re-derivation at some point.
2. Sync: two-stage — angle-activity xcorr, then signed-flexion xcorr refine
   (total lag ≈ +1.6 s, flexion lock r=0.45).

**Results (1604/1667 frames, registration rms 6.9 mm):**

| axis | plane+shoulder r / RMSE | ICP r / RMSE |
|------|------------------------|--------------|
| flexion | −0.17 / 9.8° | **0.45** / 9.7° |
| lateral | 0.03 / 9.8° | **0.38** / 14.0° |
| axial | 0.25 / 14.6° | **0.52** / 10.7° |

ICP beats the plane+shoulder pipeline on every axis, traces are visibly smooth, and
flexion/axial shapes match mocap well. **Rigid-registration concept validated.**

Remaining error sources (why r isn't higher yet):
- ICP loses grip during the big axial turn ~40–43 s (front shell self-occludes vs the
  neutral view) → converges to a wrong local minimum with plausible rms, so the
  rms/match-fraction gate doesn't catch it. Fix ideas: frame-to-frame odometry with
  keyframes, or multi-view neutral model.
- Residual sync jitter (mocap lock still only r≈0.45 on flexion).
- Lateral axis weakest (14° RMSE) — smallest real motion, most affected by cloth.

### 7. The three data bugs that hid the agreement — 2026-07-24

The 06/07 comparisons kept failing for reasons that had nothing to do with the camera
method. Found and fixed in this order:

**7a. Time sync: use the GPIO wire, not cross-correlation.**
`cam0_timestamp.msgpack` column 0 is the GPIO pin-17 bit — HIGH while mocap records.
Rising edge at camera frame 93, falling at 1623 → high span 51.03 s vs mocap CSV
50.89 s (match). So mocap `seconds=0` ≡ camera frame 93. All previous xcorr-based
lags were wrong (they found +3.0…+6.4 s; the true wall-clock offset between the two
machines is −2.38 s). Helper: `m6.load_sync_flags()`; used by 07.

**7b. The earlier charuco-basis "patch" was circular — REVERTED.**
The ~138° "basis error" S estimated on 2026-07-24 (earlier session) was fitted under
the wrong xcorr sync, and its "verification" reused the same wrong sync. With GPIO
sync that patch made everything worse. `charuco_basis.toml` [mocap] restored from
`charuco_basis_backup_pre_s_fix.toml`. Lesson: never patch a calibration file with a
correction fitted on unsynced/corrupted data.

**7c. Mocap CSV corrupted by two Motive re-lock teleports.**
The solved trunk rigid body jumps 326 mm across ONE dropped frame (10 ms) at mocap
t=1.72 s and 267 mm across 30 ms at t=26.37 s — physically impossible → Motive lost
the body during brief occlusion and re-solved it at a ghost pose (other reflective
markers, e.g. the T-frame, are in the scene). Marker distances stay exactly constant
(solved output), so only position jumps reveal it. Result: three segments with
arbitrary constant pose offsets between them; the old "neutral" was in segment A
while the motion was in B/C → rotation-from-neutral garbage → every full-recording
comparison was doomed. 07 now auto-splits segments (`SEG_JUMP_M=0.10`) and uses the
longest one (B: 1.7–26.4 s, ~24.6 s, 734 camera frames).

**Validated result (segment B, GPIO sync, both zeroed at segment start).**
A residual frame rotation S (~110–130°, the true basis rotation error) is fitted by
Kabsch on rotation-vector pairs from the FIRST half of the segment and scored on the
held-out SECOND half:

| axis | held-out r | held-out RMSE |
|------|-----------|---------------|
| flexion | **+0.81** | 2.6° |
| lateral | **+0.96** | 4.8° |
| axial | **+0.95** | 1.3° |

Basis-free sanity: rotation magnitude r=0.55 over the whole segment; without any S
the axial axis already showed r=−0.90 (pure axis flip → signal was always there).
**ICP rigid registration is validated as the camera trunk-angle method.**

Open items:
- ICP still dies after the big axial turn at ~41 s (wrong local minimum, stuck to
  end of recording). Not visible in segment-B validation; fix later (multi-view
  neutral model / better re-acquisition) or avoid such turns in protocol.
- Camera shows 15–18° bouts at ~9.5 s and ~14 s where mocap shows only ~4° —
  suspect arm movement contaminating the torso cloud; investigate mask/cloud there.
- The charuco [mocap] basis rotation is genuinely wrong (~110–130°); re-derive it
  properly in a dedicated recording (T-frame + board simultaneously visible, no
  occlusions) instead of trusting the data-fitted S long-term.
- Re-record validation with a mocap take free of re-lock teleports (watch marker
  count in Motive; remove/mask stray reflective objects).

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
