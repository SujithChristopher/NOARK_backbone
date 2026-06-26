# Stereo Camera: Calibration, Verification & Findings

Dual OV9281 (160° FOV) stereo rig for single-tag / multi-tag AprilTag pose
estimation. This document covers the stereo calibration approach, the
motion-capture verification methodology, and the measured findings on whether
(and how) a second camera reduces pose jitter.

All numbers below come from the recording
`data/dual_data/dual_cam_single_aprl_50mm_t0` (a 50 mm AprilTag, id 12, moved
in front of the rig with simultaneous OptiTrack ground truth), calibrated with
`data/calibration/dual_160/calib_cz30_dual_v2/stereo_calibration.toml`.

---

## 1. Hardware & data layout

- **Cameras**: 2 × OV9281, 1280×800, fisheye 160° FOV.
- **Baseline**: ~78 mm (mostly along X), cameras near-parallel (stereo
  rotation ≈ 0.7°).
- **Marker**: AprilTag `DICT_APRILTAG_36h11`, 50 mm.
- **Ground truth**: OptiTrack rigid body, hardware sync pulse recorded in the
  camera timestamp stream (column 0).

Recording folders (`data/dual_data/<name>/`):

```
cam0_frame.msgpack       cam0 frames (msgpack + msgpack_numpy)
cam0_timestamp.msgpack   [sync_pulse, timestamp] per frame
cam1_frame.msgpack       cam1 frames
cam1_timestamp.msgpack
<name>.csv               OptiTrack rigid-body export (mocap ground truth)
```

Calibration folders (`data/calibration/dual_160/<name>/`):

```
chessb_corners_cam0_frame.pkl   detected chessboard corners (01_corner_detection.py)
chessb_corners_cam1_frame.pkl
stereo_calibration.toml         K0/D0, K1/D1, stereo R/T, epipolar error
```

---

## 2. Stereo calibration

Scripts: `notebooks/calibration/dual_notebooks/01_corner_detection.py` →
`02_dual_calibration.py`.

### Why not `cv2.fisheye.stereoCalibrate`

`cv2.fisheye.stereoCalibrate` has an unconditional
`CV_Assert(abs_max < 1e10)` inside `CalibrateExtrinsics` that fires on many
real-world planar chessboard datasets regardless of flags. We avoid it with an
equivalent decomposed pipeline (`02_dual_calibration.py`):

1. Calibrate each camera individually (`cv2.fisheye.calibrate`) → K, D.
2. Filter frames by per-frame reprojection error (keep best ~80%), re-calibrate
   for tighter K, D.
3. For paired frames, recover per-view extrinsics with **fixed** K/D via
   `cv2.fisheye.undistortPoints` + `cv2.solvePnP`.
4. Per paired view compute stereo `R = R1 · R0ᵀ`, `T = t1 − R·t0`.
5. Aggregate: rotation mean + translation median, with MAD outlier rejection.
6. Validate with epipolar error (fundamental matrix from R, T, K).

### Calibration result (`calib_cz30_dual_v2`)

| | fx | fy | cx | cy | per-cam RMS |
|---|---|---|---|---|---|
| cam0 | 593.3 | 594.5 | 627.9 | 422.5 | 0.144 px |
| cam1 | 591.8 | 593.8 | 622.7 | 384.6 | 0.149 px |

- **Stereo T**: `[-77.9, -2.5, -2.0] mm` → baseline **78.0 mm**
- **Stereo R** (euler): ≈ 0.7° (near-parallel)
- **Epipolar error**: **0.82 px** (mean) — solid enough to trust downstream.

> Units note: the TOML stores **T in millimetres**. Verification code converts
> to metres (`/1000`) so triangulated points and mono PnP share units.

---

## 3. Verification methodology

Script: `notebooks/calibration/dual_notebooks/verification_dual_camera.py`.
Compares camera pose estimates against OptiTrack on every synced frame.

### Pose estimators (per frame)

| Name | Points | Method |
|---|---|---|
| `cam0` | 4 | single-view planar `solvePnP` on cam0 |
| `cam1` | 4 | single-view planar `solvePnP` on cam1 |
| `stereo_tri` | 8 | triangulate 4 corners with baseline → Kabsch-fit tag model (algebraic, with bad-corner rejection) |
| `stereo_pnp` | 8 | **joint multi-view PnP** — one pose minimising reprojection in *both* cameras (maximum-likelihood) |

`stereo_pnp` is initialised from the `stereo_tri` solution and refined with
`scipy.optimize.least_squares` (LM), projecting the tag model into cam0 (pose
`X`) and cam1 (pose composed with stereo `R,T` via `cv2.composeRT`).

### Pipeline steps

1. **Detect** tags on each frame; refine corners with `cv2.cornerSubPix`.
2. **Pair** cam0↔cam1 frames by nearest timestamp; reject pairs more than half
   a frame period apart.
3. **Sync window**: trim to the active capture segment using the cam0 sync
   pulse.
4. **Mocap**: tag centre = mean of 4 rigid-body markers; orientation from the
   rigid-body quaternion. Interpolate position (`interp1d`) and orientation
   (`Slerp`) onto the camera timeline.
5. **Sub-frame lag correction**: cross-correlate camera vs mocap speed signals
   to find and remove the residual time offset before scoring.
6. **Kabsch align** each estimator's trajectory to mocap, then score.

### Metrics

- **Translation accuracy (RMS)**: `sqrt(mean((cam − mocap)²))` per axis, after
  Kabsch alignment. Includes drift/bias and dynamic frames.
- **Translation jitter**: std of the **frame-to-frame change of the residual**,
  restricted to **static-hold frames** (mocap speed < 3 mm/frame). This
  isolates high-frequency tracking noise from real motion and slow bias.
  - Expression: with residual `rᵢ = p̂ᵢ − mᵢ`, jitter = `std(rᵢ₊₁ − rᵢ)` over
    static steps. (A pure first-difference inflates variance by ~√2 vs the
    underlying per-frame σ — consistent across estimators, so comparisons are
    fair; divide by √2 for absolute σ.)
- **Rotation accuracy (RMS)**: frame-invariant. Cam and mocap live in different
  world/body frames (`R_moc = G · R_cam · B`), so orientation **increments**
  conjugate (`dM = G · dC · Gᵀ`). We recover `G` by aligning the rotation axes
  of the increments, then score the residual conjugation angle. (A naive
  relative-to-first comparison gives a bogus ~45° because it ignores `G`.)
- **Rotation jitter**: median frame-to-frame geodesic angle during static holds
  (median is robust to occasional planar-flip outliers).

---

## 3.5 Pose algorithms — full reconstruction recipe

Everything needed to re-implement the four estimators from scratch in another
project. OpenCV ≥ 4.7 (the `cv2.fisheye.*` and `cv2.aruco.ArucoDetector` API).

### 3.5.0 Conventions & shared inputs

- **Coordinate frame**: right-handed, camera looks down **+Z**, X right, Y down
  (OpenCV convention). A pose `(rvec, tvec)` maps **model → camera**:
  `X_cam = R(rvec) · X_model + tvec`, where `R = cv2.Rodrigues(rvec)`.
- **Intrinsics**: each camera has `K` (3×3) and fisheye distortion `D` (4×1,
  `[k1,k2,k3,k4]`, the Kannala–Brandt model used by `cv2.fisheye`).
- **Stereo extrinsics** `(R_st, T_st)` map **cam0 → cam1**:
  `X_cam1 = R_st · X_cam0 + T_st`. Stored in the TOML; **T is in mm → divide by
  1000** to work in metres (so it matches the tag size).
- **Tag model** (object points, metres), with corner order matching the AprilTag
  detector output **TL, TR, BR, BL**:
  ```
  L = MARKER_LENGTH                      # e.g. 0.05 m
  MARKER_PTS = [[-L/2, +L/2, 0],         # TL
               [+L/2, +L/2, 0],          # TR
               [+L/2, -L/2, 0],          # BR
               [-L/2, -L/2, 0]]          # BL
  ```
- **Corners**: `cv2.aruco.ArucoDetector.detectMarkers` → refine each marker's 4
  corners with `cv2.cornerSubPix` (window 5×5, 40 iters, eps 0.01) on the
  grayscale image. Corner array shape `(4, 2)`, float.

### 3.5.1 `cam0` / `cam1` — single-view planar PnP

The estimator name in code is "mono". Algorithm:

```
# corners: (4,2) pixel coords from one camera; K, D for that camera
und = cv2.fisheye.undistortPoints(corners.reshape(-1,1,2), K, D, P=K)
# und are now pixel coords in an ideal pinhole camera with matrix K, zero dist
ok, rvec, tvec = cv2.solvePnP(MARKER_PTS, und, K, distCoeffs=None,
                              flags=cv2.SOLVEPNP_ITERATIVE)
R = cv2.Rodrigues(rvec)[0]
# tag-centre position = tvec (model is centred at origin)
```

- `undistortPoints(..., P=K)` removes fisheye distortion and re-projects to the
  same `K`, so we can call the **pinhole** `solvePnP` with `distCoeffs=None`.
- `SOLVEPNP_ITERATIVE` is Levenberg–Marquardt reprojection minimisation seeded
  by a planar homography (DLT). **Weakness**: for a small coplanar tag the depth
  (Z) and out-of-plane tilt are weakly observable → jitter + occasional 180°
  flip ambiguity. This is exactly what stereo fixes.

### 3.5.2 `stereo_tri` — triangulation + Kabsch (algebraic)

Triangulate each of the 4 corners with the stereo baseline, then fit the rigid
tag model to the 4 metric 3D points.

```
# normalise corners to z=1 image plane (no P argument → normalized coords)
n0 = cv2.fisheye.undistortPoints(c0.reshape(-1,1,2), K0, D0).reshape(-1,2).T  # (2,4)
n1 = cv2.fisheye.undistortPoints(c1.reshape(-1,1,2), K1, D1).reshape(-1,2).T  # (2,4)
P0 = [I | 0]                         # 3×4, cam0 is the reference
P1 = [R_st | T_st]                   # 3×4, T_st in METRES
Xh = cv2.triangulatePoints(P0, P1, n0, n1)   # 4×4 homogeneous
pts3d = (Xh[:3] / Xh[3]).T                    # (4,3) metres, in cam0 frame
R, t = kabsch(MARKER_PTS, pts3d)              # fit model → world (see 3.5.4)
# bad-corner rejection: if one corner's fit residual >> median, drop it & refit
res = norm((R @ MARKER_PTS.T).T + t - pts3d, axis=1)
if res.max() > 3*median(res): refit kabsch on the best 3 corners
# tag-centre position = t ; orientation = R
```

- Because `n0, n1` are **normalized** (z=1) coordinates, the projection matrices
  use identity intrinsics; depth comes from the baseline (disparity), not from
  the tag's perspective. That is the source of the depth improvement.
- This is algebraic (DLT) + a rigid fit; it does not minimise pixel
  reprojection error, hence it is slightly noisier than 3.5.3.

### 3.5.3 `stereo_pnp` — joint multi-view PnP (maximum likelihood) ← recommended

One 6-DOF pose minimising **reprojection error in both cameras simultaneously**
(8 image points, 16 residuals). Initialise from the `stereo_tri` solution.

```
rvec1_st = cv2.Rodrigues(R_st)[0]            # stereo rotation as rvec
def residual(p):                              # p = [rvec(3), tvec(3)]
    rvec, tvec = p[:3], p[3:]
    # cam0: model → cam0 directly
    proj0 = project(MARKER_PTS, rvec,  tvec,  K0, D0)        # (4,2)
    # cam1: compose model→cam0 with cam0→cam1
    rvec1, tvec1 = cv2.composeRT(rvec, tvec, rvec1_st, T_st)[:2]
    proj1 = project(MARKER_PTS, rvec1, tvec1, K1, D1)        # (4,2)
    return concat([(proj0 - c0).ravel(), (proj1 - c1).ravel()])  # length 16

x0 = concat([rvec_init, tvec_init])           # from stereo_tri (Rodrigues(R), t)
sol = scipy.optimize.least_squares(residual, x0, method="lm")
rvec, tvec = sol.x[:3], sol.x[3:]
R = cv2.Rodrigues(rvec)[0]
```

- `project(...)` is the camera projection model. For the **corner-undistort**
  pipeline use the fisheye model on the raw corners:
  `cv2.fisheye.projectPoints(obj.reshape(-1,1,3), rvec, tvec, K, D)`.
  For the **full-image-undistort** pipeline the corners are already pinhole, so
  use `cv2.projectPoints(obj, rvec, tvec, K_new, zeros)`.
- `cv2.composeRT(rvecA, tvecA, rvecB, tvecB)` returns the transform equal to
  applying A (model→cam0) then B (cam0→cam1). Order matters.
- `least_squares(..., method="lm")` is Levenberg–Marquardt; the 16 residuals
  over 6 unknowns are well-conditioned given two viewpoints, which removes the
  planar depth/flip weakness. This is the maximum-likelihood pose under
  isotropic Gaussian pixel noise.

### 3.5.4 Kabsch (rigid point-set alignment)

Closed-form least-squares rotation+translation mapping `src → dst`
(`dst ≈ R·src + t`). Used both to fit the tag model in 3.5.2 and to align a
whole trajectory to mocap before scoring.

```
cs, cd = mean(src), mean(dst)
H = (src - cs).T @ (dst - cd)         # 3×3 covariance
U, S, Vt = svd(H)
R = Vt.T @ U.T
if det(R) < 0:                        # reflection fix
    Vt[2] *= -1
    R = Vt.T @ U.T
t = cd - R @ cs
```

### 3.5.5 Which to use

`stereo_pnp` on **corner-undistorted** points (3.5.1's `undistortPoints`, no
full-image remap). Best translation accuracy and rotation; ~half the
single-camera jitter. Use `stereo_tri` only to initialise it.

---

## 4. Findings

### 4.1 Does a second camera reduce jitter? — Yes, mainly in depth

Static-hold metrics (corner-undistort pipeline, sub-pixel corners, lag-corrected):

| Estimator | Trans jitter (mm) | Z (depth) jitter | Rot RMS (°) | Rot jitter (°) |
|---|---|---|---|---|
| cam0 (4 pt) | 9.2 | 7.9 | 16.2 | 0.35 |
| cam1 (4 pt) | 7.9 | 6.3 | 16.1 | 0.37 |
| stereo_tri (8 pt) | 6.5 | 5.5 | 18.4 | 0.70 |
| **stereo_pnp (8 pt, joint)** | **4.9** | **4.0** | **10.2** | **0.30** |

**Headline:** joint stereo PnP roughly **halves** single-camera jitter
(9.2 → 4.9 mm, −47%), with the win concentrated in **depth** (Z 7.9 → 4.0 mm,
−49%) and **orientation** (16° → 10°).

**Why** (theory, confirmed by the data): single-camera planar PnP on a small
tag is near-orthographic, so the optical-axis distance (depth) and out-of-plane
tilt are weakly observable → large Z/rotation jitter and occasional flip
ambiguity. A second camera with a known baseline constrains depth via disparity
and resolves the flip. The gain is **real but bounded by the 78 mm baseline** —
at arm's length the disparity is small.

**Joint PnP beats triangulation** consistently (4.9 vs 6.5 mm). The
triangulate-then-centroid estimate discards orientation and is sensitive to a
single bad corner; the joint reprojection fit is the maximum-likelihood
estimate and is worth the extra cost.

### 4.2 Corner-undistort vs full-image-undistort

Script: `verification_dual_undistort_compare.py`. Two undistortion pipelines:

- **corner**: detect tags on the raw fisheye frame, undistort only the corner
  *points* (`cv2.fisheye.undistortPoints`). No image resampling.
- **fullimg**: remap the whole image to a pinhole model
  (`initUndistortRectifyMap` + `remap`, `balance=1.0`), detect on the
  undistorted image.

| estimator | transRMS (mm) | transJIT (mm) | rotRMS (°) | rotJIT (°) |
|---|---|---|---|---|
| **corner**:cam0 | **11.6** | 9.5 | **9.0** | 0.34 |
| fullimg:cam0 | 18.5 | 8.4 | 20.9 | 0.86 |
| **corner**:cam1 | **14.4** | 8.2 | **8.8** | 0.36 |
| fullimg:cam1 | 19.8 | 9.8 | 20.4 | 0.95 |
| corner:stereo_tri | 11.6 | 6.8 | **11.3** | 0.68 |
| **fullimg**:stereo_tri | **8.5** | **4.5** | 24.7 | 3.16 |
| **corner:stereo_pnp** | **7.7** | 5.0 | **5.7** | 0.30 |
| fullimg:stereo_pnp | 8.0 | **4.3** | 10.2 | 0.82 |

**Verdict: use corner-undistort.**

- **Rotation: corner crushes full-image** (6–11° vs 20–25°, and 4–5× lower
  rotation jitter). With `balance=1.0` the full-image remap stretches the
  fisheye edge pixels hard, biasing planar-PnP orientation.
- **Translation: a wash.** Full-image edges out stereo *jitter* slightly
  (4.3 vs 5.0 mm) but is worse on single-cam *accuracy* (cam0 RMS 18.5 vs 11.6).
  For the production estimator (`stereo_pnp`) translation is a tie (~8 mm).
- **Cost:** full-image pays a per-frame bilinear `remap`; corner-undistort
  touches only 4 points. No reason to take the resampling hit.

`corner:stereo_pnp` is the best overall cell: best translation accuracy *and*
best rotation.

> Caveat: `balance=1.0` is unfavourable to full-image (keeps the most distorted
> edge pixels). `balance=0.0` would crop to the well-behaved centre and likely
> narrow the rotation gap, at the cost of FOV. Not yet swept.

---

## 5. Recommendations

1. **Production estimator: joint multi-view PnP (`stereo_pnp`)** on
   **corner-undistorted** points. Best accuracy and rotation; ~half the
   single-camera jitter.
2. **Always sub-pixel refine** corners (`cv2.cornerSubPix`) — cheap, lowers the
   noise floor for every estimator.
3. **Report jitter on static holds**, not whole trajectories — otherwise real
   motion and sub-frame sync error masquerade as jitter.
4. **Don't full-image-undistort** for pose; detect on raw and undistort points.

---

## 6. Limitations & future work (the real ceiling)

The 78 mm baseline caps the depth gain. Bigger wins, roughly in order:

1. **Multi-tag rig** — the verified recording is a *single* tag. The production
   bracket is ids `[4, 8, 12, 14, 20]` = up to 40 points across 2 cameras. A
   single joint PnP over all visible tags would collapse jitter well below
   4.9 mm. Needs a dual recording of the full bracket + mocap.
2. **Wider baseline** (~150–200 mm) — depth sensitivity scales with baseline.
3. **Stereo bundle-adjust** R, T over the tag frames to tighten extrinsics
   below the current 0.82 px epipolar error.
4. **Sweep `balance`** for the full-image pipeline to confirm corner-undistort
   stays ahead at all FOV crops.

---

## 7. File index

| File | Purpose |
|---|---|
| `notebooks/calibration/dual_notebooks/01_corner_detection.py` | Detect chessboard corners in both cameras → `.pkl` |
| `notebooks/calibration/dual_notebooks/02_dual_calibration.py` | Decomposed fisheye stereo calibration → `stereo_calibration.toml` |
| `notebooks/calibration/dual_notebooks/verification_dual_camera.py` | 4-estimator stereo verification vs mocap (main) |
| `notebooks/calibration/dual_notebooks/verification_dual_undistort_compare.py` | corner vs full-image undistort comparison |
| `notebooks/calibration/dual_notebooks/verification_single_camera.py` | original single-camera mocap verification |

Output plots are written next to the verified recording:
`verify_trajectories.png`, `verify_jitter_rms.png`, `verify_jitter_vs_depth.png`,
`verify_xz_trajectory.png`, `verify_undistort_compare.png`.
