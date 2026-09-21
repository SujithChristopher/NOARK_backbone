# Static jitter versus tag-subset geometry — design

Date: 2026-09-21
Status: approved design, not yet implemented
Data: `data/dome/sep18_26/dome_static_burst_sep18_26`

## Problem

`05_static_jitter.py` compares jitter across camera count and tag count only.
Within a tag-count cell the *geometry* of the chosen tags is uncontrolled:
`select_burst_tags` takes the largest, most-visible tags in each burst, so a
"2-tag" number is whichever pair happened to look best, with its baseline,
facet orientations and distance all free to vary between bursts.

The open question is whether tag separation and relative orientation change
jitter, and by how much — for example whether fusing two widely separated tags
is better or worse than two adjacent ones.

## What the dome can and cannot answer

Tag geometry from `data/dome/sep18_26/dome_rb_def/rigidbody_calibration.toml`:

- 21 detected ids, core ring structure: tag 1 at the pole, ~59 mm to ring 1
  (tags 2-5), ~110 mm to ring 2 (tags 7-13).
- Pair baseline range 59-197 mm.
- Pair normal-angle range 29.7-135.8 deg.
- **corr(baseline, normal angle) = 0.98.**

The tags lie on a sphere of radius R ~ 115 mm, where chord = 2 R sin(dtheta/2)
exactly. Baseline and orientation spread are therefore one variable, not two,
for any pair on this dome. No regression on this take can separate them.

Two consequences, both of which the work must state rather than paper over:

1. Real data yields the *fused* effect of separation and orientation. Where two
   predictors cannot be separated they are fitted as a single composite spread
   term and the output says so.
2. Separating them requires geometry the dome does not have, which is what the
   simulator is for. Subsets of 3+ tags give partial decoupling (equal max
   baseline, different coplanarity) and are worth reporting, but they are
   lower-powered than the simulated sweep.

## Data budget

Detection cache `jitter_detections.pkl` over 2500 frames (50 bursts x 50):

- cam0 6.93 tags/frame (min 0, max 10), cam1 6.98.
- 6.11 tags visible in **both** cameras per frame, max 9.

Enumerating subsets of size 1-4 gives roughly 90 subsets per burst mono and
~60 stereo, so about 9000 (burst x subset x camera-config) cells. A 50-frame
standard deviation carries ~10% sampling error, so log-jitter has ~0.1 SD
measurement noise — adequate for regression.

## Decisions taken

| Decision | Choice |
|---|---|
| Purpose | Predictive design model, not just a ranking |
| Evaluation point | One fixed dome point for every subset; lever arm is part of the answer and is carried as an explicit predictor |
| Response | Position **and** rotation jitter |
| Collinearity | Report the confound from real data **and** decouple it in simulation |
| Camera configs | cam0 mono and stereo |
| Enumeration | All visible subsets of size 1-4 |
| Filtering | None. No temporal filtering anywhere, same as `05` |
| Subset selection | Frozen per burst. Never re-selected per frame — switching estimators inside a burst lands in the standard deviation as if it were jitter |

## Component 1: `jitter_model/common.py`

`05_static_jitter.py` and `06_movement_error.py` already carry verbatim copies
of the detection cache I/O, the tag-board construction, and the three pose
estimators. A third copy is where they start to drift, so these move into a
shared module first:

- detection cache load/build, timestamp records, camera pairing
- `marker_corners_reference` / `marker_to_reference` construction
- `stack_correspondences`, `raw_reprojection_rmse`, `single_tag_pose`,
  `seed_pose`, `mono_board_pose`, `stereo_board_pose`
- session/calibration loading

`05` and `06` are updated to import from it. Their numeric output must not
change; this is verified by re-running both and diffing their CSVs against the
committed ones.

## Component 2: `jitter_model/08_subset_geometry.py`

### Loop

For each burst that passes the existing stillness gate
(`MAX_BURST_MOCAP_TRAVEL_M`, `MIN_FRAMES_PER_BURST`):

1. Tags visible in at least `TAG_PRESENCE_FRACTION` of that burst's frames.
2. Every subset of size 1-4 of those tags, frozen for the burst.
3. Every frame solved with the shared estimators: `mono_board_pose` on cam0,
   `stereo_board_pose` for the stereo config (stereo subsets require the whole
   set visible in both cameras).

### Response variables

Position, at one fixed physical point for every subset:

    X_i = R_i @ P_FIXED + t_i
    pos_jitter_mm = norm(std(X_i, axis=0, ddof=1)) * 1000

`P_FIXED` defaults to the tag-1 origin (zero in board frame) so numbers stay
comparable with `05`; it is a module constant and may be repointed at the dome
centre.

Rotation, via a chordal mean so nothing wraps:

    R_bar     = SVD/quaternion mean of R_i
    dtheta_i  = rotvec(R_bar.T @ R_i)
    rot_jitter_mdeg = norm(std(dtheta_i, axis=0, ddof=1)) in millidegrees

Per-axis components of both are kept alongside the norms.

### Row schema (`subset_geometry_cells.csv`)

Keys: `burst`, `camera_config` (cam0 or stereo), `marker_ids`, `n_tags`,
`frames`.

Exact geometry, from the rigid-body TOML, constant across bursts:

- `max_baseline_mm` — max pairwise tag-centre distance
- `rms_radius_mm` — RMS tag-centre distance from the subset centroid
- `min_singular_mm` — smallest singular value of the mean-centred corner cloud
  (non-coplanarity; near zero for one tag or coplanar tags)
- `max_normal_angle_deg`, `mean_normal_angle_deg` — pairwise tag-normal angles
- `lever_mm` — distance from subset centroid to `P_FIXED`
- `n_corners` — 4 * n_tags

Pose-dependent, from the burst median pose:

- `distance_m`, `mean_incidence_deg`, `min_incidence_deg`,
  `mean_apparent_size_px`, `reprojection_px`

Response: `pos_jitter_mm` + 3 axes, `rot_jitter_mdeg` + 3 axes.

### Model

Stage 1, within burst. `log(pos_jitter_mm)` on the geometry predictors with a
per-burst intercept, implemented as within-burst demeaning followed by
`numpy.linalg.lstsq` (no new dependency). The fixed effect absorbs pose,
distance, lighting and stillness exactly, so each geometry coefficient is
identified only from subsets compared against each other inside the same burst
and the same 50 frames. Weights proportional to `2 * (frames - 1)`, the known
sampling precision of a standard deviation. Standard errors clustered by burst.
Repeated with `log(rot_jitter_mdeg)` as the response.

Stage 2, between bursts. `distance_m` and the incidence terms barely vary
within a burst, so they are estimated from burst-level means of the stage-1
residuals.

Collinearity is reported, never hidden: a printed predictor correlation matrix,
the measured baseline/normal-angle correlation over the surviving subsets, and
variance-inflation factors. Predictors whose VIF exceeds a stated threshold are
collapsed into one composite spread term and the script prints that it did so.

### Caveats the script must print

- **Selection effect.** A subset exists only when its tags are visible, and
  visibility correlates with incidence angle. Burst fixed effects do not remove
  this.
- **Lever identification.** `lever_mm` is constant per subset across bursts and
  is identified only relative to `P_FIXED`. With a fixed evaluation point the
  lever coefficient and the baseline coefficient are partly collinear, since
  wider subsets sit further from tag 1. Reporting the baseline effect without
  the lever term in the model would attribute "far from the reported point" to
  "widely separated".

### Outputs

- `subset_geometry_cells.csv`, `subset_geometry_model.csv` (coefficients, CIs,
  VIFs)
- Figures: jitter vs baseline coloured by `n_tags`; jitter vs lever; predictor
  correlation matrix; partial-residual plot per predictor; position and
  rotation side by side.

## Component 3: `jitter_model/09_layout_simulator.py`

### Engine

For a layout (tag poses in board frame), a camera pose and a corner-noise level
sigma_px: project corners with `cv2.fisheye.projectPoints` using the real K and
D, add Gaussian noise, re-solve with the **same** shared estimators, repeat M
times, take the spread at `P_FIXED`. Same estimator and same evaluation point
as component 2, so simulated and measured numbers are directly comparable.

### sigma_px calibration

Fitted from real reprojection residuals per burst, then modelled as a function
of apparent tag size — corner noise grows as tags shrink, so a single global
constant would misstate the far bursts. The size relationship is fitted on the
cells produced by component 2.

### Validation gate

Every real cell from component 2 is replayed through the simulator at its own
geometry, distance and sigma_px. Simulated vs measured jitter is plotted
log-log. The simulator is accepted only if the slope is approximately 1 and the
scatter is comparable to the ~10% sampling noise of the measured stds.

If validation fails the script reports the failure and stops. An unvalidated
simulator is not used as a design tool, and a systematic gap is itself a
finding: it would mean corner noise is not the dominant error source and that
calibration error or detector bias carries the rest.

### Design sweep

Only after the gate passes:

- Sphere radius R swept from flat (very large R) to the real 115 mm, **holding
  baseline fixed** — this is the construction that breaks the r = 0.98 tie.
- Baseline swept at fixed curvature.
- Both crossed with n_tags 1-4 and with working distance.

Output: for a given tag count and working distance, the separation and facet
tilt that minimise jitter at the tracked point.

All filenames and figure titles from this script are labelled as simulation.

## Parallelization

Per-burst work units in a `concurrent.futures.ProcessPoolExecutor`, `tqdm` over
`as_completed`. Two Windows specifics are designed around rather than
discovered later:

- Windows uses **spawn**, so a child re-imports the parent module. These are
  cell-style scripts, and a re-import would re-run the whole analysis in every
  worker. The worker therefore lives in its own small module,
  `jitter_model/_subset_worker.py`, importing only `common.py`, and the main
  script keeps its compute call behind a `__main__` guard. Cell-by-cell
  execution in a kernel still works; it takes the serial path.
- `cv2.setNumThreads(0)` in each worker. OpenCV's internal threads otherwise
  contend with the pool and the run gets slower, not faster.

Workers load the detection cache in a pool initializer rather than receiving
3.2 MB per task. Expected speedup is near-linear in core count: the stereo pass
should fall from roughly 15 min single-threaded to a couple of minutes.

## Success criteria

1. `05` and `06` produce identical CSV output after the `common.py` extraction.
2. `08` produces a cells CSV of the expected order (~9000 rows) and a model
   whose baseline and lever coefficients carry cluster-robust CIs.
3. The collinearity between baseline and normal angle is measured and printed,
   and any composite term is named in the output.
4. `09` either passes its validation gate and produces the decoupled sweep, or
   fails it and says so without producing design recommendations.

## Out of scope

- Any temporal filtering.
- Per-frame subset re-selection.
- Re-running detection: the cached detections are reused as-is.
- Changes to the movement-error analysis in `06` beyond the shared-module
  extraction.
