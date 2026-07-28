# ICP trunk tracking — theory and derivation

Companion to [`07_icp_trunk.py`](07_icp_trunk.py) (estimator + validation) and
[`08_icp_video.py`](08_icp_video.py) (renderer). Geometry front-end lives in
[`06_trunk_axis.py`](06_trunk_axis.py).

This document explains *why* rigid registration is the right estimator for a seated
trunk, derives the math actually implemented, and states where the method is
ill-conditioned. Empirical results and the running work log are in
[`THINGS_DONE.md`](THINGS_DONE.md).

---

## 1. The problem, and why landmarks fail

We want the trunk's 3-DoF orientation — flexion, lateral bend, axial twist — for a
person **seated at a desk**. Hips, pelvis, and lower spine are occluded. Everything a
standard skeleton estimator relies on for trunk orientation (hip-to-shoulder vector,
pelvis frame) is unavailable.

The baseline pipeline (06) builds a trunk frame from what *is* visible:

```
lateral  x = (L_shoulder - R_shoulder) / |·|          MediaPipe, triangulated
anterior a = torso plane normal (RANSAC, EMA-smoothed) stereo cloud
up       u = a x x
R_t = [x | u | a]  in SO(3)
```

Two structural weaknesses:

1. **Axial twist is nearly unobservable.** It shows up only as foreshortening of the
   shoulder segment — a second-order effect. Measured: r = 0.42, RMSE 12.8°.
2. **The plane normal is a 2-DoF quantity fitted to a curved surface.** A plane has no
   information about rotation *about* its own normal, and the torso's curvature makes
   the fit sensitive to which patch of the mask survives segmentation.

Both weaknesses come from the same root cause: we compress a dense measurement
(~2000 3-D points on the torso) into 3 numbers (2 shoulders + 1 normal) and then
estimate a rotation from those 3 numbers.

**The ICP alternative:** the torso is (approximately) a rigid body. Its full 6-DoF pose
is already encoded in the point cloud. Estimate the rotation by aligning today's cloud
to a reference cloud directly — no landmarks, no plane, no anatomical assumptions. Use
all ~2000 points as constraints instead of 3.

---

## 2. Input: what the point cloud is

Produced per frame by `06_trunk_axis.py::_geom_chunk` → `stereo_step`, cached in the
`cloud` key. The chain:

| Step | Operation | Purpose |
|---|---|---|
| Rectify | `cv2.fisheye.stereoRectify` on the 160° pair | epipolar lines → rows |
| Disparity | SGBM, `numDisparities=160`, `blockSize=5`, 3-way mode | dense correspondence |
| Denoise | bilateral filter on disparity; validity gated on **raw** disparity | smooth Z without blurring the invalid-pixel boundary |
| Reproject | `reprojectImageTo3D(disp, Q)` / 1000 | metric 3-D, rectified-cam0 frame |
| Mask | YOLO-seg `torso` minus dilated `arm`, then eroded by `0.06·√area` | drop arms and the curved flanks |
| Front shell | keep `z ≤ percentile(z, 5) + 0.10 m` | frontmost depth layer only |
| Downsample | voxel 1 cm, mean per voxel, cap 2000 pts | uniform density, bounded cost |
| Frame | `p_board = R0ᵀ (p_cam0 − t0)`, ChArUco basis | fixed world frame, camera-motion-free |

Two choices matter theoretically:

**Front shell.** For a convex-ish torso, the frontmost depth cap is roughly the tangent
patch facing the camera. Keeping only this band removes the grazing-angle flanks where
stereo error blows up (disparity error → depth error scales as `Z²/(f·B)` and grazing
surfaces smear across many disparities).

**Voxel downsampling before registration.** ICP's least-squares objective is a sum over
points; without resampling, densely-imaged regions (near, fronto-parallel) would
silently dominate the fit. Uniform voxel means give every 1 cm³ of torso equal say —
this is a *spatial* reweighting, not just a speedup.

Note the chest-mounted reflective marker object is deliberately **kept** in the cloud.
It is rigid with the trunk, so for registration it is a feature (extra curvature,
breaks symmetry). The plane pipeline had to dodge it (`TRUNK_DROP_M = 0.12`); ICP
benefits from it.

---

## 3. Rigid registration: the estimator

### 3.1 Objective

Let the **neutral** (reference) cloud be `Q = {q_j}` — pooled from the first
`N_NEUTRAL = 15` frames, voxel-downsampled to ≤6000 points. Let frame *i*'s cloud be
`P = {p_k}`.

We seek `(R, t) ∈ SE(3)` with `R ∈ SO(3)` minimizing

```
E(R, t) = Σ_k  w_k · ρ( d( R p_k + t ,  Q ) )
```

where `d(·, Q)` is distance to the reference *surface* and `ρ` is a robust loss.

Two things are unknown simultaneously: the transform **and** which reference point each
source point corresponds to. ICP resolves this by alternating minimization — the
classic Besl–McKay scheme, formally an EM-like coordinate descent:

```
E-step  (correspondence): with (R,t) fixed, c(k) = argmin_j |R p_k + t − q_j|
M-step  (transform):      with c fixed, solve for (R,t) in closed form or one Gauss-Newton step
```

Each step is non-increasing in `E`, so the iteration converges — but only to a **local**
minimum. That property is the source of both the method's accuracy and its one serious
failure mode (§7).

### 3.2 E-step — correspondence and trimming

`scipy.spatial.cKDTree(Q).query(moved, k=1)` gives nearest neighbours in O(log |Q|)
each. Then a hard trim:

```python
d, idx = tree.query(moved, k=1)
m = d < dist            # keep only matches closer than the current threshold
```

`ρ` is therefore the truncated quadratic

```
ρ(d) = d²      if d < τ
       const   otherwise      (zero gradient → point ignored)
```

This is **trimmed ICP**. It matters because the correspondence set is not a bijection —
occlusion, segmentation differences, and the front-shell cut mean many source points
genuinely have no counterpart. Without trimming, those points pull the fit toward a
wrong solution with full leverage.

The threshold is **annealed**:

```python
ICP_DISTS = (0.06, 0.03, 0.02)   # metres
```

Twenty iterations at τ=6 cm (wide capture radius, tolerant of a poor initial guess),
then 3 cm, then 2 cm (tight, only true correspondences survive). This is graduated
non-convexity in spirit: start with a smoothed, wide-basin objective; sharpen it once
you are near the right minimum. Registration is abandoned if fewer than
`ICP_MIN_MATCH = 200` points survive the trim.

### 3.3 M-step, point-to-point — the Kabsch/Procrustes solution

With correspondences `(p_k, q_k)` and weights `w_k` (normalised to sum 1), minimise

```
E(R, t) = Σ_k w_k |R p_k + t − q_k|²
```

**Translation.** ∂E/∂t = 0 gives `t = q̄ − R p̄` with `p̄ = Σ w_k p_k`, `q̄ = Σ w_k q_k`.
Substituting back centres both clouds and eliminates `t` entirely. Write
`p'_k = p_k − p̄`, `q'_k = q_k − q̄`.

**Rotation.** The residual objective expands to

```
Σ_k w_k |R p'_k − q'_k|²  =  Σ w_k |p'_k|² + Σ w_k |q'_k|² − 2 tr(Rᵀ H),
                            H = Σ_k w_k p'_k q'_kᵀ
```

using `‖Rp‖ = ‖p‖` (R orthogonal). Only the last term depends on R, so minimising E is
**maximising `tr(Rᵀ H)`** — the orthogonal Procrustes problem.

Take the SVD `H = U Σ Vᵀ`. Then `tr(Rᵀ H) = tr(Rᵀ U Σ Vᵀ) = tr(Vᵀ Rᵀ U Σ) = tr(Z Σ)`
with `Z = Vᵀ Rᵀ U` orthogonal. Since `Σ = diag(σ₁,σ₂,σ₃)` with `σ_i ≥ 0` and any
orthogonal `Z` has `|Z_ii| ≤ 1`, the trace `Σ_i Z_ii σ_i` is maximised at `Z = I`,
giving `R = V Uᵀ`.

**Reflection guard.** `V Uᵀ` is orthogonal but may have determinant −1 (a reflection,
not a rotation) — this happens with noisy or near-degenerate configurations. Constrain
`det R = +1` by flipping the sign of the smallest-singular-value direction:

```python
D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
R = Vt.T @ D @ U.T                          #  R = V D Uᵀ
```

That is `kabsch()` verbatim. It is a *closed-form global* optimum of the M-step — no
iteration, no initial guess needed inside the step.

### 3.4 M-step, point-to-plane — what is actually used

Point-to-point penalises sliding along the surface, which is wrong: two scans of the
same smooth torso have no reason to sample the same physical points. Point-to-plane
(Chen–Medioni) penalises only the component of the residual **along the surface normal**:

```
E(R, t) = Σ_k w_k ( n_kᵀ (R p_k + t − q_k) )²
```

Tangential slip is free, so the estimator converges in far fewer iterations and is much
less biased by non-overlapping sampling. This is the default path in `icp()`
(`dst_normals` is always supplied for both the neutral cloud and keyframes).

**Normals** come from local PCA (`estimate_normals`): for each point take the `k = 15`
nearest neighbours, form the 3×3 covariance

```
C = (1/k) Σ (x_j − x̄)(x_j − x̄)ᵀ
```

and take the eigenvector of the **smallest** eigenvalue — the direction of least spread,
i.e. the surface normal. `np.linalg.eigh` returns ascending eigenvalues, so it is column
0. Signs are then made consistent by orienting outward from the cloud centroid, valid
because the front shell is convex-ish:

```python
flip = np.einsum("ni,ni->n", normals, outward) < 0
normals[flip] *= -1
```

**Linearisation.** The point-to-plane objective has no closed form, so take one
Gauss-Newton step per iteration. Parametrise the update multiplicatively about the
current estimate `R`:

```
R⁺ = exp(δ^) R  ≈  (I + δ^) R      for small |δ|
```

where `δ ∈ ℝ³` is a rotation vector and `δ^` its skew-symmetric matrix. Writing
`r_k = R p_k` (rotation applied, translation not):

```
R⁺ p_k + t⁺ − q_k ≈ (r_k + t − q_k) + δ × r_k + Δt
```

Project onto `n_k` and use the scalar triple-product identity
`n · (δ × r) = δ · (r × n)`:

```
residual_k(δ, Δt) ≈ n_kᵀ(r_k + t − q_k)  +  (r_k × n_k)ᵀ δ  +  n_kᵀ Δt
                    └──── e_k (const) ──┘  └──── J_k [δ; Δt] ────┘
```

So each correspondence contributes one row `J_k = [ (r_k × n_k)ᵀ , n_kᵀ ] ∈ ℝ^{1×6}`,
and the step solves the 6-parameter weighted linear least squares `min ‖√W (J x + e)‖²`:

```python
resid = np.einsum("ni,ni->n", n, rp + t - q)         # e
A = np.concatenate([np.cross(rp, n), n], axis=1)     # J
x, *_ = np.linalg.lstsq(A * sw[:, None], -resid * sw, rcond=None)
return x[:3], x[3:]                                  # δ, Δt
```

Then `R ← exp(δ^) R` via `Rotation.from_rotvec(δ).as_matrix() @ R`, and `t ← t + Δt`.
Multiplicative update on the left keeps `R` exactly in SO(3) — no re-orthonormalisation
drift.

**Convergence test** — geodesic distance on SO(3) between successive iterates:

```
θ = arccos( (tr(R⁺ Rᵀ) − 1) / 2 )
```

`R⁺ Rᵀ` is the incremental rotation; its trace is `1 + 2cos θ`. Stop when θ < 0.01°.

### 3.5 Conditioning — where the information comes from

The Gauss-Newton normal matrix `JᵀWJ` (6×6) is the Fisher information of the pose given
the correspondences. Its small eigenvalues are the unobservable directions.

For point-to-plane, the rotational block is built from `r_k × n_k`. Consequences:

- A **perfectly planar** surface has all `n_k` parallel ⇒ `r_k × n_k` spans only 2
  dimensions ⇒ rotation about the normal is unobservable, plus in-plane translation.
  Three DoF gone. This is exactly the degeneracy the plane pipeline suffers from.
- A **curved** surface produces varying `n_k` and recovers the missing directions. The
  torso's mediolateral curvature plus the chest marker object supply this.

So ICP's advantage over the plane fit is not "more points" per se — it is that ICP
*uses the curvature* the plane fit throws away. The axial axis, worst for the plane
method (r = 0.42), is the one the curvature constrains, and ICP scores r = 0.89 on it.

---

## 4. Temporal structure: warm start, keyframes, odometry

Each frame's registration is initialised from the previous frame's accepted transform:

```python
A = (np.eye(3), np.zeros(3))               # cur -> neutral, warm start
res = icp(src, neu, tree, *A, dst_normals=neu_normals)
```

At ~30 fps a trunk moves a fraction of a degree per frame, so the warm start lands deep
inside the correct basin of attraction. This is what makes a purely local method work
without any global initialisation (no FPFH/RANSAC pre-alignment).

**Direct registration** (frame → neutral) is preferred whenever it converges:

```python
DIRECT_RMS_M, DIRECT_FRAC = 0.030, 0.30    # accept if rms ≤ 3cm and ≥30% of pts matched
```

**Keyframe odometry** is the fallback, used when the trunk has turned far enough that
the live front shell no longer overlaps the neutral view (self-occlusion — a genuinely
different piece of the body surface is visible). Then register frame → last-good
keyframe and compose:

```
A_cur→neutral = A_key→neutral ∘ A_cur→key
     R = R_k R_rel ,   t = R_k t_rel + t_k
```

with tighter acceptance (`ODO_RMS_M = 0.010`, `ODO_FRAC = 0.50`), because composition
propagates error.

**Why direct is preferred so strongly.** Odometry is a product of estimates. Errors
compose multiplicatively and there is no absolute reference to pull the estimate back —
a random walk on SO(3) with no restoring force. Empirically it twisted the whole series
by ~138° when allowed to run for 700+ frames. Hence `ODO_MAX_RUN = 30` (~1 s) and loose
direct thresholds: *any* convergence to neutral beats accumulating drift. In the final
run odometry fires **zero** times (1604 direct of 1667).

---

## 5. From transform to anatomical angles

### 5.1 Body rotation

`A = (R, t)` maps the **current** cloud into the **neutral** cloud's frame. So the
rotation the body underwent from neutral to now is the inverse:

```python
Rb_icp[i] = A[0].T            # R_b : neutral -> current
```

Translation `t` is discarded — the subject shifting in the chair is not a trunk angle.

### 5.2 Anatomical axes

ICP knows nothing about anatomy; it returns a rotation in the ChArUco board frame. The
anatomical labels come from the *neutral* trunk frame `R_neu` computed by the 06
pipeline (mean over the first 15 frames, `Rotation.mean()` — the chordal / Fréchet mean
on SO(3), i.e. Procrustes-projected average of the rotation matrices):

```python
R_t(i) = R_b(i) @ R_neu
```

This is a clean division of labour, and worth stating explicitly:
**ICP supplies the rotation; the shoulder-and-plane pipeline supplies only the axis
labelling, once, at neutral.** The jitter-prone landmark estimate is used exactly once
on a 15-frame average, where its noise averages down, and never again per-frame.

### 5.3 Angle extraction

`06_trunk_axis.py::trunk_angles`, with `R_neu = [x₀ | u₀ | a₀]` and
`R_t = [x | u | a]`:

```
flexion  = atan2( u·a₀ , u·u₀ )     forward/back tilt of the up-axis, in the neutral sagittal plane
lateral  = atan2( u·x₀ , u·u₀ )     side tilt of the up-axis,       in the neutral frontal plane
axial    = atan2( x·a₀ , x·x₀ )     twist of the lateral axis about up
```

Geometrically: project the current up-axis onto the neutral sagittal and frontal planes
and read off its inclination; project the current lateral axis to read the twist.

**Caveat, stated plainly:** this is *not* an Euler decomposition. The three angles are
independent projections, not a sequential factorisation of `R_b`, and they do not
compose. For small-to-moderate rotations (which is the entire seated regime) they agree
with the intuitive clinical definitions and are individually well-behaved; for large
compound rotations they are only approximately separable. The frame-invariant
`|rotvec(R_b)|` (total rotation magnitude) is reported alongside precisely because it
carries no convention at all.

### 5.4 Plausibility gate

```python
ROT_MAX_DEG = 45.0
if rot_deg(got[0]) > ROT_MAX_DEG:
    gated[i] = True
```

with `rot_deg(R) = arccos((tr R − 1)/2)` — the geodesic distance from identity, the
frame-invariant total rotation.

The reasoning is a prior, not a residual test. A seated person at a desk does not reach
45° of total trunk rotation from neutral. A registration reporting more has converged
to a wrong minimum, regardless of how good its residual looks. This is the only place in
the estimator where anatomy enters as a constraint, and it is deliberately a *hard
physical bound*, not a tuned threshold.

Critically, the gated pose is still kept as the warm start for the next frame
(`A = got; continue`), so the tracker can walk back out of a bad basin rather than being
stranded at the last good pose.

---

## 6. Validation against mocap — the convention problem

### 6.1 Hardware sync

Column 0 of `cam0_timestamp.msgpack` is the GPIO pin-17 bit, held HIGH while Motive
records. The first rising edge **is** mocap `seconds = 0`:

```python
rise = int(np.argmax(sync == 1))
mt = ts0[rise] + mocap_seconds
```

No cross-correlation. The two clocks share a wire; estimating the offset from the data
would fold sync error into the accuracy figure being measured.

### 6.2 Teleport segmentation

Motive's solved rigid body can vanish for 1–3 frames during occlusion and reappear at a
ghost pose hundreds of mm away (this take: 326 mm across 10 ms). Rotations on either
side of a re-lock carry different constant offsets, so validation uses only the longest
teleport-free segment, cut wherever

```
|m₁(t_{j+1}) − m₁(t_j)| > SEG_JUMP_M = 0.10 m
```

Result: segments `0.0–1.7s, 1.7–26.4s, 26.4–50.9s` → the middle one is used.

### 6.3 Convention-free comparison

The camera trunk frame and the mocap rigid-body frame are built from different things
(shoulders + plane normal vs. Marker1/Marker4/Marker2). Their axis labellings, handedness
conventions and zero poses differ by an unknown constant rotation. Comparing them
directly measures that constant, not the tracking.

Fix: compare only **rotation from each system's own zero**:

```
camera:  R_i(t) = R_b(t) R_z,iᵀ
mocap:   R_m(t) = R_f(t) R_z,mᵀ
```

where `R_z,·` is the Fréchet mean over a common zero window. Any constant frame offset
cancels in the right-multiplied inverse. What remains is a pure **change of basis**
between the two: if the same physical rotation is expressed in two frames related by
`S`, then

```
R_i(t) = S R_m(t) Sᵀ
```

`S` is the residual rotation between the camera's board frame and the mocap frame after
the ChArUco basis has been applied — i.e. the ChArUco basis error.

### 6.4 Estimating S

Under conjugation, the **rotation vector transforms as an ordinary vector**: if
`R_i = S R_m Sᵀ`, then `rotvec(R_i) = S · rotvec(R_m)`. (The axis rotates with the frame;
the angle is invariant.) So estimating `S` reduces to Procrustes on rotation-vector
pairs — the same `kabsch()`, no translation used:

```python
ra = Rotation.from_matrix(Rb_icp[i] @ Rz_i.T).as_rotvec()
rm = Rotation.from_matrix(R_moc[i]  @ Rz_m.T).as_rotvec()
if |ra| > 5° and |rm| > 5°: pairs.append(...)
S, _ = kabsch(rm_all, ra_all)
```

The 5° excitation gate matters: near-identity rotations have numerically ill-defined
axes (the rotvec norm → 0 and its direction is dominated by noise), so they carry no
information about `S` but plenty of noise.

**Held-out validation.** `S` is 3 free parameters fitted on data — so it is fitted on the
**first half** of the segment and scored on the **second**:

```
held-out (S fit on 1st half, scored on 2nd):
  flexion : r = +0.81   RMSE = 2.6°
  lateral : r = +0.96   RMSE = 4.7°
  axial   : r = +0.95   RMSE = 1.4°
```

This is the number that means something. `S` cannot manufacture agreement it did not
earn, because it never saw the scoring half.

### 6.5 What S also tells us

```
det(Rm_c) = +1.000    basis residual rot S = 109.8°   (half-fit differs 35.1°)
```

A 110° residual is a **diagnosis, not a nuisance parameter**: the ChArUco `[mocap]` basis
rotation in the TOML is wrong by roughly that much. The determinant check confirms it is
a proper rotation (not a mirrored/handedness bug). The 35° split-half disagreement says
`S` is not a single constant frame error — it is partly absorbing model mismatch, which
is why the numbers are reported held-out and why the basis must be **re-derived from a
dedicated clean recording, never patched from a data fit.**

---

## 7. Failure modes

### 7.1 The wrong minimum (the real one)

After ~36 s the estimator loses the trunk — and does so **confidently**:

```
ICP valid 1219/1667 (direct 1604, odometry 0, gated 385)   rms 7.2mm
```

385 of the 448 invalid frames are gate rejections, not convergence failures. At frame
1300 the registration reports **rms 12.7 mm with 84 % of points matched** — numbers that
would pass any residual-based health check — onto a pose >45° from neutral.

Mechanism: after a large axial turn, the visible front shell is a *different piece of the
body*. It has no correct alignment to the neutral cloud, but it does have a
locally-optimal one — flank curvature can be slid onto chest curvature and produce a low
residual. ICP is a local method; it has no way to know it is in the wrong basin.

Three consequences, all load-bearing:

1. **Residual is not a health metric.** rms and match-fraction are necessary, not
   sufficient. The plausibility gate is doing the real work.
2. **The fix is not a looser gate.** It is a neutral model that *covers* the turned-away
   torso — multi-view fusion, or a full torso model rather than a single-view front
   shell — or a protocol that constrains the movement range.
3. **The renderer must not hide it.** `08_icp_video.py` draws gated frames in orange with
   `REJECTED: >45 deg from neutral`, and uses `smooth_masked()` (Savitzky-Golay, then
   the NaN mask restored) so a tracking loss stays blank instead of the smoother's
   NaN interpolation clamping to the last value and drawing a confident flat line.

### 7.2 Rigid-body assumption

The torso is not rigid — breathing, scapular motion, clothing shift. These are modelled
as noise absorbed by the trimming. Registration rms of ~7 mm on a ~1400-point match is
consistent with that being small relative to stereo noise.

### 7.3 Segmentation contamination

An arm crossing the chest injects points that move independently of the trunk. The
multi-class YOLO model exists to subtract a dilated `arm` mask from `torso` for exactly
this reason. Suspected cause of the ~5 s and ~9 s camera-only 15–18° flexion bouts where
mocap shows only ~4°.

### 7.4 Zero-window dependence

Everything is relative to a 15-frame neutral. If the subject is not actually neutral
during those frames, every angle carries a constant offset. Frame-invariant magnitude
(§5.3) is the diagnostic that is immune to this.

---

## 8. Results

Longest clean mocap segment, full-segment scoring:

| axis | plane r | plane RMSE | **ICP r** | **ICP RMSE** |
|---|---|---|---|---|
| flexion | 0.33 | 9.6° | **0.27** | **5.5°** |
| lateral | 0.62 | 7.2° | **0.94** | **2.9°** |
| axial   | 0.42 | 12.8° | **0.89** | **2.9°** |

Frame-invariant rotation magnitude: r = 0.55, RMSE 4.9°.

Reading: ICP roughly **halves the error on every axis** and transforms axial twist from
unusable (0.42) to good (0.89) — the axis the plane method structurally cannot see.
Flexion's correlation stays low while its RMSE halves, which is the signature of a small
true range of motion: with little signal, `r` is dominated by noise even when absolute
error is small. RMSE is the meaningful figure there.

---

## 9. Notation

| Symbol | Code | Meaning |
|---|---|---|
| `P`, `Q` | `src`, `neu` | source (current-frame) and reference (neutral) clouds |
| `R, t` | `A = (R, t)` | rigid transform mapping **current → neutral** |
| `R_b` | `Rb_icp[i]` | body rotation neutral → current, `= Rᵀ` |
| `R_neu` | `R_neu` | neutral anatomical frame `[lateral \| up \| anterior]` from 06 |
| `R_t` | — | current trunk frame, `R_b R_neu` |
| `R_z,i`, `R_z,m` | `Rz_i`, `Rz_m` | Fréchet-mean zero pose, camera / mocap |
| `S` | `S` | residual basis rotation, camera ← mocap |
| `n_k` | `dst_normals` | reference surface normal at match `k` |
| `δ, Δt` | `dvec, dt` | Gauss-Newton rotation-vector and translation increment |
| `τ` | `ICP_DISTS` | trimming threshold (annealed 6 → 3 → 2 cm) |

---

## 10. Reproducing

```powershell
$env:TRUNK_GEOM_CACHE = "...\dual_160_trunk_ragav\geom_cache.pkl"
uv run python trunkpose\dual_notebooks\07_icp_trunk.py    # numbers + icp_trunk_angles.png
uv run python trunkpose\dual_notebooks\08_icp_video.py    # icp_trunk_video.mp4

$env:TRUNK_VIDEO_FRAMES = "24"    # 08 only: cap the RENDER; all series stay full-length
```

The geometry cache holds the per-frame clouds, so the stereo/segmentation front-end runs
once; all registration and validation experiments reuse it.

---

## 11. Key references

- P. J. Besl, N. D. McKay, *A Method for Registration of 3-D Shapes*, IEEE TPAMI 14(2),
  1992 — the alternating-minimisation ICP scheme.
- Y. Chen, G. Medioni, *Object Modelling by Registration of Multiple Range Images*,
  Image and Vision Computing 10(3), 1992 — point-to-plane metric.
- W. Kabsch, *A solution for the best rotation to relate two sets of vectors*,
  Acta Cryst. A32, 1976; A34, 1978 (determinant correction).
- K. S. Arun, T. S. Huang, S. D. Blostein, *Least-Squares Fitting of Two 3-D Point Sets*,
  IEEE TPAMI 9(5), 1987 — the SVD solution and its reflection failure case.
- D. Chetverikov et al., *The Trimmed Iterative Closest Point Algorithm*, ICPR 2002.
- S. Rusinkiewicz, M. Levoy, *Efficient Variants of the ICP Algorithm*, 3DIM 2001 —
  sampling, weighting and rejection strategies; conditioning discussion.
