# rapidtag: Recommended Usage

How to get the fastest AprilTag detection out of rapidtag without losing
accuracy, for the dual OV9281 rig (1280×800, `DICT_APRILTAG_36h11`).

All numbers below were measured on the dome static-burst recording
`data/dome/sep18_26/dome_static_burst_sep18_26` (50 bursts, 2 cameras) on the
desktop dev machine. Accuracy is measured as agreement with OpenCV's full-frame
`CORNER_REFINE_APRILTAG`, not against mocap ground truth.

---

## TL;DR

1. Install rapidtag from the submodule (`external/rapidtag`), not PyPI.
2. Use crop refinement: `corner_refinement_method = CORNER_REFINE_NONE` +
   `april_tag_refine_full_resolution = True`.
3. Crop the image to **one ROI around the whole rigid body** (not one per tag),
   copy it with `np.ascontiguousarray`, and shift the corners back.
4. Detect **both cameras in one `detect_markers_batch` call**.
5. Fall back to a full-frame batch when a camera loses tags or the pose.

Result: about **1.0 ms for both cameras**, compared with about 6.1 ms for
full-frame detection run one camera at a time. Corners are identical.

---

## 1. Install

The PyPI wheel (0.1.4) has no `april_tag_refine_full_resolution`. Install the
submodule build (0.1.12+):

```bash
uv pip install ./external/rapidtag
```

Check:

```python
assert hasattr(rapidtag.DetectorParameters(), "april_tag_refine_full_resolution")
```

## 2. Corner refinement: use crop refinement

```python
params = rapidtag.DetectorParameters()
params.corner_refinement_method = rapidtag.CORNER_REFINE_NONE
params.april_tag_refine_full_resolution = True
```

rapidtag first finds tags with its fast detector, then refits each tag's
corners with an AprilTag line fit on a small full-resolution crop around that
tag. Source: `jitter_model/10_crop_refine_accuracy.py`.

Corner error vs OpenCV full-frame AprilTag refinement:

| method | median | p95 | > 1 px | ms / frame |
|---|---|---|---|---|
| OpenCV `CORNER_REFINE_APRILTAG` (reference) | – | – | – | 180–195 |
| **rapidtag crop refine** | **0.0004 px** | **0.003 px** | **0.75–1.5 %** | **4.1** |
| rapidtag raw + `cv2.cornerSubPix` (5×5) | ~1 px | 2.3–3.5 px | 50–60 % | 3.2 |
| rapidtag raw | ~1 px | 2.3–3.5 px | 50–60 % | 3.1 |

Pose effect:
- Crop refine jitter is within ~10 % of the reference, and the median
  per-burst bias is ~0.0003 mm.
- `cornerSubPix` has similar jitter but a **systematic 1–3 mm bias**. Raw
  corners give 6–9× more jitter and 2–5 mm bias.

Known issues with crop refine:
- **Rare outliers.** About 1 % of corners are off by up to ~12 px, and the
  worst single frame pose was 15.6 mm / 7° off. A cheap guard to try (untested):
  reject a tag whose crop-refined corners move more than ~1 px from its raw
  corners.
- **Detector recall.** All rapidtag modes miss ~15 % of the tags OpenCV finds
  and add ~4–5 % extra. Crop refinement cannot recover a tag the detector missed.

## 3. Search a rigid-body ROI, not the full frame

Refinement is cheap (~0.5 ms). Most of the time goes to searching the whole
1280×800 frame. rapidtag has **no ROI parameter**: `RigidBody` /
`estimate_rigid_body_pose` only solve the pose and do not limit the search.
Crop in Python instead. The copy costs ~0.008 ms, so building the crop into
rapidtag would gain nothing.

ROI = bounding box of the rigid body's tag corners + margin × body size, with
the origin snapped down to a multiple of 8 px. Per camera, crop refine,
~500 frames:

| search area | pixels | cam0 ms | cam1 ms | tags lost vs full |
|---|---|---|---|---|
| full frame | 100 % | 3.3 | 3.2–5.6 | – |
| body ROI, margin 0.15 | 7 % | 0.86 | 0.87–1.5 | ~5 % |
| **body ROI, margin 0.5** | 16 % | **1.15** | **1.1–2.0** | **~1 %** |
| one ROI per tag, margin 0.5 × tag | 11 % | 2.20 | 2.22 | ~11 % |
| one ROI per tag, margin 1.0 × tag | 26 % | 3.10 | 3.03 | ~10 % |

Corners from any ROI match the full-frame corners (p99 diff 0.0001 px).

Use **one body ROI, not one ROI per tag**:
- Each rapidtag call costs about 0.35 ms whatever its size, so 6 tags means
  6 × that overhead.
- A per-tag ROI only searches where tags already were, so a tag that was not
  found in the last frame is never picked up again. The body ROI covers the
  gaps between tags.

## 4. Pitfalls that cost time

| do | median ms |
|---|---|
| pass a numpy slice `gray[y0:y1, x0:x1]` directly | **2.33** |
| pass `np.ascontiguousarray(gray[y0:y1, x0:x1])` | **0.80** |
| 2 cameras, two `detect_markers` calls | 1.66 |
| 2 cameras, one `detect_markers_batch` call | **1.01** |
| full frame, 2 cameras, two calls | 6.06 |
| full frame, 2 cameras, one batch call | 3.46 |

- **Always pass contiguous arrays.** For a non-contiguous view, rapidtag copies
  the pixels one at a time in Rust (`external/rapidtag/src/lib.rs`,
  `extract_frame`), which makes detection ~3× slower.
- **Batch the cameras.** `detect_markers_batch` releases the GIL and runs the
  frames in parallel on rayon threads.
- **Don't wrap rapidtag in your own thread pool.** It already uses rayon
  internally. The thread pool is pinned to the fast cores; `RAPIDTAG_CORES`
  overrides that.

## 5. Recommended loop

```python
import numpy as np
import rapidtag

DICT = "DICT_APRILTAG_36h11"
MARGIN = 0.5      # fraction of body bbox size per side; 0.15 if the ROI is pose-predicted
ALIGN = 8         # snap ROI origin to the detector's tile grid

params = rapidtag.DetectorParameters()
params.corner_refinement_method = rapidtag.CORNER_REFINE_NONE
params.april_tag_refine_full_resolution = True


def body_roi(corners, shape, margin=MARGIN):
    """Bounding box of all rig tag corners (N, 2) plus margin, clipped to the image."""
    (x0, y0), (x1, y1) = corners.min(0), corners.max(0)
    m = margin * max(x1 - x0, y1 - y0)
    h, w = shape
    x0 = int(max(0, x0 - m)) // ALIGN * ALIGN
    y0 = int(max(0, y0 - m)) // ALIGN * ALIGN
    return x0, y0, int(min(w, x1 + m + 1)), int(min(h, y1 + m + 1))


def detect(grays, rois, rig_ids):
    """grays: one gray frame per camera; rois: (x0, y0, x1, y1) or None (full frame)."""
    images, offsets = [], []
    for gray, roi in zip(grays, rois):
        if roi is None:
            images.append(gray)  # already contiguous
            offsets.append((0, 0))
        else:
            x0, y0, x1, y1 = roi
            images.append(np.ascontiguousarray(gray[y0:y1, x0:x1]))
            offsets.append((x0, y0))

    out = []
    for (corners, ids), offset in zip(
        rapidtag.detect_markers_batch(images, DICT, params), offsets
    ):
        out.append({
            int(i): np.asarray(c, np.float64).reshape(4, 2) + offset
            for i, c in zip(ids, corners)
            if int(i) in rig_ids
        })
    return out
```

Per frame:
1. Build each camera's ROI from the previous frame's tags (`body_roi`), or
   `None` for a full-frame search.
2. `detect(...)`.
3. **Fallback.** If a camera found fewer tags than last frame or the pose
   failed, search that camera on the full frame next frame. Also do a
   full-frame search every N frames to pick up tags that come into view.

### With motion: predict the ROI from the pose

The numbers above come from static bursts, where the last frame's box is
nearly exact. When the body moves, predict the box instead of reusing it, so
the margin can stay small (0.15 → ~0.86 ms per camera):

```python
# rig_corners: (M, 3) all tag corners in the rigid-body frame
# rvec/tvec: predicted pose (e.g. last pose + constant velocity)
pts = np.asarray(rapidtag.project_points(rig_corners, rvec, tvec, K, dist))
roi = body_roi(pts, gray.shape, margin=0.15)
```

Not yet validated on a movement take (e.g. `sep15_26`). Check the tag loss
rate there before choosing the margin.

---

## Status / open items

- ROI mode is not yet in the tracker; it has only been benchmarked
  (`roi_bench.py` / `roi_overhead.py`, session scratch scripts).
- Crop-refine outlier corners: cause not investigated. The raw-vs-refined
  disagreement guard is untested.
- The ~15 % recall gap vs OpenCV is a detector issue and is not yet analysed.
