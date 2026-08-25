---
type: "query"
date: "2026-08-25T06:56:41.633210+00:00"
question: "alright, when we calibrated the cameras, i think i added rotation notebooks\\calibration\\dual_notebooks\\02_dual_calibration.py will that be a problem? should we remove rotation and calibrate again?"
contributor: "graphify"
source_nodes: ["calibration/dual_notebooks/02_dual_calibration.py", "_perview_extrinsics()", "Stereo Camera: Calibration, Verification & Findings"]
---

# Q: alright, when we calibrated the cameras, i think i added rotation notebooks\calibration\dual_notebooks\02_dual_calibration.py will that be a problem? should we remove rotation and calibrate again?

## Answer

Expanded from original query via vocab: [calibration, camera, rotate, rotation, intrinsics, extrinsics, fisheye, stereo, frame, corners, transform, image]. The radxa_calib_parallel corner preprocessing in trunkpose/dual_notebooks/01_corner_detection.py applies ROTATE_180 followed by horizontal flip, which is equivalent to a vertical image flip. The jitter estimator uses raw unflipped frames, so its pixel coordinate convention does not match the K/D and stereo R/T learned from the flipped corners. Recommended: remove both preprocessing operations, regenerate corner PKLs and stereo_calibration.toml from the existing raw msgpack recordings, and validate on raw held-out corners; retain Rotation.from_matrix(...).mean() in 02_dual_calibration.py because it averages physical stereo rotations and does not rotate images. Also pair free-running cameras by nearest recorded timestamp rather than equal frame index before recalibration.

## Source Nodes

- calibration/dual_notebooks/02_dual_calibration.py
- _perview_extrinsics()
- Stereo Camera: Calibration, Verification & Findings