---
type: "query"
date: "2026-08-25T06:11:43.234410+00:00"
question: "Create jitter_model/01_plot_data.py to compare AprilTag 12 (DICT_APRILTAG_36h11, 5 cm) against the mocap rigid body in data/radxa/jitter_test, using the stereo calibration and GPIO sync concepts from the trunkpose reference notebooks."
contributor: "graphify"
source_nodes: ["02_charuco_mocap_align.py", "mocap_to_charuco_points()", "load_sync_flags()", "Rigid transform putting mocap marker positions in the board frame, FITTED from…"]
---

# Q: Create jitter_model/01_plot_data.py to compare AprilTag 12 (DICT_APRILTAG_36h11, 5 cm) against the mocap rigid body in data/radxa/jitter_test, using the stereo calibration and GPIO sync concepts from the trunkpose reference notebooks.

## Answer

Expanded from original query via graph vocab: [april, marker, mocap, rigid, sync, calibration, camera, charuco, rotation, alignment, pose, trunkpose]. Implemented a cam0 percent-cell notebook using fisheye-undistorted AprilTag pose estimation, first-high GPIO synchronization from load_sync_flags, Motive quaternion interpolation, and an offset-aware world-to-camera alignment. Full-data validation detected tag 12 in 1064/1103 frames; synchronized 920 samples; held-out RMSE 10.586 mm; axis correlations X 0.9996, Y 0.9976, Z 0.9984.

## Source Nodes

- 02_charuco_mocap_align.py
- mocap_to_charuco_points()
- load_sync_flags()
- Rigid transform putting mocap marker positions in the board frame, FITTED from…