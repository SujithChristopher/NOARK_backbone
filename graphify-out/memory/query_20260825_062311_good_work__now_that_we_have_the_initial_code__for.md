---
type: "query"
date: "2026-08-25T06:23:11.520237+00:00"
question: "good work, now that we have the initial code, for alighnment and sync pulse, i want to see how much jitter, can we reduce, when we do it with dual camera or single camera, the april tag markers also kind of form a rigid body we have more than one marker, they are located in different locations, but closeby, so can we evaluate jitter vs distance, we can use mocap as reference smootheness, for single camera - single marker, single camera - 2 / 3 marker, dual camera - single marker, dual camera 2 / 3 maker, we can use jitter_model\\01_plot_data.py as ref, we can create 02_jitter_model in the same folder, have it repl so we can review it later, how do we go about doing it? lets discuss this first"
contributor: "graphify"
source_nodes: ["verification_dual_camera.py", "pose_stereo_pnp()", "load_sync_flags()", "Stereo Camera: Calibration, Verification & Findings"]
---

# Q: good work, now that we have the initial code, for alighnment and sync pulse, i want to see how much jitter, can we reduce, when we do it with dual camera or single camera, the april tag markers also kind of form a rigid body we have more than one marker, they are located in different locations, but closeby, so can we evaluate jitter vs distance, we can use mocap as reference smootheness, for single camera - single marker, single camera - 2 / 3 marker, dual camera - single marker, dual camera 2 / 3 maker, we can use jitter_model\01_plot_data.py as ref, we can create 02_jitter_model in the same folder, have it repl so we can review it later, how do we go about doing it? lets discuss this first

## Answer

Expanded via graph vocabulary: [dual, single, stereo, triangulation, marker, mocap, distance, smooth, pose, sync, camera]. Recommended a controlled raw-estimator benchmark: build one calibrated multi-tag rigid-body model, estimate a common tag12/body-frame origin with joint PnP, compare C1-M1, C1-M2, C1-M3, C2-M1, C2-M2, and C2-M3 on common frames, then score held-out residual first-difference jitter divided by sqrt(2) during mocap-static intervals and bin by mocap-derived distance. Also report RMSE, orientation jitter, coverage, and block-bootstrap confidence intervals. The current data is feasible: five IDs are visible, distance spans roughly 0.28-0.85 m, 924 stereo pairs share at least two tags and 660 share at least three. Radxa cameras are free-running with a median nearest-frame phase gap of 16.23 ms, so dual fusion must pair by monotonic timestamps and primary jitter scoring should use static holds; GPIO aligns mocap but does not make the two exposures simultaneous.

## Source Nodes

- verification_dual_camera.py
- pose_stereo_pnp()
- load_sync_flags()
- Stereo Camera: Calibration, Verification & Findings