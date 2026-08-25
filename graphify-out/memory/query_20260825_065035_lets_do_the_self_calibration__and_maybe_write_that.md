---
type: "query"
date: "2026-08-25T06:50:35.990391+00:00"
question: "lets do the self-calibration, and maybe write that as 02_rigidbody_calib.py and our experiment as 03_jitter_model.py"
contributor: "graphify"
source_nodes: ["pose_stereo_pnp", "load_sync_flags", "kabsch", "verification_dual_camera.py"]
---

# Q: lets do the self-calibration, and maybe write that as 02_rigidbody_calib.py and our experiment as 03_jitter_model.py

## Answer

Expanded from original query via vocab: [rigid, calibration, marker, relative, transform, stereo, pnp, reprojection, kabsch, sync, mocap, distance]. Implemented jitter_model/02_rigidbody_calib.py to self-calibrate fixed AprilTag transforms in the tag-12 frame using robust relative SE(3), joint bundle adjustment, and a current-take cam0-to-cam1 extrinsic. Implemented jitter_model/03_jitter_model.py to compare C1/C2 with fixed 1/2/3-marker sets using mono joint board PnP or joint multi-view PnP, GPIO/monotonic synchronization, frozen offset-aware Motive alignment, held-out common-frame jitter, distance bins, coverage, and CSV/PNG outputs. The existing graph guided stereo transform composition, joint stereo PnP, Kabsch initialization, and GPIO sync handling.

## Source Nodes

- pose_stereo_pnp
- load_sync_flags
- kabsch
- verification_dual_camera.py