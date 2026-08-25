---
type: "query"
date: "2026-08-25T04:55:37.804486+00:00"
question: "notebooks\\calibration\\dual_notebooks\\02_dual_calibration.py seems to be a problem because of frame number mismatch, how can we solve this? do we have frame numbers? which we can select parallel frames?"
contributor: "graphify"
source_nodes: ["_paired_indices()", "dual_recorder.py", "calibration/dual_notebooks/02_dual_calibration.py"]
---

# Q: notebooks\calibration\dual_notebooks\02_dual_calibration.py seems to be a problem because of frame number mismatch, how can we solve this? do we have frame numbers? which we can select parallel frames?

## Answer

Expanded from original query via graph vocabulary: [dual, calibration, frame, frames, number, timestamp, timestamps, parallel, paired, sync, synced, recording]. The corner extraction already tracked zero-based stream indices as frame_idx. The implementation now also saves the explicit frame_id alias and SensorTimestamp as sensor_timestamp_ns for robust cross-camera pairing, while correcting sync to store the GPIO flag.

## Source Nodes

- _paired_indices()
- dual_recorder.py
- calibration/dual_notebooks/02_dual_calibration.py