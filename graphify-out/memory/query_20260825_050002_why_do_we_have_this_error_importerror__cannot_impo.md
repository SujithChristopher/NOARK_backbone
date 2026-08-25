---
type: "query"
date: "2026-08-25T05:00:02.736046+00:00"
question: "why do we have this error ImportError: cannot import name 'aruco' from 'cv2' (unknown location)"
contributor: "graphify"
source_nodes: ["calibration/dual_notebooks/01_corner_detection.py"]
---

# Q: why do we have this error ImportError: cannot import name 'aruco' from 'cv2' (unknown location)

## Answer

Expanded via graph vocabulary: [aruco, module, calibration, detection]. Local environment inspection shows cv2 resolves with loader=None as an incomplete namespace package. opencv-contrib-python 4.13.0.92 metadata is installed, and its RECORD requires cv2/__init__.py, cv2/cv2.pyd, and cv2/aruco stubs, but those files are missing. The frozen uv dependency plan has no competing OpenCV package selected. Repair by reinstalling only opencv-contrib-python with uv, then verify cv2.aruco.

## Source Nodes

- calibration/dual_notebooks/01_corner_detection.py