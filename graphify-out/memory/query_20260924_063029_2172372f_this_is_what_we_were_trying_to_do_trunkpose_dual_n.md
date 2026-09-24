---
type: "query"
date: "2026-09-24T06:30:29.218222+00:00"
question: "this is what we were trying to do trunkpose\\dual_notebooks\\experiments\\trunk_5person_test.txt, what is the cause of the error"
contributor: "graphify"
outcome: "useful"
source_nodes: ["trunkpose/dual_notebooks/trunk_angle.py", "trunkpose/dual_notebooks/07_icp_trunk.py"]
---

# Q: this is what we were trying to do trunkpose\dual_notebooks\experiments\trunk_5person_test.txt, what is the cause of the error

## Answer

Expanded from original query via graph vocab: [trunkpose, trunk, dual, person, pose, keypoints, yolo, error, errors, dimensions, ndarray, index]. The approximately 10-degree measurement error is primarily systematic ICP under-reading caused by self-occlusion and loss of overlap between each live front-torso shell and the single neutral reference cloud. As rotation grows, especially axially on a nearly cylindrical torso surface, ICP finds a locally good but physically wrong correspondence and slides the visible patch onto the neutral patch. Segmentation and hull outliers were tested and are not the present limiting causes; temporal lag is secondary for one subject, while ram has a separate frame/axis assignment problem.

## Outcome

- Signal: useful

## Source Nodes

- trunkpose/dual_notebooks/trunk_angle.py
- trunkpose/dual_notebooks/07_icp_trunk.py