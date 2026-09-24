---
type: "query"
date: "2026-09-24T06:33:45.433140+00:00"
question: "we have new model, trunkpose\\runs\\trunk_seg-7\\weights, that also includes arms during segmentation inference, will wiring this in help? then we can find the overlap and either exclude them, from the hull, or improve the estimation."
contributor: "graphify"
outcome: "useful"
source_nodes: ["torso_mask()", "run_icp_pass()", "Neutral cloud vs the live cloud mapped through the ICP transform. Overlap =…"]
---

# Q: we have new model, trunkpose\runs\trunk_seg-7\weights, that also includes arms during segmentation inference, will wiring this in help? then we can find the overlap and either exclude them, from the hull, or improve the estimation.

## Answer

Expanded via graph vocab: [trunkpose, weights, yolo, segmentation, masks, point, cloud, depth, icp, neutral, overlap, keypoints]. trunk_seg-7 is already wired into data/sep23_26/session.toml and was used by the five-person experiment. The pipeline already selects the centre torso instance and subtracts a dilated union of arm masks before constructing the stereo front-shell cloud. Better arm masks can therefore reduce articulated-arm contamination, but arm geometry should not be added to rigid trunk ICP. The missing improvement is to retain arm/torso overlap as a diagnostic and combine it with 3D neutral-shell coverage. Arm overlap can gate frames occluded by arms; neutral-shell coverage detects rotational self-occlusion. For estimation rather than exclusion, use an accumulating or multi-view torso reference, since arm masking cannot restore torso surfaces absent from the neutral cloud.

## Outcome

- Signal: useful

## Source Nodes

- torso_mask()
- run_icp_pass()
- Neutral cloud vs the live cloud mapped through the ICP transform. Overlap =…