# Trunk angle under occlusion — from torso ICP to SAM 3D Body

Working notes, 2026-09-28. Scripts `19_sam3d_export.py`, `20_sam3d_infer.py`,
`21_sam3d_score.py`. Background: the NOARK game results in
`data/sep23_26/noark_game/output/README.md`.

## 1. The problem

In the protocol takes, torso-shell ICP tracks the trunk well (woc: median error around
2°, 98% of frames solved). In the game takes it breaks down: 17–62% of in-window
frames are solved, and it produces 15–25° error spikes.

The cause is how much torso the camera can see. The error falls steadily as the
visible shell grows. At ≥800 shell points the game takes match the protocol accuracy
(1.6° median magnitude error), but only about a sixth of frames get there.

### What covers the torso

It isn't only the arms. The camera sits at table height, and **the game's AprilTag
cube, riding on the robot, gets pushed right in front of the chest**. The torso mask
subtracts arm pixels, and the cube isn't torso either, so the shell collapses. The
wording in `noark_game/output/README.md` ("the arms cover the torso") needs to include
the cube.

Which part of the torso survives also differs by subject. Median share of each
neutral-pose height band still visible during play (top = shoulders):

| subject | 0–20% | 20–40% | 40–60% | 60–80% | bottom |
|---|---|---|---|---|---|
| gokul | 70% | 66% | 31% | 23% | 9% |
| js | 90% | 55% | 13% | 22% | 2% |
| ram | 51% | 42% | 22% | 25% | 1% |
| diwakar | 0% | 0% | 5% | 3% | 15% |
| kabil | 0% | 0% | 0% | 0% | 27% |

## 2. Things tried that did not fix it

**Blending ICP with MediaPipe.** The landmark path (MediaPipe shoulders plus the torso
plane normal) takes its anterior axis from the same occluded torso plane, so it fails
on the same frames. On the gokul game take it solved more frames but was much worse:
median error 13.8/13.3/19.8° with r ≈ 0.1–0.2. Because the two fail on the same
frames, blending them doesn't help.

**Cropping to the top of the shell.** ICP was re-run on only the top 20% or 40% of the
shell height, with the matching gate lowered to 40 points. Results are total-rotation
error, median / worst 5%:

| subject | full shell | top 40% | top 20% |
|---|---|---|---|
| diwakar | 8.6 / 27.1 | 6.3 / 17.6 | 5.1 / 11.1 |
| gokul | 1.4 / 15.6 | 1.5 / 7.7 | 2.6 / 13.1 |
| js | 2.6 / 18.0 | 2.4 / 17.4 | 3.8 / 11.6 |
| kabil | 6.5 / 24.9 | 8.1 / 22.0 | 3.1 / 18.5 |
| ram | 2.8 / 15.6 | 3.2 / 13.0 | 2.1 / 8.9 |
| gokul_woc (protocol) | 2.7 / 8.1 | 2.8 / 10.5 | 4.6 / 14.6 |

Cropping trims the error spikes but hardly changes coverage. It also makes the clean
protocol take worse: a thin, flat band pins down rotation less well than the whole
chest.

## 3. Human mesh recovery (HMR)

HMR models regress a whole-body mesh from a single RGB image. They explain the arms
instead of masking them out, and they give an output on every frame.

### Candidates

- **InstantHMR v2** (NaturalPad). About 10 ms on a Galaxy S23 NPU, with an HGNetv2
  backbone. It reports 40 mm PA-MPJPE on 3DPW, but PA-MPJPE is scored after Procrustes
  alignment, which removes the global rotation our trunk angle depends on. The weights
  are "coming soon" under a research-only licence. It is distilled from SAM 3D Body,
  so SAM 3D Body sets its accuracy ceiling.
- **SAM 3D Body** (Meta). ViT-H with 631 M parameters, or DINOv3-H+ with 840 M. Gated
  on Hugging Face under the SAM License, which allows commercial use and derivative
  works. It doesn't explicitly ban training other models on its outputs, but get a
  legal read before shipping anything built that way.

### Size against our segmentation model

| model | parameters | compute per frame | speed |
|---|---|---|---|
| trunk_seg-7 (YOLOv8n-seg, 3 classes) | 3.26 M | ~12 GFLOPs @ 640 | 20 Hz on the Radxa Dragon Q6A NPU |
| InstantHMR v2 | not published (~10–30 M, my estimate) | tens of GFLOPs, estimated | 10 ms on the S23 NPU; roughly 30–60 ms on the Q6A, estimated |
| SAM 3D Body ViT-H | 631 M | hundreds of GFLOPs | 0.20 s/frame on an RTX 4060 Laptop, measured |

SAM 3D Body is about 200× our segmentation model, so it can't run on the edge device.

Shrinking a model after training doesn't close a gap that large:
- INT8 quantization: about 4× smaller.
- Pruning or low-rank factorization: about 2–5×.
- Lower input resolution: about 2–4×, and it costs accuracy.

The route that does work is **distillation**: train a small student on the big
model's outputs over unlabelled video. That is how InstantHMR was built. Our task only
needs a 3-DOF trunk rotation, so the student could be a small head on the YOLOv8n we
already run. We also have real labels, the synced mocap trunk angles, to fine-tune
and validate it.

## 4. SAM 3D Body pipeline

### Environment

The model runs in its own environment, separate from this project's `.venv`. The
checkout is facebookresearch/sam-3d-body at commit `b5c765a`, in
`E:/CMC/pyprojects/sam-3d-body`. Override that location with `SAM3D_ROOT`.

- A `pyproject.toml` there pins the same torch 2.12.1+cu130, numpy and scipy as this
  project, so `uv sync --python 3.13` reuses the cached wheels.
- Extra dependencies: `pytorch-lightning yacs einops timm roma omegaconf
  huggingface_hub braceexpand opencv-python msgpack msgpack-numpy`.
- Not needed: **detectron2**, MoGe, SAM2 or flash-attn. We pass our own box, mask and
  intrinsics, the estimator runs without its detector or FOV model, and the body
  model loads as TorchScript (`assets/mhr_model.pt`) with no `mhr` package.
- Weights: `hf download facebook/sam-3d-body-vith --local-dir
  checkpoints/sam-3d-body-vith`, after access has been granted.
- The "missing keys … head_pose.mhr.*" warning at load time is expected. Those buffers
  live in the TorchScript file, not the checkpoint.

### Run

```
uv run python trunkpose/dual_notebooks/19_sam3d_export.py --subject gokul --game   # project venv
cd E:/CMC/pyprojects/sam-3d-body
.venv/Scripts/python <repo>/trunkpose/dual_notebooks/20_sam3d_infer.py --subject gokul --game
cd <repo>
uv run python trunkpose/dual_notebooks/21_sam3d_score.py --subject gokul --game    # project venv
```

For a protocol take, drop `--game`; stride 1 is then the default. `--n 20` on 19 gives
a smoke test. Everything is written under `<recording>/sam3d/`.

### Input preparation (19)

- **Undistortion.** Frames are fisheye-undistorted to a pinhole image with balance 0,
  then rotated 180°. The saved intrinsics are for the rotated pinhole image.
- **Frame mapping.** `R_model_to_board = R0_c.T @ diag(-1,-1,1)` maps the model's
  camera frame back to the board frame, so 07's comparison applies unchanged.
- **One box per take.** It is the median of the centre-most head plus centre-most
  torso boxes, widened 35% on each side and extended 25% downwards. Per-frame boxes
  went wrong in both directions: one swallowed a bystander, and another cut off at
  mid-chest when the cube covered the torso.

### Conventions checked (21)

- **Keypoints.** The 3D keypoints plus `pred_cam_t` are in OpenCV camera axes: they
  reproject onto the model's own 2D keypoints at 0.0 px, against 55–62 px for the
  y/z-flipped alternative.
- **Joint rotations.** `pred_global_rots` are in the body model's y-up axes and need
  conjugating by diag(1,−1,−1).
  - Without the flip, the fitted-versus-geometric S gap is 145–152°.
  - With it, the gap is 62–71°. That matches ICP's own 72° on gokul_woc, so the rest
    of the gap is in our geometric reference, not in SAM.
  - Total rotation is unaffected either way.
- **Trunk joint.** The trunk is MHR joint 37, `c_spine3` (upper thoracic, at the
  sternum rigid body). The hips are behind the table and the model hallucinates the
  legs, so nothing uses the hips.

## 5. Results (gokul only so far)

Total rotation against mocap. This measure doesn't depend on how the camera and mocap
axes are aligned.

| recording | method | frames solved | median / worst 5% | r |
|---|---|---|---|---|
| gokul game | ICP | 1462 / 2774 | 1.4 / 15.6° | 0.75 |
| gokul game | **SAM 3D Body** | **2774 / 2774** | **2.0 / 5.9°** | **0.91** |
| gokul game, frames ICP failed | SAM 3D Body | 1312 | 2.6 / 6.2° | 0.90 |
| gokul_woc | ICP | 1633 | 2.7 / 8.1° | 0.93 |
| gokul_woc | **SAM 3D Body** | 1634 | **1.2 / 4.3°** | **0.99** |

By visible shell size on the game take (SAM): <200 points 2.5°, 200–400 2.9°,
400–800 2.3°, ≥800 0.7°. It still works where ICP cannot run at all.

Per axis on the game take, with the rotation flip and S fitted on the whole take
(median / worst 5%, r):

| axis | ICP | SAM 3D Body |
|---|---|---|
| flexion | 2.7 / 8.0°, r 0.43 | 3.6 / 6.6°, r 0.54 |
| lateral | 7.5 / 22.5°, r 0.39 | 0.9 / 2.8°, r 0.84 |
| axial | 1.7 / 6.4°, r 0.83 | 1.8 / 5.1°, r 0.91 |

Held-out correlations, with S fitted on the first half and scored on the second, were
+0.58 / +0.85 / +0.89. On the protocol take every axis reaches r ≥ 0.99.

**Cost:** 0.20 s per frame, 3.2 GB peak GPU memory. The gokul game and woc takes
(4,413 frames) took about 15 minutes of inference.

## 6. Caveats

- One subject so far. kabil and diwakar, where ICP coverage was worst, are the real
  test.
- The per-axis split relies on a fitted S, validated on held-out frames. Total
  rotation does not.
- The image domain differs from SAM 3D Body's training data: grayscale, 640×400,
  undistorted fisheye, seated subject behind a table.
- The camera series is smoothed with the same Savitzky–Golay filter as ICP, so the
  comparison is like for like.

## 7. Next

1. Run 19 → 20 → 21 on diwakar, js, kabil and ram (about 10 minutes of inference
   each). Pool the results, as `16_subject_agreement.py` does for ICP.
2. If they hold up, plan the edge student: a trunk-rotation head on YOLOv8n,
   pre-trained on SAM 3D Body labels over all recordings, fine-tuned and validated
   against mocap with one subject left out each time. Then quantize it to INT8 and
   check small-angle accuracy after quantization.
3. Revisit InstantHMR v2 when the weights ship: run it through the same 20/21 path.
4. Fix the occlusion wording in `noark_game/output/README.md` (the cube, not only the
   arms).
