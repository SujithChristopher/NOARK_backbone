# %% 20_sam3d_infer.py
# Stage B of the SAM 3D Body trunk test: run the model on 19's export.
#
# Runs in the SEPARATE sam-3d-body environment, not this project's venv (see
# SAM3D_BODY.md for the setup). Body decoder only (no hand refinement), with our fixed
# person box and our pinhole intrinsics -- so no detectron2 / ViTDet detector, no MoGe
# FOV estimator and no SAM2 are needed.
#
# Saves per frame: MHR70 3D/2D keypoints, the 127 per-joint global rotations, the
# global rotation, camera translation and body pose parameters, plus the timing.
# The first --overlay frames also get a keypoint overlay PNG for a sanity look.
#
#   cd E:/CMC/pyprojects/sam-3d-body
#   .venv/Scripts/python <repo>/trunkpose/dual_notebooks/20_sam3d_infer.py \
#       --subject gokul --game
#
# Writes <recording>/sam3d/sam3d_out.npz (--cam 1: 19's sam3d_cam1/ export instead;
# --dir <folder>: <folder>/sam3d_out.npz).

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SESSION = PROJECT_ROOT / "data" / "sep23_26"
# the facebookresearch/sam-3d-body checkout, with its checkpoints/ folder
SAM3D_ROOT = Path(os.environ.get("SAM3D_ROOT", PROJECT_ROOT.parents[1] / "sam-3d-body"))
SAM3D_CKPT = SAM3D_ROOT / "checkpoints" / "sam-3d-body-vith"
KEYS = ("pred_keypoints_3d", "pred_keypoints_2d", "pred_global_rots", "global_rot",
        "pred_cam_t", "body_pose_params")


def stack(lst):
    """Per-frame arrays -> one array, NaN where the model returned nothing."""
    ref = next((x for x in lst if x is not None), None)
    return np.stack([x if x is not None else np.full_like(ref, np.nan, dtype=float)
                     for x in lst]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default=str(DEFAULT_SESSION))
    ap.add_argument("--subject")
    ap.add_argument("--game", action="store_true", help="NOARK game take")
    ap.add_argument("--overlay", type=int, default=6,
                    help="write keypoint overlays for the first N frames")
    ap.add_argument("--cam", type=int, choices=(0, 1), default=0,
                    help="1 = 19's second-camera export (sam3d_cam1/)")
    ap.add_argument("--dir", help="any folder in 19's layout (meta.npz + img_*.png) instead of "
                    "--subject, e.g. minisam3d synth/export_sam3d.py's synthetic frames")
    a = ap.parse_args()
    if not (a.subject or a.dir):
        ap.error("give --subject or --dir")

    if a.dir:
        d = Path(a.dir)
    else:
        # 15 has no heavy top-level imports, so its layout resolver works in this env too
        sys.path.insert(0, str(SCRIPT_DIR))
        m15 = importlib.import_module("15_subject_icp")
        rec = m15.resolve_recording(Path(a.session).resolve(), a.subject, a.game)[0]
        d = rec / ("sam3d_cam1" if a.cam else "sam3d")
    meta = np.load(d / "meta.npz")
    K = torch.tensor(meta["K"], dtype=torch.float32)[None]

    sys.path.insert(0, str(SAM3D_ROOT))
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    model, cfg = load_sam_3d_body(str(SAM3D_CKPT / "model.ckpt"), device="cuda",
                                  mhr_path=str(SAM3D_CKPT / "assets" / "mhr_model.pt"))
    est = SAM3DBodyEstimator(model, cfg)
    print(f"GPU memory after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    res = {k: [] for k in KEYS}
    ok, times = [], []
    torch.cuda.reset_peak_memory_stats()
    for n, (i, box) in enumerate(zip(meta["idx"], meta["boxes"])):
        img = cv2.imread(str(d / f"img_{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if not np.isfinite(box).all():
            ok.append(False)
            for k in KEYS:
                res[k].append(None)
            continue
        t = time.perf_counter()
        o = est.process_one_image(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB),
                                  bboxes=box[None], cam_int=K, inference_type="body")
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        ok.append(bool(o))
        for k in KEYS:
            res[k].append(np.asarray(o[0][k]) if o else None)
        if o and n < a.overlay:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            x0, y0, x1, y1 = box.astype(int)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 1)
            for p in o[0]["pred_keypoints_2d"][:21]:
                cv2.circle(vis, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
            cv2.imwrite(str(d / f"overlay_{i:05d}.png"), vis)
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(meta['idx'])}  {np.median(times):.3f}s/frame",
                  flush=True)

    print(f"frames {len(ok)}, solved {sum(ok)}, median {np.median(times):.3f}s/frame, "
          f"peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    np.savez_compressed(d / "sam3d_out.npz", idx=meta["idx"], ok=np.array(ok),
                        sec_per_frame=np.array(times),
                        **{k: stack(v) for k, v in res.items()})
    print(f"-> {d / 'sam3d_out.npz'}")


if __name__ == "__main__":
    main()
