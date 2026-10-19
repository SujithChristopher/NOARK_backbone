# %% 19_sam3d_export.py
# Stage A of the SAM 3D Body trunk test: recording -> what the model needs.
#
# SAM 3D Body assumes a pinhole camera and an upright image; ours is a fisheye sensor
# mounted upside down. So each reference-camera frame is fisheye-undistorted to a
# pinhole image (balance 0 -> no black border) and rotated 180, and the intrinsics of
# that rotated pinhole image are saved alongside, together with the rotation that maps
# its camera frame back to the board frame (so 21 can hand the mesh trunk rotation to
# 07's comparison in the frame it expects).
#
# The person box comes from our own segmentation model: centre-most head + centre-most
# torso instance per frame, then ONE box per take (median, widened for the shoulders
# and arms, extended down to the table). The subject is seated, so a per-frame box only
# changes when the game cube or an arm covers the torso -- exactly when a per-frame
# crop would go wrong (it swallowed a bystander, or cut off at mid-chest).
#
# Only frames inside the GPIO-high window (mocap recording) are exported.
#
# --cam 1 exports the OTHER camera of the pair instead (its own intrinsics; the stereo
# rotation folded into R_model_to_board, so its answers land in the same board frame)
# into <recording>/sam3d_cam1/ -- the input of 30_sam3d_dualcam.py.
#
#   uv run python trunkpose/dual_notebooks/19_sam3d_export.py --subject gokul --game
#   uv run python trunkpose/dual_notebooks/19_sam3d_export.py --subject gokul_woc
#   ... --n 20      20 evenly spaced frames, for a smoke test
#
# Writes <recording>/sam3d/img_<frame>.png and <recording>/sam3d/meta.npz.
# Next: 20_sam3d_infer.py (in the sam-3d-body environment), then 21_sam3d_score.py.

import argparse
import importlib
import os
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SESSION = PROJECT_ROOT / "data" / "sep23_26"
sys.path.insert(0, str(SCRIPT_DIR))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default=str(DEFAULT_SESSION))
    ap.add_argument("--subject", required=True)
    ap.add_argument("--game", action="store_true", help="NOARK game take")
    ap.add_argument("--stride", type=int, default=None,
                    help="frame stride (default 3 for --game, 1 otherwise)")
    ap.add_argument("--n", type=int, default=0,
                    help="export only N evenly spaced in-window frames (0 = all)")
    ap.add_argument("--cam", type=int, choices=(0, 1), default=0,
                    help="1 = the second camera of the pair (writes sam3d_cam1/)")
    a = ap.parse_args()

    m15 = importlib.import_module("15_subject_icp")
    rec, csv, _ = m15.resolve_recording(Path(a.session).resolve(), a.subject, a.game)
    stride = a.stride or (3 if a.game else 1)
    # 06 resolves its configuration at import time from these
    os.environ.update(TRUNK_RECORDING_NAME=a.subject, TRUNK_RECORDING_DIR=str(rec),
                      TRUNK_MOCAP_CSV=str(csv), TRUNK_FRAME_STRIDE=str(stride))
    m6 = importlib.import_module("06_trunk_axis")

    W, H = m6.CAM_SIZE
    K0, D0, K1, D1, R_st, _ = m6.load_stereo(m6.STEREO_TOML)
    Kc, Dc = (K1, D1) if a.cam else (K0, D0)
    R0_c, _, _, _ = m6.load_charuco_basis(m6.CHARUCO_TOML)
    Kp = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        Kc, Dc, (W, H), np.eye(3), balance=0.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(Kc, Dc, np.eye(3), Kp, (W, H),
                                                     cv2.CV_16SC2)
    # a 180 deg image rotation is the camera rotated 180 about its optical axis
    Kr = Kp.copy()
    if m6.ROT180:
        Kr[0, 2], Kr[1, 2] = W - 1 - Kp[0, 2], H - 1 - Kp[1, 2]
    Rz = np.diag([-1.0, -1.0, 1.0]) if m6.ROT180 else np.eye(3)
    # model camera frame -> board frame:  v_board = R0_c.T @ Rz @ v_model. The stereo
    # pair is p_cam1 = R_st p_cam0 + T, so the second camera goes through R_st.T first.
    R_model_to_board = R0_c.T @ (R_st.T if a.cam else np.eye(3)) @ Rz

    def upright(img, interp):
        u = cv2.remap(img, map1, map2, interp)
        return cv2.rotate(u, cv2.ROTATE_180) if m6.ROT180 else u

    sync = m6.load_sync_flags(m6.CAM0_TIMES)
    high = np.where(sync == 1)[0]
    if not len(high):
        raise SystemExit(f"{a.subject}: no GPIO pulse in the camera timestamps")
    lo_i, hi_i = int(high[0]), int(high[-1]) + 1
    want = np.arange(lo_i, hi_i)
    if a.n:
        want = want[np.linspace(0, len(want) - 1, a.n).round().astype(int)]
    want_set = set(want.tolist())

    seg = m6.load_seg_model()
    out = rec / ("sam3d_cam1" if a.cam else "sam3d")
    out.mkdir(exist_ok=True)
    idx_l, box_l = [], []
    for i, f0 in enumerate(m6.iter_frames(m6.CAM1_FRAMES if a.cam else m6.CAM0_FRAMES)):
        if i >= hi_i:
            break
        if i not in want_set:
            continue
        g0 = f0 if f0.ndim == 2 else cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY)
        res = seg.predict(cv2.cvtColor(m6.for_inference(g0), cv2.COLOR_GRAY2RGB),
                          conf=m6.SEG_CONF, verbose=False)[0]
        # subject = centre-most head and centre-most torso (arms reach out and
        # bystanders sit off-centre, so neither may widen the box)
        body = np.zeros((H, W), np.uint8)
        if res.masks is not None:
            best = {}
            for m, cid in zip(res.masks.data.cpu().numpy(),
                              res.boxes.cls.cpu().numpy().astype(int)):
                nm = res.names[cid]
                if nm not in ("torso", "chest", "head"):
                    continue
                mm = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
                if not mm.any():
                    continue
                ys, xs = np.nonzero(mm)
                d = float(np.hypot(xs.mean() - W / 2, ys.mean() - H / 2))
                key = "head" if nm == "head" else "torso"
                if key not in best or d < best[key][0]:
                    best[key] = (d, mm)
            for _, mm in best.values():
                body[mm] = 255
        body = upright(m6.from_inference_mask(body), cv2.INTER_NEAREST)
        ys, xs = np.nonzero(body)
        box_l.append(np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)
                     if len(xs) >= 200 else np.full(4, np.nan))
        cv2.imwrite(str(out / f"img_{i:05d}.png"), upright(g0, cv2.INTER_LINEAR))
        idx_l.append(i)
        if len(idx_l) % 200 == 0:
            print(f"  {len(idx_l)}/{len(want)}", flush=True)

    fb = np.array(box_l)
    x0, y0, x1, y1 = np.nanmedian(fb, axis=0)
    bw, bh = x1 - x0, y1 - y0
    fixed = np.array([max(0, x0 - 0.35 * bw), max(0, y0 - 0.10 * bh),
                      min(W - 1, x1 + 0.35 * bw), min(H - 1, y1 + 0.25 * bh)])
    np.savez(out / "meta.npz", idx=np.array(idx_l),
             boxes=np.tile(fixed, (len(idx_l), 1)), boxes_frame=fb, K=Kr,
             R_model_to_board=R_model_to_board, size=np.array([W, H]),
             gpio=np.array([lo_i, hi_i]), stride=stride, cam=a.cam)
    print(f"{a.subject}: {len(idx_l)} frames -> {out}  "
          f"(fx={Kr[0, 0]:.0f} cx={Kr[0, 2]:.0f} cy={Kr[1, 2]:.0f}; "
          f"no head/torso on {int(np.isnan(fb[:, 0]).sum())})")


if __name__ == "__main__":
    main()
