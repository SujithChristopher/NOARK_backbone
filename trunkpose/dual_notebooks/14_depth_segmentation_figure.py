# %% 14_depth_segmentation_figure.py
# Poster panel: the stereo depth map behind the ICP pipeline, alone and with the
# torso region outlined.
#
#   left   depth (disparity reprojected to metres) for the rectified cam0 view
#   right  same map + the outline of the torso front shell that feeds ICP
#
# Two outlines, because they are not the same thing:
#   * YOLO torso mask -- the raw segmentation, re-run on this frame
#   * torso front shell -- what ICP actually consumes: that mask eroded by
#     MASK_ERODE_FRAC, arms subtracted, only the frontmost SHELL_M kept. Recovered
#     by projecting the cached cloud back into the rectified image.
# The gap between them is the erosion + front-shell gate, and it is the point.
#
#   TRUNK_FIG_FRAME=<i>  pick the frame (default: the neutral pose)
#   TRUNK_SEG_WEIGHTS    YOLO weights (inherited from 06)
#   uv run python trunkpose/dual_notebooks/14_depth_segmentation_figure.py

import importlib
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cv2
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
m12 = importlib.import_module("12_poster_figures")
m6, m7 = m12.m6, m12.m7

OUT_PNG = m6.RECORDING_DIR / "depth_segmentation_figure.png"
C_SHELL = "#FFFFFF"      # ICP input outline -- dashed, see below
# Both outlines are white and told apart by dash pattern: the ramp runs the
# whole rainbow, so no single hue contrasts everywhere, and the torso sits at
# the dark end where a black line vanishes.
C_SEG = "#FFFFFF"        # raw YOLO mask outline -- solid
CMAP = "jet"             # starts at blue, so near depth never collides with the
                         # black no-data; "nipy_spectral" and "turbo" also work
NODATA = "black"         # unmatched pixels, outside the ramp entirely
MIN_BLOB_PX = 300
CLIP_ABOVE_TORSO_M = None  # metres behind the torso to end the ramp;
                           # None = full scene range (more room colour,
                           # less contrast across the torso itself)


def depth_metres(depth_u8):
    """Undo 06's uint8 packing: 0 = no data, 1..255 = near..far."""
    z = np.full(depth_u8.shape, np.nan, dtype=np.float64)
    ok = depth_u8 > 0
    z[ok] = (m6.DEPTH_MIN_M + (depth_u8[ok].astype(np.float64) - 1) / 253.0
             * (m6.SCENE_DEPTH_MAX_M - m6.DEPTH_MIN_M))
    return z


def contours_of(mask_small):
    cnts, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2) for c in cnts if cv2.contourArea(c) >= MIN_BLOB_PX]


def pipeline_masks(i, maps0, maps1, Q, shape):
    """Re-run 06's mask path on frame `i` and return (raw YOLO, front shell).

    This repeats the mask half of ``m6.stereo_step`` rather than reading the cached
    cloud back: the cloud is voxel-downsampled, so reprojecting it needs a dilation
    to close the gaps, and that dilation cancels exactly the erosion the figure is
    meant to show. Both masks come back as rectified, downscaled pixel masks."""
    from ultralytics import YOLO

    pair = None
    for k, (f0, f1) in enumerate(m6.iter_frame_pairs(m6.CAM0_FRAMES, m6.CAM1_FRAMES)):
        if k == i:
            pair = (f0, f1)
            break
    if pair is None:
        raise SystemExit(f"Frame {i} not in {m6.CAM0_FRAMES}")
    g0, g1 = (f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in pair)

    mask0 = m6.torso_mask(YOLO(str(m6.SEG_WEIGHTS)), g0, m6.CAM_SIZE)
    if not mask0.any():
        raise SystemExit(f"YOLO ({m6.SEG_WEIGHTS.name}) found no torso on frame {i}")
    mrect = cv2.remap(mask0, *maps0, cv2.INTER_NEAREST)

    rect0 = cv2.remap(g0, *maps0, cv2.INTER_LINEAR)
    rect1 = cv2.remap(g1, *maps1, cv2.INTER_LINEAR)
    sgbm = m6.make_sgbm()
    disp_raw = sgbm.compute(rect0, rect1).astype(np.float32) / 16.0
    disp = cv2.bilateralFilter(disp_raw, m6.BILATERAL_D, m6.BILATERAL_SIGMA_COLOR,
                               m6.BILATERAL_SIGMA_SPACE)
    pts = cv2.reprojectImageTo3D(disp, Q) / 1000.0
    zc = pts[..., 2]
    valid = (disp_raw > m6.SGBM_MIN_DISP) & np.isfinite(pts).all(axis=2)

    eroded = mrect.copy()
    area = int((mrect > 0).sum())
    if area > 0:
        ksz = max(3, int(round(np.sqrt(area) * m6.MASK_ERODE_FRAC))) | 1
        eroded = cv2.erode(eroded,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)))
    torso_ok = valid & (eroded > 0) & (zc > m6.DEPTH_MIN_M) & (zc < m6.DEPTH_MAX_M)
    shell = np.zeros_like(mrect)
    if torso_ok.any():
        front = np.percentile(zc[torso_ok], m6.FRONT_PCTL)
        shell[torso_ok & (zc <= front + m6.SHELL_M)] = 255

    h, w = shape
    small = lambda m: cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return small(mrect), small(shell)


def main():
    m12.poster_style()
    inp = m7.load_inputs()
    frames, n, ts0 = inp["frames"], inp["n"], inp["ts0"]

    env_frame = int(os.environ.get("TRUNK_FIG_FRAME", "-1"))
    if env_frame >= 0:
        i = env_frame
        zwin = [0]
    else:
        # neutral pose: the frame the whole comparison is zeroed on
        _Rp, R_neu, _fp, _lp, _ap, _ = m6.assemble(frames)
        icp = m7.run_icp_pass(frames, R_neu, n)
        cmp_ = m7.compare_mocap(inp, icp["Rb_icp"], R_neu)
        zwin = cmp_["zwin"]
        i = int(zwin[0])
    fr = frames[i]
    if fr["depth"] is None or fr["cloud"] is None:
        raise SystemExit(f"Frame {i} has no cached depth/cloud.")
    t_s = float(((ts0[i] - ts0[zwin[0]]) / np.timedelta64(1, "s")))

    z = depth_metres(fr["depth"])
    finite = np.isfinite(z)
    vmin = float(np.percentile(z[finite], 1))
    print(f"frame {i}  t={t_s:+.1f}s  valid depth {100 * finite.mean():.0f}%  "
          f"range {vmin:.2f}-{np.nanmax(z):.2f} m")

    K0, D0, K1, D1, R, T_mm = m6.load_stereo(m6.STEREO_TOML)
    _R1, Q, maps0, maps1 = m6.build_rectification(K0, D0, K1, D1, R, T_mm,
                                                  m6.CAM_SIZE)
    seg_mask, mask = pipeline_masks(i, maps0, maps1, Q, z.shape)
    seg_cnts, cnts = contours_of(seg_mask), contours_of(mask)
    print(f"YOLO mask {int((seg_mask > 0).sum())} px -> front shell "
          f"{int((mask > 0).sum())} px "
          f"({100 * (mask > 0).sum() / max((seg_mask > 0).sum(), 1):.0f}% kept)")

    # Scale the ramp to the subject, not to the room: over the full 0.25-3.4 m
    # range the torso lands in the bottom third of the colormap and reads flat.
    # Everything past the clip saturates, which is the correct reading ("far wall").
    in_torso = finite & (mask > 0)
    z_torso = float(np.median(z[in_torso])) if in_torso.any() else float(np.nanmedian(z))
    vmax = float(np.percentile(z[finite], 99))
    if CLIP_ABOVE_TORSO_M is not None:
        vmax = min(vmax, z_torso + CLIP_ABOVE_TORSO_M)
    print(f"torso median depth {z_torso:.2f} m -> colour range {vmin:.2f}-{vmax:.2f} m")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2))
    for k, ax in enumerate(axes):
        im = ax.imshow(np.ma.masked_invalid(z), cmap=CMAP, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        im.cmap.set_bad(NODATA)            # 0 in the packed map = no stereo match
        if k == 1:
            for col, lw, ls, group in ((C_SEG, 2.4, "-", seg_cnts),
                                       (C_SHELL, 1.8, (0, (4, 2)), cnts)):
                for c in group:
                    loop = np.vstack([c, c[:1]])
                    ax.plot(loop[:, 0], loop[:, 1], color=col, lw=lw, ls=ls)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("#999999")
        ax.set_title(("Stereo depth map" if k == 0
                      else "Depth map + torso segmentation"), pad=6)
        ax.text(0.015, 0.97, "AB"[k], transform=ax.transAxes, fontsize=18,
                fontweight="bold", va="top", color="white")
    cb = fig.colorbar(im, ax=axes, shrink=0.88, pad=0.015, aspect=24)
    cb.set_label("depth (m)" if CLIP_ABOVE_TORSO_M is None
                 else f"depth (m, clipped at {vmax:.1f} m)", fontsize=13)
    cb.ax.tick_params(labelsize=11)
    axes[1].legend(handles=[
        Line2D([], [], color=C_SEG, lw=3, label="YOLO torso mask"),
        Line2D([], [], color=C_SHELL, lw=2, ls=(0, (4, 2)),
               label="Front shell (ICP input)"),
    ], loc="lower right", frameon=True, facecolor="#555555", framealpha=0.9,
        edgecolor="none", fontsize=11, labelcolor="white")
    fig.suptitle(f"Rectified cam0 stereo depth, neutral pose "
                 f"(frame {i}, t = {t_s:+.1f} s)", y=0.99)
    fig.savefig(OUT_PNG)
    print(f"Depth figure -> {OUT_PNG}")


if __name__ == "__main__":
    main()
