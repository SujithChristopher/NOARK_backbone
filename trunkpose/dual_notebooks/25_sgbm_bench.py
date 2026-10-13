# %% 25_sgbm_bench.py
# Which SGBM / rectification settings give the best torso depth on full-res captures?
#
# 24_rig_orientation_sim.py predicted, for the sep23 framing, that
#   * rectifying with fov_scale ~0.82 instead of 1.0 keeps the image centre at the
#     source focal (06 shrinks it 597 -> 488 px at full res): ~1.4x
#   * full 1280x800 instead of the 640x400 binned capture: ~4x
# This bench measures both on a real recording, plus the SGBM knobs, without mocap.
#
# The same sampled frames go through every variant. The torso mask comes from 06's
# torso_mask (trunk_seg-7) on the full-res reference image, once per frame; the binned
# variants use the same frames area-downsampled 2x (what the sensor binning does).
# Per variant and frame, the torso shell is cut exactly as 06 does (eroded mask,
# depth gate, mode shell). Metrics, all ground-truth free:
#   fill     valid-disparity torso px / torso mask px (rectified reference image)
#   npts     shell points (06's confidence signal; >= 800 was SAM3D-level accuracy)
#   rough_mm     depth minus a 1.5 cm masked Gaussian of itself: high-frequency noise
#                at a fixed physical scale, robust std over torso px, mm
#   rough_vox_mm the same residual averaged inside 06's 1 cm voxels -- what ICP gets
#   temp_mm  frame-to-frame depth noise on consecutive frames: per-pixel dz over the
#            common valid torso px, median removed (rigid shift), 1.4826*MAD/sqrt(2)
#   merit    rough_vox_mm / sqrt(npts_vox) -- plane/ICP noise floor, lower is better
#   ms       SGBM (+ bilateral) time per frame on this machine
#
# Run:
#   python trunkpose/dual_notebooks/25_sgbm_bench.py --rec <recording dir> [--n 30]
# The recording dir needs a session.toml at or above it (06's rules: size, swap, rot180).
import argparse
import csv
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CALIB = PROJECT_ROOT / "data/calibration/dual_160/calib_radxa_sep18_26/stereo_calibration.toml"
DEFAULT_SEG = PROJECT_ROOT / "trunkpose/runs/trunk_seg-7/weights/best.pt"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--rec", required=True, help="recording dir (cam*_frame msgpack)")
    p.add_argument("--n", type=int, default=30, help="frame pairs to sample (each + its successor)")
    p.add_argument("--skip", type=int, default=60, help="frames to skip at the start")
    p.add_argument("--max-frames", type=int, default=None,
                   help="only sample from the first N frames (e.g. chunks already copied)")
    p.add_argument("--variants", default="all", help="comma list of variant names, or all")
    p.add_argument("--out", default=None)
    return p.parse_args()


def setup(a):
    os.environ["TRUNK_RECORDING_DIR"] = str(Path(a.rec).resolve())
    os.environ["TRUNK_RECORDING_NAME"] = Path(a.rec).name
    os.environ.setdefault("TRUNK_STEREO_TOML", str(DEFAULT_CALIB))
    os.environ.setdefault("TRUNK_SEG_WEIGHTS", str(DEFAULT_SEG))
    os.environ.setdefault("TRUNK_CHARUCO_TOML", "none")
    sys.path.insert(0, str(SCRIPT_DIR))
    return importlib.import_module("06_trunk_axis")


# %% Variants
# res: 1.0 = full sensor, 0.5 = 2x binned. fs: rectification fov_scale. The SGBM keys
# override 06's make_sgbm() settings; anything absent keeps 06's value.
BASE_SGBM = dict(block=5, num_disp=160, p1=8, p2=32, uniq=10, speckle_win=100,
                 speckle_range=2, prefilter=0, mode="3way", bilateral=True, roi=False)
VARIANTS = {
    # what sep23 ran (binned, 06 as-is) and what the oct9 capture gives 06 unchanged
    "half_06":        dict(res=0.5, fs=1.0),
    "half_fs082":     dict(res=0.5, fs=0.82),
    "full_06":        dict(res=1.0, fs=1.0),
    "full_fs082":     dict(res=1.0, fs=0.82),
    # SGBM knobs, all on full_fs082
    "f82_blk3":       dict(res=1.0, fs=0.82, block=3),
    "f82_blk7":       dict(res=1.0, fs=0.82, block=7),
    "f82_blk9":       dict(res=1.0, fs=0.82, block=9),
    "f82_blk11":      dict(res=1.0, fs=0.82, block=11),
    "f82_p2x2":       dict(res=1.0, fs=0.82, p2=64),
    "f82_p1x2_p2x4":  dict(res=1.0, fs=0.82, p1=16, p2=128),
    "f82_uniq5":      dict(res=1.0, fs=0.82, uniq=5),
    "f82_uniq15":     dict(res=1.0, fs=0.82, uniq=15),
    "f82_prefilt31":  dict(res=1.0, fs=0.82, prefilter=31),
    "f82_hh":         dict(res=1.0, fs=0.82, mode="hh"),
    "f82_nobilat":    dict(res=1.0, fs=0.82, bilateral=False),
    "f82_speckle_r4": dict(res=1.0, fs=0.82, speckle_range=4),
    "f82_nd96_roi":   dict(res=1.0, fs=0.82, num_disp=96, roi=True),
}


def make_sgbm(cv2, v):
    mode = {"3way": cv2.STEREO_SGBM_MODE_SGBM_3WAY, "hh": cv2.STEREO_SGBM_MODE_HH,
            "sgbm": cv2.STEREO_SGBM_MODE_SGBM}[v["mode"]]
    b = v["block"]
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=v["num_disp"], blockSize=b,
        P1=v["p1"] * b * b, P2=v["p2"] * b * b, disp12MaxDiff=1,
        uniquenessRatio=v["uniq"], speckleWindowSize=v["speckle_win"],
        speckleRange=v["speckle_range"], preFilterCap=v["prefilter"], mode=mode)


# %% Frames
def sample_pairs(m6, n, skip, max_frames=None):
    """(i, gray0_i, gray1_i, gray0_i+1, gray1_i+1) for n evenly spread i, full res.

    Reads the streams once; only the sampled frames are kept in memory."""
    import cv2

    n_total = max_frames or len(m6.load_frame_times(m6.CAM0_TIMES))
    picks = np.linspace(skip, max(skip, n_total - 2), n).astype(int)
    want = set(picks) | set(picks + 1)
    keep = {}
    for i, (f0, f1) in enumerate(m6.iter_frame_pairs(m6.CAM0_FRAMES, m6.CAM1_FRAMES)):
        if i in want:
            g0 = f0 if f0.ndim == 2 else cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY)
            g1 = f1 if f1.ndim == 2 else cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
            keep[i] = (g0, g1)
        if i > picks[-1]:
            break
    return [(i, *keep[i], *keep[i + 1]) for i in picks if i in keep and i + 1 in keep], n_total


# %% Depth for one variant
class Variant:
    def __init__(self, m6, name, cfg, full_size):
        import cv2

        self.cv2, self.m6, self.name = cv2, m6, name
        self.v = {**BASE_SGBM, **cfg}
        s = self.v["res"]
        self.size = (int(round(full_size[0] * s)), int(round(full_size[1] * s)))
        K0, D0, K1, D1, R, T = m6.load_stereo(m6.STEREO_TOML, size=self.size)
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
            K0, D0, K1, D1, self.size, R, T, flags=cv2.CALIB_ZERO_DISPARITY,
            balance=0.0, fov_scale=self.v["fs"])
        self.maps0 = cv2.fisheye.initUndistortRectifyMap(K0, D0, R1, P1, self.size, cv2.CV_16SC2)
        self.maps1 = cv2.fisheye.initUndistortRectifyMap(K1, D1, R2, P2, self.size, cv2.CV_16SC2)
        self.Q, self.f_rect, self.f_src = Q, P1[0, 0], K0[0, 0]
        self.sgbm = make_sgbm(cv2, self.v)

    def prep(self, g):
        cv2 = self.cv2
        if self.v["res"] != 1.0:
            g = cv2.resize(g, self.size, interpolation=cv2.INTER_AREA)
        return g

    def depth(self, g0, g1, mask_full):
        """(z map in m [nan = invalid], eroded rect torso mask, sgbm ms)."""
        cv2, m6 = self.cv2, self.m6
        g0, g1 = self.prep(g0), self.prep(g1)
        mask = cv2.resize(mask_full, self.size, interpolation=cv2.INTER_NEAREST)
        r0 = cv2.remap(g0, *self.maps0, cv2.INTER_LINEAR)
        r1 = cv2.remap(g1, *self.maps1, cv2.INTER_LINEAR)
        mrect = cv2.remap(mask, *self.maps0, cv2.INTER_NEAREST)
        area = int((mrect > 0).sum())
        if area > 0:
            ksz = max(3, int(round(np.sqrt(area) * m6.MASK_ERODE_FRAC))) | 1
            mrect = cv2.erode(mrect, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)))

        t0 = time.perf_counter()
        if self.v["roi"] and (mrect > 0).any():
            # SGBM on a band around the torso only: rows of the mask bbox, columns from
            # num_disp left of it (the matcher needs that much of the right image)
            ys, xs = np.nonzero(mrect)
            pad = 16
            y0, y1 = max(0, ys.min() - pad), min(self.size[1], ys.max() + pad + 1)
            x0 = max(0, xs.min() - pad - self.v["num_disp"])
            x1 = min(self.size[0], xs.max() + pad + 1)
            disp_raw = np.full(r0.shape, -1.0, np.float32)
            disp_raw[y0:y1, x0:x1] = self.sgbm.compute(
                np.ascontiguousarray(r0[y0:y1, x0:x1]),
                np.ascontiguousarray(r1[y0:y1, x0:x1])).astype(np.float32) / 16.0
        else:
            disp_raw = self.sgbm.compute(r0, r1).astype(np.float32) / 16.0
        disp = disp_raw
        if self.v["bilateral"]:
            # 06's sigma is in px of disparity at 640x400 binned; scale with resolution
            # so the filter keeps the same metric smoothing
            s = self.v["res"] / 0.5 * self.f_rect / self.f_src
            disp = cv2.bilateralFilter(disp_raw, m6.BILATERAL_D,
                                       m6.BILATERAL_SIGMA_COLOR * s, m6.BILATERAL_SIGMA_SPACE)
        ms = (time.perf_counter() - t0) * 1000
        pts = cv2.reprojectImageTo3D(disp, self.Q) / 1000.0
        z = pts[..., 2].copy()
        valid = (disp_raw > 0) & np.isfinite(pts).all(axis=2) \
            & (z > m6.DEPTH_MIN_M) & (z < m6.DEPTH_MAX_M)
        z[~valid] = np.nan
        return pts, z, mrect > 0, ms


def roughness(cv2, pts, z, ok, f_rect, scale_m=0.015):
    """High-frequency depth noise at a fixed physical scale, so resolutions compare.

    z minus a masked Gaussian of ~scale_m (in px at the torso's depth); robust std over
    pixels, and over 06's 1 cm voxels after averaging the residual inside each voxel
    (the cloud ICP gets). Returns (px_mm, vox_mm)."""
    if ok.sum() < 100:
        return np.nan, np.nan
    zf = np.where(ok, z, 0).astype(np.float32)
    sig = 0.015 * f_rect / float(np.median(z[ok])) * scale_m / 0.015
    num = cv2.GaussianBlur(zf, (0, 0), sig)
    den = cv2.GaussianBlur(ok.astype(np.float32), (0, 0), sig)
    use = ok & (den > 0.5)
    r = z[use] - num[use] / den[use]
    px = 1.4826 * np.median(np.abs(r - np.median(r)))
    keys = np.floor(pts[use] / 0.01).astype(np.int64)
    _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    rv = np.bincount(inv.ravel(), weights=r) / cnt
    vox = 1.4826 * np.median(np.abs(rv - np.median(rv)))
    return float(px), float(vox)


def score(var, pairs, masks):
    rows = []
    for (i, a0, a1, b0, b1), (ma, mb) in zip(pairs, masks):
        pts_a, za, ra, ms = var.depth(a0, a1, ma)
        _, zb, rb, _ = var.depth(b0, b1, mb)
        if ra.sum() == 0:
            continue
        ok = ra & np.isfinite(za)
        shell = var.m6.shell_points(pts_a[ok])
        common = ok & rb & np.isfinite(zb)
        temp = np.nan
        if common.sum() > 100:
            dz = zb[common] - za[common]
            dz -= np.median(dz)
            temp = 1.4826 * np.median(np.abs(dz)) / np.sqrt(2)
        vox = var.m6.voxel_downsample(shell)
        rpx, rvox = roughness(var.cv2, pts_a, za, ok, var.f_rect)
        rows.append(dict(frame=i, fill=ok.sum() / ra.sum(), npts=len(shell), npts_vox=len(vox),
                         rough_mm=rpx * 1000, rough_vox_mm=rvox * 1000,
                         temp_mm=temp * 1000, ms=ms))
    return rows


# %% Main
def main():
    a = parse()
    m6 = setup(a)
    import cv2

    out = Path(a.out) if a.out else SCRIPT_DIR / "outputs" / "25_sgbm_bench" / Path(a.rec).name
    out.mkdir(parents=True, exist_ok=True)
    full_size = m6.CAM_SIZE
    print(f"rec {a.rec}\n size {full_size} swap={m6.SWAP_CAMS} rot180={m6.ROT180} "
          f"calib {m6.STEREO_TOML}")
    pairs, n_total = sample_pairs(m6, a.n, a.skip, a.max_frames)
    print(f" {n_total} frames, {len(pairs)} sampled pairs")

    seg = m6.load_seg_model()
    masks = [(m6.torso_mask(seg, a0, full_size), m6.torso_mask(seg, b0, full_size))
             for _, a0, _, b0, _ in pairs]
    has = [int(ma.any()) for ma, _ in masks]
    print(f" torso found in {sum(has)}/{len(has)} sampled frames")

    names = list(VARIANTS) if a.variants == "all" else a.variants.split(",")
    summary, per_frame = [], []
    for name in names:
        var = Variant(m6, name, VARIANTS[name], full_size)
        rows = score(var, pairs, masks)
        for r in rows:
            per_frame.append({"variant": name, **r})
        med = {k: float(np.nanmedian([r[k] for r in rows])) if rows else np.nan
               for k in ("fill", "npts", "npts_vox", "rough_mm", "rough_vox_mm", "temp_mm", "ms")}
        med["merit"] = med["rough_vox_mm"] / np.sqrt(max(med["npts_vox"], 1))
        summary.append({"variant": name, "f_rect": var.f_rect, **med})
        print(f"  {name:<15} f_rect={var.f_rect:6.1f} fill={med['fill']:.2f} "
              f"npts={med['npts']:7.0f} vox={med['npts_vox']:6.0f} rough={med['rough_mm']:5.2f} rvox={med['rough_vox_mm']:5.2f}mm "
              f"temp={med['temp_mm']:5.2f}mm merit={med['merit']:.4f} {med['ms']:6.1f}ms",
              flush=True)

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with open(out / "per_frame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_frame[0]))
        w.writeheader()
        w.writerows(per_frame)
    write_figure(m6, pairs, masks, full_size, out)
    print(f"wrote {out}")


def write_figure(m6, pairs, masks, full_size, out):
    """Depth of the torso for the first frame with a mask: half_06 vs full_06 vs full_fs082."""
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = next((j for j, (ma, _) in enumerate(masks) if ma.any()), None)
    if k is None:
        return
    _, a0, a1, _, _ = pairs[k]
    show = ("half_06", "full_06", "full_fs082")
    fig, axs = plt.subplots(1, len(show) + 1, figsize=(5 * (len(show) + 1), 4))
    var = Variant(m6, "full_fs082", VARIANTS["full_fs082"], full_size)
    axs[0].imshow(cv2.remap(a0, *var.maps0, cv2.INTER_LINEAR), cmap="gray")
    axs[0].set_title("rectified reference (fov_scale 0.82)")
    zs = []
    for name in show:
        v = Variant(m6, name, VARIANTS[name], full_size)
        _, z, r, _ = v.depth(a0, a1, masks[k][0])
        z = np.where(r, z, np.nan)
        zs.append((name, z))
    lo = np.nanpercentile(np.concatenate([z[np.isfinite(z)] for _, z in zs]), 2)
    hi = np.nanpercentile(np.concatenate([z[np.isfinite(z)] for _, z in zs]), 98)
    for ax, (name, z) in zip(axs[1:], zs):
        im = ax.imshow(z, cmap="turbo", vmin=lo, vmax=hi)
        ys, xs = np.nonzero(np.isfinite(z))
        if len(ys):
            ax.set_xlim(xs.min() - 20, xs.max() + 20)
            ax.set_ylim(ys.max() + 20, ys.min() - 20)
        ax.set_title(f"{name}: torso depth (m)")
    fig.colorbar(im, ax=axs[1:].tolist(), shrink=0.8)
    fig.savefig(out / "torso_depth.png", dpi=110)


if __name__ == "__main__":
    main()
