# %% 29_gain_sgbm_compare.py
# Is SGBM worse on one take than another because of the sensor gain?
#
# s3_lee's wc (std_movement) and game recordings look like they were taken at different
# analogue gain, and the recorder doesn't store gain. Whole-take SGBM metrics mix gain
# with motion and occlusion, so this compares only the first --n frames after each
# take's GPIO rise, where the subject is assumed to sit in the neutral pose: same pose,
# same torso, static -- what is left is the image itself.
#
# Per take (reference camera, torso mask from 06's segmenter, rectified as 06 does):
#   image    mean0/mean1  torso intensity per camera; lr_ratio = mean1/mean0
#            clip         fraction of torso px at <=2 or >=253
#            noise        temporal intensity noise: per-pixel std over the window
#                         (per-frame mean removed), median over torso px
#            texture      5x5 local intensity std (SGBM block scale), median over torso
#            snr          texture / noise
#   depth    fill, npts, npts_vox, rough_mm, rough_vox_mm as 25_sgbm_bench (median over
#            the window); temp_mm = per-pixel depth std over the window, frame offset
#            removed, median over torso px; merit = rough_vox_mm / sqrt(npts_vox)
#
# Two counterfactuals, both against the first --takes entry (the reference):
#   <take>@gain   the take's frames scaled to the reference's torso mean (a digital
#                 gain: brightness matched, SNR unchanged). If this recovers the
#                 reference's depth metrics, the gap was brightness/contrast vs SGBM's
#                 fixed P1/P2/uniqueness, not sensor noise.
#   ref~<take>    the reference's frames scaled to the take's mean and given extra
#                 Gaussian noise up to the take's noise. If this reproduces the take's
#                 depth metrics, gain (brightness + noise) explains the whole gap.
#
# Run:
#   python trunkpose/dual_notebooks/29_gain_sgbm_compare.py \
#       --session data/trunk_take_oct/oct9_26 --takes s3_lee_wc s3_lee_game s3_lee_game_wtc
import argparse
import csv
import importlib
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
m15 = importlib.import_module("15_subject_icp")
m25 = importlib.import_module("25_sgbm_bench")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--session", default="data/trunk_take_oct/oct9_26")
    p.add_argument("--takes", nargs="+", default=["s3_lee_wc", "s3_lee_game", "s3_lee_game_wtc"],
                   help="take names as 15 addresses them; the first is the reference")
    p.add_argument("--n", type=int, default=10, help="frames after the sync rise")
    p.add_argument("--skip", type=int, default=0, help="frames to skip after the rise")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args()


# %% Per-take setup
_m6 = None


def load_06(rec, env):
    """06 configured for this recording (module-level config, so reload per take)."""
    global _m6
    os.environ["TRUNK_RECORDING_DIR"] = str(Path(rec).resolve())
    os.environ["TRUNK_RECORDING_NAME"] = Path(rec).name
    os.environ.pop("TRUNK_SYNC_PULSE", None)
    os.environ.pop("TRUNK_TAKE", None)
    os.environ.update(env)
    _m6 = importlib.reload(_m6) if _m6 else importlib.import_module("06_trunk_axis")
    return _m6


def neutral_frames(m6, n, skip):
    """(rise, [(gray0, gray1)]) for the n frames starting skip after the take's rise."""
    import cv2

    rise = m6.sync_rise(m6.load_sync_flags(m6.stream_paths("cam0_timestamp")))
    lo, hi = rise + skip, rise + skip + n
    out = []
    for i, (f0, f1) in enumerate(m6.iter_frame_pairs(m6.CAM0_FRAMES, m6.CAM1_FRAMES)):
        if i >= hi:
            break
        if i >= lo:
            g0 = f0 if f0.ndim == 2 else cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY)
            g1 = f1 if f1.ndim == 2 else cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
            out.append((g0, g1))
    return rise, out


# %% Metrics
def image_stats(cv2, var, frames, masks):
    """Torso intensity stats on the rectified pair; cam1's torso is the rect mask moved
    by the median torso disparity (the masks come from the reference camera only)."""
    r0 = np.stack([cv2.remap(g0, *var.maps0, cv2.INTER_LINEAR) for g0, _ in frames]).astype(np.float32)
    r1 = np.stack([cv2.remap(g1, *var.maps1, cv2.INTER_LINEAR) for _, g1 in frames]).astype(np.float32)
    m = np.logical_and.reduce(masks)            # torso px in every frame of the window
    if m.sum() < 100:
        return None
    _, z, _, _ = var.depth(*frames[0], var.mask_full[0])
    f, b = var.f_rect, abs(1.0 / var.Q[3, 2]) / 1000.0
    d = int(round(f * b / np.nanmedian(np.where(m, z, np.nan))))
    m1 = np.zeros_like(m)
    m1[:, : m.shape[1] - d] = m[:, d:]

    v = r0[:, m]
    dev = v - v.mean(axis=1, keepdims=True)                 # per-frame mean (gain drift) out
    noise = float(np.median((dev - dev.mean(axis=0)).std(axis=0, ddof=1)))
    tex = []
    for img in r0:
        mu = cv2.blur(img, (5, 5))
        sd = np.sqrt(np.maximum(cv2.blur(img * img, (5, 5)) - mu * mu, 0))
        tex.append(np.median(sd[m]))
    mean0, mean1 = float(r0[:, m].mean()), float(r1[:, m1].mean())
    clip = float(((v <= 2) | (v >= 253)).mean())
    texture = float(np.median(tex))
    return dict(mean0=mean0, mean1=mean1, lr_ratio=mean1 / mean0, clip=clip,
                noise=noise, texture=texture, snr=texture / max(noise, 1e-6))


def depth_stats(var, m6, frames):
    rows, zs, oks = [], [], []
    for (g0, g1), mfull in zip(frames, var.mask_full):
        pts, z, r, _ = var.depth(g0, g1, mfull)
        ok = r & np.isfinite(z)
        shell = m6.shell_points(pts[ok])
        vox = m6.voxel_downsample(shell)
        rpx, rvox = m25.roughness(var.cv2, pts, z, ok, var.f_rect)
        rows.append(dict(fill=ok.sum() / max(r.sum(), 1), npts=len(shell), npts_vox=len(vox),
                         rough_mm=rpx * 1000, rough_vox_mm=rvox * 1000))
        zs.append(z)
        oks.append(ok)
    med = {k: float(np.nanmedian([r[k] for r in rows])) for k in rows[0]}
    common = np.logical_and.reduce(oks)
    temp = np.nan
    if common.sum() > 100 and len(zs) > 1:
        zc = np.stack([z[common] for z in zs])
        zc -= np.median(zc, axis=1, keepdims=True)          # rigid per-frame shift out
        temp = float(np.median(zc.std(axis=0, ddof=1))) * 1000
    med["temp_mm"] = temp
    med["merit"] = med["rough_vox_mm"] / np.sqrt(max(med["npts_vox"], 1))
    return med


def scale(frames, k):
    return [tuple(np.clip(g.astype(np.float32) * k, 0, 255).astype(np.uint8) for g in p)
            for p in frames]


def add_noise(frames, sigma, rng):
    if not sigma > 0:
        return frames
    return [tuple(np.clip(g.astype(np.float32) + rng.normal(0, sigma, g.shape), 0, 255)
                  .astype(np.uint8) for g in p) for p in frames]


# %% Main
def main():
    a = parse()
    import cv2

    session = (PROJECT_ROOT / a.session) if not Path(a.session).is_absolute() else Path(a.session)
    out = Path(a.out) if a.out else SCRIPT_DIR / "outputs" / "29_gain_sgbm" / "_".join(a.takes)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    takes = {}
    for t in a.takes:
        rec, _, _ = m15.resolve_recording(session, t, game=False)
        m6 = load_06(rec, m15.take_env(session, t))
        rise, frames = neutral_frames(m6, a.n, a.skip)
        seg = m6.load_seg_model()
        mask_full = [m6.torso_mask(seg, g0, m6.CAM_SIZE) for g0, _ in frames]
        var = m25.Variant(m6, t, dict(res=1.0, fs=m6.RECT_FOV_SCALE), m6.CAM_SIZE)
        var.mask_full = mask_full
        masks = [var.depth(g0, g1, mf)[2] for (g0, g1), mf in zip(frames, mask_full)]
        print(f"[{t}] {rec.name} pulse {m6.SYNC_PULSE} rise {rise}: {len(frames)} frames, "
              f"torso in {sum(int(mm.any()) for mm in masks)}", flush=True)
        takes[t] = dict(m6=m6, var=var, frames=frames, masks=masks)

    def evaluate(name, t, frames):
        d = takes[t]
        img = image_stats(cv2, d["var"], frames, d["masks"]) or {}
        dep = depth_stats(d["var"], d["m6"], frames)
        row = {"case": name, **img, **dep}
        print(f"  {name:<28} " + " ".join(
            f"{k}={row[k]:.3g}" for k in ("mean0", "lr_ratio", "clip", "noise", "texture", "snr",
                                          "fill", "npts", "rough_vox_mm", "temp_mm", "merit")
            if k in row), flush=True)
        return row

    rows = [evaluate(t, t, takes[t]["frames"]) for t in a.takes]
    base = {r["case"]: r for r in rows}
    ref = a.takes[0]
    for t in a.takes[1:]:
        k = base[ref]["mean0"] / base[t]["mean0"]
        rows.append(evaluate(f"{t}@gain", t, scale(takes[t]["frames"], k)))
        # reference pushed to this take's gain: brightness first, then the noise the
        # take has beyond the scaled reference's own (scaling scales its noise too)
        k = 1 / k
        extra = np.sqrt(max(base[t]["noise"] ** 2 - (base[ref]["noise"] * k) ** 2, 0))
        sim = add_noise(scale(takes[ref]["frames"], k), extra, rng)
        rows.append(evaluate(f"{ref}~{t}", ref, sim))

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    write_figure(cv2, takes, out)
    print(f"wrote {out}")


def write_figure(cv2, takes, out):
    """First neutral frame per take: rectified torso crop (same grey scale) and depth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, len(takes), figsize=(4.5 * len(takes), 8), squeeze=False)
    for j, (t, d) in enumerate(takes.items()):
        var, (g0, g1), m = d["var"], d["frames"][0], d["masks"][0]
        r0 = cv2.remap(g0, *var.maps0, cv2.INTER_LINEAR)
        _, z, _, _ = var.depth(g0, g1, var.mask_full[0])
        ys, xs = np.nonzero(m)
        if not len(ys):
            continue
        sl = (slice(max(ys.min() - 40, 0), ys.max() + 40), slice(max(xs.min() - 40, 0), xs.max() + 40))
        axs[0, j].imshow(r0[sl], cmap="gray", vmin=0, vmax=255)
        axs[0, j].set_title(f"{t}\nmean {r0[m].mean():.0f}")
        zz = np.where(m, z, np.nan)[sl]
        lo, hi = np.nanpercentile(zz, [2, 98]) if np.isfinite(zz).any() else (0, 1)
        im = axs[1, j].imshow(zz, cmap="turbo", vmin=lo, vmax=hi)
        axs[1, j].set_title(f"torso depth, fill {np.isfinite(zz).sum() / max(m[sl].sum(), 1):.2f}")
        fig.colorbar(im, ax=axs[1, j], shrink=0.7)
    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "neutral_frames.png", dpi=110)


if __name__ == "__main__":
    main()
