# %% 22_depth_fusion.py
# Does monocular depth (Ultralytics YOLO26n-depth) fused with the stereo SGBM depth
# make torso-shell ICP better on the NOARK game takes?
#
# One pass over the recording redoes 06's torso mask + SGBM and, alongside it, runs
# the mono depth model on the rectified reference image. Mono depth is relative, so
# per frame it is mapped onto the stereo depth with a robust log-affine fit
#   log Z_stereo = a * log Z_mono + b
# on the torso pixels SGBM did solve. Four torso clouds come out of the same masks:
#
#   A  sgbm     06's cloud, rebuilt here (sanity check against the geometry cache)
#   B  holefill SGBM where valid, fitted mono depth only in SGBM's holes
#   C  blend    log-depth blend weighted by local image texture (SGBM's reliability)
#   D  mono     fitted mono depth alone; stereo only supplies the scale
#
# Each variant then goes through the unchanged 07 path (ICP to the neutral cloud ->
# GPIO sync -> teleport split -> S fit) and is scored the way 21_sam3d_score.py
# scores SAM 3D Body, so the numbers sit next to its table in SAM3D_BODY.md.
#
# The pass also logs, per frame, how much torso the segmentation still sees and how
# much of it SGBM solved, which attributes every ICP failure to occlusion (mono depth
# cannot help: the torso surface is not in the image) or to stereo holes (it can).
#
#   uv run python trunkpose/dual_notebooks/22_depth_fusion.py --subject gokul --game
#
# Output: <rec>/depth_fusion/fusion_cache.pkl (pass), fusion_scores.npz, and
# <session>/noark_game/output/<subject>_depth_fusion.png

import argparse
import importlib
import os
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SESSION = PROJECT_ROOT / "data" / "sep23_26"
DEPTH_WEIGHTS = PROJECT_ROOT / "trunkpose" / "runs" / "yolo26n-depth.pt"

VARIANTS = {"A": "sgbm", "B": "holefill", "C": "blend", "D": "mono"}
AX = ("flexion", "lateral", "axial", "magnitude")
KEYS = ("flex_iz", "lat_iz", "axi_iz", "mag_i")
MKEYS = ("mflex", "mlat", "maxi", "mag_m")

# mono -> stereo fit
FIT_MIN_PX = 300          # torso pixels SGBM must have solved to fit on this frame
FIT_CARRY = 10            # else reuse the last torso fit for up to this many frames
FIT_TRIM_MAD = 2.5        # robust refit: drop |resid| > this * MAD
FIT_MAX_RESID = 0.08      # reject a fit whose MAD residual (log units ~ rel. depth) exceeds this
# texture confidence for the blend: mean Sobel magnitude (gray levels) over a box
TEX_BOX = 7
TEX_FULL = 25.0           # texture at which SGBM gets full weight

# failure attribution (variant A, in-window frames)
VIS_BINS = (0.0, 0.35, 0.6, 0.85, 10.0)     # torso seg area / neutral median
VALID_BINS = (0.0, 0.3, 0.6, 0.8, 1.01)     # SGBM-solved share of the eroded mask


def parse():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default=str(DEFAULT_SESSION))
    ap.add_argument("--subject", required=True)
    ap.add_argument("--game", action="store_true", help="NOARK game take")
    ap.add_argument("--stride", type=int, default=None,
                    help="frame stride (default 3 for --game, 1 otherwise); must match "
                         "the geometry cache")
    ap.add_argument("--variants", default="ABCD")
    ap.add_argument("--recompute", action="store_true", help="redo the depth pass")
    return ap.parse_args()


# %% Setup (06 resolves its paths at import time, so the env goes first)
def setup(a):
    m15 = importlib.import_module("15_subject_icp")
    rec_dir, mocap_csv, out_dir = m15.resolve_recording(Path(a.session), a.subject, a.game)
    stride = a.stride or (3 if a.game else 1)
    os.environ["TRUNK_RECORDING_NAME"] = a.subject
    os.environ["TRUNK_RECORDING_DIR"] = str(rec_dir)
    os.environ["TRUNK_MOCAP_CSV"] = str(mocap_csv)
    os.environ["TRUNK_FRAME_STRIDE"] = str(stride)
    m6 = importlib.import_module("06_trunk_axis")
    m7 = importlib.import_module("07_icp_trunk")
    os.environ.setdefault("TRUNK_GEOM_CACHE", str(m6.default_geom_cache()))
    if a.game:
        sync = m6.load_sync_flags(m6.CAM0_TIMES)
        m6.NEUTRAL_START = int(np.argmax(sync == 1))
    return m6, m7, rec_dir, out_dir


# %% Depth pass
def torso_and_arm(m6, seg_model, gray, size):
    """06.torso_mask, but also returning the torso before arm removal, so occlusion by
    arms can be told apart from occlusion by everything else (the game cube)."""
    import cv2

    W, H = size
    res = seg_model.predict(cv2.cvtColor(m6.for_inference(gray), cv2.COLOR_GRAY2RGB),
                            conf=m6.SEG_CONF, verbose=False)[0]
    torso = np.zeros((H, W), np.uint8)
    arm = np.zeros((H, W), np.uint8)
    if res.masks is not None:
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        best, best_d = None, None
        for m, cid in zip(res.masks.data.cpu().numpy(), cls_ids):
            r = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
            if not r.any():
                continue
            if res.names[cid] in ("torso", "chest"):
                ys, xs = np.nonzero(r)
                d = float(np.hypot(xs.mean() - W / 2, ys.mean() - H / 2))
                if best_d is None or d < best_d:
                    best, best_d = r, d
            elif res.names[cid] == "arm":
                arm[r] = 255
        if best is not None:
            torso[best] = 255
    full = torso.copy()
    if arm.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (m6.ARM_EXCLUDE_DILATE_PX,) * 2)
        torso[cv2.dilate(arm, k) > 0] = 0
    return m6.from_inference_mask(full), m6.from_inference_mask(torso)


def mono_depth(m6, model, rect_gray):
    """Mono depth on the rectified reference image (so it is pixel-aligned with the
    disparity), in the same upright orientation the networks expect."""
    import cv2

    rgb = cv2.cvtColor(m6.for_inference(rect_gray), cv2.COLOR_GRAY2RGB)
    d = model.predict(rgb, verbose=False)[0].depth.data   # DepthMap -> (H, W) metres
    d = d.cpu().numpy() if hasattr(d, "cpu") else np.asarray(d)
    d = d.astype(np.float32)
    if d.shape != rect_gray.shape:
        d = cv2.resize(d, rect_gray.shape[::-1], interpolation=cv2.INTER_LINEAR)
    return m6.from_inference_mask(d)


def robust_logfit(lm, ls):
    """(a, b, mad) for ls ~ a*lm + b, scale-only (a = 1), MAD-trimmed.

    A free slope is unidentifiable here: the torso spans a few cm of depth, so log Z
    barely varies over the fit pixels and a swung 0.1..1.3 between neighbouring frames
    in the smoke test. The depth head is metric-calibrated already, so keep its shape
    and take only the scale from stereo."""
    keep = np.ones(len(lm), bool)
    b = mad = np.nan
    for _ in range(3):
        if keep.sum() < 50:
            return None
        b = float(np.median(ls[keep] - lm[keep]))
        r = ls - (lm + b)
        mad = float(np.median(np.abs(r[keep])) * 1.4826)
        keep = np.abs(r) <= max(FIT_TRIM_MAD * mad, 1e-3)
    return 1.0, b, mad


def shell_cloud(m6, Z, ok, mrect_eroded, xx, yy, cam, R1, R0_c, t0_c, rng):
    """06.stereo_step's torso selection + 06's cloud post-processing on a depth map."""
    f, cx, cy = cam
    torso_ok = ok & (mrect_eroded > 0) & (Z > m6.DEPTH_MIN_M) & (Z < m6.DEPTH_MAX_M)
    z = Z[torso_ok]
    tp = np.stack([(xx[torso_ok] - cx) * z / f, (yy[torso_ok] - cy) * z / f, z], 1)
    if len(tp):
        zc = np.percentile(tp[:, 2], m6.FRONT_PCTL)
        tp = tp[tp[:, 2] <= zc + m6.SHELL_M]
    tp = tp @ R1
    if len(tp) < 50:
        return None, len(tp)
    ds = m6.voxel_downsample(tp, rng=rng)
    return m6.to_board_bulk(ds, R0_c, t0_c).astype(np.float32), len(tp)


def depth_pass(m6, n):
    import cv2
    from tqdm.auto import tqdm
    from ultralytics import YOLO

    K0, D0, K1, D1, R, T_mm = m6.load_stereo(m6.STEREO_TOML)
    R0_c, t0_c, _, _ = m6.load_charuco_basis(m6.CHARUCO_TOML)
    R1, Q, maps0, maps1 = m6.build_rectification(K0, D0, K1, D1, R, T_mm, m6.CAM_SIZE)
    cam = (float(Q[2, 3]), float(-Q[0, 3]), float(-Q[1, 3]))   # f, cx, cy (rectified)
    sgbm = m6.make_sgbm()
    seg = m6.load_seg_model()
    dmodel = YOLO(str(DEPTH_WEIGHTS))
    rng = np.random.default_rng(0)
    W, H = m6.CAM_SIZE
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    clouds = {k: [None] * n for k in VARIANTS}
    npts = {k: np.zeros(n, int) for k in VARIANTS}
    diag = {k: np.full(n, np.nan) for k in (
        "torso_px", "torso_noarm_px", "eroded_px", "valid_px", "fit_a", "fit_b",
        "fit_mad", "fit_src")}   # fit_src: 0 torso, 1 carried, nan none
    last_fit, last_fit_i = None, -10 ** 9
    checked = False

    for i, (f0, f1) in enumerate(tqdm(m6.iter_frame_pairs(m6.CAM0_FRAMES, m6.CAM1_FRAMES),
                                      total=n, desc="depth pass")):
        if i >= n:
            break
        g0 = f0 if f0.ndim == 2 else cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY)
        g1 = f1 if f1.ndim == 2 else cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)

        full, mask0 = torso_and_arm(m6, seg, g0, m6.CAM_SIZE)
        rect0 = cv2.remap(g0, *maps0, cv2.INTER_LINEAR)
        rect1 = cv2.remap(g1, *maps1, cv2.INTER_LINEAR)
        disp_raw = sgbm.compute(rect0, rect1).astype(np.float32) / 16.0
        disp = cv2.bilateralFilter(disp_raw, m6.BILATERAL_D, m6.BILATERAL_SIGMA_COLOR,
                                   m6.BILATERAL_SIGMA_SPACE)
        pts = cv2.reprojectImageTo3D(disp, Q) / 1000.0
        valid = (disp_raw > m6.SGBM_MIN_DISP) & np.isfinite(pts).all(axis=2)
        # bilateral smoothing can push a valid edge pixel to Z <= 0; 06 drops those via
        # its depth gate, here they must also stay out of the log-domain fit and blend
        Zs = np.where(valid & (pts[..., 2] > 0), pts[..., 2], np.nan).astype(np.float32)
        if not checked and valid.any():
            # the back-projection below must reproduce reprojectImageTo3D exactly
            f, cx, _ = cam
            v = valid & (Zs > m6.DEPTH_MIN_M) & (Zs < m6.DEPTH_MAX_M)
            err = np.abs((xx[v] - cx) * Zs[v] / f - pts[..., 0][v]).max()
            assert err < 1e-4, f"back-projection mismatch {err:.5f} m"
            checked = True

        # masks exactly as 06.stereo_step builds them
        frect = cv2.remap(full, *maps0, cv2.INTER_NEAREST)
        mrect = cv2.remap(mask0, *maps0, cv2.INTER_NEAREST)
        diag["torso_px"][i] = int((frect > 0).sum())
        area = int((mrect > 0).sum())
        diag["torso_noarm_px"][i] = area
        if area > 0:
            ksz = max(3, int(round(np.sqrt(area) * m6.MASK_ERODE_FRAC))) | 1
            mrect = cv2.erode(mrect, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)))
        ero = mrect > 0
        diag["eroded_px"][i] = int(ero.sum())
        in_rng = valid & (Zs > m6.DEPTH_MIN_M) & (Zs < m6.DEPTH_MAX_M)
        diag["valid_px"][i] = int((in_rng & ero).sum())

        # ---- mono depth, log-affine fit to SGBM on the torso pixels it solved ----
        Zm_raw = mono_depth(m6, dmodel, rect0)
        good_m = np.isfinite(Zm_raw) & (Zm_raw > 0)
        sel = in_rng & (frect > 0) & good_m          # whole torso (arms in) for the fit
        fit = None
        if sel.sum() >= FIT_MIN_PX:
            fit = robust_logfit(np.log(Zm_raw[sel]), np.log(Zs[sel]))
            if fit is not None and fit[2] > FIT_MAX_RESID:
                fit = None
            if fit is not None:
                last_fit, last_fit_i = fit, i
                diag["fit_src"][i] = 0
        if fit is None and last_fit is not None and i - last_fit_i <= FIT_CARRY:
            fit = last_fit
            diag["fit_src"][i] = 1
        if fit is not None:
            diag["fit_a"][i], diag["fit_b"][i], diag["fit_mad"][i] = fit
            Zm = np.where(good_m, np.exp(fit[0] * np.log(np.where(good_m, Zm_raw, 1.0))
                                         + fit[1]), np.nan).astype(np.float32)
        else:
            Zm = np.full_like(Zs, np.nan)

        # ---- the four depth maps ----
        sv = np.isfinite(Zs)
        mv = np.isfinite(Zm)
        gx = cv2.Sobel(rect0, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(rect0, cv2.CV_32F, 0, 1, ksize=3)
        tex = cv2.blur(np.hypot(gx, gy), (TEX_BOX, TEX_BOX))
        w = np.clip(tex / TEX_FULL, 0.0, 1.0)
        both = sv & mv
        Zc = np.where(sv, Zs, Zm)
        Zc[both] = np.exp(w[both] * np.log(Zs[both]) + (1 - w[both]) * np.log(Zm[both]))
        maps = {
            "A": (Zs, sv),
            "B": (np.where(sv, Zs, Zm), sv | mv),
            "C": (Zc, sv | mv),
            "D": (Zm, mv),
        }
        for k, (Z, ok) in maps.items():
            clouds[k][i], npts[k][i] = shell_cloud(
                m6, np.nan_to_num(Z, nan=0.0), ok, mrect, xx, yy, cam, R1, R0_c, t0_c, rng)
    return dict(clouds=clouds, npts=npts, diag=diag)


def cache_key(m6, n):
    return dict(n=n, stride=m6.FRAME_STRIDE, geom=m6.geom_cache_key(),
                depth=m6._fingerprint(DEPTH_WEIGHTS),
                fit=[FIT_MIN_PX, FIT_CARRY, FIT_TRIM_MAD, FIT_MAX_RESID, TEX_BOX, TEX_FULL])


def load_or_run_pass(m6, n, rec_dir, recompute):
    path = rec_dir / "depth_fusion" / "fusion_cache.pkl"
    key = cache_key(m6, n)
    if path.exists() and not recompute:
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        if d.get("key") == key:
            print(f"depth pass: cached ({path})")
            return d
        print("depth pass: cache key changed, recomputing")
    d = depth_pass(m6, n)
    d["key"] = key
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(d, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"depth pass -> {path}")
    return d


# %% Scoring
def cell(x, y, m):
    """median / p95 |error|, r, n (as 21_sam3d_score.stat)."""
    if m.sum() < 30:
        return dict(n=int(m.sum()))
    e = np.abs(x[m] - y[m])
    return dict(med=float(np.median(e)), p95=float(np.percentile(e, 95)),
                r=float(np.corrcoef(x[m], y[m])[0, 1]), n=int(m.sum()))


def fmt(c):
    if "med" not in c:
        return f"{'n<30':>24}"
    return f"{c['med']:5.1f}/{c['p95']:5.1f} r{c['r']:+.2f} n{c['n']:>5}"


def score_variant(m6, m7, inp, frames, clouds, R_neu):
    n = inp["n"]
    fv = [dict(fr, cloud=clouds[i]) for i, fr in enumerate(frames)]
    icp = m7.run_icp_pass(fv, R_neu, n)
    cmp_ = m7.compare_mocap(inp, icp["Rb_icp"], R_neu)
    in_win = cmp_["in_win"]
    out = dict(in_win=in_win, gated=icp["gated"], solved=np.array(
        [r is not None for r in icp["Rb_icp"]]))
    for ax, k, mk in zip(AX, KEYS, MKEYS):
        raw = cmp_[k]
        out[f"cam_{ax}"] = m6.smooth_series(raw)
        out[f"valid_{ax}"] = np.isfinite(raw) & in_win
        out[f"moc_{ax}"] = cmp_[mk]
        # jitter: what Savitzky-Golay removes, on solved frames
        out[f"jit_{ax}"] = float(np.nanstd((raw - out[f"cam_{ax}"])[out[f"valid_{ax}"]]))
    return out


def attribute_failures(diag, res_A, neu_slice):
    """Cross-tab of solved / failed in-window frames by visible torso and SGBM fill."""
    vis_ref = np.nanmedian(diag["torso_px"][neu_slice])
    vis = diag["torso_px"] / vis_ref
    vfrac = diag["valid_px"] / np.maximum(diag["eroded_px"], 1)
    vfrac[diag["eroded_px"] == 0] = 0.0
    win = res_A["in_win"]
    ok = res_A["valid_magnitude"]
    print(f"\nFailure attribution, variant A (in-window {int(win.sum())} frames, "
          f"neutral torso {vis_ref:.0f} px)")
    print("  rows: visible torso / neutral; cols: SGBM-solved share of eroded mask")
    print("  each cell: failed / total")
    hdr = "".join(f"{f'{lo:.1f}-{hi:.1f}' if hi < 1.01 else f'{lo:.1f}+':>14}"
                  for lo, hi in zip(VALID_BINS[:-1], VALID_BINS[1:]))
    print(f"  {'visible':<12}{hdr}{'all':>14}")
    for vlo, vhi in zip(VIS_BINS[:-1], VIS_BINS[1:]):
        row = win & (vis >= vlo) & (vis < vhi)
        lab = f"{vlo:.2f}-{vhi:.2f}" if vhi < 10 else f"{vlo:.2f}+"
        cells = ""
        for flo, fhi in zip(VALID_BINS[:-1], VALID_BINS[1:]):
            c = row & (vfrac >= flo) & (vfrac < fhi)
            cells += f"{f'{int((c & ~ok).sum())}/{int(c.sum())}':>14}"
        print(f"  {lab:<12}{cells}{f'{int((row & ~ok).sum())}/{int(row.sum())}':>14}")
    fail = win & ~ok
    occl = fail & (vis < VIS_BINS[1])
    holes = fail & (vis >= VIS_BINS[1]) & (vfrac < VALID_BINS[2])
    print(f"  failed {int(fail.sum())}: occluded (<{VIS_BINS[1]:.0%} torso visible) "
          f"{int(occl.sum())}, stereo holes (visible but <{VALID_BINS[2]:.0%} solved) "
          f"{int(holes.sum())}, other (registration/gate) "
          f"{int((fail & ~occl & ~holes).sum())}")
    return dict(vis=vis, vfrac=vfrac, occl=occl, holes=holes)


def main():
    a = parse()
    sys.path.insert(0, str(SCRIPT_DIR))
    m6, m7, rec_dir, out_dir = setup(a)
    print(f"[{a.subject}] {rec_dir}  stride={m6.FRAME_STRIDE} "
          f"neutral_start={m6.NEUTRAL_START}")

    inp = m7.load_inputs()
    frames, n = inp["frames"], inp["n"]
    R_neu = m6.assemble(frames)[1]
    fp = load_or_run_pass(m6, n, rec_dir, a.recompute)

    # sanity: A must rebuild 06's clouds (same masks, same SGBM)
    n_cache = np.array([0 if fr["cloud"] is None else len(fr["cloud"]) for fr in frames])
    n_A = np.array([0 if c is None else len(c) for c in fp["clouds"]["A"]])
    print(f"A vs geometry cache: clouds present {int((n_A > 0).sum())} vs "
          f"{int((n_cache > 0).sum())}, median |d npts| "
          f"{np.median(np.abs(n_A - n_cache)):.0f}")
    d = fp["diag"]
    print(f"mono fit: torso {int((d['fit_src'] == 0).sum())}, carried "
          f"{int((d['fit_src'] == 1).sum())}, none {int(np.isnan(d['fit_src']).sum())}; "
          f"median a={np.nanmedian(d['fit_a']):.2f} "
          f"resid(MAD log)={np.nanmedian(d['fit_mad']):.3f}")

    res = {}
    for k in a.variants:
        print(f"\n===== variant {k} ({VARIANTS[k]}) =====")
        res[k] = score_variant(m6, m7, inp, frames, fp["clouds"][k], R_neu)

    neu = slice(m6.NEUTRAL_START, m6.NEUTRAL_START + m6.N_NEUTRAL)
    att = attribute_failures(d, res["A"], neu) if "A" in res else None

    sam_path = rec_dir / "sam3d" / "sam3d_vs_mocap.npz"
    sam = np.load(sam_path) if sam_path.exists() else None

    # ---- summary tables ----
    first = next(iter(res.values()))
    win = first["in_win"]
    base_ok = res["A"]["valid_magnitude"] if "A" in res else np.zeros(n, bool)
    print(f"\n|error| median/p95 deg, r, n   (in-window {int(win.sum())} frames)")
    for ax in AX:
        print(f"\n  {ax}")
        print(f"  {'method':<12}{'all solved':>26}{'A-solved frames':>26}"
              f"{'A-failed frames':>26}{'jitter':>8}")
        for k, r in res.items():
            ok = r[f"valid_{ax}"] & np.isfinite(r[f"moc_{ax}"])
            c_all = cell(r[f"cam_{ax}"], r[f"moc_{ax}"], ok)
            c_bs = cell(r[f"cam_{ax}"], r[f"moc_{ax}"], ok & base_ok)
            c_bf = cell(r[f"cam_{ax}"], r[f"moc_{ax}"], ok & ~base_ok)
            print(f"  {k} {VARIANTS[k]:<10}{fmt(c_all):>26}{fmt(c_bs):>26}"
                  f"{fmt(c_bf):>26}{r[f'jit_{ax}']:>8.2f}")
        if sam is not None and len(sam[f"sam_{ax}"]) == n:
            sv = sam[f"valid_{ax}"] & np.isfinite(sam[f"moc_{ax}"])
            print(f"  {'SAM3D':<12}{fmt(cell(sam[f'sam_{ax}'], sam[f'moc_{ax}'], sv)):>26}"
                  f"{fmt(cell(sam[f'sam_{ax}'], sam[f'moc_{ax}'], sv & base_ok)):>26}"
                  f"{fmt(cell(sam[f'sam_{ax}'], sam[f'moc_{ax}'], sv & ~base_ok)):>26}")

    print("\ncoverage (magnitude solved / in-window):")
    for k, r in res.items():
        nv = int(r["valid_magnitude"].sum())
        print(f"  {k} {VARIANTS[k]:<10}{nv:>6}/{int(win.sum())} ({nv / win.sum():.0%})"
              f"  gated {int((r['gated'] & win).sum())}")
        if att is not None and k != "A":
            new = r["valid_magnitude"] & ~base_ok
            print(f"      newly solved {int(new.sum())}: of A's occluded failures "
                  f"{int((new & att['occl']).sum())}/{int(att['occl'].sum())}, "
                  f"stereo-hole failures {int((new & att['holes']).sum())}/"
                  f"{int(att['holes'].sum())}")

    # ---- save + plot ----
    dst = rec_dir / "depth_fusion" / "fusion_scores.npz"
    flat = {f"{k}_{f}": v for k, r in res.items() for f, v in r.items()}
    np.savez_compressed(dst, **flat, **{f"diag_{k}": v for k, v in d.items()},
                        **{f"npts_{k}": v for k, v in fp["npts"].items()})
    print(f"\nscores -> {dst}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts0 = inp["ts0"]
    i0 = int(np.argmax(win))
    t = ((ts0 - ts0[i0]) / np.timedelta64(1, "s")).astype(float)
    fig, axes = plt.subplots(len(res) + 1, 1, figsize=(14, 2.2 * (len(res) + 1)),
                             sharex=True)
    for ax_, (k, r) in zip(axes, res.items()):
        mo = np.where(win, r["moc_magnitude"], np.nan)
        cm = np.where(r["valid_magnitude"], r["cam_magnitude"], np.nan)
        ax_.plot(t, mo, color="orange", lw=1, label="mocap")
        ax_.plot(t, cm, color="tab:green", lw=1, label=f"ICP {k} {VARIANTS[k]}")
        nv = int(r["valid_magnitude"].sum())
        ax_.set_title(f"{k} {VARIANTS[k]}: {nv}/{int(win.sum())} frames", fontsize=9,
                      loc="left")
        ax_.set_ylabel("rot. mag (deg)")
        ax_.grid(alpha=0.3)
    ax_ = axes[-1]
    for k in res:
        ax_.plot(t, np.where(win, fp["npts"][k], np.nan), lw=0.8, label=f"{k} shell pts")
    ax_.set_ylabel("shell points")
    ax_.set_xlabel("time (s)")
    ax_.legend(fontsize=8, ncol=4)
    ax_.grid(alpha=0.3)
    fig.suptitle(f"{a.subject}{' game' if a.game else ''}: torso ICP with mono-depth fusion")
    fig.tight_layout()
    png = out_dir / f"{a.subject}{'' if a.game else '_woc'}_depth_fusion.png"
    fig.savefig(png, dpi=110)
    print(f"plot -> {png}")


if __name__ == "__main__":
    main()
