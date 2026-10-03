# %% 23_shell_icp.py
# Why torso ICP loses ~half the NOARK game frames, and two edge-cheap fixes.
#
# 22_depth_fusion.py showed the failures are not missing depth: failed game frames
# have as many SGBM-solved torso pixels as solved ones (median 2.7k vs 2.9k), but
# 06's front-shell filter keeps only 49% of them (97% on solved frames). The filter
# keeps points within SHELL_M behind the FRONT_PCTL-th depth percentile, so cube or
# hand pixels leaking into the torso mask become "the front" and the chest is cut.
#
# Shell filters (applied to the same cached torso points):
#   front    06's rule (sanity: must reproduce 07's numbers)
#   mode     dominant depth layer: histogram peak, tracked over time, +/- a window
#   cluster  largest 1-D depth cluster (gap-split), then 06's front cap inside it
# ICP:
#   plain    07's direct-to-neutral loop (seeds, plausibility gate)
#   vis      same, but the neutral cloud is first cut to the part that projects into
#            the current frame's visible torso mask at the warm-start pose, so a
#            partly hidden torso is matched against the same part of the neutral
#
# Both are a few ms of CPU per frame, i.e. they cost nothing on the Q6A.
#
#   uv run python trunkpose/dual_notebooks/23_shell_icp.py --subject gokul --game --pass-only
#   uv run python trunkpose/dual_notebooks/23_shell_icp.py --subject gokul --game --shell mode --icp vis
#   uv run python trunkpose/dual_notebooks/23_shell_icp.py --subject gokul --game --report
#
# Output: <rec>/shell_icp/torso_cache.pkl, <rec>/shell_icp/<shell>_<icp>.npz

import argparse
import importlib
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
m22 = importlib.import_module("22_depth_fusion")

TORSO_MAX_PTS = 6000      # cached torso points per frame (random subsample)
# mode shell
MODE_BIN_M = 0.01
MODE_FRONT_M = 0.10       # keep this far in front of the dominant layer ...
MODE_BACK_M = 0.10        # ... and this far behind it
MODE_PEAK_FRAC = 0.3      # secondary peaks this tall may win if nearer the last one
# cluster shell
CLUSTER_GAP_M = 0.03
# vis ICP
VIS_DILATE_PX = 7
VIS_MIN_PTS = 300         # fall back to the full neutral below this

REASONS = ("ok", "no_cloud", "min_match", "rms", "frac", "gated")


def parse():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default=str(m22.DEFAULT_SESSION))
    ap.add_argument("--subject", required=True)
    ap.add_argument("--game", action="store_true")
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--pass-only", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--shell", default="front", choices=("front", "mode", "mode0", "cluster"))
    ap.add_argument("--icp", default="plain", choices=("plain", "vis"))
    ap.add_argument("--report", action="store_true", help="tabulate saved variants")
    return ap.parse_args()


# %% Torso pass: pre-shell torso points + visible-torso mask, per frame
def torso_pass(m6, n):
    import cv2
    from tqdm.auto import tqdm

    K0, D0, K1, D1, R, T_mm = m6.load_stereo(m6.STEREO_TOML)
    R1, Q, maps0, maps1 = m6.build_rectification(K0, D0, K1, D1, R, T_mm, m6.CAM_SIZE)
    sgbm = m6.make_sgbm()
    seg = m6.load_seg_model()
    rng = np.random.default_rng(0)
    pts_l, mask_l = [None] * n, [None] * n
    for i, (f0, f1) in enumerate(tqdm(m6.iter_frame_pairs(m6.CAM0_FRAMES, m6.CAM1_FRAMES),
                                      total=n, desc="torso pass")):
        if i >= n:
            break
        g0 = f0 if f0.ndim == 2 else cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY)
        g1 = f1 if f1.ndim == 2 else cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        _, mask0 = m22.torso_and_arm(m6, seg, g0, m6.CAM_SIZE)
        rect0 = cv2.remap(g0, *maps0, cv2.INTER_LINEAR)
        rect1 = cv2.remap(g1, *maps1, cv2.INTER_LINEAR)
        disp_raw = sgbm.compute(rect0, rect1).astype(np.float32) / 16.0
        disp = cv2.bilateralFilter(disp_raw, m6.BILATERAL_D, m6.BILATERAL_SIGMA_COLOR,
                                   m6.BILATERAL_SIGMA_SPACE)
        pts = cv2.reprojectImageTo3D(disp, Q) / 1000.0
        z = pts[..., 2]
        valid = (disp_raw > m6.SGBM_MIN_DISP) & np.isfinite(pts).all(axis=2)
        mrect = cv2.remap(mask0, *maps0, cv2.INTER_NEAREST)
        mask_l[i] = np.packbits(mrect > 0)          # visible torso, before erosion
        area = int((mrect > 0).sum())
        if area > 0:
            ksz = max(3, int(round(np.sqrt(area) * m6.MASK_ERODE_FRAC))) | 1
            mrect = cv2.erode(mrect, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)))
        ok = valid & (mrect > 0) & (z > m6.DEPTH_MIN_M) & (z < m6.DEPTH_MAX_M)
        tp = pts[ok]
        if len(tp) > TORSO_MAX_PTS:
            tp = tp[rng.choice(len(tp), TORSO_MAX_PTS, replace=False)]
        pts_l[i] = tp.astype(np.float32)            # rectified-cam0 frame, metres
    return dict(pts=pts_l, mask=mask_l, R1=R1, Q=Q, shape=tuple(m6.CAM_SIZE[::-1]))


def load_or_run_pass(m6, n, rec_dir, recompute):
    path = rec_dir / "shell_icp" / "torso_cache.pkl"
    key = dict(n=n, stride=m6.FRAME_STRIDE, geom=m6.geom_cache_key(), max=TORSO_MAX_PTS)
    if path.exists() and not recompute:
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        if d.get("key") == key:
            return d
        print("torso pass: cache key changed, recomputing")
    d = torso_pass(m6, n)
    d["key"] = key
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(d, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"torso pass -> {path}")
    return d


# %% Shell filters (on rectified-frame points; return the kept subset)
def shell_front(m6, tp, state):
    zc = np.percentile(tp[:, 2], m6.FRONT_PCTL)
    return tp[tp[:, 2] <= zc + m6.SHELL_M]


def shell_mode(m6, tp, state):
    z = tp[:, 2]
    # >= 3 bins: np.convolve(mode="same") returns max(len(h), 3) values, so a frame
    # whose torso spans under 3 bins would index past `centres`
    edges = np.arange(z.min(), max(z.max() + 2 * MODE_BIN_M, z.min() + 3.5 * MODE_BIN_M),
                      MODE_BIN_M)
    h, _ = np.histogram(z, edges)
    h = np.convolve(h, np.ones(3) / 3, mode="same")
    centres = (edges[:-1] + edges[1:]) / 2
    peaks = [k for k in range(len(h)) if h[k] >= MODE_PEAK_FRAC * h.max()
             and (k == 0 or h[k] >= h[k - 1]) and (k == len(h) - 1 or h[k] >= h[k + 1])]
    prev = state.get("z_prev")
    if prev is not None and peaks:
        k = min(peaks, key=lambda k: abs(centres[k] - prev))
    else:
        k = int(np.argmax(h))
    zp = centres[k]
    state["z_cand"] = zp
    return tp[(z >= zp - MODE_FRONT_M) & (z <= zp + MODE_BACK_M)]


def shell_cluster(m6, tp, state):
    z = tp[:, 2]
    order = np.argsort(z)
    zs = z[order]
    cuts = np.where(np.diff(zs) > CLUSTER_GAP_M)[0]
    bounds = np.concatenate(([0], cuts + 1, [len(zs)]))
    segs = [(a, b) for a, b in zip(bounds[:-1], bounds[1:])]
    prev = state.get("z_prev")
    big = max(b - a for a, b in segs)
    cand = [s for s in segs if s[1] - s[0] >= MODE_PEAK_FRAC * big]
    if prev is not None:
        a, b = min(cand, key=lambda s: abs(np.median(zs[s[0]:s[1]]) - prev))
    else:
        a, b = max(segs, key=lambda s: s[1] - s[0])
    sel = tp[order[a:b]]
    state["z_cand"] = float(np.median(sel[:, 2]))
    return shell_front(m6, sel, state)


def shell_mode0(m6, tp, state):
    """mode without tracking (the per-frame, stateless form 06 can run in parallel)."""
    return shell_mode(m6, tp, {})


SHELLS = dict(front=shell_front, mode=shell_mode, mode0=shell_mode0, cluster=shell_cluster)


def to_cloud(m6, tp, R1, rng):
    tp = tp @ R1
    if len(tp) < 50:
        return None
    return m6.voxel_downsample(tp, rng=rng).astype(np.float32)   # board == cam0 here


# %% ICP (07's direct loop, with per-frame failure reasons and optional visibility crop)
def run_icp(m6, m7, tc, shell, use_vis, n):
    import cv2
    from scipy.spatial import cKDTree

    R0_c, t0_c, _, _ = m6.load_charuco_basis(m6.CHARUCO_TOML)
    R1, Q = tc["R1"], tc["Q"]
    f, cx, cy = float(Q[2, 3]), float(-Q[0, 3]), float(-Q[1, 3])
    H, W = tc["shape"]
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (VIS_DILATE_PX,) * 2)
    rng = np.random.default_rng(0)
    shell_fn = SHELLS[shell]

    def board(pts_cam0):
        return m6.to_board_bulk(pts_cam0, R0_c, t0_c)

    # clouds; the tracked shells carry state from the last ACCEPTED frame, so the
    # first pass (to build the neutral) runs them untracked over the neutral window
    nz = slice(m6.NEUTRAL_START, m6.NEUTRAL_START + m6.N_NEUTRAL)
    neu = []
    for i in range(nz.start, min(nz.stop, n)):
        tp = tc["pts"][i]
        if tp is not None and len(tp) >= 50:
            c = to_cloud(m6, shell_fn(m6, tp, {}), R1, rng)
            if c is not None:
                neu.append(board(c))
    neu = m6.voxel_downsample(np.concatenate(neu).astype(np.float64),
                              max_pts=m7.NEUTRAL_MAX_PTS)
    neu_n = m7.estimate_normals(neu)
    tree_full = cKDTree(neu)
    print(f"[{shell}/{'vis' if use_vis else 'plain'}] neutral {len(neu)} pts")

    def rot_deg(Rm):
        return np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1)))

    Rb = [None] * n
    reason = np.full(n, "no_cloud", dtype=object)
    npts = np.zeros(n, int)
    keep = np.full(n, np.nan)
    vis_frac = np.full(n, np.nan)
    A = (np.eye(3), np.zeros(3))
    last_good = None
    state = {}
    for i in range(n):
        tp = tc["pts"][i]
        if tp is None or len(tp) < 50:
            continue
        sh = shell_fn(m6, tp, state)
        keep[i] = len(sh) / len(tp)
        src = to_cloud(m6, sh, R1, rng)
        if src is None:
            continue
        src = board(src).astype(np.float64)
        npts[i] = len(src)

        dst, tree, dn = neu, tree_full, neu_n
        if use_vis:
            # neutral points at the warm-start pose, projected into this frame
            Rw, tw = A                                  # cur -> neutral
            cur = (neu - tw) @ Rw                       # neutral -> cur (board)
            pr = (cur @ R0_c.T + t0_c) @ R1.T           # -> rectified cam0
            u = np.round(f * pr[:, 0] / pr[:, 2] + cx).astype(int)
            v = np.round(f * pr[:, 1] / pr[:, 2] + cy).astype(int)
            inb = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (pr[:, 2] > 0)
            m = np.unpackbits(tc["mask"][i])[: H * W].reshape(H, W)
            m = cv2.dilate(m, kern) > 0
            sel = np.zeros(len(neu), bool)
            sel[inb] = m[v[inb], u[inb]]
            vis_frac[i] = sel.mean()
            if sel.sum() >= VIS_MIN_PTS:
                dst, dn = neu[sel], neu_n[sel]
                tree = cKDTree(dst)

        seeds = [A]
        for cand in ((last_good if last_good else None), (np.eye(3), np.zeros(3))):
            if cand is not None and not any(np.allclose(cand[0], s0) and np.allclose(cand[1], s1)
                                            for s0, s1 in seeds):
                seeds.append(cand)
        best, why = None, "min_match"
        rank = {"min_match": 0, "rms": 1, "frac": 2, "gated": 3}
        for R_s, t_s in seeds:
            res = m7.icp(src, dst, tree, R_s, t_s, dst_normals=dn)
            if not res:
                continue
            w = None
            if res[2] > m7.DIRECT_RMS_M:
                w = "rms"
            elif res[3] < m7.DIRECT_FRAC * len(src):
                w = "frac"
            elif rot_deg(res[0]) > m7.ROT_MAX_DEG:
                w = "gated"
            if w:
                why = w if rank[w] > rank[why] else why
                continue
            if best is None or res[2] < best[2]:
                best = res
        if best is None:
            reason[i] = why
            continue
        reason[i] = "ok"
        A = (best[0], best[1])
        last_good = A
        if "z_cand" in state:
            state["z_prev"] = state["z_cand"]
        Rb[i] = A[0].T
    return dict(Rb=Rb, reason=reason, npts=npts, keep=keep, vis_frac=vis_frac)


# %% Main
def main():
    a = parse()
    m6, m7, rec_dir, _ = m22.setup(a)
    out = rec_dir / "shell_icp"
    if a.report:
        return report(a, m6, rec_dir, out)
    inp = m7.load_inputs()
    frames, n = inp["frames"], inp["n"]
    tc = load_or_run_pass(m6, n, rec_dir, a.recompute)
    if a.pass_only:
        return
    R_neu = m6.assemble(frames)[1]
    r = run_icp(m6, m7, tc, a.shell, a.icp == "vis", n)
    cmp_ = m7.compare_mocap(inp, r["Rb"], R_neu)
    win = cmp_["in_win"]
    res = dict(in_win=win, reason=r["reason"].astype(str), npts=r["npts"],
               keep=r["keep"], vis_frac=r["vis_frac"])
    for ax, k, mk in zip(m22.AX, m22.KEYS, m22.MKEYS):
        raw = cmp_[k]
        res[f"cam_{ax}"] = m6.smooth_series(raw)
        res[f"valid_{ax}"] = np.isfinite(raw) & win
        res[f"moc_{ax}"] = cmp_[mk]
        res[f"jit_{ax}"] = float(np.nanstd((raw - res[f"cam_{ax}"])[res[f"valid_{ax}"]]))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{a.shell}_{a.icp}.npz", **res)
    rs = res["reason"][win]
    print("in-window reasons: " + ", ".join(f"{k} {int((rs == k).sum())}" for k in REASONS))
    print(f"-> {out / f'{a.shell}_{a.icp}.npz'}")


def report(a, m6, rec_dir, out):
    files = sorted(out.glob("*_*.npz"))
    if not files:
        raise SystemExit(f"no variants in {out}")
    res = {p.stem: dict(np.load(p, allow_pickle=True)) for p in files}
    base = res.get("front_plain")
    base_ok = base["valid_magnitude"] if base is not None else None
    sam_p = rec_dir / "sam3d" / "sam3d_vs_mocap.npz"
    sam = np.load(sam_p) if sam_p.exists() else None
    win = next(iter(res.values()))["in_win"]
    print(f"{a.subject}{' game' if a.game else ''}: in-window {int(win.sum())} frames")
    print(f"\n{'variant':<16}{'solved':>8}" + "".join(f"{k:>10}" for k in REASONS[1:])
          + f"{'shell keep':>12}")
    for nm, r in res.items():
        rs = r["reason"][win]
        print(f"{nm:<16}{int(r['valid_magnitude'].sum()) / win.sum():>8.0%}"
              + "".join(f"{int((rs == k).sum()):>10}" for k in REASONS[1:])
              + f"{np.nanmedian(r['keep'][win]):>12.2f}")
    for ax in m22.AX:
        print(f"\n  {ax}: |error| median/p95, r, n")
        print(f"  {'variant':<16}{'all solved':>26}{'front_plain solved':>26}"
              f"{'front_plain failed':>26}{'jitter':>8}")
        for nm, r in res.items():
            ok = r[f"valid_{ax}"] & np.isfinite(r[f"moc_{ax}"])
            bo = base_ok if base_ok is not None else np.zeros_like(ok)
            print(f"  {nm:<16}"
                  f"{m22.fmt(m22.cell(r[f'cam_{ax}'], r[f'moc_{ax}'], ok)):>26}"
                  f"{m22.fmt(m22.cell(r[f'cam_{ax}'], r[f'moc_{ax}'], ok & bo)):>26}"
                  f"{m22.fmt(m22.cell(r[f'cam_{ax}'], r[f'moc_{ax}'], ok & ~bo)):>26}"
                  f"{float(r[f'jit_{ax}']):>8.2f}")
        if sam is not None and len(sam[f"sam_{ax}"]) == len(win):
            sv = sam[f"valid_{ax}"] & np.isfinite(sam[f"moc_{ax}"])
            bo = base_ok if base_ok is not None else np.zeros_like(sv)
            print(f"  {'SAM3D':<16}"
                  f"{m22.fmt(m22.cell(sam[f'sam_{ax}'], sam[f'moc_{ax}'], sv)):>26}"
                  f"{m22.fmt(m22.cell(sam[f'sam_{ax}'], sam[f'moc_{ax}'], sv & bo)):>26}"
                  f"{m22.fmt(m22.cell(sam[f'sam_{ax}'], sam[f'moc_{ax}'], sv & ~bo)):>26}")


if __name__ == "__main__":
    main()
