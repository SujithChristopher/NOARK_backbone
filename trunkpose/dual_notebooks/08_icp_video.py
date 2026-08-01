# %% 08_icp_video.py
# 2x2 video of the ICP rigid-registration trunk tracker (the method validated in
# 07_icp_trunk.py). Same data path as 06_trunk_axis.py -- it reuses 06's geometry
# cache and 07's ICP + mocap-comparison passes, so what you see is exactly the
# series that produced the validation numbers.
#
# 2x2 layout:
#   TL cam0 + ICP trunk axes         | TR ICP alignment check (neutral vs registered)
#   BL 3D board view (cloud + axes)  | BR realtime 3-angle plots (ICP/mocap/plane)
#
# The TR panel is the point of this video: grey = neutral cloud, cyan = the current
# frame's cloud pushed through the ICP transform. They overlap when registration is
# healthy and visibly slide apart when ICP falls into a wrong minimum.
#
#   $env:TRUNK_GEOM_CACHE=".../geom_cache.pkl"
#   uv run python trunkpose/dual_notebooks/08_icp_video.py
#
# Output: icp_trunk_video.mp4

import importlib
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import numpy as np
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
m6 = importlib.import_module("06_trunk_axis")
m7 = importlib.import_module("07_icp_trunk")

ANGLE_PLOT_MODE = os.environ.get("TRUNK_ANGLE_PLOT_MODE", "all").lower()
if ANGLE_PLOT_MODE not in ("all", "icp_only", "mocap_only"):
    raise ValueError("TRUNK_ANGLE_PLOT_MODE must be all, icp_only, or mocap_only")
SHOW_ICP_ANGLES = ANGLE_PLOT_MODE in ("all", "icp_only")
SHOW_PLANE_ANGLES = ANGLE_PLOT_MODE == "all"
_video_suffix = "" if ANGLE_PLOT_MODE == "all" else f"_{ANGLE_PLOT_MODE}"
OUT_VIDEO = m6.RECORDING_DIR / f"icp_trunk_video{_video_suffix}.mp4"
QUAD_W, QUAD_H = m6.QUAD_W, m6.QUAD_H
N_WORKERS = m6.N_WORKERS
ALIGN_MATCH_M = 0.03         # nn distance counted as "matched" in the live rms readout
CLOUD_PT_SIZE = 1.5
AXIS_LEN_M = m6.AXIS_LEN_M

_V = None  # per-worker render bundle


# %% Panels
def _init_render(bundle):
    global _V
    _V = dict(bundle)
    _V["tree"] = cKDTree(_V["neutral"])


def _panel_cam(g0, vf, i):
    """cam0 grayscale + MediaPipe shoulders + the ICP trunk frame reprojected."""
    v = _V
    img = cv2.cvtColor(g0, cv2.COLOR_GRAY2BGR)
    for px in vf["sh_px"].values():
        if px is not None:
            cv2.circle(img, (int(px[0]), int(px[1])), 6, (0, 255, 0), -1)
    R_t, ls, rs = vf["Rt"], vf["ls"], vf["rs"]
    if R_t is not None and ls is not None and rs is not None:
        origin_b = m6.trunk_origin(ls, rs, R_t)
        ends = [origin_b] + [origin_b + R_t[:, c] * AXIS_LEN_M for c in (0, 1, 2)]
        pc = np.array([m6.board_to_cam0(p, v["R0_c"], v["t0_c"]) for p in ends])
        ip, _ = cv2.fisheye.projectPoints(
            pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), v["K0"], v["D0"])
        ip = ip.reshape(-1, 2).astype(int)
        for c, name in zip((1, 2, 3), ("lateral", "up", "anterior")):
            cv2.line(img, tuple(ip[0]), tuple(ip[c]), m6.AXIS_COLORS_BGR[name], 3)
    img = cv2.resize(img, (QUAD_W, QUAD_H))
    cv2.putText(img, "cam0 + ICP trunk frame", (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"frame {i}   t {v['t'][i]:+.2f}s", (12, QUAD_H - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
    return img


def _panel_align(fig, ax, vf):
    """Neutral cloud vs the live cloud mapped through the ICP transform.

    Overlap = healthy registration. The printed rms/match% are recomputed here from
    the stored transform, so a frozen (wrong-minimum) pose shows up as a bad rms even
    though the angle traces look smooth."""
    v = _V
    ax.cla()
    m6._axes3d_style(ax, v["lim_n"])
    ax.view_init(elev=14, azim=-72)
    neu = v["neutral"]
    # drawn fat + translucent so it stays visible as a halo under the cyan overlay
    ax.scatter(neu[:, 0], neu[:, 1], neu[:, 2], c="#cccccc", s=CLOUD_PT_SIZE * 4,
               alpha=0.35, depthshade=False)
    txt, col = "NO REGISTRATION", "red"
    if vf["A"] is not None and vf["cloud"] is not None:
        R, t = vf["A"]
        moved = vf["cloud"].astype(np.float64) @ R.T + t
        # orange = ICP converged but the pose was thrown out by the >ROT_MAX_DEG
        # plausibility gate, i.e. a wrong minimum. It still aligns two clouds nicely,
        # which is exactly why rms alone cannot detect this failure.
        ax.scatter(moved[:, 0], moved[:, 1], moved[:, 2],
                   c="orange" if vf["gated"] else "cyan", s=CLOUD_PT_SIZE,
                   depthshade=False)
        d, _ = v["tree"].query(moved, k=1)
        m = d < ALIGN_MATCH_M
        if m.any():
            txt = (f"ICP rms {np.sqrt((d[m] ** 2).mean()) * 1000:.1f} mm   "
                   f"matched {100 * m.mean():.0f}%")
            col = "white"
            if vf["gated"]:
                txt += "   REJECTED: >45 deg from neutral"
                col = "orange"
    ax.set_title(f"alignment: neutral (grey) vs registered (cyan)\n{txt}",
                 color=col, fontsize=8)
    return cv2.resize(m6._fig_to_bgr(fig), (QUAD_W, QUAD_H))


def _panel_board(fig, ax, vf):
    """Torso cloud in the board frame with the ICP trunk axes and mocap markers."""
    v = _V
    ax.cla()
    m6._axes3d_style(ax, v["lim_b"])
    ax.view_init(elev=18, azim=-70)
    if vf["cloud"] is not None:
        c = vf["cloud"]
        ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=c[:, 1], cmap="viridis",
                   s=CLOUD_PT_SIZE, depthshade=False)
    if vf["moc"] is not None:
        mk = np.array(vf["moc"])
        ax.scatter(mk[:, 0], mk[:, 1], mk[:, 2], c="orange", marker="x", s=30,
                   depthshade=False)
        for j in (1, 2):
            ax.plot(*np.stack([mk[0], mk[j]]).T, color="orange", lw=1.2, ls="--")
    R_t, ls, rs = vf["Rt"], vf["ls"], vf["rs"]
    if R_t is not None:
        origin = (m6.trunk_origin(ls, rs, R_t) if (ls is not None and rs is not None)
                  else (vf["cloud"].mean(axis=0) if vf["cloud"] is not None else None))
        if origin is not None:
            for col, name in zip((0, 1, 2), ("lateral", "up", "anterior")):
                ax.quiver(*origin, *(R_t[:, col] * AXIS_LEN_M),
                          color=m6.AXIS_COLORS_MPL[name], lw=2)
    ax.scatter(0, 0, 0, c="red", marker="^", s=35)
    ax.set_title("torso cloud + ICP axes + mocap markers (board frame)",
                 color="white", fontsize=8)
    return cv2.resize(m6._fig_to_bgr(fig), (QUAD_W, QUAD_H))


def _panel_angles(fig, axes, i):
    v = _V
    t = v["t"]
    fig.patch.set_facecolor("#111111")
    for k, (ax, name) in enumerate(zip(axes, ("flexion", "lateral", "axial"))):
        icp, moc, pla = v["icp"][k], v["mocap"][k], v["plane"][k]
        ax.cla()
        ax.plot(t, moc, color="white", lw=0.9, ls="--", alpha=0.8, label="mocap")
        if SHOW_PLANE_ANGLES:
            ax.plot(t, pla, color="tab:blue", lw=0.7, alpha=0.5,
                    label="plane+shoulder")
        if SHOW_ICP_ANGLES:
            ax.plot(t, icp, color=m6.ANGLE_COLORS[name], lw=1.4, label="ICP")
        ax.axvline(t[i], color="white", lw=1)
        ax.axhline(0, color="gray", lw=0.5)
        ms = f"{moc[i]:+.0f}" if np.isfinite(moc[i]) else "--"
        if SHOW_ICP_ANGLES:
            cs = f"{icp[i]:+.0f}" if np.isfinite(icp[i]) else "--"
            value_label = f"icp {cs}  moc {ms}"
        else:
            value_label = f"moc {ms}"
        ax.set_ylabel(f"{name}\n{value_label}",
                      color=m6.ANGLE_COLORS[name], fontsize=8)
        ax.set_facecolor("#111111")
        ax.tick_params(colors="gray", labelsize=6)
        ax.set_xlim(t[0], t[-1])
        if k == 0:
            ax.legend(loc="upper right", fontsize=6, facecolor="#222222",
                      labelcolor="white", framealpha=0.6)
    axes[-1].set_xlabel("time (s)", color="gray", fontsize=8)
    return cv2.resize(m6._fig_to_bgr(fig), (QUAD_W, QUAD_H))


# %% Render pass (parallel, chunked like 06)
def _render_chunk(args):
    start, end, tmp_path, vfs = args
    fig3 = plt.figure(figsize=(QUAD_W / 100, QUAD_H / 100), dpi=100)
    ax_tr = fig3.add_subplot(111, projection="3d")
    fig4 = plt.figure(figsize=(QUAD_W / 100, QUAD_H / 100), dpi=100)
    ax_bl = fig4.add_subplot(111, projection="3d")
    for f in (fig3, fig4):
        f.patch.set_facecolor("#111111")
    fig_br, axes_br = plt.subplots(3, 1, figsize=(QUAD_W / 100, QUAD_H / 100), dpi=100)

    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             _V["fps"], (QUAD_W * 2, QUAD_H * 2))
    fi = 0
    for i, g0 in enumerate(m6.iter_frames(m6.CAM0_FRAMES)):
        if i < start:
            continue
        if i >= end:
            break
        g0 = g0 if g0.ndim == 2 else cv2.cvtColor(g0, cv2.COLOR_RGB2GRAY)
        vf = vfs[fi]; fi += 1
        top = cv2.hconcat([_panel_cam(g0, vf, i), _panel_align(fig3, ax_tr, vf)])
        bot = cv2.hconcat([_panel_board(fig4, ax_bl, vf),
                           _panel_angles(fig_br, axes_br, i)])
        writer.write(cv2.vconcat([top, bot]))
    writer.release()
    for f in (fig3, fig4, fig_br):
        plt.close(f)
    return start, tmp_path


def run_render(vfs, bundle, n):
    ranges = [(int(r[0]), int(r[-1]) + 1) for r in np.array_split(np.arange(n), N_WORKERS)]
    ranges = [r for r in ranges if r[1] > r[0]]
    tmpdir = tempfile.mkdtemp(prefix="icp_render_")
    tasks = [(s, e, str(Path(tmpdir) / f"seg_{k:03d}.mp4"), vfs[s:e])
             for k, (s, e) in enumerate(ranges)]

    if N_WORKERS <= 1:
        _init_render(bundle)
        segs = [_render_chunk(t_) for t_ in tqdm(tasks, desc="render")]
    else:
        with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_render,
                                 initargs=(bundle,)) as ex:
            segs = list(tqdm(ex.map(_render_chunk, tasks), total=len(tasks),
                             desc="render(chunks)"))
    segs.sort(key=lambda s: s[0])

    writer = cv2.VideoWriter(str(OUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"),
                             bundle["fps"], (QUAD_W * 2, QUAD_H * 2))
    for _, path in segs:
        cap = cv2.VideoCapture(path)
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            writer.write(fr)
        cap.release()
        os.remove(path)
    writer.release()
    os.rmdir(tmpdir)


# %% Main
def smooth_masked(a):
    """Savitzky-Golay smoothing that keeps NaN as NaN.

    06's smoother interpolates across gaps, which is right for short dropouts but at
    the end of this recording turns a total tracking loss into a flat line that reads
    as a held pose. In a video that is an outright lie, so gaps stay blank here."""
    s = m6.smooth_series(a)
    s[~np.isfinite(a)] = np.nan
    return s


def cloud_limits(clouds, pad=0.1):
    pts = np.concatenate([c for c in clouds if c is not None])
    lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
    return {a: (lo[k] - pad, hi[k] + pad) for k, a in enumerate("xyz")}


def main():
    print(f"Workers: {N_WORKERS}")
    inp = m7.load_inputs()
    frames, n, ts0 = inp["frames"], inp["n"], inp["ts0"]
    K0, D0 = m6.load_stereo(m6.STEREO_TOML)[:2]

    _R_t_list, R_neu, flex_p, lat_p, axi_p, _ = m6.assemble(frames)
    icp = m7.run_icp_pass(frames, R_neu, n)
    cmp_ = m7.compare_mocap(inp, icp["Rb_icp"], R_neu)
    Rz_i, zwin, midx = cmp_["Rz_i"], cmp_["zwin"], cmp_["midx"]
    gated = icp["gated"]

    # ICP angles for EVERY frame (07 blanks them outside the mocap segment for
    # scoring; the video wants the whole recording), same zeroing as the validation.
    Rt_list = [None] * n
    flex_v = np.full(n, np.nan); lat_v = np.full(n, np.nan); axi_v = np.full(n, np.nan)
    for i, Rb in enumerate(icp["Rb_icp"]):
        if Rb is None:
            continue
        Rt_list[i] = (Rb @ Rz_i.T) @ R_neu
        flex_v[i], lat_v[i], axi_v[i] = m6.trunk_angles(Rt_list[i], R_neu)

    plane = [smooth_masked(a) for a in (flex_p, lat_p, axi_p)]
    for arr in plane:
        off = np.nanmean(arr[zwin])
        if np.isfinite(off):
            arr -= off

    # per-frame render payload (drops depth maps / mocap dict -> smaller pickles)
    mob, mxb, mzb = cmp_["mob"], cmp_["mxb"], cmp_["mzb"]
    vfs = []
    for i, fr in enumerate(frames):
        j = midx[i]
        moc = (mob[j], mxb[j], mzb[j])
        vfs.append(dict(cloud=fr["cloud"], sh_px=fr["sh_px"],
                        ls=fr["cam_pts"]["L_shoulder"], rs=fr["cam_pts"]["R_shoulder"],
                        A=icp["A_list"][i] or icp["A_raw"][i], gated=bool(gated[i]),
                        Rt=Rt_list[i],
                        moc=moc if all(np.isfinite(p).all() for p in moc) else None))

    t = ((ts0 - ts0[zwin[0]]) / np.timedelta64(1, "s")).astype(np.float64)
    dt = np.diff(t)
    fps = float(1.0 / np.median(dt[dt > 0])) if (dt > 0).any() else 15.0
    bundle = dict(
        K0=K0, D0=D0, R0_c=inp["R0_c"], t0_c=inp["t0_c"], fps=fps, t=t,
        neutral=icp["neutral"],
        lim_n=cloud_limits([icp["neutral"]]),
        lim_b=cloud_limits([f["cloud"] for f in frames]),
        icp=[smooth_masked(a) for a in (flex_v, lat_v, axi_v)],
        mocap=[cmp_["mflex"], cmp_["mlat"], cmp_["maxi"]],
        plane=plane)

    # TRUNK_VIDEO_FRAMES caps the RENDER only; every series above is still computed
    # over the whole recording, so a short preview shows the same traces as a full run.
    n_render = int(os.environ.get("TRUNK_VIDEO_FRAMES", "0")) or n
    n_render = min(n_render, n)
    print(f"Rendering {n_render}/{n} frames at ~{fps:.2f} fps -> {OUT_VIDEO}")
    run_render(vfs, bundle, n_render)
    print(f"Done -> {OUT_VIDEO}")


if __name__ == "__main__":
    main()
