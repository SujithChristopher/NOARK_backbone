# %% 13_hull_displacement_designs.py
# Four candidate poster panels for ONE frame -- the peak axial-rotation pose -- each
# answering "how far is it displaced?" a different way. Pick one, throw the rest.
#
#   D1 residual heat map    registered points colored by distance to the neutral
#                           surface + mm colorbar + residual histogram inset
#   D2 ghost hulls          raw pre-ICP hull (red) -> registered hull (green) over
#                           the neutral hull (grey), labelled with the rigid motion
#                           ICP measured (deg + centroid travel)
#   D3 residual quiver      arrows from each registered point to its nearest neutral
#                           surface point, exaggerated, with a true-scale bar
#   D4 heat map + section   D1 plus a horizontal cut through both hulls in the
#                           neutral trunk basis, with the outline gap annotated
#
# D1/D3/D4 show the misfit LEFT AFTER registration; D2 shows the motion ICP REMOVED.
#
#   uv run python trunkpose/dual_notebooks/13_hull_displacement_designs.py
# (reuses 12_poster_figures' loader, stubs and poster style)

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
m12 = importlib.import_module("12_poster_figures")
m6, m7 = m12.m6, m12.m7

OUT_DIR = m6.RECORDING_DIR
MATCH_M = m12.MATCH_M
PAD_M = 0.02
C_REG, C_NEU = m12.C_REG, m12.C_NEU
C_RAW = "#D55E00"        # raw, unregistered pose
CMAP = "magma_r"
QUIVER_N = 200
QUIVER_EX = 4.0          # arrow exaggeration; stated on the panel
QUIVER_PCTL = 95         # skip the occluded-flank outliers so arrows stay in frame
SECTION_HALF_M = 0.015   # slab half-thickness for the D4 cut


# %% shared geometry for the chosen frame
def frame_bundle():
    inp = m7.load_inputs()
    frames, n, ts0 = inp["frames"], inp["n"], inp["ts0"]
    _Rp, R_neu, _fp, _lp, _ap, _ = m6.assemble(frames)
    icp = m7.run_icp_pass(frames, R_neu, n)
    cmp_ = m7.compare_mocap(inp, icp["Rb_icp"], R_neu)

    neutral = icp["neutral"]
    tree = cKDTree(neutral)
    in_win, zwin = cmp_["in_win"], cmp_["zwin"]
    axi = m6.smooth_series(cmp_["axi_iz"])

    ok = np.array([a is not None for a in icp["A_list"]]) & in_win
    i = int(np.nanargmax(np.where(ok, np.abs(axi), np.nan)))
    R, t = icp["A_list"][i]
    raw = frames[i]["cloud"].astype(np.float64)
    moved = raw @ R.T + t
    d, idx = tree.query(moved, k=1)
    t_s = ((ts0 - ts0[zwin[0]]) / np.timedelta64(1, "s")).astype(np.float64)

    rot = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    print(f"frame {i}  t={t_s[i]:+.1f}s  axial {axi[i]:+.1f} deg  "
          f"ICP rot {rot:.1f} deg  rms {np.sqrt((d[d < MATCH_M] ** 2).mean()) * 1000:.1f} mm")
    return dict(neutral=neutral, raw=raw, moved=moved, nn=neutral[idx], d=d,
                R_neu=R_neu, rot=rot, axi=float(axi[i]), t=float(t_s[i]), i=i)


def limits(clouds, pad=PAD_M):
    pts = np.concatenate(clouds)
    lo, hi = np.percentile(pts, 0.5, axis=0), np.percentile(pts, 99.5, axis=0)
    return [(lo[k] - pad, hi[k] + pad) for k in range(3)]


def hull_tris(pts):
    h = ConvexHull(pts)
    return pts[h.simplices]


def style3d(ax, lim, zoom=1.35):
    ax.set_xlim(lim[0])
    ax.set_ylim(lim[1])
    ax.set_zlim(lim[2])
    ax.set_box_aspect([lim[k][1] - lim[k][0] for k in range(3)], zoom=zoom)
    ax.view_init(elev=14, azim=-72)
    ax.set_xlabel("x (m)", labelpad=-4, fontsize=10)
    ax.set_ylabel("y (m)", labelpad=-4, fontsize=10)
    ax.set_zlabel("z (m)", labelpad=-6, fontsize=10)
    ax.tick_params(labelsize=8, pad=-2)
    ax.grid(alpha=0.25)


def neutral_hull(ax, neutral, alpha=0.20):
    ax.add_collection3d(Poly3DCollection(hull_tris(neutral), facecolor=C_NEU,
                                         alpha=alpha, edgecolor=C_NEU,
                                         linewidths=0.25))


def resid_stats(d):
    mm = d * 1000
    return dict(p50=float(np.median(mm)), p95=float(np.percentile(mm, 95)),
                rms=float(np.sqrt((mm[d < MATCH_M] ** 2).mean())), mm=mm)


def caption(b, extra=""):
    return (f"peak axial rotation {b['axi']:+.0f}$\\degree$   t = {b['t']:+.1f} s"
            + (f"   {extra}" if extra else ""))


# %% D1 -- residual heat map + histogram inset
def design1(b, path=None, fig=None, ax=None, hist=True, cb_rect=(0.86, 0.30, 0.020, 0.40)):
    st = resid_stats(b["d"])
    vmax = float(np.percentile(st["mm"], 97))
    own = fig is None
    if own:
        fig = plt.figure(figsize=(9.0, 6.6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_position([0.02, 0.04, 0.80, 0.84])
    neutral_hull(ax, b["neutral"])
    sc = ax.scatter(b["moved"][:, 0], b["moved"][:, 1], b["moved"][:, 2],
                    c=st["mm"], cmap=CMAP, vmin=0, vmax=vmax, s=6,
                    depthshade=False, edgecolors="none")
    style3d(ax, limits([b["neutral"], b["moved"]]))
    # colorbar on its own axes -- an ax-attached one eats into the 3D box and
    # lands on top of the z tick labels
    cb = fig.colorbar(sc, cax=fig.add_axes(cb_rect))
    cb.set_label("distance to neutral surface (mm)", fontsize=12)
    cb.ax.tick_params(labelsize=10)
    ax.set_title("Registered torso surface vs neutral hull", pad=0)
    ax.text2D(0.5, 0.99, caption(b, f"median {st['p50']:.1f} mm   "
                                    f"p95 {st['p95']:.1f} mm   rms {st['rms']:.1f} mm"),
              transform=ax.transAxes, ha="center", va="top", fontsize=12,
              color="#333333")
    if hist:
        hx = fig.add_axes([0.60, 0.60, 0.23, 0.20])
        hx.hist(st["mm"], bins=40, range=(0, vmax * 1.2), color=C_REG, alpha=0.85)
        for v, c, lab in ((st["p50"], "#333333", "p50"), (st["p95"], C_RAW, "p95")):
            hx.axvline(v, color=c, lw=1.4, ls="--")
            hx.text(v, hx.get_ylim()[1] * 0.92, f" {lab} {v:.1f}", fontsize=9,
                    color=c, ha="left", va="top")
        hx.set_xlabel("residual (mm)", fontsize=10)
        hx.set_ylabel("points", fontsize=10)
        hx.tick_params(labelsize=9)
        hx.patch.set_alpha(0.85)
        hx.spines[["top", "right"]].set_visible(False)
    if own:
        fig.savefig(path)
        plt.close(fig)
        print(f"D1 -> {path}")


# %% D2 -- ghost hulls: what ICP removed
def design2(b, path):
    # Short and wide: the torso hulls are much wider than they are tall in this
    # view, so a square figure is mostly empty margin.
    fig = plt.figure(figsize=(9.0, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0.02, 0.14, 0.94, 0.78])
    neutral_hull(ax, b["neutral"], alpha=0.18)
    for pts, col, lw in ((b["raw"], C_RAW, 0.35), (b["moved"], C_REG, 0.35)):
        ax.add_collection3d(Poly3DCollection(hull_tris(pts), facecolor=col,
                                             alpha=0.16, edgecolor=col,
                                             linewidths=lw))
    c_raw, c_mov = b["raw"].mean(axis=0), b["moved"].mean(axis=0)
    travel = float(np.linalg.norm(c_mov - c_raw)) * 1000
    ax.quiver(*c_raw, *(c_mov - c_raw), color="#111111", lw=2.2,
              arrow_length_ratio=0.18)
    mid = (c_raw + c_mov) / 2
    ax.text(*mid, f"  {b['rot']:.0f}$\\degree$ / {travel:.0f} mm", fontsize=13,
            color="#111111")
    for pts, col in ((b["raw"], C_RAW), (b["moved"], C_REG)):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=col, s=1.6, alpha=0.5,
                   depthshade=False, edgecolors="none")
    style3d(ax, limits([b["neutral"], b["raw"], b["moved"]]), zoom=1.45)
    ax.set_title("Rigid motion recovered by ICP", pad=0)
    ax.text2D(0.5, 0.99, caption(b), transform=ax.transAxes, ha="center",
              va="top", fontsize=12, color="#333333")
    # legend gets its own strip at the bottom; the axes box is lifted to 0.14 so
    # the x-axis label, which hangs below the 3D box, does not land on it
    fig.legend(handles=[
        Line2D([], [], color=C_NEU, lw=3, label="Neutral hull"),
        Line2D([], [], color=C_RAW, lw=3, label="Live cloud, raw pose"),
        Line2D([], [], color=C_REG, lw=3, label="Live cloud, registered"),
    ], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005),
        fontsize=12)
    fig.savefig(path)
    plt.close(fig)
    print(f"D2 -> {path}")


# %% D3 -- residual quiver
def design3(b, path):
    st = resid_stats(b["d"])
    fig = plt.figure(figsize=(9.0, 6.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0.02, 0.04, 0.80, 0.84])
    neutral_hull(ax, b["neutral"])
    lim = limits([b["neutral"], b["moved"]])
    # Arrows are drawn only where the residual is inside p95: the occluded flank
    # runs to ~40 mm, which at any useful exaggeration shoots clean off the axes.
    cap = float(np.percentile(b["d"], QUIVER_PCTL))
    pool = np.where(b["d"] <= cap)[0]
    rng = np.random.default_rng(0)
    sel = rng.choice(pool, size=min(QUIVER_N, len(pool)), replace=False)
    p, v = b["moved"][sel], (b["nn"] - b["moved"])[sel]
    sc = ax.scatter(b["moved"][:, 0], b["moved"][:, 1], b["moved"][:, 2],
                    c=st["mm"], cmap=CMAP, vmin=0, vmax=float(cap * 1000),
                    s=4, alpha=0.7, depthshade=False, edgecolors="none")
    ax.quiver(p[:, 0], p[:, 1], p[:, 2],
              v[:, 0] * QUIVER_EX, v[:, 1] * QUIVER_EX, v[:, 2] * QUIVER_EX,
              color="#111111", lw=0.8, arrow_length_ratio=0.35, alpha=0.85)
    cb = fig.colorbar(sc, cax=fig.add_axes((0.86, 0.30, 0.020, 0.40)))
    cb.set_label("distance to neutral surface (mm)", fontsize=12)
    cb.ax.tick_params(labelsize=10)
    # true-scale reference drawn at the same exaggeration as the arrows
    x0 = lim[0][0] + 0.03
    y0, z0 = lim[1][1] - 0.03, lim[2][0] + 0.015
    ax.plot([x0, x0 + 0.010 * QUIVER_EX], [y0, y0], [z0, z0], color="#111111", lw=3)
    ax.text(x0, y0, z0 - 0.020, f"10 mm (x{QUIVER_EX:.0f})", fontsize=11,
            color="#111111")
    style3d(ax, lim)
    ax.set_title("Point-to-surface residual after registration", pad=0)
    ax.text2D(0.5, 0.99, caption(b, f"median {st['p50']:.1f} mm   "
                                    f"p95 {st['p95']:.1f} mm   arrows x{QUIVER_EX:.0f}, "
                                    f"drawn below p{QUIVER_PCTL}"),
              transform=ax.transAxes, ha="center", va="top", fontsize=12,
              color="#333333")
    fig.savefig(path)
    plt.close(fig)
    print(f"D3 -> {path}")


# %% D4 -- heat map + horizontal section through both hulls
def seg_dist(pts, poly):
    """Min distance from each point to a closed 2D polygon (vertex loop)."""
    a, bb = poly, np.roll(poly, -1, axis=0)
    ab = bb - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom[denom == 0] = 1e-12
    out = np.empty(len(pts))
    for k, p in enumerate(pts):
        s = np.clip(np.einsum("ij,ij->i", p - a, ab) / denom, 0, 1)
        out[k] = np.min(np.linalg.norm(a + s[:, None] * ab - p, axis=1))
    return out


def design4(b, path):
    fig = plt.figure(figsize=(14.5, 6.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.set_position([0.01, 0.06, 0.40, 0.82])
    design1(b, fig=fig, ax=ax, hist=False, cb_rect=(0.435, 0.28, 0.011, 0.44))

    lat, up, ant = b["R_neu"][:, 0], b["R_neu"][:, 1], b["R_neu"][:, 2]
    h_n, h_m = b["neutral"] @ up, b["moved"] @ up
    h0 = float(np.median(h_n))
    sec_n = b["neutral"][np.abs(h_n - h0) < SECTION_HALF_M]
    sec_m = b["moved"][np.abs(h_m - h0) < SECTION_HALF_M]

    ax2 = fig.add_subplot(1, 2, 2)
    txt = "section empty at this height"
    for pts, col, lab in ((sec_n, C_NEU, "Neutral"), (sec_m, C_REG, "Registered")):
        if len(pts) < 4:
            continue
        xy = np.column_stack([pts @ lat, pts @ ant])
        loop = xy[ConvexHull(xy).vertices]
        ax2.fill(*np.vstack([loop, loop[:1]]).T, color=col, alpha=0.18)
        ax2.plot(*np.vstack([loop, loop[:1]]).T, color=col, lw=2.2, label=lab)
        ax2.scatter(xy[:, 0], xy[:, 1], c=col, s=6, alpha=0.7, edgecolors="none")
    if len(sec_n) >= 4 and len(sec_m) >= 4:
        xy_n = np.column_stack([sec_n @ lat, sec_n @ ant])
        xy_m = np.column_stack([sec_m @ lat, sec_m @ ant])
        loop_n = xy_n[ConvexHull(xy_n).vertices]
        loop_m = xy_m[ConvexHull(xy_m).vertices]
        g = seg_dist(loop_m, loop_n) * 1000
        k = int(np.argmax(g))
        ax2.annotate(f"max outline gap {g[k]:.0f} mm\nmedian {np.median(g):.0f} mm",
                     xy=loop_m[k], xytext=(0.04, 0.06), textcoords="axes fraction",
                     fontsize=12, color="#111111",
                     arrowprops=dict(arrowstyle="->", color="#111111", lw=1.4))
        txt = ""
    ax2.set_aspect("equal")
    ax2.set_xlabel("lateral (m)")
    ax2.set_ylabel("anterior (m)")
    ax2.set_title(f"Horizontal section, slab $\\pm${SECTION_HALF_M * 1000:.0f} mm",
                  pad=8)
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper right", frameon=False)
    if txt:
        ax2.text(0.5, 0.5, txt, transform=ax2.transAxes, ha="center")
    ax2.set_position([0.55, 0.12, 0.42, 0.76])
    fig.savefig(path)
    plt.close(fig)
    print(f"D4 -> {path}")


# %% Main
def main():
    m12.poster_style()
    b = frame_bundle()
    design1(b, OUT_DIR / "design1_residual_heatmap.png")
    design2(b, OUT_DIR / "design2_ghost_hulls.png")
    design3(b, OUT_DIR / "design3_residual_quiver.png")
    design4(b, OUT_DIR / "design4_heatmap_section.png")


if __name__ == "__main__":
    main()
