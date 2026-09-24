# %% 12_poster_figures.py
# Two poster-grade static figures from the validated ICP pipeline (07_icp_trunk.py),
# same data path / sync / zeroing as the validation run -- nothing is recomputed
# differently here, only drawn for print.
#
#   A) icp_hull_registration.png  3D convex-hull registration, tight fit:
#      neutral torso hull (grey, filled) vs the live cloud pushed through the ICP
#      transform (green hull wireframe + points). Overlap == healthy registration.
#   B) icp_vs_mocap_angles.png    3-row trunk-angle traces, ICP vs mocap only.
#
#   $env:TRUNK_GEOM_CACHE="...\data\july27\july_27_sujith_no_comp\geom_cache.pkl"
#   uv run python trunkpose/dual_notebooks/12_poster_figures.py

import importlib
import os
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


def _stub_inference_deps():
    """Let 06 import without mediapipe/ultralytics installed.

    This script only ever draws from the cached geometry pass, so the pose/seg
    models are never called -- but 06 imports them at module level. If the cache
    turns out to be missing or stale, main() stops with an explicit message
    instead of letting a stub blow up somewhere deep in the geometry pass."""
    import types

    class _Stub(types.ModuleType):
        def __getattr__(self, name):
            raise RuntimeError(
                f"{self.__name__}.{name} needs the real package: install "
                "mediapipe + ultralytics (uv sync) to recompute geometry")

    for name in ("mediapipe", "mediapipe.tasks", "mediapipe.tasks.python",
                 "mediapipe.tasks.python.vision", "ultralytics"):
        mod = _Stub(name)
        mod.__path__ = []          # import machinery asks packages for this
        sys.modules.setdefault(name, mod)
    sys.modules["mediapipe"].tasks = sys.modules["mediapipe.tasks"]
    sys.modules["mediapipe.tasks"].python = sys.modules["mediapipe.tasks.python"]
    sys.modules["mediapipe.tasks.python"].BaseOptions = object
    sys.modules["mediapipe.tasks.python"].vision = \
        sys.modules["mediapipe.tasks.python.vision"]
    sys.modules["ultralytics"].YOLO = object


try:
    import mediapipe  # noqa: F401
    import ultralytics  # noqa: F401
except ImportError:
    _stub_inference_deps()

m6 = importlib.import_module("06_trunk_axis")
m7 = importlib.import_module("07_icp_trunk")

if not os.environ.get("TRUNK_GEOM_CACHE"):
    for cand in (m6.RECORDING_DIR / "geom_cache_v2.pkl",
                 m6.RECORDING_DIR / "geom_cache.pkl"):
        if cand.exists():
            os.environ["TRUNK_GEOM_CACHE"] = str(cand)
            break

OUT_HULL = m6.RECORDING_DIR / "icp_hull_registration.png"
OUT_ANG = m6.RECORDING_DIR / "icp_vs_mocap_angles.png"
MATCH_M = 0.03          # nn distance counted as matched in the printed rms
HULL_PAD_M = 0.02       # axis pad around the hulls -- small on purpose (tight fit)

# Okabe-Ito, colorblind-safe. Mocap stays black so it reads in greyscale too.
C_ICP = {"flexion": "#D55E00", "lateral": "#E69F00", "axial": "#0072B2"}
C_MOCAP = "#000000"
C_REG = "#009E73"       # registered cloud/hull
C_NEU = "#9a9a9a"       # neutral hull


def poster_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 13, "axes.labelsize": 15, "axes.titlesize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 13,
        "axes.linewidth": 1.2, "lines.linewidth": 2.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.dpi": 400,
    })


# %% Hull helpers
def hull_faces(pts):
    h = ConvexHull(pts)
    return pts[h.simplices], h


def draw_panel(ax, neutral, moved, lim, title, sub, subcol):
    # Hull faces carry their own faint edges -- a separate wireframe collection
    # reads as noise at poster size because every back-facing triangle shows
    # through the translucent surface.
    tri_n, _ = hull_faces(neutral)
    ax.add_collection3d(Poly3DCollection(tri_n, facecolor=C_NEU, alpha=0.22,
                                         edgecolor=C_NEU, linewidths=0.25))
    if moved is not None:
        tri_m, _ = hull_faces(moved)
        ax.add_collection3d(Poly3DCollection(tri_m, facecolor=C_REG, alpha=0.20,
                                             edgecolor=C_REG, linewidths=0.35))
        ax.scatter(moved[:, 0], moved[:, 1], moved[:, 2], c=C_REG, s=2.0,
                   alpha=0.75, depthshade=False, edgecolors="none")
    ax.set_xlim(lim[0])
    ax.set_ylim(lim[1])
    ax.set_zlim(lim[2])
    ax.set_box_aspect([lim[k][1] - lim[k][0] for k in range(3)], zoom=1.35)
    ax.view_init(elev=14, azim=-72)
    ax.set_xlabel("x (m)", labelpad=-4, fontsize=10)
    ax.set_ylabel("y (m)", labelpad=-4, fontsize=10)
    ax.set_zlabel("z (m)", labelpad=-6, fontsize=10)
    ax.tick_params(labelsize=8, pad=-2)
    ax.grid(alpha=0.25)
    ax.set_title(title, pad=0)
    ax.text2D(0.5, 0.995, sub, transform=ax.transAxes, ha="center", va="top",
              fontsize=12, color=subcol)


# %% Main
def main():
    poster_style()
    cache = os.environ.get("TRUNK_GEOM_CACHE", "")
    if not (cache and Path(cache).exists()):
        raise SystemExit(
            "No geometry cache. Run 07_icp_trunk.py once (with mediapipe + "
            "ultralytics installed) or set TRUNK_GEOM_CACHE.")
    inp = m7.load_inputs()
    frames, n, ts0 = inp["frames"], inp["n"], inp["ts0"]

    _Rp, R_neu, _fp, _lp, _ap, _ = m6.assemble(frames)
    icp = m7.run_icp_pass(frames, R_neu, n)
    cmp_ = m7.compare_mocap(inp, icp["Rb_icp"], R_neu)

    neutral = icp["neutral"]
    tree = cKDTree(neutral)
    in_win, zwin = cmp_["in_win"], cmp_["zwin"]
    flex = m6.smooth_series(cmp_["flex_iz"])
    lat = m6.smooth_series(cmp_["lat_iz"])
    axi = m6.smooth_series(cmp_["axi_iz"])
    icp_ser = (flex, lat, axi)
    moc_ser = (cmp_["mflex"], cmp_["mlat"], cmp_["maxi"])
    names = ("flexion", "lateral", "axial")
    labels = ("Flexion / extension", "Lateral bending", "Axial rotation")

    # ---- A) hull registration, tight fit ----
    # Panels: the neutral pose itself, then the largest accepted excursion in
    # flexion and in axial rotation -- the poses where a wrong minimum would show.
    def pick(series):
        ok = np.array([a is not None for a in icp["A_list"]]) & in_win
        s = np.where(ok, np.abs(series), np.nan)
        return int(np.nanargmax(s)) if np.isfinite(s).any() else None

    picks = [(zwin[0], "Neutral pose")]
    for series, lab in ((flex, "Peak flexion"), (axi, "Peak axial rotation")):
        i = pick(series)
        if i is not None:
            picks.append((i, f"{lab} ({series[i]:+.0f} deg)"))

    t_all = ((ts0 - ts0[zwin[0]]) / np.timedelta64(1, "s")).astype(np.float64)
    panels = []
    allpts = [neutral]
    for i, lab in picks:
        A, cloud = icp["A_list"][i], frames[i]["cloud"]
        moved = None
        sub, subcol = "no registration", "#b00000"
        if A is not None and cloud is not None:
            R, t = A
            moved = cloud.astype(np.float64) @ R.T + t
            d, _ = tree.query(moved, k=1)
            m = d < MATCH_M
            sub = (f"ICP rms {np.sqrt((d[m] ** 2).mean()) * 1000:.1f} mm   "
                   f"inliers {100 * m.mean():.0f}%   t = {t_all[i]:+.1f} s")
            subcol = "#333333"
            allpts.append(moved)
        panels.append((moved, lab, sub, subcol))

    pts = np.concatenate(allpts)
    lo, hi = np.percentile(pts, 0.5, axis=0), np.percentile(pts, 99.5, axis=0)
    lim = [(lo[k] - HULL_PAD_M, hi[k] + HULL_PAD_M) for k in range(3)]

    fig = plt.figure(figsize=(15, 5.0))
    for k, (moved, lab, sub, subcol) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), k + 1, projection="3d")
        draw_panel(ax, neutral, moved, lim, lab, sub, subcol)
        ax.text2D(0.01, 0.99, "ABC"[k], transform=ax.transAxes, fontsize=18,
                  fontweight="bold", va="top")
    handles = [
        Line2D([], [], color=C_NEU, lw=3, label="Neutral torso hull"),
        Line2D([], [], color=C_REG, lw=3, label="Registered live hull (ICP)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Rigid registration of the torso surface: live cloud mapped onto "
                 "the neutral hull", y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(OUT_HULL)
    print(f"Hull figure -> {OUT_HULL}")
    plt.close(fig)

    # ---- B) angle traces, ICP vs mocap only ----
    def agree(a, b):
        m = np.isfinite(a) & np.isfinite(b) & in_win
        if m.sum() < 50:
            return np.nan, np.nan, int(m.sum())
        return (float(np.corrcoef(a[m], b[m])[0, 1]),
                float(np.sqrt(((a[m] - b[m]) ** 2).mean())), int(m.sum()))

    w = np.where(in_win)[0]
    t0v, t1v = t_all[w[0]], t_all[w[-1]]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    print("\nICP vs mocap (clean segment):")
    for k, ax in enumerate(axes):
        r, e, cnt = agree(icp_ser[k], moc_ser[k])
        print(f"  {labels[k]:22s} r={r:+.3f}  RMSE={e:5.2f} deg  n={cnt}")
        ax.plot(t_all, moc_ser[k], color=C_MOCAP, ls="--", lw=1.6, alpha=0.9,
                label="Mocap (OptiTrack)")
        ax.plot(t_all, icp_ser[k], color=C_ICP[names[k]], lw=2.2,
                label="Camera ICP")
        ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=0)
        ax.set_ylabel(f"{labels[k]}\n(deg)")
        ax.set_xlim(t0v, t1v)
        ax.grid(alpha=0.25)
        ax.text(0.995, 0.04, f"r = {r:+.2f}   RMSE = {e:.1f} deg",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=13)
        if k == 0:
            ax.legend(loc="upper right", ncol=2, frameon=False)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Trunk angles: markerless camera ICP vs motion capture", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_ANG)
    print(f"\nAngle figure -> {OUT_ANG}")
    plt.close(fig)


if __name__ == "__main__":
    main()
