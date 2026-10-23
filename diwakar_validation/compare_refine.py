"""Refinement layers on Diwakar's edge refinement against AprilTag refinement: static jitter and accuracy
on the T1 grid holds, error while moving on T2 / T3.

Corner sources (each with its own T0 jig, all tags as one rigid body):
  april        rapidtag CORNER_REFINE_APRILTAG on the fisheye crop (detect_corners.py; the report's)
  ud_april     the same on the undistorted image (detect_paths.py)
  logged       Diwakar's tracker as recorded (CONTOUR + edge_refine on the undistorted image)
  ud_edge      the same re-detected (detect_paths.py)
  ud_edge_trk  edge + tracking layer: missed tags re-found from their last corners (detect_grid.py)
  ud_grid      edge -> grid layer: homography fit to all pattern transitions (detect_grid.py)
  ud_grid_trk  both layers
  ud_edge_drop, ud_grid_drop   ud_edge / ud_grid with the failed edge refinements left out (no re-found
               tags), on ud_edge's / ud_grid's jig, so those tags are the only difference
T1: hold jitter = median 50-frame window SD (static_jitter_T1.hold_rows), every source on the frames where
all of them give a pose; quantiles over holds, since the question is the spread, not only the median.
Accuracy: hold-median grip vs mocap, registration (analyze_T1.fit) fitted on the other T1 run.
T2 / T3: registration fitted on both T1 runs, frozen; per-sample XZ error on the samples every source
solved; mocap read MOCAP_LAG_S after the camera timestamp (the 7-8 ms camera-mocap offset).
Only ud_edge / ud_april lack T2 / T3 corners (detect_paths.py ran T0 / T1), so they sit out the moving part.

Outputs: out/refine/{holds.csv, jitter.csv, accuracy.csv, moving.csv, jitter.png}.
Prerequisites: detect_grid.py, build_jig_T0.py <source> and compute_poses.py <source> for the new sources.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import static_jitter_T1 as sj
from analyze_T1 import fit
from common import DATA, OUT, Rotation, load_mocap, mocap_frame_of, take_dir

T1 = sj.TAKES
SOURCES = ["april", "ud_april", "logged", "ud_edge", "ud_edge_drop", "ud_edge_trk", "ud_grid", "ud_grid_drop",
           "ud_grid_trk"]
MOVING_SOURCES = ["april", "logged", "ud_edge_drop", "ud_edge_trk", "ud_grid", "ud_grid_drop", "ud_grid_trk"]
POSE_DIR = {(s, "jig"): s for s in SOURCES}
SOLVERS = sj.SOLVERS
QUANTILES = (25, 50, 75, 90, 95)
MOCAP_LAG_S = 0.0076
JUMP_M = 0.02
DYN = sorted(p.name for p in DATA.glob("2026-10-03_T[23]_*"))
LABEL = {"april": "AprilTag (fisheye)", "ud_april": "AprilTag (ud)", "logged": "edge (logged)",
         "ud_edge": "edge (ud)", "ud_edge_drop": "edge, failed left out", "ud_edge_trk": "edge + track",
         "ud_grid": "edge + grid", "ud_grid_drop": "edge + grid, failed left out", "ud_grid_trk": "edge + track + grid"}
COLORS = {"april": "#eb6834", "ud_april": "#f2a37f", "logged": "#8fb5e6", "ud_edge": "#2a78d6",
          "ud_edge_drop": "#5aa0f0", "ud_edge_trk": "#1baf7a", "ud_grid": "#eda100", "ud_grid_drop": "#f5c75a",
          "ud_grid_trk": "#7a3fb8"}
DST = OUT / "refine"
DST.mkdir(exist_ok=True)


def gm(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def jitter_table(use):
    rows = []
    for (c, s), g in use.groupby(["corners", "solver"], sort=False):
        j = g.jitter_3d_mm
        rows.append({"corners": c, "solver": s, "n_holds": len(g), "gm_mm": gm(j),
                     **{f"p{q}_mm": float(np.percentile(j, q)) for q in QUANTILES}})
    out = pd.DataFrame(rows)
    # paired against AprilTag: share of holds where the source is worse
    for ref in ("april", "ud_april"):
        w = use.pivot_table(index=["take", "hold"], columns=["corners", "solver"], values="jitter_3d_mm")
        out[f"worse_than_{ref}"] = [float((w[(c, s)] > w[(ref, s)]).mean()) for c, s in zip(out.corners, out.solver)]
    return out


def accuracy_table(use):
    """Held-out across the T1 runs: per-hold XZ error, RMSE and spread."""
    rows = []
    for (c, s), g in use.groupby(["corners", "solver"], sort=False):
        g = g.rename(columns={"gx": f"{s}_x", "gy": f"{s}_y", "gz": f"{s}_z"})
        errs = []
        for ev in T1:
            tr, te = g[g["take"] != ev], g[g["take"] == ev]
            x = fit(tr, s)
            R = Rotation.from_rotvec(x[:3]).as_matrix()
            Rb = Rotation.from_quat(te[["mc_qx", "mc_qy", "mc_qz", "mc_qw"]].to_numpy()).as_matrix()
            cw = te[[f"{s}_x", f"{s}_y", f"{s}_z"]].to_numpy() @ R.T + x[3:6]
            mw = te[["mc_x", "mc_y", "mc_z"]].to_numpy() + np.einsum("nij,j->ni", Rb, x[6:9])
            errs.append((cw - mw) * 1000)
        e = np.concatenate(errs)[:, [0, 2]]
        r = np.hypot(*e.T)
        rows.append({"corners": c, "solver": s, "n_holds": len(e), "xz_rmse_mm": float(np.sqrt((r**2).mean())),
                     "x_sd_mm": float(e[:, 0].std()), "z_sd_mm": float(e[:, 1].std()),
                     "p50_mm": float(np.median(r)), "p90_mm": float(np.percentile(r, 90)), "max_mm": float(r.max())})
    return pd.DataFrame(rows)


def mocap_at(name, t, x):
    """Mocap grip point (body offset x[6:9]) at camera times t via the GPIO edges; NaN at gaps / jumps."""
    mc = load_mocap(name)
    f = mocap_frame_of(name, t)
    v = mc["valid"].copy()
    jump = np.r_[False, np.linalg.norm(np.diff(mc["p"], axis=0), axis=1) > JUMP_M]
    v &= ~jump & ~np.r_[jump[1:], False]
    q = mc["p"] + np.einsum("nij,j->ni", np.nan_to_num(mc["R"]), x[6:9])
    out = np.full((len(t), 3), np.nan)
    ok = np.isfinite(f) & (f >= 0) & (f < len(q) - 1)
    a = np.floor(f[ok]).astype(int)
    good = v[a] & v[a + 1]
    k = np.flatnonzero(ok)[good]
    a = a[good]
    w = (f[k] - a)[:, None]
    out[k] = q[a] * (1 - w) + q[a + 1] * w
    return out


def moving_table(use):
    regs = {}
    for (c, s), g in use[use.corners.isin(MOVING_SOURCES)].groupby(["corners", "solver"]):
        regs[(c, s)] = fit(g.rename(columns={"gx": f"{s}_x", "gy": f"{s}_y", "gz": f"{s}_z"}), s)
    rows = []
    for name in DYN:
        img = pd.read_csv(take_dir(name) / "images.csv", usecols=["sample"])["sample"].unique()
        poses = {c: pd.read_csv(OUT / "poses" / c / f"{name}.csv").set_index("sample") for c in MOVING_SOURCES}
        for s in SOLVERS:
            common = set(img)
            for p in poses.values():
                common &= set(p.index[p[f"{s}_gx"].notna()])
            common = sorted(common)
            for c, p in poses.items():
                x = regs[(c, s)]
                p = p.loc[common]
                t = p["t_cam1_s" if s == "cam1" else "t_cam0_s"].to_numpy()
                g = p[[f"{s}_gx", f"{s}_gy", f"{s}_gz"]].to_numpy() @ Rotation.from_rotvec(x[:3]).as_matrix().T + x[3:6]
                e = ((g - mocap_at(name, t + MOCAP_LAG_S, x)) * 1000)[:, [0, 2]]
                e = e[np.isfinite(e).all(1)]
                r = np.hypot(*e.T)
                rows.append({"take": name, "task": "_".join(name.split("_")[1:-1]), "corners": c, "solver": s, "n": len(r),
                             "xz_rmse_mm": float(np.sqrt((r**2).mean())), "p50_mm": float(np.median(r)),
                             "p90_mm": float(np.percentile(r, 90)), "p99_mm": float(np.percentile(r, 99))})
    return pd.DataFrame(rows)


def fig_jitter(use):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, s in zip(axs, SOLVERS):
        for c in SOURCES:
            j = np.sort(use[(use.corners == c) & (use.solver == s)].jitter_3d_mm.to_numpy())
            ax.plot(j, np.linspace(0, 1, len(j)), color=COLORS[c], lw=2, label=LABEL[c])
        ax.set_xscale("log")
        ax.set_xlabel("hold jitter, 3D window SD (mm)")
        ax.set_title({"cam0": "cam0 solvePnP", "cam1": "cam1 solvePnP", "joint": "joint stereo"}[s])
    axs[0].set_ylabel("share of holds")
    axs[0].legend(loc="lower right")
    fig.suptitle("T1 static jitter by corner refinement, rigid jig (cumulative over holds)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(DST / "jitter.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.DataFrame([r for n in T1 for r in sj.hold_rows(n, POSE_DIR, with_ref=False,
                                                            depth_key=("ud_edge", "jig", "joint"))])
    df["moved"] = df.mocap_3d_mm > sj.MOCAP_MOVED_MM
    df.to_csv(DST / "holds.csv", index=False)
    use = df[~df.moved]
    jt, acc, mv = jitter_table(use), accuracy_table(use), moving_table(use)
    jt.to_csv(DST / "jitter.csv", index=False)
    acc.to_csv(DST / "accuracy.csv", index=False)
    mv.to_csv(DST / "moving.csv", index=False)
    fig_jitter(use)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("\nT1 hold jitter, 3D (mm)\n", jt.round(3).to_string(index=False))
        print("\nT1 held-out accuracy, XZ (mm)\n", acc.round(3).to_string(index=False))
        m = mv.groupby(["corners", "solver"], sort=False)[["xz_rmse_mm", "p50_mm", "p90_mm", "p99_mm"]].mean()
        print("\nT2/T3 moving, XZ (mm), mean over takes\n", m.round(3).to_string())
