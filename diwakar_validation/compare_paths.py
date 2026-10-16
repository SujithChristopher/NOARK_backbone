"""Undistort-the-image ("ud") vs detect on fisheye + undistort the corners ("raw"), T1 grid holds.

Corners from detect_paths.py (same detector settings on both paths), refinements none / contour /
edge (contour + Diwakar's edge_refine) / april (rapidtag). Each corner source has its own T0 jig
(build_jig_T0.py <path>_<refine>); "tag12" = tag 12 alone. Poses from compute_poses.py.

1. Corner level: the same tag seen by both paths, corner difference (px, undistorted image) and
   tag side length ratio ud / raw, binned by distance from the principal point (fisheye distortion
   grows outward).
2. Static jitter per hold (static_jitter_T1.hold_rows: 50-frame windows), every condition on the
   frames where all 16 corner x rb variants give a pose for that solver.
3. Accuracy: hold-median grip vs mocap, rigid registration + body offset (analyze_T1.fit) fitted on
   one T1 run and scored on the other; XZ RMSE.

Outputs: out/paths/{corners.csv, holds.csv, summary.csv, corners.png, jitter.png, accuracy.png}.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import static_jitter_T1 as sj
from analyze_T1 import fit
from common import OUT, UV_COLS, Rotation, take_dir

TAKES = sj.TAKES
PATHS = ["ud", "raw"]
REFINES = ["none", "contour", "edge", "april"]
RBS = ["jig", "tag12"]
SOLVERS = sj.SOLVERS
POSE_DIR = {(f"{p}_{r}", rb): f"{p}_{r}" + ("" if rb == "jig" else "_tag12") for p in PATHS for r in REFINES for rb in RBS}
DST = OUT / "paths"
DST.mkdir(exist_ok=True)
REF_LABEL = {"none": "no refine", "contour": "contour", "edge": "Diwakar edge", "april": "AprilTag"}
PATH_LABEL = {"ud": "undistort image, then detect", "raw": "detect on fisheye, undistort corners"}
COLORS = {"none": "#1baf7a", "contour": "#eda100", "edge": "#2a78d6", "april": "#eb6834"}
INK, MUTED = sj.INK, sj.MUTED
RADIUS_EDGES = [0, 150, 250, 350, 450, 800]


# ---------- 1. corner level ----------


def side_lengths(uv):
    return np.linalg.norm(uv - np.roll(uv, -1, axis=1), axis=2).mean(1)


def corner_table():
    rows = []
    for name in TAKES:
        meta = json.loads((take_dir(name) / "meta.json").read_text())
        pp = {k: np.array(meta["undistorted"][f"camera_matrix_cam{k}"])[:2, 2] for k in (0, 1)}
        for r in REFINES:
            a = pd.read_csv(OUT / "corners" / f"{name}_ud_{r}.csv")
            b = pd.read_csv(OUT / "corners" / f"{name}_raw_{r}.csv")
            j = a.merge(b, on=["sample", "cam", "marker_id"], suffixes=("_ud", "_raw"))
            ua = j[[c + "_ud" for c in UV_COLS]].to_numpy().reshape(-1, 4, 2)
            ub = j[[c + "_raw" for c in UV_COLS]].to_numpy().reshape(-1, 4, 2)
            c = ua.mean(1)
            rad = np.linalg.norm(c - np.stack([pp[k] for k in j["cam"]]), axis=1)
            rows.append(pd.DataFrame({"take": name, "refine": r, "cam": j["cam"], "marker_id": j["marker_id"],
                                      "radius_px": rad, "side_px": side_lengths(ua),
                                      "corner_diff_px": np.linalg.norm(ua - ub, axis=2).mean(1),
                                      "centre_diff_px": np.linalg.norm(c - ub.mean(1), axis=1),
                                      "side_ratio": side_lengths(ua) / side_lengths(ub),
                                      "n_ud": len(a), "n_raw": len(b)}))
    return pd.concat(rows, ignore_index=True)


# ---------- 3. accuracy ----------


def heldout(df):
    """XZ RMSE (mm) per (corners, rb, solver), registration fitted on the other T1 run."""
    out = []
    for (c, rb, s), g in df.groupby(["corners", "rb", "solver"], sort=False):
        g = g.rename(columns={"gx": f"{s}_x", "gy": f"{s}_y", "gz": f"{s}_z"})
        errs = []
        for ev in TAKES:
            tr, te = g[g["take"] != ev], g[g["take"] == ev]
            x = fit(tr, s)
            R = Rotation.from_rotvec(x[:3]).as_matrix()
            Rb = Rotation.from_quat(te[["mc_qx", "mc_qy", "mc_qz", "mc_qw"]].to_numpy()).as_matrix()
            cw = te[[f"{s}_x", f"{s}_y", f"{s}_z"]].to_numpy() @ R.T + x[3:6]
            mw = te[["mc_x", "mc_y", "mc_z"]].to_numpy() + np.einsum("nij,j->ni", Rb, x[6:9])
            errs.append((cw - mw) * 1000)
        e = np.concatenate(errs)
        out.append({"corners": c, "rb": rb, "solver": s, "n_holds": len(e),
                    "x_rmse_mm": float(np.sqrt((e[:, 0] ** 2).mean())), "z_rmse_mm": float(np.sqrt((e[:, 2] ** 2).mean())),
                    "xz_rmse_mm": float(np.sqrt((e[:, [0, 2]] ** 2).sum(1).mean()))})
    return pd.DataFrame(out)


# ---------- figures ----------


def fig_corners(ct):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    mid = [(a + b) / 2 for a, b in zip(RADIUS_EDGES[:-1], RADIUS_EDGES[1:])]
    for r in REFINES:
        d = ct[ct.refine == r].assign(bin=lambda x: pd.cut(x.radius_px, RADIUS_EDGES))
        g = d.groupby("bin", observed=False)
        axs[0].plot(mid, g.corner_diff_px.median(), marker="o", lw=2, color=COLORS[r], label=REF_LABEL[r])
        axs[1].plot(mid, (g.side_ratio.median() - 1) * 100, marker="o", lw=2, color=COLORS[r], label=REF_LABEL[r])
    axs[0].set_ylabel("|corner ud - raw| (px), median")
    axs[1].set_ylabel("tag side, ud vs raw (%), median")
    axs[1].axhline(0, color=MUTED, lw=0.8)
    for a in axs:
        a.set_xlabel("tag centre distance from principal point (px, undistorted image)")
    axs[0].legend()
    fig.suptitle("Same tag, same frame: undistorted-image corners vs undistorted fisheye corners", x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(DST / "corners.png", dpi=160)
    plt.close(fig)


def fig_pairs(sm, col, fname, ylabel, title, log=True):
    fig, axs = plt.subplots(2, 3, figsize=(10, 6.4), sharey="row" if not log else True)
    for i, rb in enumerate(RBS):
        for k, s in enumerate(SOLVERS):
            ax = axs[i, k]
            for x, r in enumerate(REFINES):
                v = [sm[(sm.corners == f"{p}_{r}") & (sm.rb == rb) & (sm.solver == s)][col].iloc[0] for p in PATHS]
                ax.plot([x - 0.15, x + 0.15], v, color=COLORS[r], lw=1)
                ax.plot(x - 0.15, v[0], "o", ms=8, color=COLORS[r])
                ax.plot(x + 0.15, v[1], "s", ms=7, color=COLORS[r], mfc="white", mew=2)
                for xx, vv, ha in ((x - 0.22, v[0], "right"), (x + 0.22, v[1], "left")):
                    ax.annotate(f"{vv:.3f}" if vv < 1 else f"{vv:.2f}", (xx, vv), fontsize=6.5, ha=ha, va="center",
                                color=INK)
            ax.set_xticks(range(len(REFINES)), [REF_LABEL[r] for r in REFINES], fontsize=7.5)
            ax.set_xlim(-0.7, len(REFINES) - 0.3)
            ax.set_title(f"{ {'cam0': 'cam0 solvePnP', 'cam1': 'cam1 solvePnP', 'joint': 'joint stereo'}[s]}, "
                         f"{'rigid body' if rb == 'jig' else 'tag 12 only'}")
            if log:
                ax.set_yscale("log")
        axs[i, 0].set_ylabel(ylabel)
    h = [plt.Line2D([], [], marker="o", ls="", color=MUTED, ms=7, label=PATH_LABEL["ud"]),
         plt.Line2D([], [], marker="s", ls="", color=MUTED, mfc="white", mew=2, ms=7, label=PATH_LABEL["raw"])]
    fig.legend(handles=h, loc="upper right", ncol=2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(DST / fname, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ct = corner_table()
    ct.to_csv(DST / "corners.csv", index=False)
    fig_corners(ct)
    print(ct.groupby("refine").agg(tags=("corner_diff_px", "size"), diff_med_px=("corner_diff_px", "median"),
                                   diff_p90_px=("corner_diff_px", lambda x: x.quantile(0.9)),
                                   centre_med_px=("centre_diff_px", "median"),
                                   side_pct=("side_ratio", lambda x: (x.median() - 1) * 100)).round(3).to_string())
    print("detections ud / raw:", ct.groupby("refine")[["n_ud", "n_raw"]].first().sum().to_dict())

    df = pd.DataFrame([r for n in TAKES for r in sj.hold_rows(n, POSE_DIR, with_ref=False,
                                                              depth_key=("ud_edge", "jig", "joint"))])
    df["moved"] = df.mocap_3d_mm > sj.MOCAP_MOVED_MM
    df.to_csv(DST / "holds.csv", index=False)
    use = df[~df.moved]
    sm = sj.summarize(use).merge(heldout(use), on=["corners", "rb", "solver"], suffixes=("", "_acc"))
    sm.to_csv(DST / "summary.csv", index=False)
    fig_pairs(sm, "jitter_3d_med_mm", "jitter.png", "hold jitter, 3D SD (mm)",
              "T1 static jitter: undistort image vs undistort corners")
    fig_pairs(sm, "xz_rmse_mm", "accuracy.png", "XZ RMSE vs mocap (mm), held-out",
              "T1 accuracy: undistort image vs undistort corners", log=False)
    with pd.option_context("display.width", 200):
        print(sm[["corners", "rb", "solver", "n_holds", "jitter_3d_med_mm", "jitter_z_med_mm", "drift_3d_med_mm",
                  "x_rmse_mm", "z_rmse_mm", "xz_rmse_mm"]].round(3).to_string(index=False))
