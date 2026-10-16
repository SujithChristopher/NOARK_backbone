"""Static jitter on the T1 grid holds, dome-style (jitter_model/05_static_jitter.py).

Each T1 hold (marks.csv hold_start..hold_end, trimmed TRIM_S each side) is a still pose, so the
spread of the estimated grip point inside it is estimator jitter. To match the dome bursts
(50 frames), every hold is cut into WINDOW-frame windows; hold jitter = median window SD (ddof=1).
The SD over the whole trimmed hold is kept separately as `drift` (slow settling shows up there).

Conditions, all scored on the same frames per solver:
- corners: logged (Diwakar's tracker: aruco CONTOUR + edge_refine), april (rapidtag
  CORNER_REFINE_APRILTAG), none (raw OpenCV quads)
- rb: jig (all visible tags, rigid body bundle-adjusted from each source's own T0 corners)
  or tag12 (reference tag alone)
- solver: cam0 / cam1 (single-camera solvePnP: IPPE for 4 points, SQPnP + LM otherwise),
  joint (stereo Gauss-Newton over both cameras' corners)
- reference: Diwakar's exact pipeline = samples.csv (logged corners, shipped jig,
  estimate_board_pose_dual_gn with Huber 1 px).

Frames: for each solver, a sample counts only if every corner x rb variant produced a pose
(so tag 12 is visible to that solver in every corner source). The reference uses the joint set.
Position SD is reported in mocap axes (rotation from out/T1_registration.json 'joint|both';
Motive Y-up: X across, Y height, Z depth). Mocap rigid-body SD over the same windows = floor.

Outputs: out/static_jitter/{holds.csv, summary.csv, vs_distance.csv, summary.png, vs_distance.png,
axes.png}. Prerequisite: compute_poses.py for logged, april, none, tag12only, april_tag12only,
none_tag12only on the T1 takes.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_T1 import TRIM_S, load_holds
from common import OUT, Rotation, load_jig, load_mocap, load_samples, mocap_frame_of, shipped_jig_path

TAKES = ["2026-10-03_T1_grid_r1", "2026-10-03_T1_grid_r2"]
WINDOW = 50
MIN_WINDOWS = 1
CORNERS = ["logged", "april", "none"]
RBS = ["jig", "tag12"]
SOLVERS = ["cam0", "cam1", "joint"]
POSE_DIR = {("logged", "jig"): "logged", ("logged", "tag12"): "tag12only",
            ("april", "jig"): "april", ("april", "tag12"): "april_tag12only",
            ("none", "jig"): "none", ("none", "tag12"): "none_tag12only"}
REF = ("diwakar", "shipped", "joint")  # (corners, rb, solver) label for samples.csv
DEPTH_EDGES_M = np.arange(0.30, 0.701, 0.05)
MOCAP_MOVED_MM = 0.2  # hold whose mocap window SD exceeds this is flagged as moved

DST = OUT / "static_jitter"
DST.mkdir(exist_ok=True)
AXES = ["x", "y", "z"]
AXIS_LABEL = {"x": "X across", "y": "Y height", "z": "Z depth"}
CORNER_LABEL = {"logged": "Diwakar edge refine", "april": "AprilTag refine", "none": "no refine",
                "diwakar": "Diwakar pipeline (samples.csv)"}
COLORS = {"logged": "#2a78d6", "april": "#eb6834", "none": "#1baf7a"}
INK, MUTED = "#0b0b0b", "#8a8984"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "legend.frameon": False})

R_CAM_TO_MOCAP = Rotation.from_rotvec(json.loads((OUT / "T1_registration.json").read_text())["joint|both"][:3]).as_matrix()


def load_take(name, pose_dir=POSE_DIR, with_ref=True):
    """Dict (corners, rb, solver) -> DataFrame[sample, t, gx, gy, gz, qx..qw], on the common frames per solver.
    pose_dir: (corners, rb) -> out/poses/<dir>. with_ref adds Diwakar's samples.csv pose as REF."""
    poses = {k: pd.read_csv(OUT / "poses" / d / f"{name}.csv").set_index("sample") for k, d in pose_dir.items()}
    out = {}
    common = {}
    for s in SOLVERS:
        ok = None
        for p in poses.values():
            have = set(p.index[p[f"{s}_gx"].notna()])
            ok = have if ok is None else ok & have
        common[s] = np.array(sorted(ok))
        for (c, rb), p in poses.items():
            d = p.loc[common[s], ["t_cam0_s"] + [f"{s}_{k}" for k in ("gx", "gy", "gz", "qx", "qy", "qz", "qw")]]
            d.columns = ["t", "gx", "gy", "gz", "qx", "qy", "qz", "qw"]
            out[(c, rb, s)] = d.reset_index()
    if not with_ref:
        return out, common
    # Diwakar's own output: samples.csv joint pose + shipped grip
    sm = load_samples(name).set_index("sample").loc[common["joint"]]
    sm = sm[np.linalg.norm(sm[["qx", "qy", "qz", "qw"]].fillna(0).to_numpy(), axis=1) > 0.5]
    R = Rotation.from_quat(sm[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()
    g = np.einsum("nij,j->ni", R, load_jig(shipped_jig_path())["grip"]) + sm[["tx_m", "ty_m", "tz_m"]].to_numpy()
    out[REF] = pd.DataFrame({"sample": sm.index, "t": sm["t_cam0_s"].to_numpy(), "gx": g[:, 0], "gy": g[:, 1],
                             "gz": g[:, 2], **{f"q{c}": sm[f"q{c}"].to_numpy() for c in "xyzw"}})
    return out, common


def window_stats(g, q):
    """Per-window SD of grip point (mm, mocap axes) and rotation (deg). g (n,3) cam0 metres, q (n,4)."""
    gm = g @ R_CAM_TO_MOCAP.T * 1000.0
    pos, rot = [], []
    for a in range(0, len(g) - WINDOW + 1, WINDOW):
        sl = slice(a, a + WINDOW)
        pos.append(gm[sl].std(0, ddof=1))
        Rw = Rotation.from_quat(q[sl])
        dev = (Rw * Rw.mean().inv()).as_rotvec()
        rot.append(float(np.degrees(np.sqrt((dev**2).sum(1).sum() / (WINDOW - 1)))))  # RMS angle about window mean
    return np.array(pos), np.array(rot)


def mocap_windows(mc, name, t):
    """Mocap rigid-body position SD (mm) over the same windows, matched by GPIO sync."""
    out = []
    for a in range(0, len(t) - WINDOW + 1, WINDOW):
        f0, f1 = mocap_frame_of(name, np.array([t[a], t[a + WINDOW - 1]]))
        if not np.isfinite([f0, f1]).all():
            continue
        fr = np.arange(int(np.ceil(f0)), int(np.floor(f1)) + 1)
        fr = fr[mc["valid"][fr]]
        if len(fr) > 10:
            out.append(np.linalg.norm(mc["p"][fr].std(0, ddof=1)) * 1000.0)
    return float(np.median(out)) if out else np.nan


def hold_rows(name, pose_dir=POSE_DIR, with_ref=True, depth_key=("logged", "jig", "joint")):
    """One row per (hold, condition): window jitter, drift, depth, mocap floor, plus the hold's median grip
    point (cam0 frame) and mean mocap pose, so the same table also feeds a registration fit."""
    data, common = load_take(name, pose_dir, with_ref)
    mc = load_mocap(name)
    ref_depth = data[depth_key]
    rows = []
    for i, h in load_holds(name).iterrows():
        a, b = h.t0 + TRIM_S, h.t1 - TRIM_S
        rd = ref_depth[(ref_depth.t >= a) & (ref_depth.t <= b)]
        depth = float(rd.gz.median()) if len(rd) else np.nan
        mc_floor = mocap_windows(mc, name, rd.t.to_numpy()) if len(rd) >= WINDOW else np.nan
        f0, f1 = mocap_frame_of(name, np.array([a, b]))
        fr = np.arange(int(np.ceil(f0)), int(np.floor(f1)) + 1)
        fr = fr[mc["valid"][fr]]
        mc_pose = {"mc_x": mc["p"][fr, 0].mean(), "mc_y": mc["p"][fr, 1].mean(), "mc_z": mc["p"][fr, 2].mean(),
                   **{f"mc_q{c}": v for c, v in zip("xyzw", Rotation.from_matrix(mc["R"][fr]).mean().as_quat())}}
        for key, d in data.items():
            w = d[(d.t >= a) & (d.t <= b)]
            if len(w) < WINDOW * MIN_WINDOWS:
                continue
            g = w[["gx", "gy", "gz"]].to_numpy()
            pos, rot = window_stats(g, w[["qx", "qy", "qz", "qw"]].to_numpy())
            med = np.median(pos, 0)
            rows.append({"take": name, "hold": i, "corners": key[0], "rb": key[1], "solver": key[2],
                         "depth_m": depth, "n_frames": len(w), "n_windows": len(pos),
                         **{f"jitter_{ax}_mm": v for ax, v in zip(AXES, med)},
                         "jitter_3d_mm": float(np.median(np.linalg.norm(pos, axis=1))),
                         "jitter_rot_deg": float(np.median(rot)),
                         "drift_3d_mm": float(np.linalg.norm((g @ R_CAM_TO_MOCAP.T).std(0, ddof=1)) * 1000.0),
                         "mocap_3d_mm": mc_floor,
                         **{f"g{ax}": v for ax, v in zip(AXES, np.median(g, 0))}, **mc_pose})
    return rows


def q(x, p):
    return float(np.nanpercentile(x, p)) if len(x) else np.nan


def summarize(df):
    rows = []
    for key, g in df.groupby(["corners", "rb", "solver"], sort=False):
        rows.append({"corners": key[0], "rb": key[1], "solver": key[2], "n_holds": len(g),
                     "jitter_3d_med_mm": g.jitter_3d_mm.median(), "jitter_3d_p25_mm": q(g.jitter_3d_mm, 25),
                     "jitter_3d_p75_mm": q(g.jitter_3d_mm, 75), "jitter_3d_p90_mm": q(g.jitter_3d_mm, 90),
                     **{f"jitter_{ax}_med_mm": g[f"jitter_{ax}_mm"].median() for ax in AXES},
                     "jitter_rot_med_deg": g.jitter_rot_deg.median(), "drift_3d_med_mm": g.drift_3d_mm.median(),
                     "mocap_3d_med_mm": g.mocap_3d_mm.median()})
    return pd.DataFrame(rows)


def by_depth(df):
    df = df.assign(bin=pd.cut(df.depth_m, DEPTH_EDGES_M))
    out = (df.groupby(["corners", "rb", "solver", "bin"], observed=True)
           .agg(n_holds=("jitter_3d_mm", "size"), jitter_3d_med_mm=("jitter_3d_mm", "median"),
                jitter_z_med_mm=("jitter_z_mm", "median"), mocap_3d_med_mm=("mocap_3d_mm", "median"))
           .reset_index())
    out["depth_mid_m"] = out["bin"].map(lambda b: b.mid).astype(float)
    return out


# ---------- figures ----------


def fig_summary(sm, floor):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, s in zip(axs, SOLVERS):
        for i, c in enumerate(CORNERS):
            for j, rb in enumerate(RBS):
                r = sm[(sm.corners == c) & (sm.rb == rb) & (sm.solver == s)].iloc[0]
                x = i + (j - 0.5) * 0.3
                ax.plot([x, x], [r.jitter_3d_p25_mm, r.jitter_3d_p75_mm], color=COLORS[c], lw=2, solid_capstyle="round")
                ax.plot(x, r.jitter_3d_med_mm, "o", ms=8, color=COLORS[c], mfc=COLORS[c] if rb == "jig" else "white",
                        mew=2, zorder=3)
                ax.annotate(f"{r.jitter_3d_med_mm:.3f}", (x, r.jitter_3d_med_mm), xytext=(5, 0),
                            textcoords="offset points", fontsize=7, color=INK, va="center")
        if s == "joint":
            r = sm[(sm.corners == REF[0])].iloc[0]
            ax.axhline(r.jitter_3d_med_mm, color=INK, lw=1, ls="--")
            ax.text(2.45, r.jitter_3d_med_mm, f"Diwakar pipeline {r.jitter_3d_med_mm:.3f}", ha="right", va="bottom",
                    fontsize=7, color=INK)
        ax.axhline(floor, color=MUTED, lw=1, ls=":")
        ax.text(2.45, floor, f"mocap {floor:.3f}", ha="right", va="top", fontsize=7, color=MUTED)
        ax.set_xticks(range(len(CORNERS)), [CORNER_LABEL[c] for c in CORNERS], rotation=15)
        ax.set_title({"cam0": "cam0 solvePnP", "cam1": "cam1 solvePnP", "joint": "joint stereo"}[s])
        ax.set_yscale("log")
    axs[0].set_ylabel("hold jitter, 3D SD (mm)\nmedian, bar = IQR")
    h = [plt.Line2D([], [], marker="o", ls="", color=MUTED, mfc=MUTED, ms=7, label="rigid body (jig)"),
         plt.Line2D([], [], marker="o", ls="", color=MUTED, mfc="white", mew=2, ms=7, label="tag 12 only")]
    fig.legend(handles=h, loc="upper right", ncol=2)
    fig.suptitle(f"T1 static jitter, {WINDOW}-frame windows", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(DST / "summary.png", dpi=160)
    plt.close(fig)


def fig_distance(bd):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, s in zip(axs, SOLVERS):
        for c in CORNERS:
            for rb in RBS:
                d = bd[(bd.corners == c) & (bd.rb == rb) & (bd.solver == s) & (bd.n_holds >= 3)]
                ax.plot(d.depth_mid_m, d.jitter_3d_med_mm, marker="o", color=COLORS[c], lw=2, ms=5,
                        ls="-" if rb == "jig" else "--", mfc=COLORS[c] if rb == "jig" else "white")
        d = bd[(bd.corners == REF[0]) & (bd.n_holds >= 3)]
        if s == "joint":
            ax.plot(d.depth_mid_m, d.jitter_3d_med_mm, "-", color=INK, lw=1)
        d = bd[(bd.corners == "logged") & (bd.rb == "jig") & (bd.solver == "joint") & (bd.n_holds >= 3)]
        ax.plot(d.depth_mid_m, d.mocap_3d_med_mm, ":", color=MUTED, lw=1.5)
        ax.set_title({"cam0": "cam0 solvePnP", "cam1": "cam1 solvePnP", "joint": "joint stereo"}[s])
        ax.set_xlabel("grip depth from cam0 (m)")
        ax.set_yscale("log")
    axs[0].set_ylabel("hold jitter, 3D SD (mm), median per 50 mm bin")
    h = [plt.Line2D([], [], color=COLORS[c], lw=2, label=CORNER_LABEL[c]) for c in CORNERS]
    h += [plt.Line2D([], [], color=MUTED, lw=2, ls="-", label="rigid body"),
          plt.Line2D([], [], color=MUTED, lw=2, ls="--", label="tag 12 only"),
          plt.Line2D([], [], color=INK, lw=1, label="Diwakar pipeline"),
          plt.Line2D([], [], color=MUTED, lw=1.5, ls=":", label="mocap")]
    fig.legend(handles=h, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(DST / "vs_distance.png", dpi=160)
    plt.close(fig)


def fig_axes(sm):
    keys = [(c, rb, "joint") for c in CORNERS for rb in RBS] + [REF]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, a in zip(axs, AXES):
        vals = [sm[(sm.corners == k[0]) & (sm.rb == k[1]) & (sm.solver == k[2])][f"jitter_{a}_med_mm"].iloc[0] for k in keys]
        cols = [COLORS.get(k[0], INK) for k in keys]
        bars = ax.bar(range(len(keys)), vals, color=cols, edgecolor="white", linewidth=2)
        for b, k in zip(bars, keys):
            if k[1] == "tag12":
                b.set_hatch("///")
                b.set_facecolor("white")
                b.set_edgecolor(COLORS[k[0]])
        for x, v in enumerate(vals):
            ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_xticks(range(len(keys)), [f"{k[0]}\n{k[1]}" if k != REF else "Diwakar\npipeline" for k in keys], fontsize=6.5)
        ax.set_title(f"mocap {AXIS_LABEL[a]}")
        ax.set_yscale("log")
    axs[0].set_ylabel("hold jitter SD (mm), joint stereo")
    fig.tight_layout()
    fig.savefig(DST / "axes.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.DataFrame([r for name in TAKES for r in hold_rows(name)])
    moved = df.mocap_3d_mm > MOCAP_MOVED_MM
    print(f"holds flagged moved (mocap window SD > {MOCAP_MOVED_MM} mm): {df.loc[moved, ['take', 'hold']].drop_duplicates().shape[0]}")
    df["moved"] = moved
    df.to_csv(DST / "holds.csv", index=False)
    use = df[~df.moved]
    sm = summarize(use)
    sm.to_csv(DST / "summary.csv", index=False)
    bd = by_depth(use)
    bd.drop(columns="bin").to_csv(DST / "vs_distance.csv", index=False)
    floor = float(use.mocap_3d_mm.median())
    fig_summary(sm, floor)
    fig_distance(bd)
    fig_axes(sm)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(sm[["corners", "rb", "solver", "n_holds", "jitter_3d_med_mm", "jitter_3d_p90_mm", "jitter_x_med_mm",
                  "jitter_y_med_mm", "jitter_z_med_mm", "jitter_rot_med_deg", "drift_3d_med_mm"]].round(4).to_string(index=False))
        print(f"mocap floor (median window 3D SD): {floor:.4f} mm")
