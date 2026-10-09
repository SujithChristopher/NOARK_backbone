"""Camera grip point vs mocap, X and Z error, for every take, three corner sources x three solve methods.

Sources: logged (tracker's corners.csv), none / contour (re-detected, detect_corners.py).
Methods: cam0, cam1 (single camera PnP), joint (stereo). T0 jig throughout.
Only samples that have recorded images are used, so all sources see the same frames
(T1: every sample, T2/T3: every 2nd sample).

Registration per (source, method): rigid camera->mocap + grip offset on the mocap body, fit on the
T1 grid holds. T1 errors are held-out across runs (fit r1 -> test r2 and back); T2/T3 use the fit
on both T1 runs, frozen.

Outputs: out/summary_errors.csv, out/plots/<take>.png, out/plots/summary.png.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_T1 import TRIM_S, fit, load_holds, stats
from common import DATA, OUT, Rotation, load_mocap, mocap_frame_of, take_dir

import sys  # noqa: E402

# study "corners": corner refinement (each source with its own T0 jig)
# study "jig": rigid body (jig) variants, all on the tracker's logged corners
# study "jig_april": with / without rigid body on rapidtag apriltag-refined corners
STUDY = sys.argv[1] if len(sys.argv) > 1 else "corners"
SOURCES = {"corners": ["logged", "none", "contour", "april"], "jig": ["tag12only", "shipped", "logged", "T0all"],
           "jig_april": ["april_tag12only", "april"]}[STUDY]
LABEL = {"logged": "logged" if STUDY == "corners" else "T0 jig (10-03)", "none": "none", "contour": "contour", "april": "apriltag (rapidtag)" if STUDY == "corners" else "T0 jig",
         "tag12only": "tag 12 only", "april_tag12only": "tag 12 only", "shipped": "shipped jig", "T0all": "T0 jig (all T0)"}
METHODS = ["cam0", "cam1", "joint"]
COLORS = dict(zip(SOURCES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
PLOTS = OUT / ("plots" if STUDY == "corners" else f"plots_{STUDY}")
SUFFIX = "" if STUDY == "corners" else f"_{STUDY}"
NS = len(SOURCES)
BW = 0.8 / NS


def off(i):
    return (i - (NS - 1) / 2) * BW
INK, MUTED = "#0b0b0b", "#8a8984"
T1 = ["2026-10-03_T1_grid_r1", "2026-10-03_T1_grid_r2"]
DYN = sorted(p.name for p in DATA.glob("2026-*") if "_T1_" in p.name or "_T0_" in p.name) and \
    sorted(p.name for p in DATA.glob("2026-*") if "_T2_" in p.name or "_T3_" in p.name)
JUMP_M = 0.02  # mocap step > 20 mm in 10 ms = teleport / glitch
PLOTS.mkdir(exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "legend.frameon": False})


def load_poses(name, source):
    p = pd.read_csv(OUT / "poses" / source / f"{name}.csv")
    img = pd.read_csv(take_dir(name) / "images.csv", usecols=["sample"])["sample"].unique()
    p = p[p["sample"].isin(img)].reset_index(drop=True)
    if STUDY.startswith("jig"):  # score every jig on the frames where tag 12 alone also gives a pose
        ref_dir = "tag12only" if STUDY == "jig" else "april_tag12only"
        ref = pd.read_csv(OUT / "poses" / ref_dir / f"{name}.csv").set_index("sample")
        for m in METHODS:
            have = p["sample"].map(ref[f"{m}_gx"].notna()).fillna(False).to_numpy(bool)
            p.loc[~have, [c for c in p.columns if c.startswith(f"{m}_")]] = np.nan
    return p


def hold_table(name, source):
    poses = load_poses(name, source)
    mc = load_mocap(name)
    rows = []
    for i, h in load_holds(name).iterrows():
        a, b = h.t0 + TRIM_S, h.t1 - TRIM_S
        w = poses[(poses.t_cam0_s >= a) & (poses.t_cam0_s <= b)]
        f0, f1 = mocap_frame_of(name, np.array([a, b]))
        fr = np.arange(int(np.ceil(f0)), int(np.floor(f1)) + 1)
        fr = fr[mc["valid"][fr]]
        row = {"take": name, "hold": i}
        for m in METHODS:
            v = w[[f"{m}_gx", f"{m}_gy", f"{m}_gz"]].dropna().to_numpy()
            med = np.median(v, 0) if len(v) >= 20 else np.full(3, np.nan)
            row.update({f"{m}_{ax}": val for ax, val in zip("xyz", med)})
        p = mc["p"][fr]
        row.update({"mc_x": p[:, 0].mean(), "mc_y": p[:, 1].mean(), "mc_z": p[:, 2].mean()})
        row.update({f"mc_q{c}": v for c, v in zip("xyzw", Rotation.from_matrix(mc["R"][fr]).mean().as_quat())})
        rows.append(row)
    return pd.DataFrame(rows)


def to_world(g, x):
    return g @ Rotation.from_rotvec(x[:3]).as_matrix().T + x[3:6]


def mocap_at(name, t, x):
    """Mocap grip point (body offset x[6:9]) at camera times t via GPIO sync; NaN near gaps / jumps."""
    mc = load_mocap(name)
    f = mocap_frame_of(name, t)
    v = mc["valid"].copy()
    jump = np.r_[False, np.linalg.norm(np.diff(mc["p"], axis=0), axis=1) > JUMP_M]
    v &= ~jump & ~np.r_[jump[1:], False]
    idx = np.flatnonzero(v)
    q = mc["p"] + np.einsum("nij,j->ni", np.nan_to_num(mc["R"]), x[6:9])
    out = np.full((len(t), 3), np.nan)
    ok = np.isfinite(f) & (f >= idx[0]) & (f <= idx[-1])
    lo = np.floor(f[ok]).astype(int)
    good = v[lo] & v[np.minimum(lo + 1, len(v) - 1)]
    k = np.flatnonzero(ok)[good]
    fr = f[k]
    a = np.floor(fr).astype(int)
    w = (fr - a)[:, None]
    p = q[a] * (1 - w) + q[np.minimum(a + 1, len(q) - 1)] * w
    out[k] = p
    return out


def lag_scan(name, poses, x, m, lags=range(-30, 31, 2)):
    """Diagnostic only: XZ RMSE vs shift (mocap frames) of the GPIO alignment. Expect minimum at 0."""
    t = poses.t_cam0_s.to_numpy()
    g = to_world(poses[[f"{m}_gx", f"{m}_gy", f"{m}_gz"]].to_numpy(), x)
    res = []
    for L in lags:
        mw = mocap_at(name, t + L * 0.01, x)
        e = (g - mw)[:, [0, 2]]
        e = e[np.isfinite(e).all(1)]
        res.append(np.sqrt((e**2).sum(1).mean()) * 1000)
    return list(lags)[int(np.argmin(res))], res


# ---------- registration on T1 holds ----------
holds = {s: pd.concat([hold_table(n, s) for n in T1], ignore_index=True) for s in SOURCES}
def ok_rows(h, m):
    return h[np.isfinite(h[[f"{m}_x", f"{m}_y", f"{m}_z"]].to_numpy()).all(1)]


regs = {(s, m, f): fit(ok_rows(holds[s] if f == "both" else holds[s][holds[s]["take"] == f], m), m)
        for s in SOURCES for m in METHODS for f in T1 + ["both"]}

rows = []
# ---------- T1: held-out across runs ----------
for ev in T1:
    other = [t for t in T1 if t != ev][0]
    fig, ax = plt.subplots(2, max(3, NS), figsize=(5 * max(3, NS), 8.5), constrained_layout=True)
    errs = {}
    for s in SOURCES:
        for m in METHODS:
            h = ok_rows(holds[s][holds[s]["take"] == ev], m)
            x = regs[(s, m, other)]
            g = to_world(h[[f"{m}_x", f"{m}_y", f"{m}_z"]].to_numpy(), x)
            Rb = Rotation.from_quat(h[["mc_qx", "mc_qy", "mc_qz", "mc_qw"]].to_numpy()).as_matrix()
            mw = h[["mc_x", "mc_y", "mc_z"]].to_numpy() + np.einsum("nij,j->ni", Rb, x[6:9])
            e = (g - mw) * 1000
            errs[(s, m)] = (mw * 1000, e)
            r = {"take": ev, "task": "T1", "source": s, "method": m, "n": len(e), "lag_best": np.nan}
            for ax_, k in (("x", 0), ("z", 2)):
                r.update({f"{ax_}_{q}": v for q, v in stats(e[:, k] / 1000).items()})
            r["xz_rmse_mm"] = float(np.sqrt((e[:, [0, 2]] ** 2).sum(1).mean()))
            rows.append(r)
    for j, s in enumerate(SOURCES):  # error maps, joint
        mw, e = errs[(s, "joint")]
        a = ax[0, j]
        a.quiver(mw[:, 0], mw[:, 2], e[:, 0], e[:, 2], angles="xy", scale_units="xy", scale=0.05,
                 color=COLORS[s], width=0.004)
        a.scatter(mw[:, 0], mw[:, 2], s=6, color=MUTED, zorder=3)
        a.set_title(f"joint, {LABEL[s]}: hold error (arrows x20)")
        a.set_xlabel("mocap X (mm)")
        a.set_ylabel("mocap Z (mm)")
        a.set_aspect("equal")
    for j, (lab, k) in enumerate((("X", 0), ("Z", 2))):
        a = ax[1, j]
        for m_i, m in enumerate(METHODS):
            for s_i, s in enumerate(SOURCES):
                e = errs[(s, m)][1][:, k]
                a.bar(m_i + off(s_i), np.sqrt((e**2).mean()), BW * 0.92, color=COLORS[s],
                      label=LABEL[s] if m_i == 0 else None)
        a.set_xticks(range(3), METHODS)
        a.set_ylabel(f"{lab} RMSE (mm)")
        a.set_title(f"{lab} error, held-out registration (fit on {other[-2:]})")
        a.legend(title=STUDY)
    a = ax[1, 2]
    for s in SOURCES:
        e = np.hypot(*errs[(s, "joint")][1][:, [0, 2]].T)
        a.plot(np.sort(e), np.linspace(0, 1, len(e)), color=COLORS[s], lw=2, label=LABEL[s])
    a.set_xlabel("joint XZ error per hold (mm)")
    a.set_ylabel("fraction of holds")
    a.set_title("cumulative distribution, joint")
    a.legend(title=STUDY)
    for a in ax[1, 3:]:
        a.axis("off")
    fig.suptitle(f"{ev}: grid holds, camera grip point vs mocap", fontsize=12)
    fig.savefig(PLOTS / f"{ev}.png", dpi=110)
    plt.close(fig)
    print("plotted", ev, flush=True)

# ---------- T2 / T3: frozen T1 registration ----------
for name in DYN:
    fig = plt.figure(figsize=(5 * max(3, NS), 10), constrained_layout=True)
    gs = fig.add_gridspec(3, max(3, NS))
    series = {}
    for s in SOURCES:
        poses = load_poses(name, s)
        for m in METHODS:
            x = regs[(s, m, "both")]
            t = poses.t_cam0_s.to_numpy() if m != "cam1" else poses.t_cam1_s.to_numpy()
            g = to_world(poses[[f"{m}_gx", f"{m}_gy", f"{m}_gz"]].to_numpy(), x)
            mw = mocap_at(name, t, x)
            ok = np.isfinite(g).all(1) & np.isfinite(mw).all(1)
            e = (g - mw)[ok] * 1000
            series[(s, m)] = (t[ok] - t[0], g[ok] * 1000, mw[ok] * 1000, e)
            lag = lag_scan(name, poses, x, m)[0] if m == "joint" else np.nan
            r = {"take": name, "task": name.split("_")[1], "source": s, "method": m, "n": int(ok.sum()), "lag_best": lag}
            for ax_, k in (("x", 0), ("z", 2)):
                r.update({f"{ax_}_{q}": v for q, v in stats(e[:, k] / 1000).items()})
            r["xz_rmse_mm"] = float(np.sqrt((e[:, [0, 2]] ** 2).sum(1).mean()))
            rows.append(r)
    for j, s in enumerate(SOURCES):  # paths
        a = fig.add_subplot(gs[0, j])
        tt, g, mw, e = series[(s, "joint")]
        a.plot(mw[:, 0], mw[:, 2], color=INK, lw=2, label="mocap")
        a.plot(g[:, 0], g[:, 2], color=COLORS[s], lw=1, label=f"joint, {LABEL[s]}")
        a.set_aspect("equal")
        a.set_xlabel("X (mm)")
        a.set_ylabel("Z (mm)")
        a.set_title(f"path: joint, {LABEL[s]}")
        a.legend()
    for i, (lab, k) in enumerate((("X", 0), ("Z", 2))):
        a = fig.add_subplot(gs[1, i])
        for s in SOURCES:
            tt, _, _, e = series[(s, "joint")]
            a.plot(tt, e[:, k], color=COLORS[s], lw=1, label=LABEL[s])
        a.axhline(0, color=MUTED, lw=0.8)
        a.set_xlabel("time (s)")
        a.set_ylabel(f"{lab} error (mm)")
        a.set_title(f"joint {lab} error over time")
        a.legend(title=STUDY)
    a = fig.add_subplot(gs[1, 2])
    tt, _, mw, _ = series[(SOURCES[-1], "joint")]
    a.plot(tt, mw[:, 0] - mw[:, 0].mean(), color=INK, lw=1.5, label="X")
    a.plot(tt, mw[:, 2] - mw[:, 2].mean(), color=MUTED, lw=1.5, label="Z")
    a.set_xlabel("time (s)")
    a.set_ylabel("mocap position, centred (mm)")
    a.set_title("movement (mocap)")
    a.legend()
    for j, (lab, k) in enumerate((("X", 0), ("Z", 2))):
        a = fig.add_subplot(gs[2, j])
        for m_i, m in enumerate(METHODS):
            for s_i, s in enumerate(SOURCES):
                e = series[(s, m)][3][:, k]
                a.bar(m_i + off(s_i), np.sqrt((e**2).mean()), BW * 0.92, color=COLORS[s], label=LABEL[s] if m_i == 0 else None)
        a.set_xticks(range(3), METHODS)
        a.set_ylabel(f"{lab} RMSE (mm)")
        a.set_title(f"{lab} RMSE by method (registration frozen from T1)")
        a.legend(title=STUDY)
    a = fig.add_subplot(gs[2, 2])
    for s in SOURCES:
        e = np.hypot(*series[(s, "joint")][3][:, [0, 2]].T)
        a.plot(np.sort(e), np.linspace(0, 1, len(e)), color=COLORS[s], lw=2, label=LABEL[s])
    a.set_xlabel("joint XZ error per sample (mm)")
    a.set_ylabel("fraction of samples")
    a.set_title("cumulative distribution, joint")
    a.legend(title=STUDY)
    fig.suptitle(f"{name}: camera grip point vs mocap", fontsize=12)
    fig.savefig(PLOTS / f"{name}.png", dpi=110)
    plt.close(fig)
    print("plotted", name, flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / f"summary_errors{SUFFIX}.csv", index=False)

# ---------- summary across takes ----------
fig, ax = plt.subplots(3, 1, figsize=(15, 11), constrained_layout=True)
takes = list(dict.fromkeys(res["take"]))
for i, m in enumerate(METHODS):
    a = ax[i]
    for s_i, s in enumerate(SOURCES):
        v = res[(res.method == m) & (res.source == s)].set_index("take").loc[takes, "xz_rmse_mm"]
        a.bar(np.arange(len(takes)) + off(s_i), v, BW * 0.92, color=COLORS[s], label=LABEL[s])
    a.set_xticks(range(len(takes)), [t.replace("2026-10-03_", "") for t in takes], rotation=40, ha="right")
    a.set_ylabel("XZ RMSE (mm)")
    a.set_title(f"{m}: XZ RMSE per take")
    a.legend(title=STUDY, ncol=3)
fig.savefig(PLOTS / "summary.png", dpi=110)
pd.set_option("display.width", 250)
piv = res.pivot_table(index=["task"], columns=["method", "source"], values="xz_rmse_mm", aggfunc="mean")
print(piv.round(2).to_string())
print(res.pivot_table(index="task", columns=["method", "source"], values=["x_rmse_mm", "z_rmse_mm"], aggfunc="mean").round(2).T.to_string())
print("lag_best (joint):", res[res.method == "joint"].groupby("source").lag_best.describe()[["mean", "min", "max"]].to_string())
