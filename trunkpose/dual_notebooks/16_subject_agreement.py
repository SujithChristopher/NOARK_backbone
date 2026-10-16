# %% 16_subject_agreement.py
# Group agreement figure: camera ICP vs mocap trunk angles across every subject of a
# session, pooled from the per-subject icp_vs_mocap.npz files written by
# 15_subject_icp.py.
#
#   <output>/icp_vs_mocap_agreement.png   scatter + Bland-Altman, 3 axes x N subjects
#   <output>/subject_rmse_bar.png         anonymized per-subject RMSE bar chart
#   <output>/error_distribution.png       per-subject |error| boxplots, all axes
#   <output>/agreement_summary.csv        per-subject per-axis r / RMSE / bias / LoA
#
# Only frames where BOTH systems are valid are scored -- the ICP series carries NaN
# wherever registration was rejected, and those frames are dropped here rather than
# interpolated, so a subject with poor coverage shows up as a small n, not as a
# smooth line that was never measured.
#
# --condition picks which arm of the session to pool (sep23 recorded every subject
# twice, _wc with trunk compensation and _woc without) and suffixes the output files so
# the two never overwrite each other.
#
#   uv run python trunkpose/dual_notebooks/16_subject_agreement.py
#   uv run python trunkpose/dual_notebooks/16_subject_agreement.py --condition woc

import argparse
import csv
import importlib
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

# The three anatomical axes plus the total rotation magnitude. Magnitude is the
# frame-invariant one: it needs no residual camera<->mocap rotation S, so it survives
# even when the per-axis split is unstable (see S_diff in the summary table).
AXES = ("flexion", "lateral", "axial", "magnitude")
# Okabe-Ito, colorblind-safe; one colour per subject, reused in both rows.
SUBJ_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
               "#56B4E9", "#F0E442", "#000000"]
MAX_SCATTER = 1500      # points drawn per subject per axis (all points are still scored)
# flexion / lateral / axial, used by both the bar chart and the boxplots
AXIS_COLORS = ("#0072B2", "#009E73", "#D55E00")
# Which camera method is scored against mocap (--method). "sam3d" reads the teacher's
# <rec>/sam3d/sam3d_vs_mocap.npz and "student" reads <student_dir>/<take>/student/
# sam3d_vs_mocap.npz (both written by 21_sam3d_score.py); their sam_* series are renamed
# to icp_* so every figure below runs unchanged. Set in main().
SOURCE = {"method": "icp", "student_dir": None}
METHOD_LABEL = {"icp": "camera ICP", "sam3d": "SAM 3D Body", "student": "miniSAM3D student"}


def load_scored(rec, subject, result, take=""):
    """The per-take result npz for SOURCE["method"], with the camera series as icp_*.
    None when that method has no result for this take. `take` is the _<take> suffix of a
    game take that shares its recording with another (15.game_take)."""
    method = SOURCE["method"]
    if method == "icp":
        f = rec / f"icp_vs_mocap{result}{take}.npz"
    elif method == "sam3d":
        f = rec / "sam3d" / f"sam3d_vs_mocap{take}.npz"
    else:
        f = Path(SOURCE["student_dir"]) / subject / "student" / f"sam3d_vs_mocap{take}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    if method == "icp":
        return d
    out = {k.replace("sam_", "icp_", 1) if k.startswith("sam_") else k: d[k] for k in d.files}
    # 21 stores the S disagreement as S_diff and no fitted-S angle or frame count
    out.update(S_diff_deg=float(d["S_diff"]), S_deg=float("nan"), n_frames=len(d["in_win"]))
    return out


def load_subjects(session_dir, subjects, condition="wc", game=False, result=""):
    """(name, npz) per subject that actually has results, in the given order.

    The game takes keep their npz beside their frames, five levels down in the
    recorder's tree, so 15 resolves the path for both layouts.
    """
    import toml

    m15 = importlib.import_module("15_subject_icp")
    if not subjects:
        if game:
            subjects = m15.game_subjects(session_dir)
        elif m15.game_take(session_dir, f"_{condition}"):
            # condition names a game take (oct9/oct10 [game] takes: game, game_wtc)
            subjects = [s for s in m15.game_subjects(session_dir)
                        if m15.game_take(session_dir, s)[1] == condition]
        else:
            cfg_path = session_dir / "session.toml"
            cfg = toml.load(cfg_path) if cfg_path.exists() else {}
            # oct9/oct10 keep [subjects] as one display string, not wc/woc lists
            listed = cfg.get("subjects", {})
            subjects = listed.get(condition, []) if isinstance(listed, dict) else []
            if not subjects:
                take_root = (session_dir / "std_movement"
                             if (session_dir / "std_movement").is_dir() else session_dir)
                subjects = sorted(
                    p.parent.name
                    for p in take_root.glob(f"*_{condition}/icp_vs_mocap{result}.npz"))
    out = []
    for s in subjects:
        take = m15.game_take(session_dir, s)
        d = load_scored(m15.resolve_recording(session_dir, s, game)[0], s, result,
                        f"_{take[1]}" if take else "")
        if d is None:
            print(f"skip {s}: no {SOURCE['method']} result")
            continue
        out.append((s, d))
    return out


def load_sessions(session_dirs, subjects, condition="wc", game=False, result=""):
    """load_subjects over several sessions (oct9 + oct10 are one protocol split
    across two days), concatenated in the order the sessions are given."""
    out = []
    for sd in session_dirs:
        names = subjects
        if subjects:
            # an explicit list spans every session: keep the ones recorded in this one
            m15 = importlib.import_module("15_subject_icp")
            names = [s for s in subjects if m15.resolve_recording(sd, s, game)[0].is_dir()]
            if not names:
                continue
        out += load_subjects(sd, names, condition, game, result)
    if not out:
        raise SystemExit("No per-subject results under "
                         + ", ".join(str(sd) for sd in session_dirs))
    return out


def paired(d, axis):
    """(mocap, icp) on frames where both systems are valid inside the mocap window.

    `valid_*` marks the frames ICP actually solved; the stored series are smoothed,
    and smoothing interpolates across the gaps, so filtering on isfinite alone would
    silently score interpolation against mocap."""
    a, b = d[f"icp_{axis}"], d[f"moc_{axis}"]
    m = d[f"valid_{axis}"] & np.isfinite(b) & d["in_win"]
    return b[m], a[m]


def stats(moc, icp):
    d = icp - moc
    # RMSE is dominated by the occluded tail, so the median and p95 of |error| are
    # reported beside it: the median is what a typical frame costs, p95 the bad ones.
    ad = np.abs(d)
    return {"n": len(d), "r": float(np.corrcoef(moc, icp)[0, 1]),
            "rmse": float(np.sqrt((d ** 2).mean())),
            "median_abs": float(np.median(ad)), "p95_abs": float(np.percentile(ad, 95)),
            "bias": float(d.mean()), "sd": float(d.std(ddof=1)),
            "rom_moc": float(np.percentile(moc, 97.5) - np.percentile(moc, 2.5)),
            "rom_icp": float(np.percentile(icp, 97.5) - np.percentile(icp, 2.5)),
            # slope of camera on mocap: < 1 means the camera under-reads the range
            "slope": float(np.polyfit(moc, icp, 1)[0])}


def abs_errors(d, axis):
    moc, icp = paired(d, axis)
    return np.abs(icp - moc) if len(moc) >= 50 else None


def short_name(subject):
    """3-letter tick label from a take name: s3_jerome_wc -> Jer, s3_js_woc -> JS."""
    stem = subject.removeprefix("s3_")
    for take in ("_game_wtc", "_game", "_woc", "_wc"):   # longest first
        if stem.endswith(take):
            stem = stem[: -len(take)]
            break
    return stem.upper() if len(stem) <= 2 else stem[:3].capitalize()


def error_boxplots(data, anon, out_png, title, other=None, cond_names=None, ymax=None):
    """|error| per subject, with the anatomical axes side by side inside each subject.

    Box = IQR, line = median (printed), whiskers = 5th/95th percentile. Fliers are left
    off: with ~1500 frames per subject they draw as a solid smear, and the 95th whisker
    already carries the tail that RMSE is made of. These are frames, not independent
    samples -- consecutive frames are correlated -- so a box describes the error
    distribution within a take, not the spread between subjects.

    With `other`, every box is doubled: the first condition solid, the second hatched,
    matched to the same subject by name stem.
    """
    paired_mode = other is not None
    by_stem = ({s.rsplit("_", 1)[0]: d for s, d in other} if paired_mode else {})
    box_axes = [a for a in AXES if a != "magnitude"]
    per_subject = len(box_axes) * (2 if paired_mode else 1)
    step = 0.30 if paired_mode else 0.46          # spacing between boxes in a group
    span = step * (per_subject - 1)
    group = span + (1.05 if paired_mode else 0.85)  # subject-to-subject spacing

    # ~2.6 in per subject when the conditions are paired (six boxes), ~1.7 otherwise
    per_group_in = 2.6 if paired_mode else 1.7
    fig, ax = plt.subplots(figsize=(max(7.0, 1.8 + per_group_in * len(data)), 5.2))
    series, faces, pos, ticks, tick_pos = [], [], [], [], []
    for j, (subj, d) in enumerate(data):
        base = group * j
        slot = 0
        for ax_name, col in zip(box_axes, AXIS_COLORS):
            sources = [(d, False)]
            if paired_mode:
                twin = by_stem.get(subj.rsplit("_", 1)[0])
                sources = [(d, False), (twin, True)] if twin is not None else sources
            for src, hatched in sources:
                e = abs_errors(src, ax_name)
                offset = -span / 2 + step * slot
                slot += 1
                if e is None:
                    continue
                series.append(e)
                faces.append((col, hatched))
                pos.append(base + offset)
        ticks.append(anon[subj])
        tick_pos.append(base)
    if not series:
        raise SystemExit("nothing to plot")

    bp = ax.boxplot(series, positions=pos, widths=step * 0.82, whis=(5, 95),
                    showfliers=False, patch_artist=True,
                    medianprops={"color": "black", "lw": 1.5})
    for patch, (col, hatched) in zip(bp["boxes"], faces):
        patch.set_facecolor(col)
        patch.set_edgecolor(col)
        patch.set_alpha(0.45 if hatched else 0.85)
        if hatched:
            patch.set_hatch("///")
    # median printed above each box, on a plate so it stays legible over the whisker
    for x_, e in zip(pos, series):
        ax.annotate(f"{np.median(e):.1f}", (x_, np.percentile(e, 75)),
                    textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7,
                    bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none",
                          "alpha": 0.8})
    ax.set_xticks(tick_pos, ticks)
    ax.set_xlabel("Subject")
    ax.set_ylabel("absolute error (degrees)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Patch(facecolor=c, alpha=0.85, label=a.capitalize())
               for a, c in zip(box_axes, AXIS_COLORS)]
    if paired_mode:
        handles += [Patch(facecolor="0.35", alpha=0.85, label=cond_names[0]),
                    Patch(facecolor="0.35", alpha=0.45, hatch="///",
                          label=cond_names[1])]
    ax.legend(handles=handles, frameon=False, ncol=len(handles), loc="upper left",
              fontsize=9)
    # a fixed ymax keeps figures from different sessions/conditions comparable
    ax.set_ylim(0, ymax if ymax else ax.get_ylim()[1] * 1.08)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"-> {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", nargs="+", default=[str(PROJECT_ROOT / "data" / "sep23_26")],
                    help="one or more session dirs; several are pooled as one cohort "
                         "and written to <their common parent>/output")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--condition", default="wc",
                    help="which [subjects] list in session.toml to pool (wc / woc)")
    ap.add_argument("--suffix", default=None,
                    help="output filename suffix (default: '' for wc, else _<condition>)")
    ap.add_argument("--vs", default=None,
                    help="second condition to draw beside the first in the boxplots")
    ap.add_argument("--exclude", nargs="*", default=(),
                    help="subjects to drop (e.g. a take whose mocap frame is unusable)")
    ap.add_argument("--game", action="store_true",
                    help="pool the NOARK game takes under <session>/noark_game")
    ap.add_argument("--shell", default="mode", choices=("mode", "front"),
                    help="which 15 results to pool: the default 'mode' shell "
                         "(icp_vs_mocap_mode.npz) or the original 'front' one")
    ap.add_argument("--ymax", type=float, default=25.0,
                    help="error-distribution y-axis limit in degrees (0 = auto)")
    ap.add_argument("--names", action="store_true",
                    help="label subjects by 3-letter name (Jer, Lee, JS) instead of Sub1..")
    ap.add_argument("--method", default="icp", choices=tuple(METHOD_LABEL),
                    help="camera method scored against mocap")
    ap.add_argument("--student-dir",
                    default=str(PROJECT_ROOT / "trunkpose" / "runs" / "minisam3d"
                                / "r18_v2_all_octtest"),
                    help="--method student: run folder holding <take>/student/sam3d_vs_mocap.npz")
    a = ap.parse_args()
    SOURCE.update(method=a.method, student_dir=a.student_dir)
    mlabel = METHOD_LABEL[a.method]
    # mirrors 06.RESULT_SUFFIX; outputs carry it too, so the two never overwrite
    result = "" if a.shell == "front" else f"_{a.shell}"
    session_dirs = [Path(s_).resolve() for s_ in a.session]
    session_dir = (session_dirs[0] if len(session_dirs) == 1
                   else Path(os.path.commonpath(session_dirs)))
    out_dir = (session_dir / "noark_game" / "output") if a.game else (session_dir / "output")
    session_name = "+".join(sd.name for sd in session_dirs)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = a.suffix if a.suffix is not None else (
        "_game" if a.game else "" if a.condition == "wc" else f"_{a.condition}") + (
        result if a.method == "icp" else f"_{a.method}")
    cond = "game" if a.game else a.condition

    m12 = importlib.import_module("12_poster_figures")
    m12.poster_style()

    data = load_sessions(session_dirs, a.subjects, a.condition, a.game, result)
    dropped = [s_ for s_, _ in data if s_ in a.exclude]
    if dropped:
        print(f"excluded: {', '.join(dropped)}")
        data = [(s_, d) for s_, d in data if s_ not in a.exclude] or data
    # the wc-vs-woc figure spans both conditions, so it cannot use `suffix` (which
    # names one of them); it carries the exclusion in its own name instead.
    ex_tag = ("_ex_" + "_".join(sorted(s_.rsplit("_", 1)[0] for s_ in dropped))
              if dropped else "")
    label = {s: s.replace("_wc", "").replace("_woc", "") for s, _ in data}
    rng = np.random.default_rng(0)

    rows, pooled = [], {ax: ([], []) for ax in AXES}
    for s, d in data:
        for ax in AXES:
            moc, icp = paired(d, ax)
            if len(moc) < 50:
                print(f"{s} {ax}: only {len(moc)} paired frames -- skipped")
                continue
            pooled[ax][0].append(moc)
            pooled[ax][1].append(icp)
            st = stats(moc, icp)
            rows.append(dict(subject=s, axis=ax, **st,
                             loa_lo=st["bias"] - 1.96 * st["sd"],
                             loa_hi=st["bias"] + 1.96 * st["sd"],
                             coverage=float(d[f"valid_{ax}"][d["in_win"]].mean()),
                             frames=int(d["n_frames"]), S_deg=float(d["S_deg"]),
                             S_diff_deg=float(d["S_diff_deg"])))

    # ---- summary table ----
    csv_path = out_dir / f"agreement_summary{suffix}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'subject':<12}{'axis':<10}{'n':>6}{'cov':>6}{'r':>7}{'RMSE':>7}"
          f"{'med':>7}{'p95':>7}{'bias':>7}{'slope':>7}{'LoA':>16}")
    for r in rows:
        print(f"{r['subject']:<12}{r['axis']:<10}{r['n']:>6}{r['coverage'] * 100:>5.0f}%"
              f"{r['r']:>+7.2f}{r['rmse']:>6.1f}d{r['median_abs']:>6.1f}d"
              f"{r['p95_abs']:>6.1f}d{r['bias']:>+6.1f}d{r['slope']:>7.2f}"
              f"   [{r['loa_lo']:+5.1f},{r['loa_hi']:+5.1f}]")
    flagged = sorted({r["subject"] for r in rows if r["S_diff_deg"] > 30})
    if flagged:
        print("\nAxis split unreliable (fitted S disagrees with the neutral-pose S by "
              f"more than 30 deg): {', '.join(flagged)}. Their magnitude row still "
              "holds -- it does not depend on S.")
    print(f"-> {csv_path}")

    # ---- per-subject RMSE for the anatomical axes ----
    # Preserve the configured subject order. Ticks are anonymous (Sub1..) unless
    # --names; the console mapping makes either output reproducible.
    anon = {s: (short_name(s) if a.names else f"Sub{j + 1}") for j, (s, _) in enumerate(data)}
    print("tick labels: " + ", ".join(f"{anon[s]}={s}" for s, _ in data))
    bar_axes = ("flexion", "lateral", "axial")
    row_by_key = {(r["subject"], r["axis"]): r for r in rows}
    x = np.arange(len(data), dtype=float)
    width = 0.24

    fig_bar, ax_bar = plt.subplots(figsize=(9.2, 5.5))
    for k, (ax_name, color) in enumerate(zip(bar_axes, AXIS_COLORS)):
        values = [row_by_key[(s, ax_name)]["rmse"] for s, _ in data]
        bars = ax_bar.bar(x + (k - 1) * width, values, width,
                          label=ax_name.capitalize(), color=color)
        ax_bar.bar_label(bars, labels=[f"{v:.1f}" for v in values],
                         padding=3, fontsize=9)

    ax_bar.set_xticks(x, [anon[s] for s, _ in data])
    ax_bar.set_ylabel("RMSE (degrees)")
    ax_bar.set_xlabel("Subject")
    ax_bar.set_title(f"Per-subject trunk-angle error, {mlabel} vs mocap")
    ax_bar.legend(frameon=False, ncol=len(bar_axes), loc="upper left")
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.set_ylim(0, ax_bar.get_ylim()[1] * 1.12)
    fig_bar.tight_layout()
    bar_png = out_dir / f"subject_rmse_bar{suffix}.png"
    fig_bar.savefig(bar_png)
    plt.close(fig_bar)
    print(f"-> {bar_png}")

    # ---- figure: |error| distributions, all subjects together ----
    error_boxplots(data, anon, out_dir / f"error_distribution{suffix}.png",
                   f"{mlabel} vs mocap: trunk-angle error - {len(data)} subjects "
                   f"({session_name}, {cond})", ymax=a.ymax)
    if a.vs:
        error_boxplots(data, anon,
                       out_dir / f"error_distribution_{a.condition}_vs_{a.vs}{ex_tag}{result}.png",
                       f"Trunk-angle error distribution - {a.condition} vs {a.vs} "
                       f"({session_name})",
                       other=load_sessions(session_dirs, None, a.vs, a.game, result),
                       cond_names=(a.condition, a.vs), ymax=a.ymax)

    # ---- figure: scatter (top) + Bland-Altman (bottom), one column per axis ----
    fig, axes = plt.subplots(2, len(AXES), figsize=(4.8 * len(AXES), 9))
    for k, ax_name in enumerate(AXES):
        sc, ba = axes[0, k], axes[1, k]
        allm = np.concatenate(pooled[ax_name][0])
        alli = np.concatenate(pooled[ax_name][1])
        st = stats(allm, alli)
        lim = np.ceil(max(np.abs(allm).max(), np.abs(alli).max()) / 10) * 10
        lo_lim = 0.0 if ax_name == "magnitude" else -lim   # magnitude is non-negative

        for j, (s, d) in enumerate(data):
            moc, icp = paired(d, ax_name)
            if len(moc) < 50:
                continue
            idx = (rng.choice(len(moc), MAX_SCATTER, replace=False)
                   if len(moc) > MAX_SCATTER else slice(None))
            c = SUBJ_COLORS[j % len(SUBJ_COLORS)]
            sc.scatter(moc[idx], icp[idx], s=4, alpha=0.35, color=c, lw=0,
                       rasterized=True)
            ba.scatter((moc[idx] + icp[idx]) / 2, icp[idx] - moc[idx], s=4, alpha=0.35,
                       color=c, lw=0, rasterized=True)

        sc.plot([lo_lim, lim], [lo_lim, lim], color="#444444", lw=1.2, ls="--",
                zorder=3)
        # pooled least-squares fit: slope < 1 means the camera under-reads the range
        A = np.vstack([allm, np.ones_like(allm)]).T
        slope, icept = np.linalg.lstsq(A, alli, rcond=None)[0]
        sc.plot([lo_lim, lim], [slope * lo_lim + icept, slope * lim + icept],
                color="#111111", lw=1.6, zorder=4)
        sc.set_xlim(lo_lim, lim); sc.set_ylim(lo_lim, lim)
        sc.set_aspect("equal")
        sc.set_xlabel(f"mocap {ax_name} (deg)"
                      + ("\n(frame-invariant: no S)" if ax_name == "magnitude" else ""))
        if k == 0:
            sc.set_ylabel(f"{mlabel} (deg)")
        sc.set_title("rotation magnitude" if ax_name == "magnitude" else ax_name,
                     pad=8)
        sc.text(0.04, 0.96, f"r = {st['r']:+.2f}\nRMSE = {st['rmse']:.1f}$\\degree$\n"
                            f"slope = {slope:.2f}\nn = {st['n']}",
                transform=sc.transAxes, va="top", ha="left", fontsize=12)
        sc.grid(alpha=0.25)

        lo, hi = st["bias"] - 1.96 * st["sd"], st["bias"] + 1.96 * st["sd"]
        for y, ls, lw in ((st["bias"], "-", 1.6), (lo, "--", 1.2), (hi, "--", 1.2)):
            ba.axhline(y, color="#111111", ls=ls, lw=lw, zorder=3)
        ba.set_xlabel("mean of both (deg)")
        if k == 0:
            ba.set_ylabel("camera $-$ mocap (deg)")
        # headroom so the annotation never lands on the upper LoA line
        ylo, yhi = ba.get_ylim()
        ba.set_ylim(ylo, yhi + 0.22 * (yhi - ylo))
        ba.text(0.04, 0.96, f"bias = {st['bias']:+.1f}$\\degree$\n"
                            f"LoA [{lo:+.1f}, {hi:+.1f}]$\\degree$",
                transform=ba.transAxes, va="top", ha="left", fontsize=12)
        ba.grid(alpha=0.25)

    handles = [Line2D([], [], marker="o", ls="", color=SUBJ_COLORS[j % len(SUBJ_COLORS)],
                      markersize=7, label=label[s]) for j, (s, _) in enumerate(data)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Trunk angle agreement, {mlabel} vs mocap - "
                 f"{len(data)} subjects ({session_name}, {cond})", y=0.99)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    png = out_dir / f"icp_vs_mocap_agreement{suffix}.png"
    fig.savefig(png)
    print(f"-> {png}")


if __name__ == "__main__":
    main()
