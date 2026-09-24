# %% 16_subject_agreement.py
# Group agreement figure: camera ICP vs mocap trunk angles across every subject of a
# session, pooled from the per-subject icp_vs_mocap.npz files written by
# 15_subject_icp.py.
#
#   <output>/icp_vs_mocap_agreement.png   scatter + Bland-Altman, 3 axes x N subjects
#   <output>/agreement_summary.csv        per-subject per-axis r / RMSE / bias / LoA
#
# Only frames where BOTH systems are valid are scored -- the ICP series carries NaN
# wherever registration was rejected, and those frames are dropped here rather than
# interpolated, so a subject with poor coverage shows up as a small n, not as a
# smooth line that was never measured.
#
#   uv run python trunkpose/dual_notebooks/16_subject_agreement.py
#   uv run python trunkpose/dual_notebooks/16_subject_agreement.py --session data/sep23_26

import argparse
import csv
import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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


def load_subjects(session_dir, subjects):
    """(name, npz) per subject that actually has results, in the given order."""
    import toml

    if not subjects:
        cfg_path = session_dir / "session.toml"
        cfg = toml.load(cfg_path) if cfg_path.exists() else {}
        subjects = cfg.get("subjects", {}).get("wc", [])
        if not subjects:
            subjects = sorted(p.parent.name
                              for p in session_dir.glob("*/icp_vs_mocap.npz"))
    out = []
    for s in subjects:
        f = session_dir / s / "icp_vs_mocap.npz"
        if not f.exists():
            print(f"skip {s}: no {f.name} (run 15_subject_icp.py --subject {s})")
            continue
        out.append((s, np.load(f, allow_pickle=True)))
    if not out:
        raise SystemExit(f"No per-subject results under {session_dir}")
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
    return dict(n=len(d), r=float(np.corrcoef(moc, icp)[0, 1]),
                rmse=float(np.sqrt((d ** 2).mean())),
                bias=float(d.mean()), sd=float(d.std(ddof=1)),
                rom_moc=float(np.percentile(moc, 97.5) - np.percentile(moc, 2.5)),
                rom_icp=float(np.percentile(icp, 97.5) - np.percentile(icp, 2.5)),
                # slope of camera on mocap: < 1 means the camera under-reads the range
                slope=float(np.polyfit(moc, icp, 1)[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default=str(PROJECT_ROOT / "data" / "sep23_26"))
    ap.add_argument("--subjects", nargs="*", default=None)
    a = ap.parse_args()
    session_dir = Path(a.session).resolve()
    out_dir = session_dir / "output"
    out_dir.mkdir(exist_ok=True)

    m12 = importlib.import_module("12_poster_figures")
    m12.poster_style()

    data = load_subjects(session_dir, a.subjects)
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
    csv_path = out_dir / "agreement_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'subject':<12}{'axis':<10}{'n':>6}{'cov':>6}{'r':>7}{'RMSE':>7}"
          f"{'bias':>7}{'slope':>7}{'LoA':>16}{'ROM cam/mocap':>16}")
    for r in rows:
        print(f"{r['subject']:<12}{r['axis']:<10}{r['n']:>6}{r['coverage'] * 100:>5.0f}%"
              f"{r['r']:>+7.2f}{r['rmse']:>6.1f}d{r['bias']:>+6.1f}d{r['slope']:>7.2f}"
              f"   [{r['loa_lo']:+5.1f},{r['loa_hi']:+5.1f}]"
              f"{r['rom_icp']:>9.0f}/{r['rom_moc']:.0f}d")
    flagged = sorted({r["subject"] for r in rows if r["S_diff_deg"] > 30})
    if flagged:
        print("\nAxis split unreliable (fitted S disagrees with the neutral-pose S by "
              f"more than 30 deg): {', '.join(flagged)}. Their magnitude row still "
              "holds -- it does not depend on S.")
    print(f"-> {csv_path}")

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
            sc.set_ylabel("camera ICP (deg)")
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
    fig.suptitle(f"Trunk angle agreement, camera ICP vs mocap - "
                 f"{len(data)} subjects ({session_dir.name})", y=0.99)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    png = out_dir / "icp_vs_mocap_agreement.png"
    fig.savefig(png)
    print(f"-> {png}")


if __name__ == "__main__":
    main()
