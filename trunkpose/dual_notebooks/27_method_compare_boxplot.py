# %% 27_method_compare_boxplot.py
# Method comparison figure: |error| vs mocap for camera ICP, SAM 3D Body (teacher) and
# the miniSAM3D student side by side per subject, one panel per axis. The per-method
# figures are drawn here too (16.error_boxplots), so one run writes every figure for
# every condition instead of one 16 --method run per method:
#
#   <session parent>/output/method_compare_<condition>[_names].png       all methods
#   <session parent>/output/error_distribution_<condition>_<method>[_names].png  one each
#   <session parent>/output/method_pooled_<condition>[_unseen].png  all subjects pooled
#
# Each method is scored on its own valid in-window frames (16.paired), so ICP's boxes
# leave out the frames it failed to register while SAM/student cover every frame.
# The student is drawn only where its result exists -- by default the run folder holds
# just the subjects it never trained on.
#
#   uv run python trunkpose/dual_notebooks/27_method_compare_boxplot.py \
#       --session data/trunk_take_oct/oct9_26 data/trunk_take_oct/oct10_26 --names
#
# The game takes are conditions too (session.toml [game] takes), scored at stride 6:
#
#   ... --session data/trunk_take_oct/oct10_26 --condition game game_wtc --names

import argparse
import importlib
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
m16 = importlib.import_module("16_subject_agreement")

METHODS = ("icp", "sam3d", "student")
# Okabe-Ito, distinct from 16's per-axis colours so the two figure types are not confused
METHOD_COLORS = {"icp": "#E69F00", "sam3d": "#56B4E9", "student": "#CC79A7"}


def pooled_figure(by_method, subjects, ymax, out_png, title):
    """|error| pooled over `subjects` (every scored frame, so longer takes weigh more):
    flexion / lateral / axial / magnitude groups, one box per method inside each."""
    step = 0.27
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    series, pos, cols = [], [], []
    for i, ax_name in enumerate(m16.AXES):
        for k, m in enumerate(METHODS):
            parts = [m16.abs_errors(by_method[m][s], ax_name)
                     for s in subjects if s in by_method[m]]
            parts = [e for e in parts if e is not None]
            if not parts:
                continue
            series.append(np.concatenate(parts))
            pos.append(i + (k - 1) * step)
            cols.append(METHOD_COLORS[m])
    bp = ax.boxplot(series, positions=pos, widths=step * 0.82, whis=(5, 95),
                    showfliers=False, patch_artist=True,
                    medianprops={"color": "black", "lw": 1.5})
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_edgecolor(c)
        patch.set_alpha(0.85)
    for x_, e in zip(pos, series):
        ax.annotate(f"{np.median(e):.1f}", (x_, np.percentile(e, 75)),
                    textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8,
                    bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none",
                          "alpha": 0.8})
    ax.set_xticks(range(len(m16.AXES)), [a_.capitalize() for a_ in m16.AXES])
    ax.set_ylabel("absolute error (degrees)")
    ax.set_ylim(0, ymax if ymax else ax.get_ylim()[1] * 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    n = {m: sum(s in by_method[m] for s in subjects) for m in METHODS}
    ax.legend(handles=[Patch(facecolor=METHOD_COLORS[m], alpha=0.85,
                             label=f"{m16.METHOD_LABEL[m]} (n={n[m]})") for m in METHODS],
              frameon=False, ncol=len(METHODS), loc="upper left", fontsize=9)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"-> {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", nargs="+", required=True)
    ap.add_argument("--condition", nargs="+", default=["wc", "woc"])
    ap.add_argument("--student-dir",
                    default=str(m16.PROJECT_ROOT / "trunkpose" / "runs" / "minisam3d"
                                / "r18_v2_all_octtest"))
    ap.add_argument("--shell", default="mode", choices=("mode", "front"))
    ap.add_argument("--ymax", type=float, default=25.0, help="y limit in degrees (0 = auto)")
    ap.add_argument("--names", action="store_true",
                    help="label subjects by 3-letter name instead of Sub1..")
    a = ap.parse_args()
    result = "" if a.shell == "front" else f"_{a.shell}"
    session_dirs = [Path(s).resolve() for s in a.session]
    root = session_dirs[0] if len(session_dirs) == 1 else Path(os.path.commonpath(session_dirs))
    out_dir = root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    m12 = importlib.import_module("12_poster_figures")
    m12.poster_style()
    for cond in a.condition:
        draw_condition(a, cond, session_dirs, out_dir, result)


def draw_condition(a, condition, session_dirs, out_dir, result):
    # ICP defines the subject list and order; the other methods are matched by take name
    by_method = {}
    for m in METHODS:
        m16.SOURCE.update(method=m, student_dir=a.student_dir)
        by_method[m] = dict(m16.load_sessions(session_dirs, None, condition, False, result))
    subjects = list(by_method["icp"])
    label = {s: (m16.short_name(s) if a.names else f"Sub{j + 1}")
             for j, s in enumerate(subjects)}
    print("tick labels: " + ", ".join(f"{label[s]}={s}" for s in subjects))
    tag = f"{condition}{'_names' if a.names else ''}"
    sess_name = "+".join(sd.name for sd in session_dirs)

    # ---- one figure per method (same layout as 16's error_distribution) ----
    for m in METHODS:
        data = [(s, by_method[m][s]) for s in subjects if s in by_method[m]]
        if not data:
            continue
        m16.error_boxplots(data, label, out_dir / f"error_distribution_{tag}_{m}.png",
                           f"{m16.METHOD_LABEL[m]} vs mocap: trunk-angle error - "
                           f"{len(data)} subjects ({sess_name}, {condition})",
                           ymax=a.ymax)

    # ---- all subjects pooled, one box per method per axis ----
    pooled_figure(by_method, subjects, a.ymax, out_dir / f"method_pooled_{condition}.png",
                  f"Trunk-angle error vs mocap, all subjects pooled ({sess_name}, {condition})")
    # the student covers only its unseen subjects; the matched figure scores every
    # method on exactly those, so the three boxes describe the same people
    unseen = [s for s in subjects if s in by_method["student"]]
    if unseen and len(unseen) < len(subjects):
        pooled_figure(by_method, unseen, a.ymax,
                      out_dir / f"method_pooled_{condition}_unseen.png",
                      f"Trunk-angle error vs mocap, student-unseen subjects only "
                      f"({sess_name}, {condition})")

    # ---- all methods side by side ----

    step, group = 0.27, 1.0
    fig, axes = plt.subplots(len(m16.AXES), 1, figsize=(1.8 + 1.25 * len(subjects), 12.5),
                             sharex=True)
    for ax, ax_name in zip(axes, m16.AXES):
        series, pos, cols = [], [], []
        for j, s in enumerate(subjects):
            for k, m in enumerate(METHODS):
                d = by_method[m].get(s)
                e = m16.abs_errors(d, ax_name) if d is not None else None
                if e is None:
                    continue
                series.append(e)
                pos.append(group * j + (k - 1) * step)
                cols.append(METHOD_COLORS[m])
        bp = ax.boxplot(series, positions=pos, widths=step * 0.82, whis=(5, 95),
                        showfliers=False, patch_artist=True,
                        medianprops={"color": "black", "lw": 1.3})
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_edgecolor(c)
            patch.set_alpha(0.85)
        for x_, e in zip(pos, series):
            ax.annotate(f"{np.median(e):.1f}", (x_, np.percentile(e, 75)),
                        textcoords="offset points", xytext=(0, 4), ha="center", fontsize=6.5,
                        bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none",
                              "alpha": 0.8})
        ax.set_ylabel(f"{ax_name}\n|error| (deg)")
        ax.set_ylim(0, a.ymax if a.ymax else ax.get_ylim()[1] * 1.08)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks([group * j for j in range(len(subjects))],
                        [label[s] for s in subjects])
    axes[-1].set_xlabel("Subject")
    axes[0].legend(handles=[Patch(facecolor=METHOD_COLORS[m], alpha=0.85,
                                  label=m16.METHOD_LABEL[m]) for m in METHODS],
                   frameon=False, ncol=len(METHODS), loc="upper left", fontsize=9)
    n_student = sum(s in by_method["student"] for s in subjects)
    fig.suptitle(f"Trunk-angle error vs mocap by method - {len(subjects)} subjects "
                 f"({sess_name}, {condition}; "
                 f"student on {n_student} unseen)", y=0.995)
    fig.tight_layout()
    out_png = out_dir / f"method_compare_{tag}.png"
    fig.savefig(out_png)
    plt.close(fig)
    print(f"-> {out_png}")


if __name__ == "__main__":
    main()
