# %% 18_game_mocap_rom.py
# How much trunk motion is there in the NOARK game takes -- mocap alone.
#
# The game recordings (data/sep23_26/noark_game) are minutes long and were not shot
# against a ChArUco board, so before any camera work it is worth knowing whether the
# subjects move enough to be worth measuring. This reads the Motive rigid body only,
# takes the pose at the START of the take as the neutral, and reports flexion /
# lateral / axial rotation about that pose per subject.
#
# The rigid-body convention (Marker2 = origin, ->Marker3 = +X lateral,
# ->Marker4 = +Z anterior) and the Motive body name come from
# data/sep23_26/session.toml, read through 06 -- the same code path the validated
# ICP-vs-mocap analysis uses, so the angles are directly comparable to those takes.
#
#   uv run python trunkpose/dual_notebooks/18_game_mocap_rom.py
#   uv run python trunkpose/dual_notebooks/18_game_mocap_rom.py --subjects gokul kabil
#
# Writes into <noark_game>/output/:
#   game_mocap_rom.csv          per subject x axis summary
#   game_mocap_traces.png       angle-vs-time, one row per subject
#   game_mocap_rom_box.png      distribution per subject, axes side by side

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SESSION = PROJECT_ROOT / "data" / "sep23_26"

# 06 resolves its configuration at import time from TRUNK_RECORDING_DIR, and it needs
# to land on the session.toml that describes THIS mocap mounting. Nothing else in 06 is
# used here, so the recording dir only has to sit under the session.
os.environ.setdefault("TRUNK_RECORDING_DIR", str(DEFAULT_SESSION / "noark_game"))
os.environ.setdefault("TRUNK_RECORDING_NAME", "noark_game")
sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.spatial.transform import Rotation

m6 = importlib.import_module("06_trunk_axis")

AXES = ("flexion", "lateral", "axial")
AXIS_COLORS = ("#0072B2", "#009E73", "#D55E00")   # Okabe-Ito, as in 16
JUMP_M = 0.10           # origin-marker step that means the solver re-locked, not motion
BIG_DEG = (10.0, 20.0, 30.0)   # thresholds the "% of time beyond" columns use


# %% Load
def capture_rate(path):
    """Motive's own frame rate from the take header (row 1, key/value pairs)."""
    head = path.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
    cells = head.split(",")
    for k, v in zip(cells[:-1], cells[1:]):
        if k.strip() == "Capture Frame Rate":
            return float(v)
    return 100.0


def longest_clean_span(o, rate):
    """Longest run of valid origin-marker rows without a solver teleport.

    Motive re-locks a rigid body after an occlusion and the body can come back with a
    different axis assignment; a jump in the origin marker between consecutive valid
    rows is the tell. Angles either side of such a jump are not on the same reference,
    so only the longest clean run is used. Returns (lo, hi) row indices inclusive.
    """
    ok = np.isfinite(o).all(axis=1)
    vidx = np.where(ok)[0]
    if len(vidx) < 2:
        raise SystemExit("no valid rigid-body rows")
    cut = np.where(np.linalg.norm(np.diff(o[vidx], axis=0), axis=1) > JUMP_M)[0]
    bounds = np.concatenate(([0], cut + 1, [len(vidx)]))
    segs = [(vidx[a], vidx[b - 1]) for a, b in zip(bounds[:-1], bounds[1:])]
    lo, hi = max(segs, key=lambda ab: ab[1] - ab[0])
    if len(segs) > 1:
        print(f"  teleport-split into {len(segs)} segment(s); longest = "
              f"{lo / rate:.1f}-{hi / rate:.1f}s "
              f"({(hi - lo + 1) / rate:.1f}s of {len(o) / rate:.1f}s)")
    return lo, hi


def angles_from_neutral(path, neutral_s):
    """Per-row trunk angles (deg) about the pose at the start of the take.

    Returns (t, flex, lat, axi, info). Rows outside the clean span, or with a marker
    dropped, come back NaN so the time axis stays honest about the gaps.
    """
    rate = capture_rate(path)
    o, mx, mz = m6.load_trunk_rb(str(path))
    n = len(o)
    t = np.arange(n) / rate
    lo, hi = longest_clean_span(o, rate)

    R = [None] * n
    for i in range(lo, hi + 1):
        R[i] = m6.mocap_trunk_frame(o[i], mx[i], mz[i])
    valid = [i for i in range(lo, hi + 1) if R[i] is not None]
    if len(valid) < 10:
        raise SystemExit(f"{path.name}: only {len(valid)} usable rows")

    # neutral = the pose the subject starts the take in, averaged over a window so
    # marker jitter does not set the reference.
    n_neu = max(3, round(neutral_s * rate))
    neu_idx = valid[:n_neu]
    R_neu = Rotation.from_matrix(np.array([R[i] for i in neu_idx])).mean().as_matrix()

    flex = np.full(n, np.nan)
    lat = np.full(n, np.nan)
    axi = np.full(n, np.nan)
    for i in valid:
        flex[i], lat[i], axi[i] = m6.trunk_angles(R[i], R_neu)

    # how still the neutral window actually was, and how far the start pose sits from
    # the pose the subject spends the take in -- if these are large the "initial pose"
    # reference is the wrong zero and every number below is shifted.
    spread = float(np.nanmax([np.nanstd(a[neu_idx]) for a in (flex, lat, axi)]))
    drift = [float(np.nanmedian(a)) for a in (flex, lat, axi)]
    info = {
        "rate": rate, "n_rows": n, "n_used": len(valid),
        "dur_s": (hi - lo + 1) / rate, "take_s": n / rate,
        "valid_frac": len(valid) / max(1, hi - lo + 1),
        "neutral_s": n_neu / rate, "neutral_sd": spread,
        "median_pose": drift,
        "marker_mm": [float(np.nanmedian(np.linalg.norm(mx - o, axis=1))) * 1000,
                      float(np.nanmedian(np.linalg.norm(mz - o, axis=1))) * 1000],
    }
    return t, flex, lat, axi, info


# %% Stats
def axis_stats(a):
    """Distribution of one axis: how far, how often, and the usable range."""
    v = a[np.isfinite(a)]
    ad = np.abs(v)
    lo, hi = np.percentile(v, [1, 99])
    row = {
        "n": len(v),
        "median_abs": float(np.median(ad)),
        "p95_abs": float(np.percentile(ad, 95)),
        "max_abs": float(ad.max()),
        "min_deg": float(v.min()), "max_deg": float(v.max()),
        "p1_deg": float(lo), "p99_deg": float(hi),
        "rom_p1_p99": float(hi - lo),
        "sd_deg": float(v.std()),
    }
    for th in BIG_DEG:
        row[f"pct_gt{int(th)}"] = float(100.0 * (ad > th).mean())
    return row


def magnitude(flex, lat, axi):
    """Total rotation away from neutral, as the norm of the three angles.

    Not a rotation magnitude in the exact sense (the three are not orthogonal
    rotation-vector components), but it answers "did the trunk move at all" in one
    number and is what the wc/woc analysis reports as `magnitude`.
    """
    return np.sqrt(flex ** 2 + lat ** 2 + axi ** 2)


# %% Figures
def trace_figure(res, out_png):
    fig, axs = plt.subplots(len(res), 1, figsize=(11.5, 2.5 * len(res)), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, (subj, d) in zip(axs, res):
        for name, col in zip(AXES, AXIS_COLORS):
            ax.plot(d["t"], d[name], lw=0.7, color=col, label=name.capitalize())
        ax.axhline(0, color="0.6", lw=0.6)
        ax.axvspan(d["t"][0], d["t"][0] + d["info"]["neutral_s"], color="0.85",
                   zorder=0)
        ax.set_ylabel("deg")
        ax.set_title(f"{subj}   {d['info']['dur_s']:.0f}s used, "
                     f"{100 * d['info']['valid_frac']:.0f}% of rows valid", fontsize=9,
                     loc="left")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
    axs[-1].set_xlabel("time in take (s)   [grey = neutral window]")
    fig.suptitle("NOARK game, trunk rotation about the pose at take start (mocap only)",
                 y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"-> {out_png}")


def box_figure(res, out_png):
    """Signed angle per subject, axes side by side. Whiskers are 5th/95th percentile."""
    step, group = 0.46, 2.0
    span = step * (len(AXES) - 1)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.8 + 1.9 * len(res)), 5.2))
    series, pos, ticks, tick_pos = [], [], [], []
    for j, (subj, d) in enumerate(res):
        base = group * j
        for k, name in enumerate(AXES):
            v = d[name][np.isfinite(d[name])]
            series.append(v)
            pos.append(base - span / 2 + step * k)
        ticks.append(subj)
        tick_pos.append(base)
    bp = ax.boxplot(series, positions=pos, widths=step * 0.8, whis=(5, 95),
                    showfliers=False, patch_artist=True,
                    medianprops={"color": "black", "lw": 1.5})
    for k, patch in enumerate(bp["boxes"]):
        col = AXIS_COLORS[k % len(AXES)]
        patch.set_facecolor(col)
        patch.set_edgecolor(col)
        patch.set_alpha(0.85)
    for x_, v in zip(pos, series):
        ax.annotate(f"{np.percentile(v, 95) - np.percentile(v, 5):.0f}",
                    (x_, np.percentile(v, 95)), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7,
                    bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none",
                          "alpha": 0.8})
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(tick_pos, ticks)
    ax.set_xlabel("Subject")
    ax.set_ylabel("angle from take-start pose (degrees)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Patch(facecolor=c, alpha=0.85, label=a.capitalize())
                       for a, c in zip(AXES, AXIS_COLORS)],
              frameon=False, ncol=3, loc="upper left", fontsize=9)
    ax.set_title("NOARK game trunk rotation (mocap) - box = IQR, whisker = 5-95%, "
                 "label = 5-95% range")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"-> {out_png}")


# %% Main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default=str(DEFAULT_SESSION))
    ap.add_argument("--games", default=None,
                    help="game session dir (default <session>/noark_game)")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="subject stems (default: every mocap/*_game.csv)")
    ap.add_argument("--neutral-sec", type=float, default=1.0,
                    help="seconds averaged at the take start to define neutral")
    a = ap.parse_args()

    games = Path(a.games) if a.games else Path(a.session) / "noark_game"
    moc_dir = games / "mocap"
    paths = sorted(moc_dir.glob("*_game.csv"))
    if a.subjects:
        want = {s.lower() for s in a.subjects}
        paths = [p for p in paths
                 if p.stem.replace("_game", "").lower() in want]
    if not paths:
        raise SystemExit(f"no mocap CSVs in {moc_dir}")
    out_dir = games / "output"
    out_dir.mkdir(exist_ok=True)

    res, rows = [], []
    for p in paths:
        subj = p.stem.replace("_game", "")
        print(f"\n== {subj}  ({p.name})")
        t, flex, lat, axi, info = angles_from_neutral(p, a.neutral_sec)
        print(f"  {info['take_s']:.0f}s take @ {info['rate']:.0f}Hz, "
              f"{info['n_used']} rows used, markers "
              f"|m2m3| {info['marker_mm'][0]:.1f}mm |m2m4| {info['marker_mm'][1]:.1f}mm")
        print(f"  neutral window {info['neutral_s']:.2f}s, jitter SD "
              f"{info['neutral_sd']:.2f} deg; take-median pose vs that neutral "
              + "/".join(f"{v:+.1f}" for v in info["median_pose"]) + " deg")
        d = {"t": t, "flexion": flex, "lateral": lat, "axial": axi,
             "magnitude": magnitude(flex, lat, axi), "info": info}
        res.append((subj, d))
        for name in (*AXES, "magnitude"):
            rows.append({"subject": subj, "axis": name, **axis_stats(d[name])})

    hdr = (f"\n{'subject':<10}{'axis':<11}{'med|.|':>8}{'p95|.|':>8}{'max|.|':>8}"
           f"{'min':>8}{'max':>8}{'ROM(1-99)':>11}"
           + "".join(f"{'>' + str(int(x)) + 'd%':>8}" for x in BIG_DEG))
    print(hdr)
    for r in rows:
        print(f"{r['subject']:<10}{r['axis']:<11}{r['median_abs']:>8.1f}"
              f"{r['p95_abs']:>8.1f}{r['max_abs']:>8.1f}{r['min_deg']:>8.1f}"
              f"{r['max_deg']:>8.1f}{r['rom_p1_p99']:>11.1f}"
              + "".join(f"{r['pct_gt' + str(int(x))]:>8.1f}" for x in BIG_DEG))

    csv = out_dir / "game_mocap_rom.csv"
    keys = list(rows[0])
    csv.write_text("\n".join([",".join(keys)]
                             + [",".join(f"{r[k]:.4f}" if isinstance(r[k], float)
                                         else str(r[k]) for k in keys) for r in rows])
                   + "\n", encoding="utf-8")
    print(f"\n-> {csv}")
    trace_figure(res, out_dir / "game_mocap_traces.png")
    box_figure(res, out_dir / "game_mocap_rom_box.png")


if __name__ == "__main__":
    main()
