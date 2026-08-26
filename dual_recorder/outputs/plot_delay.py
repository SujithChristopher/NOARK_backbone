"""Plot the inter-camera frame delay from a capture_flicker.py take.

The two OV9281s self-clock off separate 24 MHz crystals with no FSIN between
them, so their frame delay is neither zero nor constant: it sits at whatever
phase the sensors happened to start at, and walks from there at the crystals'
rate difference (tens of ppm, i.e. milliseconds per minute).

Three views of the same delay, peeled one term at a time:

    raw                     delay itself                      -> the offset
    offset removed          delay - mean(delay)               -> the ramp
    offset + skew removed   delay - (a + b*t), least squares   -> the residual

Two figures, each a line plot next to a box plot of the same numbers:

    camera_delay_raw.png    the raw delay, on its own scale (tens of ms)
    camera_delay_skew.png   the two corrected series, on a shared scale (us)

They are separate figures on purpose. The offset here is ~14 ms against a ~0.1 ms
residual, so putting all three on one axis would flatten both corrected series
onto the zero line and show nothing.

    uv run python dual_recorder/outputs/plot_delay.py
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Categorical slots of the reference palette, assigned in fixed order and held
# constant across both figures: a series keeps its hue wherever it appears.
C_RAW = "#1baf7a"  # raw delay
C_OFF = "#2a78d6"  # offset removed
C_FIX = "#eb6834"  # offset + skew removed
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d4"


def pair_by_timestamp(ts_a, ts_b):
    """Nearest-timestamp pairing - a dropped frame must not shift later pairs.

    Returns (index_into_b, valid). Frames at either end of the take can have no
    partner at all; their nearest neighbour is then frame periods away, and left
    in they would swamp both the skew fit and the box plot with delays larger
    than a whole frame period.
    """
    idx = np.clip(np.searchsorted(ts_b, ts_a), 1, len(ts_b) - 1)
    left, right = ts_b[idx - 1], ts_b[idx]
    idx = np.where(np.abs(ts_a - left) <= np.abs(ts_a - right), idx - 1, idx)
    period_ns = np.median(np.diff(ts_a)) if len(ts_a) > 1 else np.inf
    return idx, np.abs(ts_b[idx] - ts_a) <= period_ns / 2


def load(npz_path, duration):
    d = np.load(npz_path, allow_pickle=True)
    ts0 = d["cam0_sensor_ns"].astype(np.float64)
    ts1 = d["cam1_sensor_ns"].astype(np.float64)
    j, valid = pair_by_timestamp(ts0, ts1)
    t = (ts0 - ts0[0]) / 1e9
    keep = valid & (t <= duration)
    dropped = int((~valid).sum())
    if dropped:
        print(f"{dropped} frame(s) had no partner within half a frame - dropped")
    labels = [str(x) for x in d["labels"]] if "labels" in d else ["cam0", "cam1"]
    period_us = float(np.median(np.diff(ts0)) / 1e3)
    return t[keep], (ts1[j[keep]] - ts0[keep]) / 1e3, labels, period_us, d


def style(ax, zero_line=True):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=10)
    if zero_line:
        ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.grid(axis="y", color=GRID, lw=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def box(ax, data, colors, tick_labels, unit):
    bp = ax.boxplot(
        data, widths=0.5, patch_artist=True, showfliers=True,
        tick_labels=tick_labels,
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": INK_2,
                    "markeredgecolor": "none", "alpha": 0.5},
        medianprops={"color": "white", "lw": 2},
        whiskerprops={"color": INK_2, "lw": 1.2},
        capprops={"color": INK_2, "lw": 1.2},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("white")
        patch.set_linewidth(2)
    for i, y in enumerate(data, start=1):
        ax.annotate(
            f"sd {y.std():.1f} {unit}", (i, y.max()), textcoords="offset points",
            xytext=(0, 10), ha="center", fontsize=10, color=INK,
        )


def figure_raw(t, delay_us, labels, period_us, span, out):
    """The delay as measured, no correction of any kind - so the offset shows."""
    delay_ms = delay_us / 1e3
    fig, (ax_line, ax_box) = plt.subplots(
        1, 2, figsize=(13, 5), sharey=True,
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.06},
    )
    fig.patch.set_facecolor("white")
    for ax in (ax_line, ax_box):
        style(ax, zero_line=False)

    ax_line.plot(t, delay_ms, color=C_RAW, lw=2)
    ax_line.set_xlabel("time into recording (s)", color=INK_2, fontsize=11)
    ax_line.set_ylabel("inter-camera delay (ms)", color=INK_2, fontsize=11)
    ax_line.set_xlim(t[0], t[-1])
    # No explicit ylim and no reference lines out at 0 / -T/2: either would hold
    # the axis open across the whole 14 ms offset and flatten the trace to a
    # line. Autoscaled, the panel shows the raw delay's actual shape, and the
    # title carries the scale it should be read against.
    ax_line.set_title(
        f"Delay over time, no correction   mean {delay_ms.mean():+.2f} ms "
        f"= {abs(delay_us.mean()) / period_us * 100:.0f}% of a "
        f"{period_us / 1e3:.2f} ms frame",
        color=INK, fontsize=12, loc="left", pad=12,
    )

    box(ax_box, [delay_ms], [C_RAW], ["raw delay"], "ms")
    ax_box.set_title("Distribution", color=INK, fontsize=12, loc="left", pad=12)

    fig.suptitle(
        f"Inter-camera frame delay as measured, {labels[0]} -> {labels[1]}   "
        f"{len(t)} pairs over {span:.1f} s, free-running",
        color=INK, fontsize=13, x=0.008, ha="left", y=0.985,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.85, bottom=0.12)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"Wrote {out}")


def figure_corrected(t, raw, fixed, labels, meta, span, slope, slope_se, offset,
                     resolved, out):
    """The two correction stages against each other, on a shared us scale."""
    fig, (ax_line, ax_box) = plt.subplots(
        1, 2, figsize=(13, 5), sharey=True,
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.06},
    )
    fig.patch.set_facecolor("white")
    for ax in (ax_line, ax_box):
        style(ax)

    ax_line.plot(t, raw, color=C_OFF, lw=2, label="offset removed (skew present)")
    ax_line.plot(t, fixed, color=C_FIX, lw=2, label="offset + skew removed")
    ax_line.set_xlabel("time into recording (s)", color=INK_2, fontsize=11)
    ax_line.set_ylabel("inter-camera delay (us)", color=INK_2, fontsize=11)
    ax_line.set_xlim(t[0], t[-1])
    leg = ax_line.legend(frameon=False, loc="upper left", fontsize=10)
    for text in leg.get_texts():
        text.set_color(INK)
    ax_line.set_title(
        f"Delay over time   mean offset {offset:+.0f} us, "
        f"fitted skew {slope:+.2f} +/- {slope_se:.2f} ppm"
        + ("" if resolved else " (not resolved over 10 s)"),
        color=INK, fontsize=12, loc="left", pad=12,
    )

    box(ax_box, [raw, fixed], [C_OFF, C_FIX],
        ["offset\nremoved", "offset + skew\nremoved"], "us")
    ax_box.set_title("Distribution", color=INK, fontsize=12, loc="left", pad=12)

    exposure = float(meta["exposure_us"]) if "exposure_us" in meta else float("nan")
    fig.suptitle(
        f"Inter-camera frame delay, {labels[0]} -> {labels[1]}   "
        f"{len(t)} pairs over {span:.1f} s, free-running, exposure {exposure:.0f} us",
        color=INK, fontsize=13, x=0.008, ha="left", y=0.985,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.85, bottom=0.12)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", default=os.path.join(HERE, "flicker_50hz_timestamps.npz"))
    ap.add_argument("--duration", type=float, default=10.0, help="seconds to plot")
    ap.add_argument("--out", default=os.path.join(HERE, "camera_delay_skew.png"))
    ap.add_argument("--out-raw", default=os.path.join(HERE, "camera_delay_raw.png"))
    args = ap.parse_args()

    t, delay_us, labels, period_us, meta = load(args.npz, args.duration)
    if len(t) < 10:
        raise SystemExit(f"only {len(t)} frame pairs in {args.npz}")

    offset = float(delay_us.mean())
    slope, intercept = np.polyfit(t, delay_us, 1)  # us per second == ppm exactly
    raw = delay_us - offset
    fixed = delay_us - (intercept + slope * t)

    # Standard error of the slope. Over a short window the crystals' rate
    # difference is a small ramp sitting under ~100 us of per-frame timestamp
    # jitter, so the fit can easily be consistent with zero - without this
    # number the plot reads as if the skew had been measured precisely.
    span = t[-1] - t[0]
    dof = len(t) - 2
    slope_se = (
        fixed.std(ddof=2) / np.sqrt(np.sum((t - t.mean()) ** 2)) if dof > 0 else 0.0
    )
    resolved = abs(slope) > 2 * slope_se

    print(f"{len(t)} frame pairs over {span:.2f} s ({labels[0]} -> {labels[1]})")
    print(f"  frame period    {period_us / 1e3:9.3f} ms")
    print(
        f"  mean delay      {offset:+9.1f} us "
        f"({abs(offset) / period_us * 100:.1f}% of a frame)"
    )
    print(
        f"  fitted skew     {slope:+9.2f} +/- {slope_se:.2f} ppm"
        + ("" if resolved else "  (consistent with zero over this window)")
    )
    for name, y in (
        ("raw (no correction)", delay_us),
        ("offset removed", raw),
        ("offset + skew removed", fixed),
    ):
        print(
            f"  {name:<22} sd {y.std():7.2f} us   "
            f"p2p {y.max() - y.min():8.2f} us   "
            f"|max| {np.abs(y).max():9.2f} us"
        )

    figure_raw(t, delay_us, labels, period_us, span, args.out_raw)
    figure_corrected(t, raw, fixed, labels, meta, span, slope, slope_se, offset,
                     resolved, args.out)


if __name__ == "__main__":
    main()
