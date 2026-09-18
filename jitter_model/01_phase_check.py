# %% [markdown]
# # Inter-camera frame phase check
#
# The two OV9281s self-clock off separate 24 MHz oscillators with no FSIN wiring
# between them. `rcam.FrameSync` walks one sensor's phase onto the other before
# recording starts, but the crystals still differ by tens of ppm, so the two
# exposures drift apart over a take. This script measures that drift for **one**
# recording, from the frame timestamps the recorder already wrote.
#
# The phase is recomputed per frame from `sensor_ns` (column 3, the timestamp
# CAMSS stamps in its frame-done interrupt) rather than read from
# `phase_log.msgpack`, which the recorder samples only about once a second. The
# logged measurements are overlaid as a cross-check, since they carry the one
# thing the frame timestamps cannot show after the fact: which measurements
# drove a correction.
#
# Frame data is never opened — `cam*_frame.msgpack` runs to gigabytes and holds
# nothing this analysis needs.
#
# Run from the repository root, cell-by-cell or as a script:
#
# ```powershell
# uv run python jitter_model/01_phase_check.py
# uv run python jitter_model/01_phase_check.py --recording data/dome/sep18_26/dome_random_movement_100hz_sep18_26
# ```

# %% Imports
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import msgpack
import msgpack_numpy as mpn
import numpy as np
import pandas as pd

# %% Paths and settings
try:
    NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:  # running cells in an interactive kernel
    NOTEBOOK_DIR = Path.cwd()
    if NOTEBOOK_DIR.name != "jitter_model":
        NOTEBOOK_DIR = NOTEBOOK_DIR / "jitter_model"

PROJECT_ROOT = NOTEBOOK_DIR.parent

# One take per run. Edit this when working in a kernel; pass --recording when
# running as a script. Outputs land inside whichever take was analysed, so
# running the script once per recording never overwrites an earlier take.
RECORDING_DIR = (
    PROJECT_ROOT / "data" / "dome" / "sep18_26" / "dome_random_movement_sep18_26"
)

# Outputs live beside the recording, in a subfolder of the take itself, so each
# take keeps its own artefacts. Set below, once the recording path is final.
OUTPUT_SUBDIR = "phase_check"
OUTPUT_DIR = None

REFERENCE_CAMERA = "cam0"
ADJUSTED_CAMERA = "cam1"

# Histogram resolution for panel 3.
PHASE_BINS = 60


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Measure inter-camera frame phase for one dual recording."
    )
    parser.add_argument(
        "--recording",
        type=Path,
        default=None,
        help="Recording directory holding cam{0,1}_timestamp.msgpack.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override where the figure and summary CSV are written "
        "(default: <recording>/phase_check).",
    )
    return parser.parse_args(argv)


# argparse only when actually run as a script; in a kernel sys.argv is the
# kernel's own command line and would fail to parse.
if __name__ == "__main__" and "ipykernel" not in sys.modules:
    _args = _parse_args(sys.argv[1:])
    if _args.recording is not None:
        RECORDING_DIR = _args.recording
    if _args.output_dir is not None:
        OUTPUT_DIR = _args.output_dir

RECORDING_DIR = Path(RECORDING_DIR)
if not RECORDING_DIR.is_absolute():
    RECORDING_DIR = PROJECT_ROOT / RECORDING_DIR

# Default only once the recording is known, so --recording carries its outputs
# with it; an explicit --output-dir still wins.
if OUTPUT_DIR is None:
    OUTPUT_DIR = RECORDING_DIR / OUTPUT_SUBDIR
OUTPUT_DIR = Path(OUTPUT_DIR)
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

TAKE_NAME = RECORDING_DIR.name
# The folder already names the take, so the files need not repeat it.
SAVE_FIGURE = OUTPUT_DIR / "phase.png"
SAVE_SUMMARY = OUTPUT_DIR / "phase_summary.csv"

print(f"Recording : {RECORDING_DIR}")
print(f"Outputs   : {OUTPUT_DIR}")


# %% Loaders
def load_frame_times(path):
    """Return the per-frame columns of a camera timestamp file.

    Record layout written by the dual recorder:
        [sync, wall_clock_iso, monotonic_ns, sensor_ns, sequence]

    `sensor_ns` is used rather than `monotonic_ns`: both are CLOCK_MONOTONIC,
    but the latter is stamped once the frame has been unpacked on the host and
    carries 1.6-1.9 ms of scheduling jitter, which would swamp a phase measured
    in tens of microseconds.
    """
    with path.open("rb") as stream:
        records = list(msgpack.Unpacker(stream, object_hook=mpn.decode))
    if not records:
        raise ValueError(f"No timestamp records in {path}")

    sync = np.asarray([int(record[0]) for record in records], dtype=np.uint8)
    sensor_ns = np.asarray([int(record[3]) for record in records], dtype=np.int64)
    sequence = np.asarray([int(record[4]) for record in records], dtype=np.int64)
    return sync, sensor_ns, sequence


def load_phase_log(path):
    """Return the recorder's own phase measurements, or None if it wrote none.

    Record layout:
        [monotonic_ns, phase_us, jitter_us, drift_ppm, n, nudged]
    """
    if not path.exists():
        return None
    with path.open("rb") as stream:
        records = list(msgpack.Unpacker(stream, object_hook=mpn.decode))
    if not records:
        return None
    array = np.asarray(records, dtype=float)
    return pd.DataFrame(
        {
            "monotonic_ns": array[:, 0].astype(np.int64),
            "phase_us": array[:, 1],
            "jitter_us": array[:, 2],
            "drift_ppm": array[:, 3],
            "n": array[:, 4].astype(int),
            "nudged": array[:, 5].astype(int),
        }
    )


def load_sync_events(path):
    """Return the GPIO sync edges as `(monotonic_ns, value)`, or None."""
    if not path.exists():
        return None
    with path.open("rb") as stream:
        records = list(msgpack.Unpacker(stream, object_hook=mpn.decode))
    if not records:
        return None
    array = np.asarray(records, dtype=float)
    return array[:, 0].astype(np.int64), array[:, 1].astype(int)


# %% Load this take
ref_sync, ref_ns, ref_seq = load_frame_times(
    RECORDING_DIR / f"{REFERENCE_CAMERA}_timestamp.msgpack"
)
adj_sync, adj_ns, adj_seq = load_frame_times(
    RECORDING_DIR / f"{ADJUSTED_CAMERA}_timestamp.msgpack"
)
phase_log = load_phase_log(RECORDING_DIR / "phase_log.msgpack")
sync_events = load_sync_events(RECORDING_DIR / "sync_events.msgpack")

print(
    f"{REFERENCE_CAMERA}: {len(ref_ns)} frames, {ADJUSTED_CAMERA}: {len(adj_ns)} frames"
)


# %% Frame period
# Taken from the data rather than a flag, so a 30 Hz and a 100 Hz take run
# through unchanged. The median is robust to the dropped frames and stretched
# intervals that the panels below go on to show.
ref_intervals_us = np.diff(ref_ns) / 1000.0
adj_intervals_us = np.diff(adj_ns) / 1000.0
frame_period_us = float(np.median(np.concatenate([ref_intervals_us, adj_intervals_us])))
frame_rate_hz = 1e6 / frame_period_us

print(f"Frame period: {frame_period_us:.1f} us ({frame_rate_hz:.2f} Hz)")


# %% Per-frame phase
def pair_nearest(reference_ns, adjusted_ns):
    """For each reference frame, the index of the nearest adjusted-camera frame.

    Index-matching the two cameras would be wrong: they start on different
    driver sequence numbers, and either can drop a frame mid-take.
    """
    insert = np.searchsorted(adjusted_ns, reference_ns)
    left = np.clip(insert - 1, 0, len(adjusted_ns) - 1)
    right = np.clip(insert, 0, len(adjusted_ns) - 1)
    take_left = np.abs(reference_ns - adjusted_ns[left]) <= np.abs(
        reference_ns - adjusted_ns[right]
    )
    return np.where(take_left, left, right)


def wrap_to_half_period(delta_us, period_us):
    """Wrap a raw time difference into +/- half a frame period.

    Phase is only defined modulo the frame period: two sensors a full period
    apart expose in lockstep. Negative means the adjusted camera exposes first.
    """
    return ((delta_us + period_us / 2.0) % period_us) - period_us / 2.0


match = pair_nearest(ref_ns, adj_ns)
phase_us = wrap_to_half_period((adj_ns[match] - ref_ns) / 1000.0, frame_period_us)

# Frame times relative to the first reference frame, for every time axis below.
t0_ns = int(ref_ns[0])
ref_time_s = (ref_ns - t0_ns) / 1e9
adj_time_s = (adj_ns - t0_ns) / 1e9
duration_s = float(ref_time_s[-1])

phase_frac = phase_us / frame_period_us * 100.0  # percent of a frame period


# %% Dropped frames
# A gap in the driver's sequence counter means the sensor produced a frame that
# never reached the host. A gap in the timestamps alone cannot tell that apart
# from the capture thread being descheduled, which is why the counter is logged.
def dropped_frames(sequence):
    gaps = np.diff(sequence) - 1
    gaps = np.clip(gaps, 0, None)
    return int(gaps.sum()), np.flatnonzero(gaps > 0)


ref_dropped, ref_gap_idx = dropped_frames(ref_seq)
adj_dropped, adj_gap_idx = dropped_frames(adj_seq)

print(
    f"Dropped frames - {REFERENCE_CAMERA}: {ref_dropped}, {ADJUSTED_CAMERA}: {adj_dropped}"
)


# %% Smoothed phase
# Each camera's own sensor timestamp carries about 100 us of jitter, so their
# difference carries ~140 us and the per-frame trace is far noisier than the
# drift hiding in it. A one-second rolling median gives the trend at the same
# cadence the recorder's own measurements come in at, without the recorder's
# gaps. Median, not mean, so a single late frame does not drag the trend.
smoothing_frames = max(3, round(frame_rate_hz))
phase_smoothed = (
    pd.Series(phase_us)
    .rolling(smoothing_frames, center=True, min_periods=1)
    .median()
    .to_numpy()
)


# %% Drift fit
# A nudge (rcam stretching one sensor's vertical blanking) steps the phase, so a
# single fit across one would measure the correction rather than the crystals.
# The fit therefore runs on the longest stretch of the take with no nudge in it,
# which is the whole take when nothing was corrected.
def nudge_times_s(phase_log_df):
    if phase_log_df is None:
        return np.empty(0)
    nudges = phase_log_df.loc[phase_log_df["nudged"] == 1, "monotonic_ns"].to_numpy()
    return (nudges - t0_ns) / 1e9


nudge_s = nudge_times_s(phase_log)
nudge_s_in_take = nudge_s[(nudge_s >= 0) & (nudge_s <= duration_s)]

segment_edges = np.concatenate([[0.0], np.sort(nudge_s_in_take), [duration_s]])
segment_lengths = np.diff(segment_edges)
longest = int(np.argmax(segment_lengths))
fit_start_s, fit_end_s = segment_edges[longest], segment_edges[longest + 1]
fit_mask = (ref_time_s >= fit_start_s) & (ref_time_s <= fit_end_s)

if fit_mask.sum() >= 2:
    # phase in us against time in s is us/s, which is ppm as it stands.
    drift_ppm, phase_intercept_us = np.polyfit(
        ref_time_s[fit_mask], phase_us[fit_mask], 1
    )
else:
    drift_ppm, phase_intercept_us = np.nan, np.nan

# The recorder's own drift estimate, over the rows that fall inside the take.
if phase_log is not None:
    log_time_s = (phase_log["monotonic_ns"].to_numpy() - t0_ns) / 1e9
    log_in_take = (log_time_s >= 0) & (log_time_s <= duration_s)
    logged_drift_ppm = float(np.median(phase_log.loc[log_in_take, "drift_ppm"]))
    logged_jitter_us = float(np.median(phase_log.loc[log_in_take, "jitter_us"]))
    log_rows_before_take = int((~log_in_take & (log_time_s < 0)).sum())
else:
    log_time_s = None
    log_in_take = None
    logged_drift_ppm = np.nan
    logged_jitter_us = np.nan
    log_rows_before_take = 0

print(
    f"Drift: {drift_ppm:+.2f} ppm fitted over {fit_end_s - fit_start_s:.1f} s, "
    f"{logged_drift_ppm:+.2f} ppm logged (median)"
)
if log_rows_before_take:
    print(
        f"{log_rows_before_take} phase_log rows predate the first frame "
        "(pre-recording alignment) and are left off the plot."
    )


# %% Summary table
summary = pd.DataFrame(
    [
        {
            "take": TAKE_NAME,
            "frames_ref": len(ref_ns),
            "frames_adj": len(adj_ns),
            "duration_s": round(duration_s, 3),
            "frame_rate_hz": round(frame_rate_hz, 3),
            "frame_period_us": round(frame_period_us, 1),
            "phase_median_us": round(float(np.median(phase_us)), 2),
            "phase_p95_abs_us": round(float(np.percentile(np.abs(phase_us), 95)), 2),
            "phase_max_abs_us": round(float(np.max(np.abs(phase_us))), 2),
            "phase_max_abs_pct_frame": round(float(np.max(np.abs(phase_frac))), 3),
            "phase_start_us": round(float(phase_us[0]), 2),
            "phase_end_us": round(float(phase_us[-1]), 2),
            "drift_ppm_fitted": round(float(drift_ppm), 3),
            "drift_ppm_logged": round(float(logged_drift_ppm), 3),
            "drift_fit_span_s": round(float(fit_end_s - fit_start_s), 1),
            # Spread of the per-frame phase about its own rolling median: the
            # timestamping noise, separated from the slow crystal drift.
            "phase_noise_std_us": round(float(np.std(phase_us - phase_smoothed)), 2),
            "jitter_us_logged": round(float(logged_jitter_us), 2),
            "nudges": len(nudge_s_in_take),
            "dropped_ref": ref_dropped,
            "dropped_adj": adj_dropped,
            "phase_log_rows": 0 if phase_log is None else len(phase_log),
        }
    ]
)

print()
print(summary.T.to_string(header=False))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
summary.to_csv(SAVE_SUMMARY, index=False)
print(f"\nSaved summary -> {SAVE_SUMMARY}")


# %% Figure
fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)

# --- Panel 1: phase against time -------------------------------------------
ax = axes[0]
ax.plot(
    ref_time_s,
    phase_us,
    color="tab:blue",
    linewidth=0.6,
    alpha=0.35,
    label=f"per-frame phase ({ADJUSTED_CAMERA} - {REFERENCE_CAMERA})",
)
ax.plot(
    ref_time_s,
    phase_smoothed,
    color="tab:blue",
    linewidth=1.6,
    label=f"rolling median ({smoothing_frames} frames ~ 1 s)",
)
if np.isfinite(drift_ppm):
    fit_line_s = np.array([fit_start_s, fit_end_s])
    ax.plot(
        fit_line_s,
        drift_ppm * fit_line_s + phase_intercept_us,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"drift fit {drift_ppm:+.2f} ppm",
    )
if phase_log is not None and log_in_take.any():
    ax.plot(
        log_time_s[log_in_take],
        phase_log.loc[log_in_take, "phase_us"],
        linestyle="none",
        marker="o",
        markersize=3.5,
        color="tab:orange",
        alpha=0.85,
        label="recorder phase_log",
    )
for index, nudge_time in enumerate(nudge_s_in_take):
    ax.axvline(
        nudge_time,
        color="tab:red",
        linestyle="-",
        linewidth=1,
        alpha=0.7,
        label="nudge applied" if index == 0 else None,
    )
if sync_events is not None:
    sync_ns, sync_values = sync_events
    sync_s = (sync_ns - t0_ns) / 1e9
    for index, (edge_s, value) in enumerate(zip(sync_s, sync_values)):
        if 0 <= edge_s <= duration_s:
            ax.axvline(
                edge_s,
                color="0.55",
                linestyle=":",
                linewidth=1,
                label="GPIO sync edge" if index == 0 else None,
            )
ax.axhline(0, color="0.7", linewidth=0.8)
ax.set_ylabel("Phase [us]")
ax.set_title(
    f"Inter-camera phase — {ADJUSTED_CAMERA} relative to {REFERENCE_CAMERA} "
    "(negative = exposes first)"
)
ax.grid(True, alpha=0.25)
ax.legend(loc="upper left", ncols=2, fontsize=9)

# Same trace read as a fraction of the frame period: the axis that makes takes
# at different rates comparable, since a fixed microsecond offset is a much
# larger part of a 10 ms frame than of a 33 ms one.
ax_frac = ax.twinx()
low, high = ax.get_ylim()
ax_frac.set_ylim(low / frame_period_us * 100.0, high / frame_period_us * 100.0)
ax_frac.set_ylabel("Phase [% of frame period]")

# --- Panel 2: frame intervals ----------------------------------------------
ax = axes[1]
ax.plot(
    ref_time_s[1:],
    ref_intervals_us,
    color="tab:blue",
    linewidth=0.7,
    alpha=0.8,
    label=REFERENCE_CAMERA,
)
ax.plot(
    adj_time_s[1:],
    adj_intervals_us,
    color="tab:green",
    linewidth=0.7,
    alpha=0.8,
    label=ADJUSTED_CAMERA,
)
ax.axhline(
    frame_period_us,
    color="black",
    linestyle="--",
    linewidth=1,
    label=f"nominal {frame_period_us:.0f} us",
)
# Sequence gaps rugged along the bottom: a stretched interval with a gap is a
# dropped frame, one without is a nudge or a descheduled capture thread.
gap_low, gap_high = ax.get_ylim()
rug_y = gap_low + 0.02 * (gap_high - gap_low)
for gap_indices, colour, name in (
    (ref_gap_idx, "tab:blue", REFERENCE_CAMERA),
    (adj_gap_idx, "tab:green", ADJUSTED_CAMERA),
):
    if len(gap_indices):
        times = (ref_time_s if name == REFERENCE_CAMERA else adj_time_s)[
            gap_indices + 1
        ]
        ax.plot(
            times,
            np.full(len(times), rug_y),
            linestyle="none",
            marker="|",
            markersize=10,
            color=colour,
            label=f"{name} sequence gap",
        )
ax.set_ylabel("Frame interval [us]")
ax.set_title("Frame intervals from sensor timestamps — drops and stretched intervals")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper left", ncols=2, fontsize=9)

# --- Panel 3: phase distribution -------------------------------------------
ax = axes[2]
ax.hist(phase_us, bins=PHASE_BINS, color="tab:blue", alpha=0.8)
for value, colour, label in (
    (float(np.median(phase_us)), "black", "median"),
    (float(np.percentile(np.abs(phase_us), 95)), "tab:orange", "p95 |phase|"),
    (float(np.max(np.abs(phase_us))), "tab:red", "max |phase|"),
):
    ax.axvline(
        value,
        color=colour,
        linestyle="--",
        linewidth=1.2,
        label=f"{label} {value:+.1f} us",
    )
ax.set_xlabel("Phase [us]")
ax.set_ylabel("Frames")
ax.set_title("Phase distribution over the take")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right", fontsize=9)

axes[1].set_xlabel("Time after first frame [s]")
fig.suptitle(
    f"{TAKE_NAME} — {frame_rate_hz:.1f} Hz, {duration_s:.1f} s, "
    f"{len(ref_ns)} frames/camera"
)

fig.savefig(SAVE_FIGURE, dpi=180, bbox_inches="tight")
print(f"Saved figure -> {SAVE_FIGURE}")

plt.show()
