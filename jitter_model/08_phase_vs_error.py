# %% [markdown]
# # Phase against error, in one picture
#
# `07_timing_sensitivity.py` runs five tests and draws four figures. This script
# draws the one plot that carries the conclusion, for a single stereo condition,
# and nothing else.
#
# Left panel: the signed test. Latency displaces the stereo estimate *along* the
# motion by `phase x speed`, so the along-track error must follow the dashed
# slope-1 line. Per-frame points are too noisy to read, so equal-count bins with
# their interquartile range are drawn over them; the bins are what to look at.
#
# Right panel: the same measurement made against a known offset, by re-pairing
# cam1 one and two frames away (`07`'s injection test, read back from its CSV).
# It fixes the scale in mm per ms and marks where this take's real phase sits,
# which turns "the left panel is flat" into "flat, and here is how small an
# effect this plot could still have seen".
#
# Requires `06_movement_error.py` and `07_timing_sensitivity.py` to have run on
# the same recording.
#
# ```powershell
# uv run python jitter_model/08_phase_vs_error.py
# uv run python jitter_model/08_phase_vs_error.py --recording data/dome/sep18_26/dome_random_movement_100hz_sep18_26
# ```

# %% Imports
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jitter_model.common import load_timestamp_records

# %% Paths and settings
try:
    NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:  # running cells in an interactive kernel
    NOTEBOOK_DIR = Path.cwd()
    if NOTEBOOK_DIR.name != "jitter_model":
        NOTEBOOK_DIR = NOTEBOOK_DIR / "jitter_model"

PROJECT_ROOT = NOTEBOOK_DIR.parent

RECORDING_DIR = (
    PROJECT_ROOT / "data" / "dome" / "sep18_26" / "dome_random_movement_sep18_26"
)

# One stereo condition, not all four: the point is legibility, and the four
# differ only in how many tags the solve was given.
CONDITION = "C2-M4"

# Equal-count bins, so every marker carries the same weight. Twelve leaves a few
# hundred frames per bin at either take's length.
BIN_COUNT = 12

# Same cut as 07: a solve this far out is a detection failure, not a pose, and
# a handful of them would set the slope of a fit spanning under a millimetre.
GROSS_ERROR_M = 0.05

OUTPUT_SUBDIR = "phase_check"
OUTPUT_DIR = None


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="One-figure summary of phase against stereo error."
    )
    parser.add_argument("--recording", type=Path, default=None)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__" and "ipykernel" not in sys.modules:
    _args = _parse_args(sys.argv[1:])
    if _args.recording is not None:
        RECORDING_DIR = _args.recording
    if _args.condition is not None:
        CONDITION = _args.condition
    if _args.output_dir is not None:
        OUTPUT_DIR = _args.output_dir

RECORDING_DIR = Path(RECORDING_DIR)
if not RECORDING_DIR.is_absolute():
    RECORDING_DIR = PROJECT_ROOT / RECORDING_DIR
if OUTPUT_DIR is None:
    OUTPUT_DIR = RECORDING_DIR / OUTPUT_SUBDIR
OUTPUT_DIR = Path(OUTPUT_DIR)
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

TAKE_NAME = RECORDING_DIR.name
SAMPLES_CSV = RECORDING_DIR / "movement_error" / "movement_error_samples.csv"
INJECTION_CSV = RECORDING_DIR / "timing_sensitivity" / "timing_injection.csv"
SAVE_FIGURE = OUTPUT_DIR / "phase_vs_error.png"
SAVE_SUMMARY = OUTPUT_DIR / "phase_vs_error.csv"

for required in (SAMPLES_CSV, INJECTION_CSV):
    if not required.exists():
        raise FileNotFoundError(
            f"Missing {required}. Run 06_movement_error.py and "
            "07_timing_sensitivity.py on this recording first."
        )

print(f"Recording : {RECORDING_DIR}")
print(f"Condition : {CONDITION}")


# %% Per-frame phase
# Recomputed from the sensor timestamps rather than read from
# phase_log.msgpack, which the recorder samples only about once a second.
cam0 = load_timestamp_records(RECORDING_DIR / "cam0_timestamp.msgpack")
cam1 = load_timestamp_records(RECORDING_DIR / "cam1_timestamp.msgpack")
cam0_sensor_ns = cam0["sensor_ns"]
cam1_sensor_ns = cam1["sensor_ns"]
frame_period_us = float(np.median(np.diff(cam0_sensor_ns)) / 1000.0)


def nearest_index(reference_ns, other_ns):
    """Nearest `other` frame for each reference frame; the cameras free-run."""
    insert = np.searchsorted(other_ns, reference_ns)
    left = np.clip(insert - 1, 0, len(other_ns) - 1)
    right = np.clip(insert, 0, len(other_ns) - 1)
    take_left = np.abs(reference_ns - other_ns[left]) <= np.abs(
        reference_ns - other_ns[right]
    )
    return np.where(take_left, left, right)


def wrap_to_half_period(delta_us, period_us):
    """Phase is defined modulo the frame period; a whole period is lockstep."""
    return ((delta_us + period_us / 2.0) % period_us) - period_us / 2.0


match = nearest_index(cam0_sensor_ns, cam1_sensor_ns)
phase_us = wrap_to_half_period(
    (cam1_sensor_ns[match] - cam0_sensor_ns) / 1000.0, frame_period_us
)


# %% Join the per-frame errors onto the phase
all_samples = pd.read_csv(SAMPLES_CSV)
samples = all_samples.loc[all_samples["condition"] == CONDITION].sort_values("time_s")
if samples.empty:
    raise ValueError(
        f"No rows for condition {CONDITION} in {SAMPLES_CSV}. "
        f"Available: {sorted(all_samples['condition'].unique())}"
    )
samples = samples.reset_index(drop=True)
samples["phase_us"] = phase_us[samples["frame"].to_numpy()]

# Direction of travel from the mocap track this condition was scored against.
# A central difference, so the direction is not biased half a frame forward.
time_s = samples["time_s"].to_numpy()
mocap = samples[["mocap_x", "mocap_y", "mocap_z"]].to_numpy()
velocity = np.gradient(mocap, time_s, axis=0)
speed = np.linalg.norm(velocity, axis=1)
direction = velocity / np.maximum(speed[:, None], 1e-9)

error_vector = samples[["error_x_m", "error_y_m", "error_z_m"]].to_numpy()
samples["along_track_mm"] = 1000.0 * np.einsum("ij,ij->i", error_vector, direction)
# What pure latency would put on that axis: signed, because the estimate must
# lag when cam1 is late and lead when it is early.
samples["signed_mm"] = 1000.0 * samples["phase_us"] * 1e-6 * samples["speed_ms"]
samples = samples.loc[
    np.isfinite(samples["signed_mm"]) & (samples["error_m"] < GROSS_ERROR_M)
].reset_index(drop=True)

print(
    f"Frame period {frame_period_us:.0f} us; phase "
    f"{samples['phase_us'].min():+.0f} to {samples['phase_us'].max():+.0f} us "
    f"over {len(samples)} frames"
)


# %% Bin the cloud
def equal_count_bins(x, y, bin_count):
    """Median of y in equal-count bins of x, with the interquartile range."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bin_count + 1)))
    index = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for bin_index in range(len(edges) - 1):
        inside = index == bin_index
        if inside.sum() < 10:
            continue
        rows.append(
            {
                "x_median": float(np.median(x[inside])),
                "y_median": float(np.median(y[inside])),
                "y_q25": float(np.quantile(y[inside], 0.25)),
                "y_q75": float(np.quantile(y[inside], 0.75)),
                "n": int(inside.sum()),
            }
        )
    return pd.DataFrame(rows)


signed = samples["signed_mm"].to_numpy()
along = samples["along_track_mm"].to_numpy()
binned = equal_count_bins(signed, along, BIN_COUNT)
observed_slope, observed_intercept = np.polyfit(signed, along, 1)
# The correlation is the honest headline: the phase spans well under a
# millimetre here, so a slope fitted across it swings on very little.
observed_r = float(np.corrcoef(signed, along)[0, 1])


# %% The injection ladder, for scale
injection = pd.read_csv(INJECTION_CSV)
ladder = (
    injection.groupby("offset_frames")
    .agg(
        offset_ms=("offset_ms", "median"),
        shift_mm=("shift_mm", "median"),
        shift_q25=("shift_mm", lambda column: column.quantile(0.25)),
        shift_q75=("shift_mm", lambda column: column.quantile(0.75)),
        n=("shift_mm", "size"),
    )
    .reset_index()
    .sort_values("offset_ms")
)
moved = ladder.loc[ladder["offset_frames"] != 0]
# mm of position per ms of timing error, from the offsets that were injected.
sensitivity_mm_per_ms = float(
    np.median(moved["shift_mm"].to_numpy() / np.abs(moved["offset_ms"].to_numpy()))
)
real_phase_ms = float(np.median(np.abs(samples["phase_us"]))) / 1000.0
expected_from_real_phase_mm = sensitivity_mm_per_ms * real_phase_ms
noise_floor_mm = float(np.std(along))
noise_ratio = noise_floor_mm / max(expected_from_real_phase_mm, 1e-9)

print(
    f"Injection sensitivity {sensitivity_mm_per_ms:.3f} mm/ms; "
    f"real phase {real_phase_ms * 1000:.0f} us predicts "
    f"{expected_from_real_phase_mm:.3f} mm against a "
    f"{noise_floor_mm:.2f} mm along-track spread ({noise_ratio:.0f}x larger)"
)
print(
    f"Observed r {observed_r:+.3f}, slope {observed_slope:+.2f} mm/mm "
    "(pure latency would be r ~ +1 and slope +1.00)"
)


# %% Summary
summary = pd.DataFrame(
    [
        {
            "take": TAKE_NAME,
            "condition": CONDITION,
            "samples": len(samples),
            "frame_period_us": round(frame_period_us, 1),
            "phase_median_abs_us": round(real_phase_ms * 1000, 1),
            "observed_r": round(observed_r, 4),
            "observed_slope_mm_per_mm": round(float(observed_slope), 3),
            "sensitivity_mm_per_ms": round(sensitivity_mm_per_ms, 4),
            "expected_shift_mm": round(expected_from_real_phase_mm, 4),
            "along_track_noise_mm": round(noise_floor_mm, 3),
            "noise_over_expected": round(noise_ratio, 1),
        }
    ]
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
summary.to_csv(SAVE_SUMMARY, index=False)
print()
print(summary.T.to_string(header=False))


# %% Figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# --- Left: does the error follow the phase? --------------------------------
ax = axes[0]
ax.scatter(signed, along, s=4, alpha=0.12, color="tab:blue", label="per frame")
ax.errorbar(
    binned["x_median"],
    binned["y_median"],
    yerr=[
        binned["y_median"] - binned["y_q25"],
        binned["y_q75"] - binned["y_median"],
    ],
    linestyle="-",
    marker="o",
    markersize=5,
    linewidth=1.6,
    capsize=3,
    color="tab:blue",
    label=f"median of {len(binned)} equal-count bins (bars: IQR)",
)
span = np.array([signed.min(), signed.max()])
ax.plot(
    span,
    span,
    color="black",
    linestyle="--",
    linewidth=1.4,
    label="slope +1: pure latency",
)
ax.plot(
    span,
    observed_slope * span + observed_intercept,
    color="tab:red",
    linewidth=1.4,
    label=f"fit, slope {observed_slope:+.2f} (r {observed_r:+.2f})",
)
ax.axhline(0, color="0.7", linewidth=0.8)
ax.axvline(0, color="0.7", linewidth=0.8)
ax.set_xlabel("Signed phase x speed [mm]")
ax.set_ylabel("Along-track error [mm]")
ax.set_title("Error does not follow the phase")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper left", fontsize=8, markerscale=3)

# --- Right: how big an effect could this have seen? ------------------------
ax = axes[1]
ax.errorbar(
    ladder["offset_ms"],
    ladder["shift_mm"],
    yerr=[
        ladder["shift_mm"] - ladder["shift_q25"],
        ladder["shift_q75"] - ladder["shift_mm"],
    ],
    linestyle="-",
    marker="o",
    markersize=5,
    linewidth=1.6,
    capsize=3,
    color="tab:green",
    label="cam1 re-paired 1 and 2 frames away",
)
ax.axhline(
    noise_floor_mm,
    color="tab:red",
    linestyle="--",
    linewidth=1.2,
    label=f"along-track noise {noise_floor_mm:.1f} mm",
)
# The real phase, on the same axes: a sliver next to anything injected.
ax.axvspan(
    -real_phase_ms,
    real_phase_ms,
    color="tab:orange",
    alpha=0.3,
    label=f"this take's phase (+/-{real_phase_ms * 1000:.0f} us "
    f"-> {expected_from_real_phase_mm:.2f} mm)",
)
ax.set_xlabel("Injected timing offset [ms]")
ax.set_ylabel("Position shift [mm]")
ax.set_title(f"The same solve does see timing: {sensitivity_mm_per_ms:.2f} mm per ms")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper center", fontsize=8)

fig.suptitle(
    f"Phase against stereo error - {TAKE_NAME}, {CONDITION}\n"
    f"real phase would move position {expected_from_real_phase_mm:.2f} mm, "
    f"{noise_ratio:.0f}x below the noise"
)
fig.savefig(SAVE_FIGURE, dpi=180, bbox_inches="tight")
print(f"\nSaved figure -> {SAVE_FIGURE}")
print(f"Saved summary -> {SAVE_SUMMARY}")

plt.show()
