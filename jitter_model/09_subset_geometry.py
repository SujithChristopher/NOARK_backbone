# %% [markdown]
# # Subset geometry versus jitter
#
# `05_static_jitter.py` compares eight fixed conditions (1-4 tags, one or two
# cameras). This script sweeps every co-visible tag subset of every burst in
# the same `dome_static_burst_sep18_26` recording, up to `MAX_TAGS` tags, and
# fits a within-burst fixed-effects model of log jitter on that subset's exact
# board-frame geometry (baseline, spread, angle, lever) plus its solved pose
# (distance, incidence). The burst fixed effect absorbs pose, distance and
# lighting exactly, so each geometry coefficient is identified from subsets
# compared against other subsets solved in the *same* burst.
#
# A subset is chosen once per burst and frozen for every frame in it (see
# `jitter_model._subset_worker.solve_burst`); choosing per frame would let the
# estimator change inside a burst and that switching would land in the
# standard deviation as if it were jitter. No temporal filtering anywhere.
#
# The burst grouping and mocap-stillness cells below are reused verbatim from
# `05_static_jitter.py` (same recording, same `bursts.json`): they produce
# `burst_frames` (every cam0 frame index per burst) and `static_bursts` (the
# subset of bursts the dome was actually still for), which is all this script
# needs from that machinery. Per-subset solving and geometry come from
# `jitter_model._subset_worker` / `jitter_model.geometry` / `jitter_model.jitter_stats`
# (Tasks 3-10); the model comes from `jitter_model.fixed_effects` and
# `jitter_model.model_assembly` (Task 12).
#
# Run from the repository root:
#
# ```powershell
# uv run python jitter_model/09_subset_geometry.py
# ```

# %% Imports
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import msgpack
import msgpack_numpy as mpn
import numpy as np
import pandas as pd
import toml
from scipy import stats
from tqdm.auto import tqdm

from jitter_model import (
    _subset_worker,
    common,
    fixed_effects,
    geometry,
    jitter_stats,
    model_assembly,
)
from support import pd_support

# %% Paths and experiment settings
try:
    NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:  # running cell-by-cell in an interactive kernel
    NOTEBOOK_DIR = Path.cwd()
    if NOTEBOOK_DIR.name != "jitter_model":
        NOTEBOOK_DIR = NOTEBOOK_DIR / "jitter_model"

PROJECT_ROOT = NOTEBOOK_DIR.parent

RECORDING_DIR = (
    PROJECT_ROOT / "data" / "dome" / "sep18_26" / "dome_static_burst_sep18_26"
)
OUTPUT_SUBDIR = "subset_geometry"
CAMERA_NAMES = ("cam0", "cam1")
CAMERA_CONFIGS = ("cam0", "stereo")
MAX_TAGS = 4
TAG_PRESENCE_FRACTION = 0.9
MIN_FRAMES_PER_BURST = 10
MAX_BURST_MOCAP_TRAVEL_M = 0.002
MAX_PAIR_FRACTION_OF_FRAME = 0.55
# Report every subset at one physical point so the numbers are comparable;
# zero is the reference tag's origin, which is what 05 reports.
P_FIXED = np.zeros(3)
WORKERS = None  # None -> os.cpu_count()
# A predictor above this VIF is collapsed into the composite spread term.
MAX_VIF = 10.0
RESPONSES = ("pos_jitter_mm", "rot_jitter_mdeg")

MOCAP_CSV = RECORDING_DIR / f"{RECORDING_DIR.name}.csv"
BURSTS_JSON = RECORDING_DIR / "bursts.json"
SESSION_TOML = RECORDING_DIR.parent / "session.toml"
DETECTION_CACHE = RECORDING_DIR / "jitter_detections.pkl"

OUTPUT_DIR = RECORDING_DIR / OUTPUT_SUBDIR
CELLS_CSV = OUTPUT_DIR / "subset_geometry_cells.csv"
MODEL_CSV = OUTPUT_DIR / "subset_geometry_model.csv"
BASELINE_FIGURE = OUTPUT_DIR / "subset_jitter_vs_baseline.png"
LEVER_FIGURE = OUTPUT_DIR / "subset_jitter_vs_lever.png"
CORRELATION_FIGURE = OUTPUT_DIR / "subset_predictor_correlations.png"
PARTIAL_RESIDUALS_FIGURE = OUTPUT_DIR / "subset_partial_residuals.png"
POSITION_VS_ROTATION_FIGURE = OUTPUT_DIR / "subset_position_vs_rotation.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Recording : {RECORDING_DIR}")
print(f"Outputs   : {OUTPUT_DIR}")


# %% Calibration and tag rig metadata
if not SESSION_TOML.exists():
    raise FileNotFoundError(f"Missing {SESSION_TOML}")
session = toml.load(SESSION_TOML)


def session_path(key):
    """Resolve one of session.toml's calibration paths, which are relative to it."""
    return (SESSION_TOML.parent / session["calibration"][key]).resolve()


STEREO_TOML = session_path("stereo")
RIGIDBODY_TOML = session_path("rigid_body")
MOCAP_BODY_X_FROM = tuple(session["mocap"]["x_from"])
MOCAP_BODY_Z_FROM = tuple(session["mocap"]["z_from"])
MOCAP_BODY_ORIGIN = session["mocap"]["origin"]

for required_path in (STEREO_TOML, RIGIDBODY_TOML, MOCAP_CSV, BURSTS_JSON):
    if not required_path.exists():
        raise FileNotFoundError(f"Missing {required_path}")

stereo = toml.load(STEREO_TOML)
rigidbody = toml.load(RIGIDBODY_TOML)
# Only what's needed to validate/load the detection cache and to build the
# worker payload; the rig itself (`common.build_tag_rig`) and the stereo
# extrinsic are rebuilt inside each worker process from the raw TOML dicts.
TAG_SIZE_M = float(rigidbody["meta"]["tag_size_m"])
DETECT_MARKER_IDS = tuple(int(x) for x in rigidbody["meta"]["marker_ids"])


# %% Load the reusable detection cache built by 05_static_jitter.py
detection_cache = common.load_detection_cache(
    DETECTION_CACHE,
    marker_ids=DETECT_MARKER_IDS,
    tag_size_m=TAG_SIZE_M,
    recording_dir=RECORDING_DIR,
    camera_names=CAMERA_NAMES,
)
if detection_cache is None:
    raise RuntimeError(
        f"No usable detection cache at {DETECTION_CACHE}; run "
        "05_static_jitter.py on this recording first to build it."
    )
print(f"Loaded reusable detections <- {DETECTION_CACHE}")

cam0 = detection_cache["cameras"]["cam0"]
cam1 = detection_cache["cameras"]["cam1"]


# %% Pair the two free-running cameras on the shared host-monotonic clock
cam0_monotonic_ns = cam0["metadata"]["monotonic_ns"]
cam1_monotonic_ns = cam1["metadata"]["monotonic_ns"]
cam1_pair = common.pair_cameras(
    cam0_monotonic_ns, cam1_monotonic_ns, MAX_PAIR_FRACTION_OF_FRAME
)
paired_gaps_ms = np.asarray(
    [
        abs(int(cam1_monotonic_ns[match]) - int(cam0_monotonic_ns[index])) / 1e6
        for index, match in enumerate(cam1_pair)
        if match >= 0
    ]
)
print(
    f"Camera pairing: {len(paired_gaps_ms)}/{len(cam0_monotonic_ns)} frames, "
    f"median gap={np.median(paired_gaps_ms):.2f} ms, "
    f"max accepted={np.max(paired_gaps_ms):.2f} ms"
)


# %% Burst grouping
# Reused verbatim from 05_static_jitter.py's "Burst grouping" cell: same
# recording, same bursts.json, produces `burst_frames` for build_tasks below.
bursts_meta = json.loads(BURSTS_JSON.read_text())
burst_records = {int(entry["index"]): entry for entry in bursts_meta["bursts"]}
cam0_burst = cam0["metadata"]["burst"]
burst_indices = np.asarray(sorted({int(b) for b in cam0_burst if b > 0}), dtype=int)
if not len(burst_indices):
    raise RuntimeError(
        "No burst column in the camera timestamps; this take is not a burst recording"
    )
burst_frames = {
    int(index): np.flatnonzero(cam0_burst == index) for index in burst_indices
}
print(
    f"Bursts: {len(burst_indices)}, "
    f"{min(len(f) for f in burst_frames.values())}-{max(len(f) for f in burst_frames.values())} "
    "cam0 frames each"
)


# %% Mocap: fit the single camera-to-Motive time offset, then use it as a check
# Reused verbatim from 05_static_jitter.py's mocap-stillness cells: same
# recording, same session mocap body definition, produces `static_bursts`.
def read_mocap_body(path):
    """Motive take with its body frame rebuilt from the labelled markers.

    Parsing and the marker-edge basis come from ``support.pd_support`` so every
    script in the repository reads Motive exports the same way. The axes are
    orthonormalized from two measured marker edges rather than taken from
    Motive's solved quaternion, whose rigid body can re-lock at a ghost pose
    after an occlusion.
    """
    table, capture_start = pd_support.read_rigid_body_csv(str(path))
    positions, _rotations = pd_support.rigid_body_marker_frames(
        table,
        x_from=MOCAP_BODY_X_FROM,
        z_from=MOCAP_BODY_Z_FROM,
        origin=MOCAP_BODY_ORIGIN,
    )
    seconds = table["seconds"].to_numpy(dtype=np.float64)
    finite = np.isfinite(seconds) & np.isfinite(positions).all(axis=1)
    return seconds[finite], positions[finite], capture_start


mocap_seconds, mocap_positions, mocap_capture_start = read_mocap_body(MOCAP_CSV)
print(f"Mocap: {len(mocap_seconds)} frames, capture start {mocap_capture_start}")

sync_high = np.flatnonzero(cam0["metadata"]["sync"] == 1)
sync_events_path = RECORDING_DIR / "sync_events.msgpack"
if sync_events_path.exists():
    with sync_events_path.open("rb") as stream:
        sync_records = list(msgpack.Unpacker(stream, object_hook=mpn.decode))
    sync_monotonic_ns = int(sync_records[0][0])
elif len(sync_high):
    sync_monotonic_ns = int(cam0_monotonic_ns[sync_high[0]])
else:
    raise RuntimeError("No GPIO sync edge to anchor the camera clock")

burst_start_s = np.asarray(
    [
        (burst_records[i]["mono_start_ns"] - sync_monotonic_ns) / 1e9
        for i in burst_indices
    ]
)
burst_end_s = np.asarray(
    [(burst_records[i]["mono_end_ns"] - sync_monotonic_ns) / 1e9 for i in burst_indices]
)


# The GPIO rise anchors the camera clock but does not zero Motive's: in this
# take the first burst sits 5.2 s after the rise while its mocap plateau sits at
# 13.9 s. Rather than cross-correlating two signals, the known burst schedule is
# slid as a rigid template and scored on how still the mocap is inside the burst
# windows, which is a one-parameter fit with a physically meaningful optimum.
def window_travel(offset_s):
    """How far the dome moves inside each burst window, shifted by ``offset_s``.

    Travel, not speed: at 100 Hz the marker noise divided by the sample interval
    puts a stationary dome at several mm/s, which leaves almost no contrast
    between a still window and a moving one. The span a window covers has no
    such floor, and separates the two by more than two orders of magnitude.

    Windows are gathered by binary search rather than by masking the whole take
    once per window, since the search below evaluates this a thousand times.
    """
    left = np.searchsorted(mocap_seconds, burst_start_s + offset_s, side="left")
    right = np.searchsorted(mocap_seconds, burst_end_s + offset_s, side="right")
    spans = [
        float(
            np.linalg.norm(
                mocap_positions[a:b].max(axis=0) - mocap_positions[a:b].min(axis=0)
            )
        )
        for a, b in zip(left, right)
        if b - a >= MIN_FRAMES_PER_BURST
    ]
    if len(spans) < len(burst_indices) // 2:
        return np.inf
    return float(np.median(spans))


def search_offset(centre, half_width, step):
    candidates = np.arange(centre - half_width, centre + half_width + step, step)
    values = np.asarray([window_travel(offset) for offset in candidates])
    best = int(np.argmin(values))
    return float(candidates[best]), float(values[best]), values


# Coarse pass over the plausible range, then a fine pass around the winner. The
# optimum is sharp because the dome is moved between bursts, so a misaligned
# template straddles the transits and scores far worse.
coarse_offset, _, coarse_values = search_offset(15.0, 45.0, 0.1)
MOCAP_OFFSET_S, best_travel_m, _ = search_offset(coarse_offset, 0.2, 0.005)
# Accept on the absolute criterion that the fit is meant to satisfy: at the
# right offset the dome is still inside the windows, to the same tolerance a
# burst must meet to be used at all. Comparing against a typical offset instead
# would be too weak a test, because the dome is stationary for most of the take
# and most offsets therefore land mostly in dwells anyway; the worst offset is
# what a real mismatch looks like, and it is reported for scale.
worst_travel_m = float(np.nanmax(coarse_values[np.isfinite(coarse_values)]))
mocap_usable = np.isfinite(best_travel_m) and best_travel_m < MAX_BURST_MOCAP_TRAVEL_M
print(
    f"Mocap time offset: {MOCAP_OFFSET_S:+.3f} s "
    f"(median in-burst travel {1000 * best_travel_m:.2f} mm, "
    f"against {1000 * worst_travel_m:.1f} mm at the worst offset and a "
    f"{1000 * MAX_BURST_MOCAP_TRAVEL_M:.0f} mm tolerance)"
)
if not mocap_usable:
    warnings.warn(
        "Burst schedule did not lock onto the mocap stillness pattern; "
        "mocap checks and the noise floor are skipped"
    )


# %% Per-burst mocap stillness and noise floor
def mocap_window(burst_index):
    position = int(np.flatnonzero(burst_indices == burst_index)[0])
    start = burst_start_s[position] + MOCAP_OFFSET_S
    end = burst_end_s[position] + MOCAP_OFFSET_S
    inside = (mocap_seconds >= start) & (mocap_seconds <= end)
    return mocap_positions[inside]


mocap_burst = {}
if mocap_usable:
    for index in burst_indices:
        window = mocap_window(int(index))
        if len(window) < MIN_FRAMES_PER_BURST:
            continue
        mocap_burst[int(index)] = {
            "mocap_frames": len(window),
            "mocap_travel_m": float(
                np.linalg.norm(window.max(axis=0) - window.min(axis=0))
            ),
            "mocap_jitter_3d_mm": float(
                np.linalg.norm(1000.0 * np.std(window, axis=0, ddof=1))
            ),
        }
    travels = np.asarray([v["mocap_travel_m"] for v in mocap_burst.values()])
    static_bursts = {
        index
        for index, v in mocap_burst.items()
        if v["mocap_travel_m"] <= MAX_BURST_MOCAP_TRAVEL_M
    }
    print(
        f"Mocap stillness: {len(static_bursts)}/{len(mocap_burst)} bursts move less "
        f"than {1000 * MAX_BURST_MOCAP_TRAVEL_M:.0f} mm "
        f"(median travel {1000 * np.median(travels):.2f} mm)"
    )
    MOCAP_FLOOR_MM = float(
        np.median([v["mocap_jitter_3d_mm"] for v in mocap_burst.values()])
    )
    print(f"Mocap noise floor: {MOCAP_FLOOR_MM:.3f} mm median 3D jitter per burst")
else:
    static_bursts = {int(i) for i in burst_indices}
    MOCAP_FLOOR_MM = np.nan


# %% Build the work list and the worker payload
def build_tasks():
    """One task per (burst, camera config), carrying that burst's frame indices."""
    return [
        (burst, camera_config, burst_frames[burst].tolist())
        for burst in sorted(static_bursts)
        for camera_config in CAMERA_CONFIGS
    ]


def build_payload():
    return {
        "cache_path": str(DETECTION_CACHE),
        "rigidbody": rigidbody,
        "stereo": stereo,
        "camera_names": list(CAMERA_NAMES),
        "options": {
            "pairs": cam1_pair,
            "max_tags": MAX_TAGS,
            "presence_fraction": TAG_PRESENCE_FRACTION,
            "min_frames": MIN_FRAMES_PER_BURST,
            "fixed_point": P_FIXED.tolist(),
        },
    }


def run_sweep(tasks, payload, workers=None):
    """Solve every task, in a process pool unless `workers` is 1.

    The serial path exists so the parallel result can be checked against it and
    so the script still runs cell-by-cell in a kernel, where a pool would try
    to re-import __main__.
    """
    if workers == 1:
        _subset_worker.init_worker(payload)
        return [
            row
            for task in tqdm(tasks, desc="Solving subsets")
            for row in _subset_worker.worker_burst(task)
        ]

    rows = []
    with ProcessPoolExecutor(
        max_workers=workers or os.cpu_count(),
        initializer=_subset_worker.init_worker,
        initargs=(payload,),
    ) as pool:
        futures = [pool.submit(_subset_worker.worker_burst, task) for task in tasks]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Solving subsets"
        ):
            rows.extend(future.result())
    return rows


# %% Run it
#
# Decision 2 (task-11-12 briefing): the briefed guard was
# `if __name__ == "__main__" and "ipykernel" not in sys.modules: ... else: ...`
# with the `else` branch running the sweep serially. That is wrong for tests:
# `importlib.util.exec_module` gives the module the name "subset_geometry", so
# the `else` branch fires on a plain *import* and the whole multi-minute sweep
# would run just from importing the script (e.g. test_subset_sweep_parallel's
# `_load_script()`). Guarding on `__name__ == "__main__"` alone -- true for
# both a CLI run and a Jupyter kernel cell, false for an exec_module import --
# avoids that, and the ipykernel check only decides worker count (a pool
# re-imports __main__, which breaks inside a kernel).
#
# The guard is repeated on every cell below that runs the sweep/model/figures
# (rather than wrapped once around all of them) so each `# %%` cell stays a
# syntactically complete, independently re-runnable unit in a kernel -- a cell
# body indented under a PREVIOUS cell's `if` would raise an IndentationError
# if executed alone.
if __name__ == "__main__":
    workers = 1 if "ipykernel" in sys.modules else WORKERS
    cells = run_sweep(build_tasks(), build_payload(), workers)

    cell_table = (
        pd.DataFrame(cells)
        .sort_values(["camera_config", "burst", "n_tags", "marker_ids"])
        .reset_index(drop=True)
    )
    cell_table.to_csv(CELLS_CSV, index=False)
    print(f"{len(cell_table)} cells -> {CELLS_CSV}")


# %% Fit, within burst
POSE_PREDICTORS = ["distance_m", "mean_incidence_deg", "mean_apparent_size_px"]

if __name__ == "__main__":
    model_rows = []
    for camera_config in CAMERA_CONFIGS:
        subset = cell_table.loc[
            cell_table["camera_config"] == camera_config
        ].reset_index(drop=True)
        # Decision 4: the angle predictors (`max_normal_angle_deg`,
        # `mean_normal_angle_deg`) are NaN for every n_tags == 1 row. The
        # correlation matrix and the model below are both restricted to the
        # n_tags >= 2 rows where that predictor is finite; the single-tag rows
        # are summarised separately as the reference condition.
        finite_angle = np.isfinite(subset["max_normal_angle_deg"].to_numpy(dtype=float))
        finite_cells = subset.loc[finite_angle].reset_index(drop=True)
        single_cells = subset.loc[~finite_angle].reset_index(drop=True)

        design, names = model_assembly.build_design(
            finite_cells, model_assembly.GEOMETRY_PREDICTORS
        )
        correlation = pd.DataFrame(design, columns=names).corr()
        print(
            f"\n[{camera_config}] predictor correlation matrix "
            f"(n={len(finite_cells)} multi-tag rows):"
        )
        print(correlation.round(3).to_string())
        print(
            f"[{camera_config}] corr(log_max_baseline_mm, log_max_normal_angle_deg) = "
            f"{correlation.loc['log_max_baseline_mm', 'log_max_normal_angle_deg']:.3f}"
        )
        print(
            f"[{camera_config}] single-tag reference condition: {len(single_cells)} rows, "
            f"median pos_jitter_mm={single_cells['pos_jitter_mm'].median():.4f}, "
            f"median rot_jitter_mdeg={single_cells['rot_jitter_mdeg'].median():.4f}"
        )

        for response in RESPONSES:
            result, coef_names, folded, fit_cells, _single = (
                model_assembly.fit_response(subset, response, max_vif=MAX_VIF)
            )
            if folded:
                print(
                    f"[{camera_config}/{response}] folded into composite_spread: {folded}"
                )
            for predictor, coefficient, se, lo, hi, vif in zip(
                coef_names,
                result.coefficients,
                result.standard_errors,
                result.ci_low,
                result.ci_high,
                result.vif,
            ):
                model_rows.append(
                    {
                        "camera_config": camera_config,
                        "response": response,
                        "stage": "within",
                        "predictor": predictor,
                        "coefficient": float(coefficient),
                        "standard_error": float(se),
                        "ci_low": float(lo),
                        "ci_high": float(hi),
                        "vif": float(vif),
                    }
                )

            # %% Between-burst stage
            # The stage-1 residual (log response minus the fitted geometry
            # effect, left un-demeaned) still carries each burst's fixed
            # effect. Averaging it per burst and regressing those burst means
            # on that burst's typical distance and incidence checks whether
            # the geometry model's burst effect tracks the pose it was solved
            # from, the way `POSE_PREDICTORS` suggests it should.
            burst_means = (
                fit_cells.groupby("burst")
                .agg(
                    stage1_residual=("stage1_residual", "mean"),
                    distance_m=("distance_m", "mean"),
                    mean_incidence_deg=("mean_incidence_deg", "mean"),
                )
                .reset_index()
            )
            between_predictors = np.column_stack(
                [
                    np.log(
                        burst_means["distance_m"].to_numpy(dtype=float)
                        + model_assembly.LOG_FLOOR
                    ),
                    burst_means["mean_incidence_deg"].to_numpy(dtype=float),
                ]
            )
            between_names = ["log_distance_m", "mean_incidence_deg"]
            design_between = np.column_stack(
                [np.ones(len(between_predictors)), between_predictors]
            )
            y_between = burst_means["stage1_residual"].to_numpy(dtype=float)
            beta, *_ = np.linalg.lstsq(design_between, y_between, rcond=None)
            residual_between = y_between - design_between @ beta
            dof = max(len(y_between) - design_between.shape[1], 1)
            sigma2 = float(np.sum(residual_between**2) / dof)
            covariance = sigma2 * np.linalg.pinv(design_between.T @ design_between)
            se_between = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
            critical = stats.t.ppf(0.975, dof)
            between_vif = fixed_effects.variance_inflation(between_predictors)
            vif_by_name = {
                "intercept": float("nan"),
                "log_distance_m": float(between_vif[0]),
                "mean_incidence_deg": float(between_vif[1]),
            }
            for name, coefficient, se in zip(
                ["intercept", *between_names], beta, se_between
            ):
                model_rows.append(
                    {
                        "camera_config": camera_config,
                        "response": response,
                        "stage": "between",
                        "predictor": name,
                        "coefficient": float(coefficient),
                        "standard_error": float(se),
                        "ci_low": float(coefficient - critical * se),
                        "ci_high": float(coefficient + critical * se),
                        "vif": vif_by_name[name],
                    }
                )

    print(
        "Caveat: a subset exists only when its tags are visible, and visibility "
        "correlates with incidence angle. Burst fixed effects do not remove this."
    )
    print(
        "Caveat: lever_mm is constant per subset across bursts and is identified "
        "only relative to P_FIXED. Dropping it would attribute 'far from the "
        "reported point' to 'widely separated'."
    )

    model_table = pd.DataFrame(model_rows)
    model_table.to_csv(MODEL_CSV, index=False)
    print(f"{len(model_table)} model rows -> {MODEL_CSV}")


# %% Figures
if __name__ == "__main__":
    TAKE_NAME = RECORDING_DIR.name
    COLORMAP = "turbo"

    fig, axes = plt.subplots(1, len(CAMERA_CONFIGS), figsize=(12, 5), sharey=True)
    for axis, camera_config in zip(axes, CAMERA_CONFIGS):
        subset = cell_table.loc[cell_table["camera_config"] == camera_config]
        scatter = axis.scatter(
            subset["max_baseline_mm"],
            subset["pos_jitter_mm"],
            c=subset["n_tags"],
            cmap=COLORMAP,
            s=14,
            alpha=0.7,
        )
        axis.set_yscale("log")
        axis.set_xlabel("max_baseline_mm")
        axis.set_title(camera_config)
    axes[0].set_ylabel("pos_jitter_mm (log)")
    fig.colorbar(scatter, ax=axes, label="n_tags")
    fig.suptitle(f"Subset jitter versus baseline -- {TAKE_NAME}")
    fig.savefig(BASELINE_FIGURE, dpi=200, bbox_inches="tight")
    print(f"Saved {BASELINE_FIGURE}")

    fig, axes = plt.subplots(1, len(CAMERA_CONFIGS), figsize=(12, 5), sharey=True)
    for axis, camera_config in zip(axes, CAMERA_CONFIGS):
        subset = cell_table.loc[cell_table["camera_config"] == camera_config]
        scatter = axis.scatter(
            subset["lever_mm"],
            subset["pos_jitter_mm"],
            c=subset["n_tags"],
            cmap=COLORMAP,
            s=14,
            alpha=0.7,
        )
        axis.set_yscale("log")
        axis.set_xlabel("lever_mm")
        axis.set_title(camera_config)
    axes[0].set_ylabel("pos_jitter_mm (log)")
    fig.colorbar(scatter, ax=axes, label="n_tags")
    fig.suptitle(f"Subset jitter versus lever arm -- {TAKE_NAME}")
    fig.savefig(LEVER_FIGURE, dpi=200, bbox_inches="tight")
    print(f"Saved {LEVER_FIGURE}")

    fig, axes = plt.subplots(1, len(CAMERA_CONFIGS), figsize=(13, 5.5))
    for axis, camera_config in zip(axes, CAMERA_CONFIGS):
        subset = cell_table.loc[cell_table["camera_config"] == camera_config]
        finite_angle = np.isfinite(subset["max_normal_angle_deg"].to_numpy(dtype=float))
        design, names = model_assembly.build_design(
            subset.loc[finite_angle], model_assembly.GEOMETRY_PREDICTORS
        )
        correlation = pd.DataFrame(design, columns=names).corr().to_numpy()
        image = axis.imshow(correlation, cmap=COLORMAP, vmin=-1, vmax=1)
        axis.set_xticks(range(len(names)))
        axis.set_yticks(range(len(names)))
        axis.set_xticklabels(names, rotation=90, fontsize=7)
        axis.set_yticklabels(names, fontsize=7)
        axis.set_title(camera_config)
        for row in range(len(names)):
            for column in range(len(names)):
                axis.text(
                    column,
                    row,
                    f"{correlation[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                )
    fig.colorbar(image, ax=axes, label="correlation")
    fig.suptitle(f"Predictor correlations (n_tags >= 2 rows) -- {TAKE_NAME}")
    fig.savefig(CORRELATION_FIGURE, dpi=200, bbox_inches="tight")
    print(f"Saved {CORRELATION_FIGURE}")

    # Partial residuals for the richer (stereo) condition against pos_jitter_mm,
    # the primary response: y_within - sum_{j != i} coef_j x_within_j leaves
    # coef_i x_within_i + noise, so plotting it against x_i isolates that one
    # predictor's fitted slope from the others'.
    partial_camera_config = (
        "stereo" if "stereo" in CAMERA_CONFIGS else CAMERA_CONFIGS[0]
    )
    partial_response = "pos_jitter_mm"
    partial_subset = cell_table.loc[
        cell_table["camera_config"] == partial_camera_config
    ].reset_index(drop=True)
    result, coef_names, _folded, fit_cells, _single = model_assembly.fit_response(
        partial_subset, partial_response, max_vif=MAX_VIF
    )
    design, _names = model_assembly.build_design(
        fit_cells, model_assembly.GEOMETRY_PREDICTORS
    )
    design, coef_names_full, _folded_full = model_assembly.collapse_collinear(
        design, [f"log_{p}" for p in model_assembly.GEOMETRY_PREDICTORS], MAX_VIF
    )
    groups = fit_cells["burst"].to_numpy()
    weights = 2.0 * (fit_cells["frames"].to_numpy(dtype=float) - 1.0)
    x_within = fixed_effects.within_transform(design, groups, weights)
    y_within = fixed_effects.within_transform(
        np.log(fit_cells[partial_response].to_numpy(dtype=float)), groups, weights
    )
    fig, axes = plt.subplots(
        1, len(coef_names_full), figsize=(4.2 * len(coef_names_full), 4.2)
    )
    if len(coef_names_full) == 1:
        axes = [axes]
    for axis, predictor_index, predictor_name in zip(
        axes, range(len(coef_names_full)), coef_names_full
    ):
        other = [i for i in range(len(coef_names_full)) if i != predictor_index]
        partial_y = y_within - x_within[:, other] @ result.coefficients[other]
        axis.scatter(
            x_within[:, predictor_index], partial_y, s=10, alpha=0.6, color="tab:blue"
        )
        order = np.argsort(x_within[:, predictor_index])
        axis.plot(
            x_within[order, predictor_index],
            result.coefficients[predictor_index] * x_within[order, predictor_index],
            color="black",
            linewidth=1.5,
        )
        axis.set_xlabel(f"{predictor_name} (within-burst)")
        axis.set_title(predictor_name)
    axes[0].set_ylabel(f"partial residual, log({partial_response})")
    fig.suptitle(
        f"Partial residuals -- {partial_camera_config}/{partial_response} -- {TAKE_NAME}"
    )
    fig.savefig(PARTIAL_RESIDUALS_FIGURE, dpi=200, bbox_inches="tight")
    print(f"Saved {PARTIAL_RESIDUALS_FIGURE}")

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        cell_table["rot_jitter_mdeg"],
        cell_table["pos_jitter_mm"],
        c=cell_table["lever_mm"],
        cmap=COLORMAP,
        s=12,
        alpha=0.6,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rot_jitter_mdeg (log)")
    ax.set_ylabel("pos_jitter_mm (log)")
    fig.colorbar(scatter, ax=ax, label="lever_mm")
    ax.set_title(
        f"Position versus rotation jitter, coloured by lever arm -- {TAKE_NAME}"
    )
    fig.savefig(POSITION_VS_ROTATION_FIGURE, dpi=200, bbox_inches="tight")
    print(f"Saved {POSITION_VS_ROTATION_FIGURE}")
