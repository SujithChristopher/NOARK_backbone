# %% [markdown]
# # AprilTag refinement versus rapidtag crop refinement
#
# The realtime tracker in `NOARKGames/pyscripts/tracker.py` does not run the
# full-frame AprilTag line fit. rapidtag finds the tags with its fast ArUco
# detector and, with `april_tag_refine_full_resolution = True`, refits only the
# border of a small crop around each tag, in parallel. That is much faster, and
# its author reports a match to the full-resolution fit within 0.0001 px. The
# tracker nevertheless keeps it opt-in (`tracker_refine = "crop"`) because on this
# rig it once produced bad poses, so the claim needs measuring on real frames.
#
# Every frame of `dome_static_burst_sep18_26` is detected four ways:
#
# - `cv2_apriltag`: OpenCV `CORNER_REFINE_APRILTAG` on the full frame. This is the
#   `april_refine` mode of `05_static_jitter.py` and the reference here.
# - `rt_crop`: rapidtag with `april_tag_refine_full_resolution` (tracker "crop").
# - `rt_subpix`: rapidtag raw corners, then `cv2.cornerSubPix` (tracker default).
# - `rt_raw`: rapidtag raw corners, no refinement; the floor to beat.
#
# Three things are compared against the reference, none needing mocap:
#
# 1. corners: detection agreement and per-corner distance in pixels, plus the
#    in-burst corner scatter, which is detector noise before any pose solve;
# 2. poses: the same joint board PnP / stereo fit as `05b`, with every method on
#    the identical frames and tag subsets, so a difference is the corners alone;
# 3. cost: milliseconds per frame on this machine (the Pi will rank them the
#    same way but not scale the same).
#
# The dome is still inside each burst, so a burst's pose spread is jitter, and the
# gap between a method's burst-mean pose and the reference's is its systematic
# bias, which jitter cannot show. Accuracy here means agreement with the full
# AprilTag fit, not with ground truth.
#
# rapidtag must be new enough to have `april_tag_refine_full_resolution`
# (`external/rapidtag`, not the 0.1.4 wheel from PyPI):
#
# ```powershell
# uv pip install .\external\rapidtag
# uv run python jitter_model/10_crop_refine_accuracy.py
# ```

# %% Imports
import collections
import pickle
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import msgpack
import msgpack_numpy as mpn
import numpy as np
import pandas as pd
import rapidtag
import toml
from cv2 import aruco
from tqdm.auto import tqdm

from jitter_model import common

# %% Settings
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
BURSTS_JSON = RECORDING_DIR / "bursts.json"
SESSION_TOML = RECORDING_DIR.parent / "session.toml"
OUTPUT_DIR = RECORDING_DIR / "crop_refine_accuracy"
DETECTION_CACHE = RECORDING_DIR / "jitter_detections_crop_compare.pkl"
CACHE_VERSION = 1

CAMERA_NAMES = ("cam0", "cam1")
REBUILD_DETECTION_CACHE = False
MAX_FRAMES = None

# Reference first; the rest are compared against it, in this order.
REFERENCE = "cv2_apriltag"
METHODS = (REFERENCE, "rt_crop", "rt_subpix", "rt_raw")
DICTIONARY_NAME = "DICT_APRILTAG_36h11"
# cornerSubPix settings the tracker uses on rapidtag's raw corners.
SUBPIX_WINDOW = (5, 5)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)

CAMERA_COUNTS = (1, 2)
TAG_COUNTS = (1, 2, 4)
MIN_FRAMES_PER_BURST = 10
# A tag must appear, in every method, in this fraction of a burst's frames.
TAG_PRESENCE_FRACTION = 0.9
MAX_PAIR_FRACTION_OF_FRAME = 0.55
CORNER_ERROR_THRESHOLDS_PX = (0.05, 0.1, 0.5, 1.0)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Recording : {RECORDING_DIR}")
print(f"Outputs   : {OUTPUT_DIR}")

# %% Calibration, tag rig, and detectors
session = toml.load(SESSION_TOML)


def session_path(key):
    return (SESSION_TOML.parent / session["calibration"][key]).resolve()


stereo = toml.load(session_path("stereo"))
rigidbody = toml.load(session_path("rigid_body"))
camera_models = common.build_camera_models(stereo, CAMERA_NAMES)
R_STEREO, T_STEREO, _ = common.stereo_extrinsic(stereo, rigidbody)
rig = common.build_tag_rig(rigidbody)
solver = common.PoseSolver(rig, camera_models, R_STEREO, T_STEREO)
DETECT_MARKER_IDS = rig.marker_ids

cv2_parameters = aruco.DetectorParameters()
cv2_parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
cv2_detector = aruco.ArucoDetector(
    aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11), cv2_parameters
)

crop_parameters = rapidtag.DetectorParameters()
if not hasattr(crop_parameters, "april_tag_refine_full_resolution"):
    raise RuntimeError(
        f"rapidtag {getattr(rapidtag, '__version__', '?')} has no "
        "april_tag_refine_full_resolution, so there is no crop refinement to test. "
        "Install the build in external/rapidtag: uv pip install .\\external\\rapidtag"
    )
crop_parameters.corner_refinement_method = rapidtag.CORNER_REFINE_NONE
crop_parameters.april_tag_refine_full_resolution = True
raw_parameters = rapidtag.DetectorParameters()
raw_parameters.corner_refinement_method = rapidtag.CORNER_REFINE_NONE
raw_parameters.april_tag_refine_full_resolution = False
print(f"rapidtag {getattr(rapidtag, '__version__', '?')}, OpenCV {cv2.__version__}")


# %% Detection with every method on every frame
def as_detections(corners, ids):
    """Rig tags only, as {id: (4, 2) float64}."""
    found = {}
    for marker_id, marker_corners in zip(ids, corners):
        marker_id = int(marker_id)
        if marker_id in DETECT_MARKER_IDS:
            found[marker_id] = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
    return found


def subpixel(gray, detections):
    refined = {}
    for marker_id, corners in detections.items():
        points = corners.astype(np.float32).reshape(-1, 1, 2)
        cv2.cornerSubPix(gray, points, SUBPIX_WINDOW, (-1, -1), SUBPIX_CRITERIA)
        refined[marker_id] = points.reshape(4, 2).astype(np.float64)
    return refined


def detect_frame(gray):
    """Detections and elapsed milliseconds for every method on one frame."""
    results, timings = {}, {}

    start = time.perf_counter()
    corners, ids, _rejected = cv2_detector.detectMarkers(gray)
    timings[REFERENCE] = time.perf_counter() - start
    results[REFERENCE] = as_detections(corners, [] if ids is None else ids.ravel())

    start = time.perf_counter()
    corners, ids = rapidtag.detect_markers(gray, DICTIONARY_NAME, crop_parameters)
    timings["rt_crop"] = time.perf_counter() - start
    results["rt_crop"] = as_detections(corners, ids)

    start = time.perf_counter()
    corners, ids = rapidtag.detect_markers(gray, DICTIONARY_NAME, raw_parameters)
    raw = as_detections(corners, ids)
    timings["rt_raw"] = time.perf_counter() - start
    results["rt_raw"] = raw

    start = time.perf_counter()
    results["rt_subpix"] = subpixel(gray, raw)
    timings["rt_subpix"] = timings["rt_raw"] + time.perf_counter() - start

    return results, {name: 1000.0 * value for name, value in timings.items()}


def detect_camera(camera_name):
    frame_path = RECORDING_DIR / f"{camera_name}_frame.msgpack"
    metadata = common.load_timestamp_records(
        RECORDING_DIR / f"{camera_name}_timestamp.msgpack"
    )
    limit = len(metadata["sync"]) if MAX_FRAMES is None else MAX_FRAMES
    limit = min(limit, len(metadata["sync"]))
    expected = camera_models[camera_name].resolution
    detections = {name: [] for name in METHODS}
    timings = {name: [] for name in METHODS}

    with frame_path.open("rb") as stream:
        unpacker = msgpack.Unpacker(stream, object_hook=mpn.decode)
        for index, frame in tqdm(
            enumerate(unpacker), total=limit, desc=f"Detecting in {camera_name}"
        ):
            if index >= limit:
                break
            if tuple(frame.shape[1::-1]) != expected:
                raise ValueError(
                    f"{camera_name} frame {frame.shape[1::-1]} != calibration {expected}"
                )
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = np.ascontiguousarray(gray)
            results, elapsed = detect_frame(gray)
            for name in METHODS:
                detections[name].append(results[name])
                timings[name].append(elapsed[name])

    for key in metadata:
        metadata[key] = metadata[key][:limit]
    return {
        "metadata": metadata,
        "detections": detections,
        "timings_ms": {name: np.asarray(value) for name, value in timings.items()},
    }


def load_cache():
    if REBUILD_DETECTION_CACHE or not DETECTION_CACHE.exists():
        return None
    with DETECTION_CACHE.open("rb") as stream:
        cache = pickle.load(stream)
    current = (CACHE_VERSION, METHODS, tuple(DETECT_MARKER_IDS), MAX_FRAMES)
    if cache.get("key") != current:
        print("Detection cache is stale; rebuilding it")
        return None
    return cache


cache = load_cache()
if cache is None:
    cache = {
        "key": (CACHE_VERSION, METHODS, tuple(DETECT_MARKER_IDS), MAX_FRAMES),
        "cameras": {name: detect_camera(name) for name in CAMERA_NAMES},
    }
    with DETECTION_CACHE.open("wb") as stream:
        pickle.dump(cache, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved detections -> {DETECTION_CACHE}")
else:
    print(f"Loaded detections <- {DETECTION_CACHE}")

cam0, cam1 = cache["cameras"]["cam0"], cache["cameras"]["cam1"]
det0, det1 = cam0["detections"], cam1["detections"]
cam1_pair = common.pair_cameras(
    cam0["metadata"]["monotonic_ns"],
    cam1["metadata"]["monotonic_ns"],
    MAX_PAIR_FRACTION_OF_FRAME,
)

burst_column = cam0["metadata"]["burst"]
burst_indices = sorted({int(b) for b in burst_column if b > 0})
burst_frames = {b: np.flatnonzero(burst_column == b) for b in burst_indices}
print(f"Bursts: {len(burst_indices)}, frames: {len(burst_column)}")


# %% Cost: milliseconds per frame
timing_rows = []
for camera_name, camera in cache["cameras"].items():
    for name in METHODS:
        values = camera["timings_ms"][name]
        timing_rows.append(
            {
                "camera": camera_name,
                "method": name,
                "median_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95)),
            }
        )
timing = pd.DataFrame(timing_rows)
reference_ms = timing[timing["method"] == REFERENCE].set_index("camera")["median_ms"]
timing["speedup_vs_reference"] = timing.apply(
    lambda row: reference_ms[row["camera"]] / row["median_ms"], axis=1
)
timing.to_csv(OUTPUT_DIR / "timing.csv", index=False)
print("\nDetection cost per frame (this machine, single-frame calls):")
print(timing.round(2).to_string(index=False))


# %% Corners: agreement with the reference
def quad_area(corners):
    x, y = corners[:, 0], corners[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


agreement_rows, corner_rows = [], []
for camera_name, camera in cache["cameras"].items():
    detections = camera["detections"]
    burst_of = camera["metadata"]["burst"]
    for name in METHODS[1:]:
        missed = extra = reference_total = 0
        for frame_index, reference in enumerate(detections[REFERENCE]):
            other = detections[name][frame_index]
            reference_total += len(reference)
            missed += len(set(reference) - set(other))
            extra += len(set(other) - set(reference))
            for marker_id in set(reference) & set(other):
                distance = np.linalg.norm(other[marker_id] - reference[marker_id], axis=1)
                area = quad_area(reference[marker_id])
                for corner, value in enumerate(distance):
                    corner_rows.append(
                        {
                            "camera": camera_name,
                            "method": name,
                            "burst": int(burst_of[frame_index]),
                            "frame": frame_index,
                            "marker_id": marker_id,
                            "corner": corner,
                            "error_px": float(value),
                            "area_px2": area,
                        }
                    )
        agreement_rows.append(
            {
                "camera": camera_name,
                "method": name,
                "reference_tags": reference_total,
                "missed_vs_reference": missed,
                "extra_vs_reference": extra,
            }
        )

agreement = pd.DataFrame(agreement_rows)
corner_errors = pd.DataFrame(corner_rows)
agreement.to_csv(OUTPUT_DIR / "detection_agreement.csv", index=False)
print("\nDetection agreement (rig tags):")
print(agreement.to_string(index=False))


def corner_summary(group):
    values = group["error_px"].to_numpy()
    row = {
        "corners": len(values),
        "median_px": np.median(values),
        "p95_px": np.percentile(values, 95),
        "p99_px": np.percentile(values, 99),
        "max_px": values.max(),
    }
    for limit in CORNER_ERROR_THRESHOLDS_PX:
        row[f"frac_over_{limit}px"] = float((values > limit).mean())
    return pd.Series(row)


corner_table = (
    corner_errors.groupby(["camera", "method"])
    .apply(corner_summary, include_groups=False)
    .reset_index()
)
corner_table.to_csv(OUTPUT_DIR / "corner_error_vs_reference.csv", index=False)
print("\nCorner distance from the reference fit:")
print(corner_table.round(4).to_string(index=False))

# Detector noise before any pose solve: scatter of one corner inside a burst.
scatter_rows = []
for camera_name, camera in cache["cameras"].items():
    detections = camera["detections"]
    for burst in burst_indices:
        indices = burst_frames[burst]
        for name in METHODS:
            samples = collections.defaultdict(list)
            for frame_index in indices:
                for marker_id, corners in detections[name][frame_index].items():
                    samples[marker_id].append(corners)
            for marker_id, stack in samples.items():
                if len(stack) < TAG_PRESENCE_FRACTION * len(indices):
                    continue
                spread = np.std(np.asarray(stack), axis=0, ddof=1)  # (4, 2)
                scatter_rows.extend(
                    {
                        "camera": camera_name,
                        "method": name,
                        "burst": burst,
                        "marker_id": marker_id,
                        "corner": corner,
                        "scatter_px": float(np.linalg.norm(spread[corner])),
                    }
                    for corner in range(4)
                )
corner_scatter = pd.DataFrame(scatter_rows)
scatter_table = (
    corner_scatter.groupby(["camera", "method"])["scatter_px"]
    .agg(median="median", p95=lambda s: np.percentile(s, 95))
    .reset_index()
)
scatter_table.to_csv(OUTPUT_DIR / "corner_scatter_in_burst.csv", index=False)
print("\nIn-burst corner scatter (detector noise, px):")
print(scatter_table.round(4).to_string(index=False))


# %% Poses: identical frames and tag subsets for every method
def select_tags(indices, stereo_pair, tag_count):
    """Best ``tag_count`` tags for a burst, seen by every method (and camera)."""
    present = collections.Counter()
    areas = collections.defaultdict(list)
    for index in indices:
        sources = [det0[name][index] for name in METHODS]
        if stereo_pair:
            sources += [det1[name][cam1_pair[index]] for name in METHODS]
        for marker_id in set.intersection(*(set(source) for source in sources)):
            present[marker_id] += 1
            areas[marker_id].append(quad_area(det0[REFERENCE][index][marker_id]))
    candidates = [
        marker_id
        for marker_id, count in present.items()
        if count >= TAG_PRESENCE_FRACTION * len(indices)
    ]
    if len(candidates) < tag_count:
        return None
    candidates.sort(key=lambda marker_id: -float(np.median(areas[marker_id])))
    return tuple(sorted(candidates[:tag_count]))


def pose_of(name, index, stereo_pair, marker_ids):
    if stereo_pair:
        return solver.stereo_board_pose(
            det0[name][index], det1[name][cam1_pair[index]], marker_ids
        )
    return solver.mono_board_pose(det0[name][index], marker_ids, "cam0")


def solve_condition(camera_count, tag_count):
    stereo_pair = camera_count == 2
    rows = []
    for burst in burst_indices:
        indices = burst_frames[burst]
        if stereo_pair:
            indices = np.asarray([i for i in indices if cam1_pair[i] >= 0])
        if len(indices) < MIN_FRAMES_PER_BURST:
            continue
        marker_ids = select_tags(indices, stereo_pair, tag_count)
        if marker_ids is None:
            continue
        for index in indices:
            poses = {
                name: pose_of(name, index, stereo_pair, marker_ids) for name in METHODS
            }
            # All or nothing, so every method is scored on exactly the same frames.
            if any(pose is None for pose in poses.values()):
                continue
            for name, pose in poses.items():
                rows.append(
                    {
                        "condition": f"C{camera_count}-M{tag_count}",
                        "cameras": camera_count,
                        "tags": tag_count,
                        "burst": burst,
                        "frame": int(index),
                        "method": name,
                        "marker_ids": "+".join(map(str, marker_ids)),
                        "x": pose["tvec"][0],
                        "y": pose["tvec"][1],
                        "z": pose["tvec"][2],
                        "rvec_x": pose["rvec"][0],
                        "rvec_y": pose["rvec"][1],
                        "rvec_z": pose["rvec"][2],
                        "reprojection_px": pose["rmse_px"],
                    }
                )
    return pd.DataFrame(rows)


pose_tables = []
for camera_count in CAMERA_COUNTS:
    for tag_count in TAG_COUNTS:
        table = solve_condition(camera_count, tag_count)
        if table.empty:
            print(f"C{camera_count}-M{tag_count}: no burst solved, skipped")
            continue
        pose_tables.append(table)
        print(
            f"C{camera_count}-M{tag_count}: {table['frame'].nunique()} frames, "
            f"{table['burst'].nunique()} bursts"
        )
poses = pd.concat(pose_tables, ignore_index=True)
poses["distance_m"] = np.linalg.norm(poses[["x", "y", "z"]].to_numpy(), axis=1)


# %% Pose metrics
def rotation_gap_deg(rvec_a, rvec_b):
    relative = cv2.Rodrigues(rvec_a)[0].T @ cv2.Rodrigues(rvec_b)[0]
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(relative)[0])))


# Per burst: jitter is the in-burst spread; bias is how far the burst-mean pose
# sits from the reference's, which a spread statistic cannot see.
burst_rows = []
for (condition, burst, name), group in poses.groupby(["condition", "burst", "method"]):
    xyz = group[["x", "y", "z"]].to_numpy()
    rvec = group[["rvec_x", "rvec_y", "rvec_z"]].to_numpy()
    burst_rows.append(
        {
            "condition": condition,
            "burst": burst,
            "method": name,
            "frames": len(group),
            "distance_m": float(np.median(group["distance_m"])),
            "jitter_3d_mm": float(np.linalg.norm(1000.0 * np.std(xyz, axis=0, ddof=1))),
            "rot_jitter_deg": float(
                np.degrees(np.linalg.norm(np.std(rvec, axis=0, ddof=1)))
            ),
            "mean_x": xyz[:, 0].mean(),
            "mean_y": xyz[:, 1].mean(),
            "mean_z": xyz[:, 2].mean(),
            "reprojection_px": float(group["reprojection_px"].median()),
        }
    )
bursts = pd.DataFrame(burst_rows)
reference_bursts = bursts[bursts["method"] == REFERENCE].set_index(["condition", "burst"])
bursts = bursts.join(
    reference_bursts[["jitter_3d_mm", "mean_x", "mean_y", "mean_z"]].add_prefix("ref_"),
    on=["condition", "burst"],
)
bursts["bias_vs_reference_mm"] = 1000.0 * np.linalg.norm(
    bursts[["mean_x", "mean_y", "mean_z"]].to_numpy()
    - bursts[["ref_mean_x", "ref_mean_y", "ref_mean_z"]].to_numpy(),
    axis=1,
)
bursts["jitter_ratio"] = bursts["jitter_3d_mm"] / bursts["ref_jitter_3d_mm"]
bursts.to_csv(OUTPUT_DIR / "pose_bursts.csv", index=False)

# Per frame: how far each method's pose lands from the reference's on that frame.
reference_poses = poses[poses["method"] == REFERENCE].set_index(["condition", "frame"])
frame_rows = []
for name in METHODS[1:]:
    other = poses[poses["method"] == name].set_index(["condition", "frame"])
    joined = other.join(reference_poses, rsuffix="_ref", how="inner")
    delta = 1000.0 * (
        joined[["x", "y", "z"]].to_numpy() - joined[["x_ref", "y_ref", "z_ref"]].to_numpy()
    )
    rot = [
        rotation_gap_deg(a, b)
        for a, b in zip(
            joined[["rvec_x", "rvec_y", "rvec_z"]].to_numpy(),
            joined[["rvec_x_ref", "rvec_y_ref", "rvec_z_ref"]].to_numpy(),
        )
    ]
    frame_rows.append(
        pd.DataFrame(
            {
                "condition": joined.index.get_level_values("condition"),
                "frame": joined.index.get_level_values("frame"),
                "method": name,
                "distance_m": joined["distance_m"].to_numpy(),
                "pos_diff_mm": np.linalg.norm(delta, axis=1),
                "rot_diff_deg": rot,
                "reprojection_px": joined["reprojection_px"].to_numpy(),
                "reprojection_px_ref": joined["reprojection_px_ref"].to_numpy(),
            }
        )
    )
frame_diff = pd.concat(frame_rows, ignore_index=True)
frame_diff.to_csv(OUTPUT_DIR / "pose_frame_diff_vs_reference.csv", index=False)

summary = (
    bursts.groupby(["condition", "method"])
    .agg(
        bursts=("burst", "nunique"),
        jitter_3d_mm=("jitter_3d_mm", "median"),
        jitter_ratio=("jitter_ratio", "median"),
        bias_mm=("bias_vs_reference_mm", "median"),
        bias_max_mm=("bias_vs_reference_mm", "max"),
        reprojection_px=("reprojection_px", "median"),
    )
    .reset_index()
)
frame_summary = (
    frame_diff.groupby(["condition", "method"])
    .agg(
        pos_diff_median_mm=("pos_diff_mm", "median"),
        pos_diff_p95_mm=("pos_diff_mm", lambda s: np.percentile(s, 95)),
        pos_diff_max_mm=("pos_diff_mm", "max"),
        rot_diff_median_deg=("rot_diff_deg", "median"),
        rot_diff_max_deg=("rot_diff_deg", "max"),
    )
    .reset_index()
)
summary = summary.merge(frame_summary, on=["condition", "method"], how="left")
summary["condition"] = pd.Categorical(
    summary["condition"],
    categories=sorted(summary["condition"].unique(), key=lambda c: (c[:2], int(c[4:]))),
    ordered=True,
)
summary = summary.sort_values(["condition", "method"]).reset_index(drop=True)
summary.to_csv(OUTPUT_DIR / "pose_summary.csv", index=False)
print("\nPose summary (jitter and bias per burst, difference per frame):")
print(summary.round(4).to_string(index=False))

# %% Figures
colours = {
    REFERENCE: "tab:blue",
    "rt_crop": "tab:red",
    "rt_subpix": "tab:green",
    "rt_raw": "0.5",
}
fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

axis = axes[0]
for name in METHODS[1:]:
    values = np.sort(corner_errors.loc[corner_errors["method"] == name, "error_px"])
    axis.plot(
        values,
        np.arange(1, len(values) + 1) / len(values),
        color=colours[name],
        label=name,
    )
axis.set_xscale("log")
axis.set_xlabel("Corner distance from cv2_apriltag (px)")
axis.set_ylabel("Cumulative fraction of corners")
axis.set_title("Corner agreement with the full AprilTag fit")
axis.grid(True, alpha=0.25, which="both")
axis.legend()

axis = axes[1]
conditions = list(summary["condition"].cat.categories)
width = 0.8 / len(METHODS)
for position, name in enumerate(METHODS):
    rows = summary[summary["method"] == name].set_index("condition")["jitter_3d_mm"]
    axis.bar(
        np.arange(len(conditions)) + position * width,
        [rows.get(condition, np.nan) for condition in conditions],
        width,
        color=colours[name],
        label=name,
    )
axis.set_xticks(np.arange(len(conditions)) + width * (len(METHODS) - 1) / 2)
axis.set_xticklabels(conditions)
axis.set_ylabel("Median in-burst 3D jitter (mm)")
axis.set_title("Static jitter per condition")
axis.grid(True, axis="y", alpha=0.25)
axis.legend()

axis = axes[2]
crop = frame_diff[frame_diff["method"] == "rt_crop"]
for condition, group in crop.groupby("condition"):
    axis.scatter(group["distance_m"], group["pos_diff_mm"], s=4, alpha=0.35, label=condition)
axis.set_yscale("log")
axis.set_xlabel("Distance to dome (m)")
axis.set_ylabel("rt_crop pose minus cv2_apriltag pose (mm)")
axis.set_title("Per-frame pose difference, crop vs full fit")
axis.grid(True, alpha=0.25, which="both")
axis.legend(markerscale=3)

figure_path = OUTPUT_DIR / "crop_refine_accuracy.png"
fig.savefig(figure_path, dpi=160, bbox_inches="tight")
print(f"\nSaved {figure_path}")
