# %% [markdown]
# # Jitter versus camera count, marker count, and distance
#
# Compare six estimators against synchronized Motive data:
#
# - one camera with 1, 2, or 3 rigidly connected AprilTags;
# - two cameras with the same 1, 2, or 3 tags.
#
# Marker geometry comes from `02_rigidbody_calib.py`. Multi-marker estimates use
# one joint board PnP (never an average of independent `tvec`s), and dual-camera
# estimates minimize corner reprojection in both fisheye cameras simultaneously.
# No temporal filtering is used. Static jitter and dynamic movement smoothness
# are reported separately so a smooth-but-lagged estimate cannot appear better.
#
# Run from the repository root:
#
# ```powershell
# uv run python jitter_model/03_jitter_model.py
# ```

# %% Imports
from datetime import datetime
from pathlib import Path
import pickle
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
import toml
from tqdm.auto import tqdm


# %% Paths and experiment settings
try:
    NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:  # running cell-by-cell
    NOTEBOOK_DIR = Path.cwd()
    if NOTEBOOK_DIR.name != "jitter_model":
        NOTEBOOK_DIR = NOTEBOOK_DIR / "jitter_model"

PROJECT_ROOT = NOTEBOOK_DIR.parent
RECORDING_DIR = PROJECT_ROOT / "data" / "radxa" / "jitter_test"
MOCAP_CSV = RECORDING_DIR / "jitter_test.csv"
STEREO_TOML = (
    PROJECT_ROOT
    / "data"
    / "calibration"
    / "dual_160"
    / "radxa_calib_parallel"
    / "stereo_calibration.toml"
)
RIGIDBODY_TOML = NOTEBOOK_DIR / "rigidbody_calibration.toml"
DETECTION_CACHE = NOTEBOOK_DIR / "jitter_detections.pkl"

SUMMARY_CSV = NOTEBOOK_DIR / "jitter_summary.csv"
DISTANCE_CSV = NOTEBOOK_DIR / "jitter_vs_distance.csv"
SMOOTHNESS_CSV = NOTEBOOK_DIR / "movement_smoothness.csv"
DEPTH_METRICS_CSV = NOTEBOOK_DIR / "metrics_vs_depth.csv"
ALIGNMENT_TOML = NOTEBOOK_DIR / "jitter_alignment.toml"
OUTPUT_FIGURE = NOTEBOOK_DIR / "jitter_model.png"
SMOOTHNESS_FIGURE = NOTEBOOK_DIR / "movement_smoothness.png"
DEPTH_FIGURE = NOTEBOOK_DIR / "metrics_vs_depth.png"

# Fixed nested sets make marker-count comparisons reproducible. Change these
# tuples to test a different physical subset without changing the estimator.
MARKER_SETS = {
    1: (12,),
    2: (12, 20),
    3: (4, 12, 20),
}
DISTANCE_EDGES_M = np.asarray([0.25, 0.40, 0.55, 0.70, 0.90])
DEPTH_EDGES_M = np.arange(0.25, 0.901, 0.05)
STATIC_MOCAP_STEP_M = 0.003
MAX_PAIR_FRACTION_OF_FRAME = 0.55
MIN_STATIC_PAIRS = 10
MIN_DEPTH_PAIRS = 4
MAX_LAG_S = 0.100
LAG_STEPS = 81


# %% Load camera calibration, rigid-body geometry, and cached detections
for required_path in (STEREO_TOML, RIGIDBODY_TOML, DETECTION_CACHE, MOCAP_CSV):
    if not required_path.exists():
        raise FileNotFoundError(
            f"Missing {required_path}. Run jitter_model/02_rigidbody_calib.py first."
        )

stereo = toml.load(STEREO_TOML)
rigidbody = toml.load(RIGIDBODY_TOML)
with DETECTION_CACHE.open("rb") as stream:
    detections = pickle.load(stream)  # noqa: S301 - trusted local analysis cache

camera_models = {}
for camera_name in ("cam0", "cam1"):
    camera_models[camera_name] = {
        "K": np.asarray(stereo[camera_name]["camera_matrix"], dtype=np.float64),
        "D": np.asarray(stereo[camera_name]["dist_coeffs"], dtype=np.float64).reshape(
            -1, 1
        ),
    }
# Use the current-take stereo self-calibration written by notebook 02. The
# supplied TOML still supplies the intrinsics and a nominal extrinsic fallback.
if "stereo_refined" in rigidbody:
    R_STEREO = np.asarray(
        rigidbody["stereo_refined"]["rotation_cam0_to_cam1"], dtype=np.float64
    )
    T_STEREO = np.asarray(
        rigidbody["stereo_refined"]["translation_cam0_to_cam1_m"], dtype=np.float64
    )
    print("Using current-take stereo extrinsic from rigidbody_calibration.toml")
else:
    warnings.warn("No refined stereo extrinsic; falling back to calibration TOML")
    R_STEREO = np.asarray(stereo["stereo"]["R"], dtype=np.float64)
    T_STEREO = np.asarray(stereo["stereo"]["T"], dtype=np.float64) / 1000.0
RVEC_STEREO = cv2.Rodrigues(R_STEREO)[0]

TAG_SIZE_M = float(rigidbody["meta"]["tag_size_m"])
REFERENCE_ID = int(rigidbody["meta"]["reference_id"])
CALIBRATION_END_FRAME = int(rigidbody["meta"]["calibration_end_frame"])

half_size = TAG_SIZE_M / 2.0
TAG_CORNERS_LOCAL = np.asarray(
    [
        [-half_size, +half_size, 0.0],
        [+half_size, +half_size, 0.0],
        [+half_size, -half_size, 0.0],
        [-half_size, -half_size, 0.0],
    ],
    dtype=np.float64,
)

marker_corners_reference = {}
for marker_id_text, marker_data in rigidbody["markers"].items():
    marker_id = int(marker_id_text)
    marker_rotation = np.asarray(
        marker_data["rotation_marker_to_reference"], dtype=np.float64
    )
    marker_translation = np.asarray(
        marker_data["translation_marker_to_reference_m"], dtype=np.float64
    )
    marker_corners_reference[marker_id] = (
        TAG_CORNERS_LOCAL @ marker_rotation.T + marker_translation
    )

cam0 = detections["cameras"]["cam0"]
cam1 = detections["cameras"]["cam1"]


def first_sync_rise(sync):
    high = np.flatnonzero(np.asarray(sync) == 1)
    if not len(high):
        raise ValueError("No GPIO synchronization pulse in camera recording")
    return int(high[0])


sync_frame = first_sync_rise(cam0["metadata"]["sync"])
sync_monotonic_ns = int(cam0["metadata"]["monotonic_ns"][sync_frame])
cam0_monotonic_ns = np.asarray(cam0["metadata"]["monotonic_ns"], dtype=np.int64)
cam1_monotonic_ns = np.asarray(cam1["metadata"]["monotonic_ns"], dtype=np.int64)
cam0_period_ns = float(np.median(np.diff(cam0_monotonic_ns)))
max_pair_gap_ns = MAX_PAIR_FRACTION_OF_FRAME * cam0_period_ns


# %% Pair the two free-running cameras on the shared host-monotonic clock
def nearest_cam1_index(timestamp_ns):
    insertion = int(np.searchsorted(cam1_monotonic_ns, timestamp_ns))
    candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(cam1_monotonic_ns)]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda index: abs(int(cam1_monotonic_ns[index]) - int(timestamp_ns)))
    if abs(int(cam1_monotonic_ns[nearest]) - int(timestamp_ns)) > max_pair_gap_ns:
        return None
    return nearest


cam1_pair = np.asarray(
    [
        -1 if (match := nearest_cam1_index(timestamp)) is None else match
        for timestamp in cam0_monotonic_ns
    ],
    dtype=np.int32,
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


# %% Joint marker-board pose estimators
def stack_correspondences(frame_detections, marker_ids):
    if not all(marker_id in frame_detections for marker_id in marker_ids):
        return None
    return (
        np.concatenate([marker_corners_reference[marker_id] for marker_id in marker_ids]),
        np.concatenate([frame_detections[marker_id] for marker_id in marker_ids]),
    )


def raw_reprojection_rmse(object_points, image_points, rvec, tvec, camera_name):
    model = camera_models[camera_name]
    projected, _ = cv2.fisheye.projectPoints(
        object_points.reshape(-1, 1, 3), rvec, tvec, model["K"], model["D"]
    )
    residual = projected.reshape(-1, 2) - image_points
    return float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))


def mono_board_pose(frame_detections, marker_ids, camera_name="cam0"):
    correspondences = stack_correspondences(frame_detections, marker_ids)
    if correspondences is None:
        return None
    object_points, image_points_raw = correspondences
    model = camera_models[camera_name]
    image_points = cv2.fisheye.undistortPoints(
        image_points_raw.reshape(-1, 1, 2),
        model["K"],
        model["D"],
        P=model["K"],
    ).reshape(-1, 2)

    if len(marker_ids) == 1 and marker_ids[0] == REFERENCE_ID:
        solutions = cv2.solvePnPGeneric(
            object_points,
            image_points,
            model["K"],
            None,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not solutions[0]:
            return None
        candidates = []
        for rvec, tvec in zip(solutions[1], solutions[2]):
            if float(tvec.reshape(3)[2]) <= 0:
                continue
            candidates.append(
                (
                    raw_reprojection_rmse(
                        object_points, image_points_raw, rvec, tvec, camera_name
                    ),
                    rvec,
                    tvec,
                )
            )
        if not candidates:
            return None
        rmse_px, rvec, tvec = min(candidates, key=lambda item: item[0])
    else:
        # The two-tag subset is nearly planar and ITERATIVE can otherwise land
        # on its mirrored local solution. Seed it with tag 12's pose from this
        # same image; this is geometric initialization, not temporal filtering.
        reference_initial = mono_board_pose(
            frame_detections, (REFERENCE_ID,), camera_name
        )
        if reference_initial is None:
            return None
        initial_rvec = reference_initial["rvec"].reshape(3, 1).copy()
        initial_tvec = reference_initial["tvec"].reshape(3, 1).copy()
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            model["K"],
            None,
            initial_rvec,
            initial_tvec,
            True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or float(tvec.reshape(3)[2]) <= 0:
            return None
        rmse_px = raw_reprojection_rmse(
            object_points, image_points_raw, rvec, tvec, camera_name
        )
    return {
        "rvec": np.asarray(rvec).reshape(3),
        "tvec": np.asarray(tvec).reshape(3),
        "rmse_px": rmse_px,
    }


def stereo_board_pose(frame0, frame1, marker_ids):
    corr0 = stack_correspondences(frame0, marker_ids)
    corr1 = stack_correspondences(frame1, marker_ids)
    if corr0 is None or corr1 is None:
        return None
    object_points, image0 = corr0
    _object_points1, image1 = corr1
    initial = mono_board_pose(frame0, marker_ids, "cam0")
    if initial is None:
        return None

    def residual(parameters):
        rvec0 = parameters[:3].reshape(3, 1)
        tvec0 = parameters[3:].reshape(3, 1)
        projected0, _ = cv2.fisheye.projectPoints(
            object_points.reshape(-1, 1, 3),
            rvec0,
            tvec0,
            camera_models["cam0"]["K"],
            camera_models["cam0"]["D"],
        )
        rvec1, tvec1 = cv2.composeRT(
            rvec0, tvec0, RVEC_STEREO, T_STEREO.reshape(3, 1)
        )[:2]
        projected1, _ = cv2.fisheye.projectPoints(
            object_points.reshape(-1, 1, 3),
            rvec1,
            tvec1,
            camera_models["cam1"]["K"],
            camera_models["cam1"]["D"],
        )
        return np.concatenate(
            [
                (projected0.reshape(-1, 2) - image0).ravel(),
                (projected1.reshape(-1, 2) - image1).ravel(),
            ]
        )

    x0 = np.concatenate([initial["rvec"], initial["tvec"]])
    result = least_squares(
        residual,
        x0,
        # Match the already-validated stereo PnP implementation in
        # trunkpose/dual_notebooks/verification_dual_camera.py. The cam0-only
        # initial pose can have a large cam1 residual, for which a robust loss
        # incorrectly suppresses the very measurements needed to cross-baseline
        # refine the depth.
        method="lm",
        max_nfev=100,
    )
    if result.x[5] <= 0:
        return None
    point_residuals = residual(result.x).reshape(-1, 2)
    return {
        "rvec": result.x[:3],
        "tvec": result.x[3:],
        "rmse_px": float(np.sqrt(np.mean(np.sum(point_residuals**2, axis=1)))),
    }


# %% Run the six raw estimators
method_specs = {
    f"C{camera_count}-M{marker_count}": {
        "camera_count": camera_count,
        "marker_ids": marker_ids,
    }
    for camera_count in (1, 2)
    for marker_count, marker_ids in MARKER_SETS.items()
}
method_rows = {method: [] for method in method_specs}

for method, spec in method_specs.items():
    for frame_index, frame0 in enumerate(
        tqdm(cam0["detections"], desc=f"Estimating {method}")
    ):
        marker_ids = spec["marker_ids"]
        if spec["camera_count"] == 1:
            pose = mono_board_pose(frame0, marker_ids)
            timestamp_ns = int(cam0_monotonic_ns[frame_index])
            pair_gap_ms = np.nan
        else:
            match = int(cam1_pair[frame_index])
            if match < 0:
                continue
            pose = stereo_board_pose(frame0, cam1["detections"][match], marker_ids)
            timestamp_ns = (
                int(cam0_monotonic_ns[frame_index])
                + int(cam1_monotonic_ns[match])
            ) // 2
            pair_gap_ms = abs(
                int(cam1_monotonic_ns[match]) - int(cam0_monotonic_ns[frame_index])
            ) / 1e6
        if pose is None:
            continue
        method_rows[method].append(
            {
                "frame": frame_index,
                "time_s": (timestamp_ns - sync_monotonic_ns) / 1e9,
                "x": pose["tvec"][0],
                "y": pose["tvec"][1],
                "z": pose["tvec"][2],
                "reprojection_rmse_px": pose["rmse_px"],
                "pair_gap_ms": pair_gap_ms,
            }
        )

method_tables = {
    method: pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)
    for method, rows in method_rows.items()
}
for method, table in method_tables.items():
    print(
        f"{method} {method_specs[method]['marker_ids']}: {len(table)} poses, "
        f"median reprojection={table['reprojection_rmse_px'].median():.3f} px"
    )


# %% Motive loader and continuous interpolation
def read_rigid_body_csv(path):
    raw_header = pd.read_csv(path, nrows=0, dtype=str).columns.tolist()
    start_index = raw_header.index("Capture Start Time")
    capture_start = datetime.strptime(
        raw_header[start_index + 1], "%Y-%m-%d %I.%M.%S.%f %p"
    )
    table = pd.read_csv(path, skiprows=6)
    columns = {
        "Time (Seconds)": "seconds",
        "X": "qx",
        "Y": "qy",
        "Z": "qz",
        "W": "qw",
        "X.1": "px",
        "Y.1": "py",
        "Z.1": "pz",
    }
    missing = [column for column in columns if column not in table]
    if missing:
        raise ValueError(f"Unexpected Motive CSV layout; missing {missing}")
    result = table[list(columns)].rename(columns=columns).apply(pd.to_numeric, errors="coerce")
    result = result.dropna().drop_duplicates("seconds").sort_values("seconds")
    return result, capture_start


mocap, mocap_capture_start = read_rigid_body_csv(MOCAP_CSV)
mocap_times = mocap["seconds"].to_numpy(dtype=np.float64)
mocap_positions = mocap[["px", "py", "pz"]].to_numpy(dtype=np.float64)
mocap_quaternions = mocap[["qx", "qy", "qz", "qw"]].to_numpy(
    dtype=np.float64, copy=True
)
mocap_quaternions /= np.linalg.norm(mocap_quaternions, axis=1, keepdims=True)
mocap_slerp = Slerp(mocap_times, Rotation.from_quat(mocap_quaternions))


def interpolate_mocap(query_times):
    query_times = np.asarray(query_times, dtype=np.float64)
    if np.any(query_times < mocap_times[0]) or np.any(query_times > mocap_times[-1]):
        raise ValueError("Mocap interpolation requested outside synchronized overlap")
    positions = np.column_stack(
        [np.interp(query_times, mocap_times, mocap_positions[:, axis]) for axis in range(3)]
    )
    return positions, mocap_slerp(query_times)


# %% Freeze one offset-aware mocap alignment using only calibration frames
def fit_rigid_transform(source_points, target_points):
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    covariance = (source_points - source_center).T @ (target_points - target_center)
    U, _singular_values, Vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(Vt.T @ U.T))
    matrix = Vt.T @ np.diag([1.0, 1.0, sign]) @ U.T
    translation = target_center - matrix @ source_center
    return matrix, translation


baseline = method_tables["C1-M1"]
alignment_fit = baseline.loc[
    (baseline["frame"] < CALIBRATION_END_FRAME)
    & baseline["time_s"].between(mocap_times[0], mocap_times[-1])
].copy()
if len(alignment_fit) < 20:
    raise RuntimeError("Too few synchronized calibration poses for mocap alignment")
fit_mocap_positions, fit_mocap_orientations = interpolate_mocap(
    alignment_fit["time_s"].to_numpy()
)
fit_tag_positions = alignment_fit[["x", "y", "z"]].to_numpy()
initial_R, initial_t = fit_rigid_transform(fit_mocap_positions, fit_tag_positions)
x0 = np.concatenate([Rotation.from_matrix(initial_R).as_rotvec(), initial_t, np.zeros(3)])


def alignment_residual(parameters):
    world_to_camera = Rotation.from_rotvec(parameters[:3])
    tag_world = fit_mocap_positions + fit_mocap_orientations.apply(parameters[6:9])
    prediction = world_to_camera.apply(tag_world) + parameters[3:6]
    return (prediction - fit_tag_positions).ravel()


alignment_result = least_squares(
    alignment_residual,
    x0,
    loss="soft_l1",
    f_scale=0.003,
    max_nfev=2000,
)
WORLD_TO_CAMERA = Rotation.from_rotvec(alignment_result.x[:3])
WORLD_TO_CAMERA_T = alignment_result.x[3:6]
TAG_OFFSET_BODY = alignment_result.x[6:9]
alignment_fit_rmse_mm = 1000 * np.sqrt(np.mean(alignment_residual(alignment_result.x) ** 2) * 3)
print(
    f"Frozen C1-M1 mocap alignment: N={len(alignment_fit)}, "
    f"fit RMSE={alignment_fit_rmse_mm:.3f} mm, "
    f"body offset={np.round(TAG_OFFSET_BODY * 1000, 2)} mm"
)

alignment_payload = {
    "meta": {
        "fit_method": "C1-M1",
        "fit_end_frame_exclusive": CALIBRATION_END_FRAME,
        "fit_samples": len(alignment_fit),
        "fit_rmse_mm": alignment_fit_rmse_mm,
        "sync_cam0_frame": sync_frame,
    },
    "alignment": {
        "rotation_mocap_world_to_cam0": WORLD_TO_CAMERA.as_matrix().tolist(),
        "translation_mocap_world_to_cam0_m": WORLD_TO_CAMERA_T.tolist(),
        "tag12_offset_in_mocap_body_m": TAG_OFFSET_BODY.tolist(),
    },
}
with ALIGNMENT_TOML.open("w", encoding="utf-8") as stream:
    toml.dump(alignment_payload, stream)


def add_mocap_reference(table):
    valid = table["time_s"].between(mocap_times[0], mocap_times[-1])
    result = table.loc[valid].copy().reset_index(drop=True)
    positions, orientations = interpolate_mocap(result["time_s"].to_numpy())
    tag_world = positions + orientations.apply(TAG_OFFSET_BODY)
    aligned = WORLD_TO_CAMERA.apply(tag_world) + WORLD_TO_CAMERA_T
    result[["mocap_x", "mocap_y", "mocap_z"]] = aligned
    residual = result[["x", "y", "z"]].to_numpy() - aligned
    result[["residual_x", "residual_y", "residual_z"]] = residual
    result["distance_m"] = np.linalg.norm(aligned, axis=1)
    result["depth_m"] = aligned[:, 2]
    return result


evaluation_tables = {method: add_mocap_reference(table) for method, table in method_tables.items()}
for method in evaluation_tables:
    evaluation_tables[method] = evaluation_tables[method].loc[
        evaluation_tables[method]["frame"] >= CALIBRATION_END_FRAME
    ].reset_index(drop=True)


# %% Fair comparison on frames available to every method
common_frames = set.intersection(
    *(set(table["frame"].astype(int)) for table in evaluation_tables.values())
)
common_frames = np.asarray(sorted(common_frames), dtype=np.int32)
if len(common_frames) < 20:
    raise RuntimeError(f"Only {len(common_frames)} common held-out frames")
print(f"Held-out common-frame set: {len(common_frames)} frames")


def static_pair_mask(table):
    frame = table["frame"].to_numpy(dtype=np.int32)
    time_s = table["time_s"].to_numpy(dtype=np.float64)
    mocap_xyz = table[["mocap_x", "mocap_y", "mocap_z"]].to_numpy()
    return (
        (np.diff(frame) == 1)
        & (np.diff(time_s) > 0)
        & (np.diff(time_s) < 1.5 * cam0_period_ns / 1e9)
        & (np.linalg.norm(np.diff(mocap_xyz, axis=0), axis=1) < STATIC_MOCAP_STEP_M)
    )


def calculate_metrics(table):
    residual = table[["residual_x", "residual_y", "residual_z"]].to_numpy()
    pair_mask = static_pair_mask(table)
    residual_step = np.diff(residual, axis=0)[pair_mask]
    if len(residual_step) < MIN_STATIC_PAIRS:
        return None
    jitter_axis_mm = 1000.0 * np.std(residual_step, axis=0, ddof=1) / np.sqrt(2.0)
    return {
        "samples": len(table),
        "static_pairs": int(pair_mask.sum()),
        "rmse_mm": 1000.0 * np.sqrt(np.mean(np.sum(residual**2, axis=1))),
        "jitter_x_mm": jitter_axis_mm[0],
        "jitter_y_mm": jitter_axis_mm[1],
        "jitter_z_mm": jitter_axis_mm[2],
        "jitter_3d_mm": float(np.linalg.norm(jitter_axis_mm)),
        "median_reprojection_px": table["reprojection_rmse_px"].median(),
    }


summary_rows = []
common_tables = {}
for method, table in evaluation_tables.items():
    common = table.loc[table["frame"].isin(common_frames)].sort_values("frame").reset_index(drop=True)
    common_tables[method] = common
    metrics = calculate_metrics(common)
    if metrics is None:
        raise RuntimeError(f"Too few static frame pairs for {method}")
    metrics.update(
        {
            "method": method,
            "cameras": method_specs[method]["camera_count"],
            "markers": len(method_specs[method]["marker_ids"]),
            "marker_ids": "+".join(map(str, method_specs[method]["marker_ids"])),
            "native_heldout_poses": len(table),
            "coverage_percent": 100.0 * len(table) / max(1, len(cam0["detections"]) - CALIBRATION_END_FRAME),
        }
    )
    summary_rows.append(metrics)

summary = pd.DataFrame(summary_rows).sort_values(["cameras", "markers"]).reset_index(drop=True)
summary.to_csv(SUMMARY_CSV, index=False)
print("\nCommon-frame jitter summary [mm]:")
print(
    summary[
        ["method", "samples", "static_pairs", "rmse_mm", "jitter_x_mm", "jitter_y_mm", "jitter_z_mm", "jitter_3d_mm", "coverage_percent"]
    ].round(3).to_string(index=False)
)


# %% Dynamic movement smoothness on the same held-out common frames
def aligned_mocap_at_times(query_times):
    """Return the tag-12 mocap reference in cam0 coordinates."""
    positions, orientations = interpolate_mocap(query_times)
    tag_world = positions + orientations.apply(TAG_OFFSET_BODY)
    return WORLD_TO_CAMERA.apply(tag_world) + WORLD_TO_CAMERA_T


def movement_smoothness_metrics(table, estimate_lag=True):
    """Compare raw position derivatives with identically sampled mocap.

    Movement intervals are the complement of the existing static definition:
    a valid consecutive pair is dynamic when mocap moves at least 3 mm. Velocity,
    acceleration, and jerk are finite differences on the original timestamps;
    no camera or mocap trajectory is smoothed.
    """
    frame = table["frame"].to_numpy(dtype=np.int32)
    time_s = table["time_s"].to_numpy(dtype=np.float64)
    camera_xyz = table[["x", "y", "z"]].to_numpy(dtype=np.float64)
    mocap_xyz = table[["mocap_x", "mocap_y", "mocap_z"]].to_numpy(
        dtype=np.float64
    )

    dt = np.diff(time_s)
    mocap_step = np.linalg.norm(np.diff(mocap_xyz, axis=0), axis=1)
    valid_pair = (
        (np.diff(frame) == 1)
        & (dt > 0)
        & (dt < 1.5 * cam0_period_ns / 1e9)
    )
    moving_pair = valid_pair & (mocap_step >= STATIC_MOCAP_STEP_M)

    camera_velocity = np.diff(camera_xyz, axis=0) / dt[:, None]
    mocap_velocity = np.diff(mocap_xyz, axis=0) / dt[:, None]
    velocity_error = camera_velocity - mocap_velocity

    velocity_time = 0.5 * (time_s[:-1] + time_s[1:])
    velocity_dt = np.diff(velocity_time)
    valid_acceleration = valid_pair[:-1] & valid_pair[1:] & (velocity_dt > 0)
    moving_acceleration = valid_acceleration & (
        moving_pair[:-1] | moving_pair[1:]
    )
    camera_acceleration = np.diff(camera_velocity, axis=0) / velocity_dt[:, None]
    mocap_acceleration = np.diff(mocap_velocity, axis=0) / velocity_dt[:, None]
    acceleration_error = camera_acceleration - mocap_acceleration

    acceleration_time = 0.5 * (velocity_time[:-1] + velocity_time[1:])
    acceleration_dt = np.diff(acceleration_time)
    valid_jerk = (
        valid_acceleration[:-1]
        & valid_acceleration[1:]
        & (acceleration_dt > 0)
    )
    moving_jerk = valid_jerk & (
        moving_acceleration[:-1] | moving_acceleration[1:]
    )
    camera_jerk = np.diff(camera_acceleration, axis=0) / acceleration_dt[:, None]
    mocap_jerk = np.diff(mocap_acceleration, axis=0) / acceleration_dt[:, None]
    jerk_error = camera_jerk - mocap_jerk

    moving_samples = np.zeros(len(table), dtype=bool)
    moving_samples[:-1] |= moving_pair
    moving_samples[1:] |= moving_pair
    position_residual = camera_xyz - mocap_xyz

    if moving_pair.sum() < 3 or moving_samples.sum() < 3:
        return None

    def vector_rms(values):
        return float(np.sqrt(np.mean(np.sum(values**2, axis=1))))

    camera_path_m = float(
        np.sum(np.linalg.norm(np.diff(camera_xyz, axis=0)[moving_pair], axis=1))
    )
    mocap_path_m = float(np.sum(mocap_step[moving_pair]))

    # Estimate lag independently of constant positional bias. A positive result
    # means the camera trajectory follows the mocap trajectory late.
    if estimate_lag:
        lag_candidates_s = np.linspace(-MAX_LAG_S, MAX_LAG_S, LAG_STEPS)
        lag_errors = []
        camera_moving = camera_xyz[moving_samples]
        camera_moving_centered = camera_moving - camera_moving.mean(axis=0)
        moving_times = time_s[moving_samples]
        for shift_s in lag_candidates_s:
            shifted_times = moving_times + shift_s
            if (
                shifted_times[0] < mocap_times[0]
                or shifted_times[-1] > mocap_times[-1]
            ):
                lag_errors.append(np.inf)
                continue
            shifted_mocap = aligned_mocap_at_times(shifted_times)
            shifted_mocap -= shifted_mocap.mean(axis=0)
            lag_errors.append(vector_rms(camera_moving_centered - shifted_mocap))
        best_shift_s = float(lag_candidates_s[int(np.argmin(lag_errors))])
        estimated_lag_ms = -1000.0 * best_shift_s
    else:
        estimated_lag_ms = np.nan

    result = {
        "dynamic_samples": int(moving_samples.sum()),
        "dynamic_pairs": int(moving_pair.sum()),
        "acceleration_samples": int(moving_acceleration.sum()),
        "jerk_samples": int(moving_jerk.sum()),
        "dynamic_position_rmse_mm": 1000.0
        * vector_rms(position_residual[moving_samples]),
        "velocity_error_rmse_mm_s": 1000.0
        * vector_rms(velocity_error[moving_pair]),
        "path_length_ratio": camera_path_m / max(mocap_path_m, np.finfo(float).eps),
        "estimated_camera_lag_ms": estimated_lag_ms,
    }
    if moving_acceleration.any():
        result["acceleration_error_rmse_m_s2"] = vector_rms(
            acceleration_error[moving_acceleration]
        )
    else:
        result["acceleration_error_rmse_m_s2"] = np.nan
    if moving_jerk.any():
        camera_jerk_rms = vector_rms(camera_jerk[moving_jerk])
        mocap_jerk_rms = vector_rms(mocap_jerk[moving_jerk])
        result.update(
            {
                "jerk_error_rmse_m_s3": vector_rms(jerk_error[moving_jerk]),
                "camera_jerk_rms_m_s3": camera_jerk_rms,
                "mocap_jerk_rms_m_s3": mocap_jerk_rms,
                "jerk_ratio_camera_over_mocap": camera_jerk_rms
                / max(mocap_jerk_rms, np.finfo(float).eps),
            }
        )
    else:
        result.update(
            {
                "jerk_error_rmse_m_s3": np.nan,
                "camera_jerk_rms_m_s3": np.nan,
                "mocap_jerk_rms_m_s3": np.nan,
                "jerk_ratio_camera_over_mocap": np.nan,
            }
        )
    return result


smoothness_rows = []
for method, table in common_tables.items():
    metrics = movement_smoothness_metrics(table)
    if metrics is None:
        warnings.warn(f"Too few moving frame pairs for {method}")
        continue
    metrics.update(
        {
            "method": method,
            "cameras": method_specs[method]["camera_count"],
            "markers": len(method_specs[method]["marker_ids"]),
            "marker_ids": "+".join(map(str, method_specs[method]["marker_ids"])),
            "scope": "all-method common held-out frames",
        }
    )
    smoothness_rows.append(metrics)

smoothness_summary = (
    pd.DataFrame(smoothness_rows)
    .sort_values(["cameras", "markers"])
    .reset_index(drop=True)
)
smoothness_summary.to_csv(SMOOTHNESS_CSV, index=False)
print("\nDynamic movement smoothness summary:")
print(
    smoothness_summary[
        [
            "method",
            "dynamic_pairs",
            "dynamic_position_rmse_mm",
            "velocity_error_rmse_mm_s",
            "acceleration_error_rmse_m_s2",
            "jerk_error_rmse_m_s3",
            "path_length_ratio",
            "estimated_camera_lag_ms",
        ]
    ].round(3).to_string(index=False)
)


# %% Jitter versus synchronized mocap distance
# The overall bars above use the strict all-method common frame set. Distance
# curves use each method's native held-out detections, because requiring three
# tags in both cameras naturally removes most far-range frames. Pair counts are
# exported beside every bin so visibility and statistical support stay explicit.
distance_rows = []
for method, table in evaluation_tables.items():
    residual = table[["residual_x", "residual_y", "residual_z"]].to_numpy()
    pair_mask = static_pair_mask(table)
    residual_steps = np.diff(residual, axis=0)
    pair_distance = 0.5 * (
        table["distance_m"].to_numpy()[:-1] + table["distance_m"].to_numpy()[1:]
    )
    for low, high in zip(DISTANCE_EDGES_M[:-1], DISTANCE_EDGES_M[1:]):
        in_bin = pair_mask & (pair_distance >= low) & (pair_distance < high)
        steps = residual_steps[in_bin]
        if len(steps) >= MIN_STATIC_PAIRS:
            axis_jitter = 1000.0 * np.std(steps, axis=0, ddof=1) / np.sqrt(2.0)
            jitter_3d = float(np.linalg.norm(axis_jitter))
        else:
            axis_jitter = np.full(3, np.nan)
            jitter_3d = np.nan
        distance_rows.append(
            {
                "method": method,
                "scope": "native held-out detections",
                "distance_low_m": low,
                "distance_high_m": high,
                "distance_center_m": 0.5 * (low + high),
                "static_pairs": int(in_bin.sum()),
                "jitter_x_mm": axis_jitter[0],
                "jitter_y_mm": axis_jitter[1],
                "jitter_z_mm": axis_jitter[2],
                "jitter_3d_mm": jitter_3d,
            }
        )

distance_summary = pd.DataFrame(distance_rows)
distance_summary.to_csv(DISTANCE_CSV, index=False)


# %% Static jitter and dynamic smoothness versus optical-axis depth
depth_rows = []
smoothness_columns = (
    "dynamic_position_rmse_mm",
    "velocity_error_rmse_mm_s",
    "acceleration_error_rmse_m_s2",
    "jerk_error_rmse_m_s3",
    "path_length_ratio",
)
for method, table in evaluation_tables.items():
    for low, high in zip(DEPTH_EDGES_M[:-1], DEPTH_EDGES_M[1:]):
        in_bin = table["depth_m"].between(low, high, inclusive="left")
        subset = table.loc[in_bin].sort_values("frame").reset_index(drop=True)
        if subset.empty:
            continue

        residual = subset[["residual_x", "residual_y", "residual_z"]].to_numpy()
        static_mask = static_pair_mask(subset)
        residual_steps = np.diff(residual, axis=0)[static_mask]
        if len(residual_steps) >= MIN_DEPTH_PAIRS:
            axis_jitter = (
                1000.0
                * np.std(residual_steps, axis=0, ddof=1)
                / np.sqrt(2.0)
            )
            static_jitter_3d_mm = float(np.linalg.norm(axis_jitter))
        else:
            static_jitter_3d_mm = np.nan

        dynamic = movement_smoothness_metrics(subset, estimate_lag=False)
        row = {
            "method": method,
            "cameras": method_specs[method]["camera_count"],
            "markers": len(method_specs[method]["marker_ids"]),
            "marker_ids": "+".join(map(str, method_specs[method]["marker_ids"])),
            "depth_low_m": low,
            "depth_high_m": high,
            "depth_m": float(subset["depth_m"].median()),
            "pose_samples": len(subset),
            "static_pairs": int(static_mask.sum()),
            "static_jitter_3d_mm": static_jitter_3d_mm,
        }
        if dynamic is None:
            row.update({column: np.nan for column in smoothness_columns})
            row.update(
                {
                    "dynamic_samples": 0,
                    "dynamic_pairs": 0,
                    "acceleration_samples": 0,
                    "jerk_samples": 0,
                }
            )
        else:
            row.update({column: dynamic[column] for column in smoothness_columns})
            row.update(
                {
                    "dynamic_samples": dynamic["dynamic_samples"],
                    "dynamic_pairs": dynamic["dynamic_pairs"],
                    "acceleration_samples": dynamic["acceleration_samples"],
                    "jerk_samples": dynamic["jerk_samples"],
                }
            )
        depth_rows.append(row)

depth_summary = pd.DataFrame(depth_rows).sort_values(
    ["cameras", "markers", "depth_m"]
)
depth_summary.to_csv(DEPTH_METRICS_CSV, index=False)


# %% Plot overall jitter, jitter versus distance, and coverage
fig, axes = plt.subplots(1, 3, figsize=(17, 5.3), constrained_layout=True)
methods = summary["method"].tolist()
colors = ["#4c78a8" if method.startswith("C1") else "#f58518" for method in methods]

axes[0].bar(methods, summary["jitter_3d_mm"], color=colors)
axes[0].set_title("Raw position jitter on common frames")
axes[0].set_ylabel(r"$\sigma(\Delta residual)/\sqrt{2}$ [mm]")
axes[0].grid(axis="y", alpha=0.25)

marker_style = {1: ("-", "o"), 2: ("--", "s"), 3: (":", "^")}
for method, color in zip(methods, colors):
    subset = distance_summary.loc[distance_summary["method"] == method]
    marker_count = method_specs[method]["marker_ids"]
    linestyle, marker = marker_style[len(marker_count)]
    axes[1].plot(
        subset["distance_center_m"],
        subset["jitter_3d_mm"],
        marker=marker,
        linewidth=1.5,
        color=color,
        linestyle=linestyle,
        label=method,
    )
axes[1].set_title("Jitter versus camera-to-tag distance")
axes[1].set_xlabel("Mocap reference distance [m]")
axes[1].set_ylabel("3D jitter [mm]")
axes[1].grid(True, alpha=0.25)
axes[1].legend(ncols=2, fontsize=8)

axes[2].bar(methods, summary["coverage_percent"], color=colors)
axes[2].set_title("Native held-out pose coverage")
axes[2].set_ylabel("Frames with all selected tags [%]")
axes[2].set_ylim(0, 105)
axes[2].grid(axis="y", alpha=0.25)

fig.suptitle(
    "AprilTag rigid-body jitter experiment — no temporal smoothing\n"
    "C1/C2 = camera count, M1/M2/M3 = fixed nested marker count"
)
fig.savefig(OUTPUT_FIGURE, dpi=180, bbox_inches="tight")

# Dynamic metrics have different units, so use small multiples rather than a
# combined score that would hide the meaning of each quantity.
smooth_fig, smooth_axes = plt.subplots(
    2, 2, figsize=(13, 9), constrained_layout=True
)
smooth_metrics = (
    (
        "dynamic_position_rmse_mm",
        "Position tracking during movement",
        "RMSE [mm]",
    ),
    (
        "velocity_error_rmse_mm_s",
        "Velocity tracking",
        "Velocity error RMSE [mm/s]",
    ),
    (
        "acceleration_error_rmse_m_s2",
        "Acceleration smoothness",
        r"Acceleration error RMSE [m/s$^2$]",
    ),
    (
        "jerk_error_rmse_m_s3",
        "Jerk smoothness",
        r"Jerk error RMSE [m/s$^3$]",
    ),
)
for axis, (column, title, ylabel) in zip(smooth_axes.ravel(), smooth_metrics):
    axis.bar(smoothness_summary["method"], smoothness_summary[column], color=colors)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
smooth_fig.suptitle(
    "Movement smoothness against mocap — common held-out frames, no temporal smoothing\n"
    "Lower is better; derivatives use identical timestamps for camera and mocap"
)
smooth_fig.savefig(SMOOTHNESS_FIGURE, dpi=180, bbox_inches="tight")

# Match the reference plot's line-chart vocabulary while keeping each physical
# quantity in its own axis. Camera count is color; marker count is line/shape.
depth_fig, depth_axes = plt.subplots(2, 3, figsize=(17, 9.5), sharex=True)
depth_fig.subplots_adjust(
    left=0.06, right=0.99, bottom=0.08, top=0.80, wspace=0.22, hspace=0.30
)
depth_plot_metrics = (
    ("static_jitter_3d_mm", "Static jitter", "3D jitter [mm]"),
    (
        "dynamic_position_rmse_mm",
        "Position tracking during movement",
        "RMSE [mm]",
    ),
    (
        "velocity_error_rmse_mm_s",
        "Velocity smoothness",
        "Velocity error RMSE [mm/s]",
    ),
    (
        "acceleration_error_rmse_m_s2",
        "Acceleration smoothness",
        r"Acceleration error RMSE [m/s$^2$]",
    ),
    (
        "jerk_error_rmse_m_s3",
        "Jerk smoothness",
        r"Jerk error RMSE [m/s$^3$]",
    ),
    ("path_length_ratio", "Path roughness", "Camera / mocap path length"),
)
for axis, (column, title, ylabel) in zip(depth_axes.ravel(), depth_plot_metrics):
    for method, color in zip(methods, colors):
        subset = depth_summary.loc[depth_summary["method"] == method]
        marker_ids = method_specs[method]["marker_ids"]
        linestyle, marker = marker_style[len(marker_ids)]
        axis.plot(
            subset["depth_m"],
            subset[column],
            marker=marker,
            linewidth=1.5,
            color=color,
            linestyle=linestyle,
            label=method,
        )
    if column == "path_length_ratio":
        axis.axhline(1.0, color="0.45", linewidth=1, linestyle="--")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
for axis in depth_axes[-1]:
    axis.set_xlabel("Mocap optical-axis depth [m]")
handles, labels = depth_axes[0, 0].get_legend_handles_labels()
depth_fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.875),
    ncols=6,
    frameon=False,
)
depth_fig.suptitle(
    "Tracking quality versus depth — static jitter and dynamic smoothness\n"
    "C1/C2 = camera count; M1/M2/M3 = fixed marker count; gaps lack support",
    y=0.985,
)
depth_fig.savefig(DEPTH_FIGURE, dpi=180, bbox_inches="tight")

print(f"\nSaved {SUMMARY_CSV}")
print(f"Saved {DISTANCE_CSV}")
print(f"Saved {SMOOTHNESS_CSV}")
print(f"Saved {DEPTH_METRICS_CSV}")
print(f"Saved {ALIGNMENT_TOML}")
print(f"Saved {OUTPUT_FIGURE}")
print(f"Saved {SMOOTHNESS_FIGURE}")
print(f"Saved {DEPTH_FIGURE}")
plt.show()


# %% Optional inspection tables
summary
distance_summary
smoothness_summary
depth_summary
