# %% [markdown]
# # Self-calibrate the AprilTag rigid body
#
# Learn the fixed pose of tags 4, 8, 14, and 20 relative to tag 12 from
# same-camera, same-frame observations. The first part of the recording is used
# for calibration; the remainder is held out for reprojection validation.
#
# Outputs:
#
# - `rigidbody_calibration.toml`: marker-to-tag12 rigid transforms
# - `jitter_detections.pkl`: reusable sub-pixel corner detections for both cameras
# - `rigidbody_calibration.png`: geometry and held-out validation plots
#
# Run from the repository root:
#
# ```powershell
# uv run python jitter_model/02_rigidbody_calib.py
# ```

# %% Imports
from collections import Counter
from pathlib import Path
import pickle
import warnings

import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import msgpack
import msgpack_numpy as mpn
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
import toml
from tqdm.auto import tqdm


# %% Paths and settings
try:
    NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:  # running cell-by-cell in an interactive kernel
    NOTEBOOK_DIR = Path.cwd()
    if NOTEBOOK_DIR.name != "jitter_model":
        NOTEBOOK_DIR = NOTEBOOK_DIR / "jitter_model"

PROJECT_ROOT = NOTEBOOK_DIR.parent
RECORDING_DIR = PROJECT_ROOT / "data" / "radxa" / "jitter_test"
CALIBRATION_TOML = (
    PROJECT_ROOT
    / "data"
    / "calibration"
    / "dual_160"
    / "radxa_calib_parallel"
    / "stereo_calibration.toml"
)

DETECTION_CACHE = NOTEBOOK_DIR / "jitter_detections.pkl"
RIGIDBODY_TOML = NOTEBOOK_DIR / "rigidbody_calibration.toml"
OUTPUT_FIGURE = NOTEBOOK_DIR / "rigidbody_calibration.png"

CAMERA_NAMES = ("cam0", "cam1")
MARKER_IDS = (4, 8, 12, 14, 20)
REFERENCE_ID = 12
TAG_SIZE_M = 0.05

CALIBRATION_FRACTION = 0.30
MAX_POSE_REPROJECTION_RMSE_PX = 1.5
MIN_SAMPLES_PER_MARKER = 20
REBUILD_DETECTION_CACHE = False
MAX_FRAMES = None

SUBPIX_WINDOW = (5, 5)
SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    40,
    0.01,
)


# %% Calibration and tag model
stereo_calibration = toml.load(CALIBRATION_TOML)
camera_models = {}
for camera_name in CAMERA_NAMES:
    camera_data = stereo_calibration[camera_name]
    camera_models[camera_name] = {
        "K": np.asarray(camera_data["camera_matrix"], dtype=np.float64),
        "D": np.asarray(camera_data["dist_coeffs"], dtype=np.float64).reshape(
            -1, 1
        ),
        "resolution": tuple(camera_data["resolution"]),
    }

half_size = TAG_SIZE_M / 2.0
TAG_CORNERS_MARKER = np.asarray(
    [
        [-half_size, +half_size, 0.0],
        [+half_size, +half_size, 0.0],
        [+half_size, -half_size, 0.0],
        [-half_size, -half_size, 0.0],
    ],
    dtype=np.float64,
)

dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
detector_parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, detector_parameters)


# %% Recording I/O and reusable detection cache
def load_timestamp_records(path):
    """Load GPIO, wall-clock, and host-monotonic timestamps."""
    with path.open("rb") as stream:
        records = list(msgpack.Unpacker(stream, object_hook=mpn.decode))
    return {
        "sync": np.asarray([int(record[0]) for record in records], dtype=np.uint8),
        "wall_time": np.asarray(
            [record[1] for record in records], dtype="datetime64[us]"
        ),
        "monotonic_ns": np.asarray(
            [int(record[2]) for record in records], dtype=np.int64
        ),
    }


def detect_camera(camera_name, max_frames=None):
    """Detect and sub-pixel refine all configured tags in one recording."""
    frame_path = RECORDING_DIR / f"{camera_name}_frame.msgpack"
    metadata = load_timestamp_records(
        RECORDING_DIR / f"{camera_name}_timestamp.msgpack"
    )
    frame_count = len(metadata["sync"])
    limit = frame_count if max_frames is None else min(frame_count, max_frames)
    detections = []

    with frame_path.open("rb") as stream:
        unpacker = msgpack.Unpacker(stream, object_hook=mpn.decode)
        for frame_index, frame in tqdm(
            enumerate(unpacker),
            total=limit,
            desc=f"Detecting rigid-body tags in {camera_name}",
        ):
            if frame_index >= limit:
                break
            expected_resolution = camera_models[camera_name]["resolution"]
            if tuple(frame.shape[1::-1]) != expected_resolution:
                raise ValueError(
                    f"{camera_name} frame resolution {frame.shape[1::-1]} does not "
                    f"match calibration {expected_resolution}"
                )

            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _rejected = detector.detectMarkers(gray)
            frame_detections = {}
            if ids is not None:
                for marker_id, marker_corners in zip(ids.ravel(), corners):
                    marker_id = int(marker_id)
                    if marker_id not in MARKER_IDS:
                        continue
                    refined = np.asarray(marker_corners, dtype=np.float32).reshape(
                        -1, 1, 2
                    )
                    cv2.cornerSubPix(
                        gray,
                        refined,
                        SUBPIX_WINDOW,
                        (-1, -1),
                        SUBPIX_CRITERIA,
                    )
                    frame_detections[marker_id] = refined.reshape(4, 2).astype(
                        np.float64
                    )
            detections.append(frame_detections)

    for key in metadata:
        metadata[key] = metadata[key][:limit]
    return {"metadata": metadata, "detections": detections}


def cache_is_compatible(cache):
    return (
        cache.get("version") == 1
        and tuple(cache.get("marker_ids", ())) == MARKER_IDS
        and cache.get("tag_size_m") == TAG_SIZE_M
        and cache.get("recording_dir") == str(RECORDING_DIR.resolve())
        and all(camera_name in cache.get("cameras", {}) for camera_name in CAMERA_NAMES)
    )


if DETECTION_CACHE.exists() and not REBUILD_DETECTION_CACHE:
    with DETECTION_CACHE.open("rb") as stream:
        detection_cache = pickle.load(stream)  # noqa: S301 - trusted local cache
    if not cache_is_compatible(detection_cache):
        warnings.warn("Detection cache is stale; rebuilding it")
        detection_cache = None
else:
    detection_cache = None

if detection_cache is None:
    detection_cache = {
        "version": 1,
        "recording_dir": str(RECORDING_DIR.resolve()),
        "calibration_toml": str(CALIBRATION_TOML.resolve()),
        "dictionary": "DICT_APRILTAG_36h11",
        "marker_ids": MARKER_IDS,
        "tag_size_m": TAG_SIZE_M,
        "cameras": {
            camera_name: detect_camera(camera_name, MAX_FRAMES)
            for camera_name in CAMERA_NAMES
        },
    }
    with DETECTION_CACHE.open("wb") as stream:
        pickle.dump(detection_cache, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved reusable detections -> {DETECTION_CACHE}")
else:
    print(f"Loaded reusable detections <- {DETECTION_CACHE}")

for camera_name in CAMERA_NAMES:
    counts = Counter(
        marker_id
        for frame in detection_cache["cameras"][camera_name]["detections"]
        for marker_id in frame
    )
    print(f"{camera_name} detection counts: {dict(sorted(counts.items()))}")


# %% Per-marker pose on an ideal pinhole image plane
def estimate_marker_pose(corners, camera_name):
    """Return the lowest-error positive-depth IPPE pose for one square tag."""
    K = camera_models[camera_name]["K"]
    D = camera_models[camera_name]["D"]
    undistorted = cv2.fisheye.undistortPoints(
        np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2), K, D, P=K
    ).reshape(4, 2)

    result = cv2.solvePnPGeneric(
        TAG_CORNERS_MARKER,
        undistorted,
        K,
        None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not result[0]:
        return None

    candidates = []
    for rvec, tvec in zip(result[1], result[2]):
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        if tvec[2] <= 0:
            continue
        projected, _ = cv2.projectPoints(
            TAG_CORNERS_MARKER,
            rvec,
            tvec,
            K,
            None,
        )
        residual = projected.reshape(4, 2) - undistorted
        rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        candidates.append(
            {
                "R": cv2.Rodrigues(rvec)[0],
                "t": tvec,
                "rmse_px": rmse,
            }
        )
    return min(candidates, key=lambda item: item["rmse_px"]) if candidates else None


pose_cache = {camera_name: [] for camera_name in CAMERA_NAMES}
for camera_name in CAMERA_NAMES:
    for frame in tqdm(
        detection_cache["cameras"][camera_name]["detections"],
        desc=f"Estimating individual tag poses in {camera_name}",
    ):
        pose_cache[camera_name].append(
            {
                marker_id: estimate_marker_pose(corners, camera_name)
                for marker_id, corners in frame.items()
            }
        )


# %% Candidate marker->tag12 transforms from the calibration split
frame_counts = [
    len(detection_cache["cameras"][camera_name]["detections"])
    for camera_name in CAMERA_NAMES
]
calibration_end_frame = int(min(frame_counts) * CALIBRATION_FRACTION)
print(
    f"Calibration split: frames [0, {calibration_end_frame}); "
    f"held-out validation starts at frame {calibration_end_frame}."
)

relative_candidates = {marker_id: [] for marker_id in MARKER_IDS if marker_id != REFERENCE_ID}
for camera_name in CAMERA_NAMES:
    for frame_index in range(calibration_end_frame):
        frame_poses = pose_cache[camera_name][frame_index]
        reference_pose = frame_poses.get(REFERENCE_ID)
        if reference_pose is None:
            continue
        if reference_pose["rmse_px"] > MAX_POSE_REPROJECTION_RMSE_PX:
            continue

        R_camera_reference = reference_pose["R"]
        t_camera_reference = reference_pose["t"]
        for marker_id in relative_candidates:
            marker_pose = frame_poses.get(marker_id)
            if marker_pose is None:
                continue
            if marker_pose["rmse_px"] > MAX_POSE_REPROJECTION_RMSE_PX:
                continue

            # Marker point -> camera via marker pose, then camera -> tag12 frame.
            R_reference_marker = R_camera_reference.T @ marker_pose["R"]
            t_reference_marker = R_camera_reference.T @ (
                marker_pose["t"] - t_camera_reference
            )
            relative_candidates[marker_id].append(
                {
                    "R": R_reference_marker,
                    "t": t_reference_marker,
                    "camera": camera_name,
                    "frame": frame_index,
                    "pose_rmse_px": max(
                        reference_pose["rmse_px"], marker_pose["rmse_px"]
                    ),
                }
            )


# %% Robust SE(3) averaging
def robust_limit(values, sigma=3.5, floor=0.0):
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = max(1.4826 * mad, floor)
    return median + sigma * robust_sigma


def robust_average_transform(candidates):
    """Iteratively reject translation and rotation outliers, then average SE(3)."""
    if len(candidates) < MIN_SAMPLES_PER_MARKER:
        raise RuntimeError(
            f"Only {len(candidates)} relative poses; need {MIN_SAMPLES_PER_MARKER}"
        )
    rotations = Rotation.from_matrix(np.asarray([item["R"] for item in candidates]))
    translations = np.asarray([item["t"] for item in candidates], dtype=np.float64)
    keep = np.ones(len(candidates), dtype=bool)

    for _ in range(6):
        mean_rotation = rotations[keep].mean()
        rotation_error_deg = np.degrees(
            (mean_rotation.inv() * rotations).magnitude()
        )
        median_translation = np.median(translations[keep], axis=0)
        translation_error_mm = 1000.0 * np.linalg.norm(
            translations - median_translation, axis=1
        )
        rotation_threshold = robust_limit(
            rotation_error_deg[keep], floor=0.5
        )
        translation_threshold = robust_limit(
            translation_error_mm[keep], floor=0.5
        )
        new_keep = (
            (rotation_error_deg <= rotation_threshold)
            & (translation_error_mm <= translation_threshold)
        )
        if new_keep.sum() < MIN_SAMPLES_PER_MARKER:
            break
        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep

    mean_rotation = rotations[keep].mean()
    median_translation = np.median(translations[keep], axis=0)
    rotation_error_deg = np.degrees(
        (mean_rotation.inv() * rotations[keep]).magnitude()
    )
    translation_error_mm = 1000.0 * np.linalg.norm(
        translations[keep] - median_translation, axis=1
    )
    return {
        "R": mean_rotation.as_matrix(),
        "t": median_translation,
        "sample_count": len(candidates),
        "used_count": int(keep.sum()),
        "rotation_spread_deg": float(np.median(rotation_error_deg)),
        "translation_spread_mm": float(np.median(translation_error_mm)),
    }


marker_transforms = {
    REFERENCE_ID: {
        "R": np.eye(3),
        "t": np.zeros(3),
        "sample_count": 0,
        "used_count": 0,
        "rotation_spread_deg": 0.0,
        "translation_spread_mm": 0.0,
    }
}
for marker_id, candidates in relative_candidates.items():
    marker_transforms[marker_id] = robust_average_transform(candidates)
    item = marker_transforms[marker_id]
    print(
        f"tag {marker_id:2d} -> tag {REFERENCE_ID}: "
        f"used {item['used_count']}/{item['sample_count']}, "
        f"t={np.round(1000 * item['t'], 2)} mm, "
        f"spread={item['translation_spread_mm']:.2f} mm / "
        f"{item['rotation_spread_deg']:.2f} deg"
    )


# %% Joint bundle adjustment of marker geometry and per-view board poses
# A square tag's fronto-parallel pose is weakly constrained by only four corners.
# Refine every fixed marker transform together, while allowing each same-camera
# calibration view to have its own nuisance board pose. This makes all observed
# corners agree with one rigid body and removes the IPPE tilt bias from the
# initial pairwise transforms above.
optimized_marker_ids = [
    marker_id for marker_id in MARKER_IDS if marker_id != REFERENCE_ID
]
calibration_views = []
for camera_name in CAMERA_NAMES:
    camera_detections = detection_cache["cameras"][camera_name]["detections"]
    for frame_index in range(calibration_end_frame):
        frame_detections = camera_detections[frame_index]
        reference_pose = pose_cache[camera_name][frame_index].get(REFERENCE_ID)
        visible_ids = [
            marker_id for marker_id in MARKER_IDS if marker_id in frame_detections
        ]
        if (
            reference_pose is None
            or reference_pose["rmse_px"] > MAX_POSE_REPROJECTION_RMSE_PX
            or len(visible_ids) < 2
        ):
            continue
        calibration_views.append(
            {
                "camera": camera_name,
                "frame": frame_index,
                "visible_ids": visible_ids,
                "detections": frame_detections,
                "initial_R": reference_pose["R"],
                "initial_t": reference_pose["t"],
            }
        )

marker_parameter_offset = {
    marker_id: 6 * index for index, marker_id in enumerate(optimized_marker_ids)
}
view_parameter_start = 6 * len(optimized_marker_ids)

x0_parts = []
for marker_id in optimized_marker_ids:
    transform = marker_transforms[marker_id]
    x0_parts.append(Rotation.from_matrix(transform["R"]).as_rotvec())
    x0_parts.append(transform["t"])
for view in calibration_views:
    x0_parts.append(Rotation.from_matrix(view["initial_R"]).as_rotvec())
    x0_parts.append(view["initial_t"])
x0 = np.concatenate(x0_parts)


def unpack_marker_transform(parameters, marker_id):
    if marker_id == REFERENCE_ID:
        return np.eye(3), np.zeros(3)
    offset = marker_parameter_offset[marker_id]
    return (
        Rotation.from_rotvec(parameters[offset : offset + 3]).as_matrix(),
        parameters[offset + 3 : offset + 6],
    )


def bundle_residuals(parameters):
    residuals = []
    for view_index, view in enumerate(calibration_views):
        view_offset = view_parameter_start + 6 * view_index
        rvec = parameters[view_offset : view_offset + 3]
        tvec = parameters[view_offset + 3 : view_offset + 6]
        model = camera_models[view["camera"]]
        for marker_id in view["visible_ids"]:
            marker_R, marker_t = unpack_marker_transform(parameters, marker_id)
            object_points = TAG_CORNERS_MARKER @ marker_R.T + marker_t
            projected, _ = cv2.fisheye.projectPoints(
                object_points.reshape(-1, 1, 3),
                rvec,
                tvec,
                model["K"],
                model["D"],
            )
            residuals.append(
                (projected.reshape(4, 2) - view["detections"][marker_id]).ravel()
            )
    return np.concatenate(residuals)


residual_count = sum(8 * len(view["visible_ids"]) for view in calibration_views)
jacobian_sparsity = lil_matrix((residual_count, len(x0)), dtype=np.uint8)
row = 0
for view_index, view in enumerate(calibration_views):
    view_offset = view_parameter_start + 6 * view_index
    for marker_id in view["visible_ids"]:
        jacobian_sparsity[row : row + 8, view_offset : view_offset + 6] = 1
        if marker_id != REFERENCE_ID:
            marker_offset = marker_parameter_offset[marker_id]
            jacobian_sparsity[row : row + 8, marker_offset : marker_offset + 6] = 1
        row += 8

initial_bundle_rmse = float(
    np.sqrt(np.mean(bundle_residuals(x0).reshape(-1, 2) ** 2) * 2.0)
)
print(
    f"Bundle adjustment: {len(calibration_views)} views, "
    f"{residual_count // 2} corner coordinates, "
    f"initial point RMSE={initial_bundle_rmse:.3f} px"
)
bundle_result = least_squares(
    bundle_residuals,
    x0,
    jac_sparsity=jacobian_sparsity.tocsr(),
    method="trf",
    loss="soft_l1",
    f_scale=0.5,
    x_scale="jac",
    max_nfev=100,
    verbose=1,
)
bundle_rmse = float(
    np.sqrt(np.mean(bundle_residuals(bundle_result.x).reshape(-1, 2) ** 2) * 2.0)
)
print(
    f"Bundle adjustment finished: success={bundle_result.success}, "
    f"point RMSE={bundle_rmse:.3f} px"
)
for marker_id in optimized_marker_ids:
    marker_R, marker_t = unpack_marker_transform(bundle_result.x, marker_id)
    marker_transforms[marker_id]["R"] = marker_R
    marker_transforms[marker_id]["t"] = marker_t
    print(
        f"refined tag {marker_id:2d} -> tag {REFERENCE_ID}: "
        f"t={np.round(1000 * marker_t, 2)} mm"
    )


# %% Self-calibrate the current cam0 -> cam1 extrinsic from the rigid board
# The supplied TOML remains the source of both intrinsics and the nominal
# baseline. The cameras in this take have a different relative alignment, so we
# estimate the actual six-DoF stereo transform from paired calibration views.
def marker_corners_from_transform(marker_id):
    transform = marker_transforms[marker_id]
    return TAG_CORNERS_MARKER @ transform["R"].T + transform["t"]


def board_pose_for_stereo(frame_detections, camera_name, require_markers=2):
    visible_ids = [marker_id for marker_id in MARKER_IDS if marker_id in frame_detections]
    if len(visible_ids) < require_markers:
        return None
    object_points = np.concatenate(
        [marker_corners_from_transform(marker_id) for marker_id in visible_ids]
    )
    raw_points = np.concatenate([frame_detections[marker_id] for marker_id in visible_ids])
    model = camera_models[camera_name]
    undistorted = cv2.fisheye.undistortPoints(
        raw_points.reshape(-1, 1, 2), model["K"], model["D"], P=model["K"]
    ).reshape(-1, 2)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        undistorted,
        model["K"],
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success or float(tvec.reshape(3)[2]) <= 0:
        return None
    return {
        "R": cv2.Rodrigues(rvec)[0],
        "t": tvec.reshape(3),
        "rvec": rvec.reshape(3),
        "visible_ids": visible_ids,
        "object_points": object_points,
        "raw_points": raw_points,
    }


mono_ns0 = detection_cache["cameras"]["cam0"]["metadata"]["monotonic_ns"]
mono_ns1 = detection_cache["cameras"]["cam1"]["metadata"]["monotonic_ns"]
median_period_ns = float(np.median(np.diff(mono_ns0)))


def nearest_stereo_frame(timestamp_ns):
    insertion = int(np.searchsorted(mono_ns1, timestamp_ns))
    candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(mono_ns1)]
    if not candidates:
        return None
    match = min(candidates, key=lambda index: abs(int(mono_ns1[index]) - int(timestamp_ns)))
    if abs(int(mono_ns1[match]) - int(timestamp_ns)) > 0.55 * median_period_ns:
        return None
    return match


stereo_views = []
stereo_candidates = []
for frame0 in range(calibration_end_frame):
    frame1 = nearest_stereo_frame(mono_ns0[frame0])
    if frame1 is None:
        continue
    pose0 = board_pose_for_stereo(
        detection_cache["cameras"]["cam0"]["detections"][frame0], "cam0"
    )
    pose1 = board_pose_for_stereo(
        detection_cache["cameras"]["cam1"]["detections"][frame1], "cam1"
    )
    if pose0 is None or pose1 is None:
        continue
    relative_R = pose1["R"] @ pose0["R"].T
    relative_t = pose1["t"] - relative_R @ pose0["t"]
    stereo_candidates.append({"R": relative_R, "t": relative_t})
    stereo_views.append(
        {
            "frame0": frame0,
            "frame1": frame1,
            "cam0": pose0,
            "cam1": pose1,
            "gap_ms": abs(int(mono_ns1[frame1]) - int(mono_ns0[frame0])) / 1e6,
        }
    )

stereo_initial = robust_average_transform(stereo_candidates)
print(
    "Current stereo initial estimate: "
    f"T={np.round(1000 * stereo_initial['t'], 2)} mm, "
    f"N={len(stereo_views)}, median pair gap="
    f"{np.median([view['gap_ms'] for view in stereo_views]):.2f} ms"
)

# Global stereo transform followed by one nuisance tag12->cam0 pose per view.
stereo_x0_parts = [
    Rotation.from_matrix(stereo_initial["R"]).as_rotvec(),
    stereo_initial["t"],
]
for view in stereo_views:
    stereo_x0_parts.extend([view["cam0"]["rvec"], view["cam0"]["t"]])
stereo_x0 = np.concatenate(stereo_x0_parts)


def stereo_bundle_residual(parameters):
    stereo_rvec = parameters[:3].reshape(3, 1)
    stereo_tvec = parameters[3:6].reshape(3, 1)
    residuals = []
    for view_index, view in enumerate(stereo_views):
        offset = 6 + 6 * view_index
        rvec0 = parameters[offset : offset + 3].reshape(3, 1)
        tvec0 = parameters[offset + 3 : offset + 6].reshape(3, 1)
        for camera_name, observation in (("cam0", view["cam0"]),):
            model = camera_models[camera_name]
            projected, _ = cv2.fisheye.projectPoints(
                observation["object_points"].reshape(-1, 1, 3),
                rvec0,
                tvec0,
                model["K"],
                model["D"],
            )
            residuals.append((projected.reshape(-1, 2) - observation["raw_points"]).ravel())
        rvec1, tvec1 = cv2.composeRT(rvec0, tvec0, stereo_rvec, stereo_tvec)[:2]
        observation = view["cam1"]
        model = camera_models["cam1"]
        projected, _ = cv2.fisheye.projectPoints(
            observation["object_points"].reshape(-1, 1, 3),
            rvec1,
            tvec1,
            model["K"],
            model["D"],
        )
        residuals.append((projected.reshape(-1, 2) - observation["raw_points"]).ravel())
    return np.concatenate(residuals)


stereo_residual_count = sum(
    2 * (len(view["cam0"]["raw_points"]) + len(view["cam1"]["raw_points"]))
    for view in stereo_views
)
stereo_sparsity = lil_matrix((stereo_residual_count, len(stereo_x0)), dtype=np.uint8)
row = 0
for view_index, view in enumerate(stereo_views):
    view_offset = 6 + 6 * view_index
    cam0_rows = 2 * len(view["cam0"]["raw_points"])
    stereo_sparsity[row : row + cam0_rows, view_offset : view_offset + 6] = 1
    row += cam0_rows
    cam1_rows = 2 * len(view["cam1"]["raw_points"])
    stereo_sparsity[row : row + cam1_rows, :6] = 1
    stereo_sparsity[row : row + cam1_rows, view_offset : view_offset + 6] = 1
    row += cam1_rows

stereo_initial_rmse = float(
    np.sqrt(np.mean(stereo_bundle_residual(stereo_x0).reshape(-1, 2) ** 2) * 2)
)
stereo_result = least_squares(
    stereo_bundle_residual,
    stereo_x0,
    jac_sparsity=stereo_sparsity.tocsr(),
    method="trf",
    loss="soft_l1",
    f_scale=0.75,
    x_scale="jac",
    max_nfev=300,
    verbose=1,
)
refined_stereo_R = Rotation.from_rotvec(stereo_result.x[:3]).as_matrix()
refined_stereo_t = stereo_result.x[3:6]
stereo_final_rmse = float(
    np.sqrt(np.mean(stereo_bundle_residual(stereo_result.x).reshape(-1, 2) ** 2) * 2)
)
print(
    "Refined current stereo: "
    f"T={np.round(1000 * refined_stereo_t, 2)} mm, "
    f"RMSE={stereo_initial_rmse:.3f}->{stereo_final_rmse:.3f} px"
)


# %% Export the rigid-body calibration
rigidbody_payload = {
    "meta": {
        "format_version": 1,
        "recording": str(RECORDING_DIR.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "stereo_calibration": str(CALIBRATION_TOML.relative_to(PROJECT_ROOT)).replace(
            "\\", "/"
        ),
        "dictionary": "DICT_APRILTAG_36h11",
        "tag_size_m": TAG_SIZE_M,
        "reference_id": REFERENCE_ID,
        "marker_ids": list(MARKER_IDS),
        "calibration_fraction": CALIBRATION_FRACTION,
        "calibration_end_frame": calibration_end_frame,
        "bundle_views": len(calibration_views),
        "bundle_initial_rmse_px": initial_bundle_rmse,
        "bundle_final_rmse_px": bundle_rmse,
        "bundle_success": bool(bundle_result.success),
    },
    "markers": {},
    "stereo_refined": {
        "rotation_cam0_to_cam1": refined_stereo_R.tolist(),
        "translation_cam0_to_cam1_m": refined_stereo_t.tolist(),
        "views": len(stereo_views),
        "median_pair_gap_ms": float(np.median([view["gap_ms"] for view in stereo_views])),
        "initial_rmse_px": stereo_initial_rmse,
        "final_rmse_px": stereo_final_rmse,
        "success": bool(stereo_result.success),
    },
}
for marker_id in MARKER_IDS:
    item = marker_transforms[marker_id]
    rigidbody_payload["markers"][str(marker_id)] = {
        "rotation_marker_to_reference": item["R"].tolist(),
        "translation_marker_to_reference_m": item["t"].tolist(),
        "samples_total": item["sample_count"],
        "samples_used": item["used_count"],
        "translation_spread_mm": item["translation_spread_mm"],
        "rotation_spread_deg": item["rotation_spread_deg"],
    }

with RIGIDBODY_TOML.open("w", encoding="utf-8") as stream:
    toml.dump(rigidbody_payload, stream)
print(f"Saved rigid-body geometry -> {RIGIDBODY_TOML}")


# %% Held-out board reprojection validation
def marker_corners_in_reference(marker_id):
    transform = marker_transforms[marker_id]
    return TAG_CORNERS_MARKER @ transform["R"].T + transform["t"]


def estimate_board_pose(frame_detections, camera_name, require_markers=2):
    visible_ids = [marker_id for marker_id in MARKER_IDS if marker_id in frame_detections]
    if len(visible_ids) < require_markers:
        return None
    object_points = np.concatenate(
        [marker_corners_in_reference(marker_id) for marker_id in visible_ids]
    )
    image_points_raw = np.concatenate(
        [frame_detections[marker_id] for marker_id in visible_ids]
    )
    K = camera_models[camera_name]["K"]
    D = camera_models[camera_name]["D"]
    image_points = cv2.fisheye.undistortPoints(
        image_points_raw.reshape(-1, 1, 2), K, D, P=K
    ).reshape(-1, 2)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success or tvec.reshape(3)[2] <= 0:
        return None
    projected, _ = cv2.fisheye.projectPoints(
        object_points.reshape(-1, 1, 3), rvec, tvec, K, D
    )
    residual = projected.reshape(-1, 2) - image_points_raw
    return {
        "marker_count": len(visible_ids),
        "rmse_px": float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
    }


validation_rows = []
for camera_name in CAMERA_NAMES:
    heldout = detection_cache["cameras"][camera_name]["detections"][
        calibration_end_frame:
    ]
    for frame_offset, frame_detections in enumerate(
        tqdm(heldout, desc=f"Validating board geometry in {camera_name}")
    ):
        validation = estimate_board_pose(frame_detections, camera_name)
        if validation is not None:
            validation_rows.append(
                {
                    "camera": camera_name,
                    "frame": calibration_end_frame + frame_offset,
                    **validation,
                }
            )

validation_rmse = np.asarray(
    [row["rmse_px"] for row in validation_rows], dtype=np.float64
)
if not len(validation_rmse):
    raise RuntimeError("No held-out multi-marker frames available for validation")
print(
    f"Held-out board reprojection: N={len(validation_rmse)}, "
    f"median={np.median(validation_rmse):.3f} px, "
    f"p95={np.percentile(validation_rmse, 95):.3f} px"
)


# %% Plot calibrated geometry and validation error
fig = plt.figure(figsize=(14, 6), constrained_layout=True)
geometry_ax = fig.add_subplot(1, 2, 1, projection="3d")
validation_ax = fig.add_subplot(1, 2, 2)

colors = plt.cm.tab10(np.linspace(0, 1, len(MARKER_IDS)))
for marker_id, color in zip(MARKER_IDS, colors):
    corners = marker_corners_in_reference(marker_id)
    loop = np.vstack([corners, corners[0]]) * 1000.0
    geometry_ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=color, linewidth=2)
    center = marker_transforms[marker_id]["t"] * 1000.0
    geometry_ax.text(*center, str(marker_id), color=color, fontsize=11)

geometry_ax.set_title(f"Self-calibrated tag geometry in tag {REFERENCE_ID} frame")
geometry_ax.set_xlabel("X [mm]")
geometry_ax.set_ylabel("Y [mm]")
geometry_ax.set_zlabel("Z [mm]")
geometry_ax.set_box_aspect((1, 1, 1))

for camera_name, color in zip(CAMERA_NAMES, ("tab:blue", "tab:orange")):
    values = [
        row["rmse_px"] for row in validation_rows if row["camera"] == camera_name
    ]
    validation_ax.hist(
        values,
        bins=np.linspace(0, min(5.0, max(validation_rmse)), 40),
        alpha=0.55,
        label=f"{camera_name} (N={len(values)})",
        color=color,
    )
validation_ax.axvline(
    np.median(validation_rmse), color="black", linestyle="--", label="combined median"
)
validation_ax.set_title("Held-out multi-marker reprojection error")
validation_ax.set_xlabel("RMSE [px]")
validation_ax.set_ylabel("Frames")
validation_ax.grid(True, alpha=0.25)
validation_ax.legend()

fig.savefig(OUTPUT_FIGURE, dpi=180, bbox_inches="tight")
print(f"Saved calibration figure -> {OUTPUT_FIGURE}")
plt.show()


# %% Optional inspection values
rigidbody_payload
