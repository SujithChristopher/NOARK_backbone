# %% [markdown]
# # Does inter-camera timing drive the stereo error?
#
# The two sensors free-run against each other, so cam1 exposes a little before
# or after cam0. A stereo solve treats the pair as simultaneous, so while the
# board is moving the two views disagree by roughly `phase x speed`, and the
# solve absorbs that as position error.
#
# `01_phase_check.py` measures the phase and `06_movement_error.py` measures the
# error. This script asks whether one explains the other, which a plain
# correlation cannot answer: phase drifts monotonically through a take while
# speed and distance vary with the sweep, so the two are confounded from the
# start.
#
# Five tests, weakest to strongest:
#
# 1. raw correlation, for reference, with the single-camera conditions as a
#    control -- they never read cam1, so any correlation there is confounding;
# 2. partial correlation, with speed and distance regressed out;
# 3. the signed projection of the error onto the direction of travel, since
#    latency displaces the estimate *along* the motion rather than merely
#    inflating its magnitude, and must flip sign with the sign of the phase;
# 4. the paired stereo-minus-mono difference, which cancels the systematic floor
#    the two share and leaves only what is specific to the stereo solve;
# 5. an injection test: cam1 is deliberately re-paired one and two frames away,
#    the stereo pose re-solved, and the resulting displacement measured against
#    a known offset. That is an intervention rather than an observation, so it
#    calibrates how far position moves per millisecond of timing error and turns
#    a null result into a stated detection limit.
#
# Requires `06_movement_error.py` to have been run on the same recording.
#
# ```powershell
# uv run python jitter_model/07_timing_sensitivity.py
# ```

# %% Imports
import argparse
import pickle
import sys
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import msgpack
import msgpack_numpy as mpn
import numpy as np
import pandas as pd
import toml
from cv2 import aruco
from matplotlib import ticker
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from tqdm.auto import tqdm

from support import pd_support

# %% Paths and settings
try:
    NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:  # running cell-by-cell in an interactive kernel
    NOTEBOOK_DIR = Path.cwd()
    if NOTEBOOK_DIR.name != "jitter_model":
        NOTEBOOK_DIR = NOTEBOOK_DIR / "jitter_model"

PROJECT_ROOT = NOTEBOOK_DIR.parent

RECORDING_DIR = (
    PROJECT_ROOT / "data" / "dome" / "sep18_26" / "dome_random_movement_sep18_26"
)
OUTPUT_SUBDIR = "timing_sensitivity"
OUTPUT_DIR = None

CAMERA_NAMES = ("cam0", "cam1")
REBUILD_DETECTION_CACHE = False
MAX_FRAMES = None
MAX_TAGS = 4

# Corner sub-pixel refinement, matching 02_rigidbody_calib.py.
SUBPIX_WINDOW = (5, 5)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
# Pair the two free-running cameras no further apart than this much of a frame.
MAX_PAIR_FRACTION_OF_FRAME = 0.55

# Half the usable span trains the alignment, the rest is held out for the
# numbers that get reported.
ALIGNMENT_FIT_FRACTION = 0.5
# Camera and mocap durations must agree this closely for the GPIO mapping to be
# trusted without further work.
MAX_SYNC_MISMATCH_S = 0.5
# Poses this far from the aligned mocap are detection or pose-flip failures
# rather than accuracy, and are counted separately instead of averaged in.
GROSS_ERROR_M = 0.05

DISTANCE_EDGES_M = np.asarray([0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.95])
SPEED_EDGES_MS = np.asarray([0.0, 0.05, 0.10, 0.20, 0.35, 0.60, 1.00])
MIN_SAMPLES_PER_BIN = 25
# Bins for the per-condition error histograms, log-spaced across the pooled range.
HISTOGRAM_BINS = 55

ERROR_COLORMAP = "turbo"

# Which stereo condition the injection test re-solves, and how far cam1 is
# shifted. One frame is a whole period, thousands of times the real phase, which
# is the point: the effect must be made large enough to measure before the real
# one can be placed against it.
INJECTION_CONDITION_TAGS = 3
INJECTION_FRAME_OFFSETS = (-2, -1, 0, 1, 2)


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Dynamic movement accuracy for one dual recording."
    )
    parser.add_argument("--recording", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__" and "ipykernel" not in sys.modules:
    _args = _parse_args(sys.argv[1:])
    if _args.recording is not None:
        RECORDING_DIR = _args.recording
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
MOCAP_CSV = RECORDING_DIR / f"{TAKE_NAME}.csv"
SESSION_TOML = RECORDING_DIR.parent / "session.toml"
DETECTION_CACHE = RECORDING_DIR / "jitter_detections.pkl"

# 06 writes the per-frame errors this script sets out to explain.
MOVEMENT_SAMPLES_CSV = RECORDING_DIR / "movement_error" / "movement_error_samples.csv"

OBSERVED_CSV = OUTPUT_DIR / "timing_observational.csv"
INJECTION_CSV = OUTPUT_DIR / "timing_injection.csv"
PHASE_FIGURE = OUTPUT_DIR / "timing_phase_and_error.png"
SCATTER_FIGURE = OUTPUT_DIR / "timing_error_vs_predicted.png"
DIRECTIONAL_FIGURE = OUTPUT_DIR / "timing_directional.png"
INJECTION_FIGURE = OUTPUT_DIR / "timing_injection.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Recording : {RECORDING_DIR}")
print(f"Outputs   : {OUTPUT_DIR}")


# %% Session configuration
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
print(
    f"Session   : {SESSION_TOML.parent.name}, mocap body "
    f"x={MOCAP_BODY_X_FROM[0]}-{MOCAP_BODY_X_FROM[1]}, "
    f"z={MOCAP_BODY_Z_FROM[0]}-{MOCAP_BODY_Z_FROM[1]}, origin={MOCAP_BODY_ORIGIN}"
)

for required_path in (STEREO_TOML, RIGIDBODY_TOML, MOCAP_CSV):
    if not required_path.exists():
        raise FileNotFoundError(f"Missing {required_path}")

stereo = toml.load(STEREO_TOML)
rigidbody = toml.load(RIGIDBODY_TOML)

camera_models = {}
for camera_name in CAMERA_NAMES:
    camera_data = stereo[camera_name]
    camera_models[camera_name] = {
        "K": np.asarray(camera_data["camera_matrix"], dtype=np.float64),
        "D": np.asarray(camera_data["dist_coeffs"], dtype=np.float64).reshape(-1, 1),
        "resolution": tuple(camera_data["resolution"]),
    }

if "stereo_refined" in rigidbody:
    R_STEREO = np.asarray(
        rigidbody["stereo_refined"]["rotation_cam0_to_cam1"], dtype=np.float64
    )
    T_STEREO = np.asarray(
        rigidbody["stereo_refined"]["translation_cam0_to_cam1_m"], dtype=np.float64
    )
    print("Using self-calibrated stereo extrinsic from rigidbody_calibration.toml")
else:
    warnings.warn("No refined stereo extrinsic; falling back to calibration TOML")
    R_STEREO = np.asarray(stereo["stereo"]["R"], dtype=np.float64)
    T_STEREO = np.asarray(stereo["stereo"]["T"], dtype=np.float64) / 1000.0
RVEC_STEREO = cv2.Rodrigues(R_STEREO)[0]

TAG_SIZE_M = float(rigidbody["meta"]["tag_size_m"])
REFERENCE_ID = int(rigidbody["meta"]["reference_id"])
DETECT_MARKER_IDS = tuple(int(x) for x in rigidbody["meta"]["marker_ids"])

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
marker_to_reference = {}
for marker_id_text, marker_data in rigidbody["markers"].items():
    marker_rotation = np.asarray(
        marker_data["rotation_marker_to_reference"], dtype=np.float64
    )
    marker_translation = np.asarray(
        marker_data["translation_marker_to_reference_m"], dtype=np.float64
    )
    marker_corners_reference[int(marker_id_text)] = (
        TAG_CORNERS_LOCAL @ marker_rotation.T + marker_translation
    )
    marker_to_reference[int(marker_id_text)] = (marker_rotation, marker_translation)

dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
detector_parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, detector_parameters)


# %% Recording I/O and reusable detection cache
def load_timestamp_records(path):
    """Per-frame GPIO bit and clocks. `sensor_ns` is the frame-done stamp."""
    with path.open("rb") as stream:
        records = list(msgpack.Unpacker(stream, object_hook=mpn.decode))
    return {
        "sync": np.asarray([int(record[0]) for record in records], dtype=np.uint8),
        "monotonic_ns": np.asarray(
            [int(record[2]) for record in records], dtype=np.int64
        ),
        "sensor_ns": np.asarray([int(record[3]) for record in records], dtype=np.int64),
    }


def detect_camera(camera_name, max_frames=None):
    """Detect and sub-pixel refine every dome tag in one camera's recording."""
    frame_path = RECORDING_DIR / f"{camera_name}_frame.msgpack"
    metadata = load_timestamp_records(
        RECORDING_DIR / f"{camera_name}_timestamp.msgpack"
    )
    frame_count = len(metadata["sync"])
    limit = frame_count if max_frames is None else min(frame_count, max_frames)
    expected_resolution = camera_models[camera_name]["resolution"]
    detections = []

    with frame_path.open("rb") as stream:
        unpacker = msgpack.Unpacker(stream, object_hook=mpn.decode)
        for frame_index, frame in tqdm(
            enumerate(unpacker), total=limit, desc=f"Detecting tags in {camera_name}"
        ):
            if frame_index >= limit:
                break
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
                    if marker_id not in DETECT_MARKER_IDS:
                        continue
                    refined = np.asarray(marker_corners, dtype=np.float32).reshape(
                        -1, 1, 2
                    )
                    cv2.cornerSubPix(
                        gray, refined, SUBPIX_WINDOW, (-1, -1), SUBPIX_CRITERIA
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
        and tuple(cache.get("marker_ids", ())) == DETECT_MARKER_IDS
        and cache.get("tag_size_m") == TAG_SIZE_M
        and cache.get("recording_dir") == str(RECORDING_DIR.resolve())
        and all(name in cache.get("cameras", {}) for name in CAMERA_NAMES)
    )


detection_cache = None
if DETECTION_CACHE.exists() and not REBUILD_DETECTION_CACHE:
    with DETECTION_CACHE.open("rb") as stream:
        detection_cache = pickle.load(stream)
    if not cache_is_compatible(detection_cache):
        warnings.warn("Detection cache is stale; rebuilding it")
        detection_cache = None

if detection_cache is None:
    detection_cache = {
        "version": 1,
        "recording_dir": str(RECORDING_DIR.resolve()),
        "calibration_toml": str(STEREO_TOML.resolve()),
        "dictionary": "DICT_APRILTAG_36h11",
        "marker_ids": DETECT_MARKER_IDS,
        "tag_size_m": TAG_SIZE_M,
        "cameras": {name: detect_camera(name, MAX_FRAMES) for name in CAMERA_NAMES},
    }
    with DETECTION_CACHE.open("wb") as stream:
        pickle.dump(detection_cache, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved reusable detections -> {DETECTION_CACHE}")
else:
    print(f"Loaded reusable detections <- {DETECTION_CACHE}")

cam0 = detection_cache["cameras"]["cam0"]
cam1 = detection_cache["cameras"]["cam1"]


# %% Pair the two free-running cameras on the shared host-monotonic clock
cam0_monotonic_ns = cam0["metadata"]["monotonic_ns"]
cam1_monotonic_ns = cam1["metadata"]["monotonic_ns"]
cam0_period_ns = float(np.median(np.diff(cam0_monotonic_ns)))
max_pair_gap_ns = MAX_PAIR_FRACTION_OF_FRAME * cam0_period_ns


def nearest_cam1_index(timestamp_ns):
    insertion = int(np.searchsorted(cam1_monotonic_ns, timestamp_ns))
    candidates = [
        i for i in (insertion - 1, insertion) if 0 <= i < len(cam1_monotonic_ns)
    ]
    if not candidates:
        return None
    nearest = min(
        candidates, key=lambda i: abs(int(cam1_monotonic_ns[i]) - int(timestamp_ns))
    )
    if abs(int(cam1_monotonic_ns[nearest]) - int(timestamp_ns)) > max_pair_gap_ns:
        return None
    return nearest


cam1_pair = np.asarray(
    [
        -1 if (match := nearest_cam1_index(stamp)) is None else match
        for stamp in cam0_monotonic_ns
    ],
    dtype=np.int32,
)
print(
    f"Camera pairing: {int((cam1_pair >= 0).sum())}/{len(cam0_monotonic_ns)} frames, "
    f"{1e9 / cam0_period_ns:.1f} fps"
)


# %% GPIO synchronization
# Motive is gated by the pulse on these takes, so its seconds column starts at
# the rising edge. That is an assumption about the wiring, not a fact about the
# file, so it is checked: the high interval and the Motive take should be the
# same length. The static burst recording fails this check, which is how its
# missing falling edge was caught.
sync_events_path = RECORDING_DIR / "sync_events.msgpack"
sync_rise_ns = sync_fall_ns = None
if sync_events_path.exists():
    with sync_events_path.open("rb") as stream:
        sync_records = list(msgpack.Unpacker(stream, object_hook=mpn.decode))
    for record in sync_records:
        if int(record[1]) == 1 and sync_rise_ns is None:
            sync_rise_ns = int(record[0])
        elif int(record[1]) == 0 and sync_rise_ns is not None and sync_fall_ns is None:
            sync_fall_ns = int(record[0])
if sync_rise_ns is None:
    high = np.flatnonzero(cam0["metadata"]["sync"] == 1)
    if not len(high):
        raise RuntimeError("No GPIO synchronization pulse in this recording")
    sync_rise_ns = int(cam0_monotonic_ns[high[0]])

camera_time_s = (cam0_monotonic_ns - sync_rise_ns) / 1e9


# %% Motive loader
def read_mocap_body(path):
    """Motive take with its body frame rebuilt from the labelled markers.

    The axes are orthonormalized from two measured marker edges named in
    session.toml rather than taken from Motive's solved quaternion, whose rigid
    body can re-lock at a ghost pose after an occlusion.
    """
    table, capture_start = pd_support.read_rigid_body_csv(str(path))
    positions, rotations = pd_support.rigid_body_marker_frames(
        table,
        x_from=MOCAP_BODY_X_FROM,
        z_from=MOCAP_BODY_Z_FROM,
        origin=MOCAP_BODY_ORIGIN,
    )
    seconds = table["seconds"].to_numpy(dtype=np.float64)
    finite = (
        np.isfinite(seconds)
        & np.isfinite(positions).all(axis=1)
        & np.isfinite(rotations).all(axis=(1, 2))
    )
    dropped = int((~finite).sum())
    if dropped:
        print(f"Dropped {dropped} mocap frames with a missing marker")
    quaternions = Rotation.from_matrix(rotations[finite]).as_quat()
    frame = pd.DataFrame(
        {
            "seconds": seconds[finite],
            "px": positions[finite, 0],
            "py": positions[finite, 1],
            "pz": positions[finite, 2],
            "qx": quaternions[:, 0],
            "qy": quaternions[:, 1],
            "qz": quaternions[:, 2],
            "qw": quaternions[:, 3],
        }
    )
    return frame.drop_duplicates("seconds").sort_values("seconds"), capture_start


mocap, mocap_capture_start = read_mocap_body(MOCAP_CSV)
mocap_times = mocap["seconds"].to_numpy(dtype=np.float64)
mocap_positions = mocap[["px", "py", "pz"]].to_numpy(dtype=np.float64)
mocap_quaternions = mocap[["qx", "qy", "qz", "qw"]].to_numpy(
    dtype=np.float64, copy=True
)
mocap_quaternions /= np.linalg.norm(mocap_quaternions, axis=1, keepdims=True)
mocap_slerp = Slerp(mocap_times, Rotation.from_quat(mocap_quaternions))

if sync_fall_ns is not None:
    pulse_s = (sync_fall_ns - sync_rise_ns) / 1e9
    mismatch = abs(pulse_s - mocap_times[-1])
    print(
        f"GPIO pulse {pulse_s:.2f} s against a {mocap_times[-1]:.2f} s Motive take "
        f"({mismatch:.2f} s apart)"
    )
    if mismatch > MAX_SYNC_MISMATCH_S:
        raise RuntimeError(
            f"GPIO pulse and Motive take differ by {mismatch:.2f} s; the pulse does "
            "not gate this recording and the timing needs to be established first"
        )
else:
    warnings.warn("No falling edge; the GPIO mapping could not be cross-checked")


def interpolate_mocap(query_times):
    """Mocap body pose at arbitrary camera times, linear in position, slerp in rotation."""
    query_times = np.asarray(query_times, dtype=np.float64)
    positions = np.column_stack(
        [
            np.interp(query_times, mocap_times, mocap_positions[:, axis])
            for axis in range(3)
        ]
    )
    return positions, mocap_slerp(np.clip(query_times, mocap_times[0], mocap_times[-1]))


# Speed is what separates a dynamic take from a static one, so it is carried
# alongside every sample and later used as a binning axis: a residual latency
# shows up as error growing in proportion to it.
mocap_speed = np.zeros(len(mocap_times))
mocap_speed[1:] = np.linalg.norm(np.diff(mocap_positions, axis=0), axis=1) / np.maximum(
    np.diff(mocap_times), 1e-9
)
# Smoothed over ~0.1 s so a single noisy sample does not set a frame's speed bin.
speed_window = max(3, round(0.1 / np.median(np.diff(mocap_times))))
mocap_speed = (
    pd.Series(mocap_speed)
    .rolling(speed_window, center=True, min_periods=1)
    .median()
    .to_numpy()
)
print(
    f"Mocap: {len(mocap_times)} frames, speed median {mocap_speed.mean():.3f} m/s, "
    f"max {mocap_speed.max():.3f} m/s"
)


# %% Joint tag-board pose estimators
# Shared with 05_static_jitter.py: one joint PnP over every visible corner,
# and for two cameras one pose minimizing reprojection in both at once.
def stack_correspondences(frame_detections, marker_ids):
    """Every requested tag's corners as one board, or None if any is missing.

    Requiring the whole set keeps a condition's geometry fixed: a 4-tag number
    is always four tags, never whichever two happened to be visible.
    """
    if not all(marker_id in frame_detections for marker_id in marker_ids):
        return None
    return (
        np.concatenate([marker_corners_reference[m] for m in marker_ids]),
        np.concatenate([frame_detections[m] for m in marker_ids]),
    )


def raw_reprojection_rmse(object_points, image_points, rvec, tvec, camera_name):
    model = camera_models[camera_name]
    projected, _ = cv2.fisheye.projectPoints(
        object_points.reshape(-1, 1, 3), rvec, tvec, model["K"], model["D"]
    )
    residual = projected.reshape(-1, 2) - image_points
    return float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))


def single_tag_pose(frame_detections, marker_id, camera_name):
    """Board pose from one tag, solved on that tag's own square then composed.

    IPPE_SQUARE wants a planar square centred on the origin, which only the
    reference tag's corners satisfy in board coordinates. So the solve runs in
    the tag's own frame and the result is carried onto the board through the
    rigid-body transform, letting any tag stand in as the single-tag estimator.
    """
    if marker_id not in frame_detections:
        return None
    model = camera_models[camera_name]
    image_points_raw = frame_detections[marker_id]
    image_points = cv2.fisheye.undistortPoints(
        image_points_raw.reshape(-1, 1, 2), model["K"], model["D"], P=model["K"]
    ).reshape(-1, 2)
    solutions = cv2.solvePnPGeneric(
        TAG_CORNERS_LOCAL,
        image_points,
        model["K"],
        None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not solutions[0]:
        return None

    rotation_marker, translation_marker = marker_to_reference[marker_id]
    best = None
    for rvec_tag, tvec_tag in zip(solutions[1], solutions[2]):
        if float(tvec_tag.reshape(3)[2]) <= 0:
            continue
        # p_cam = R_tag p_local + t_tag and p_ref = R_m2r p_local + t_m2r, so
        # the board pose is R_tag R_m2r' with the origin shifted accordingly.
        rotation_board = cv2.Rodrigues(rvec_tag)[0] @ rotation_marker.T
        translation_board = tvec_tag.reshape(3) - rotation_board @ translation_marker
        rvec = cv2.Rodrigues(rotation_board)[0]
        tvec = translation_board.reshape(3, 1)
        error = raw_reprojection_rmse(
            marker_corners_reference[marker_id],
            image_points_raw,
            rvec,
            tvec,
            camera_name,
        )
        if best is None or error < best[0]:
            best = (error, rvec, tvec)
    if best is None:
        return None
    return {
        "rvec": np.asarray(best[1]).reshape(3),
        "tvec": np.asarray(best[2]).reshape(3),
        "rmse_px": best[0],
    }


def seed_pose(frame_detections, marker_ids, camera_name):
    """The best single-tag board pose among ``marker_ids``, to start PnP from."""
    candidates = [
        pose
        for pose in (
            single_tag_pose(frame_detections, m, camera_name) for m in marker_ids
        )
        if pose is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda pose: pose["rmse_px"])


def mono_board_pose(frame_detections, marker_ids, camera_name="cam0"):
    """One joint PnP over every visible corner of the requested tags.

    A joint solve, never an average of per-tag `tvec`s: averaging independent
    poses throws away the constraint that the tags are one rigid body, which is
    most of what the extra tags are worth.
    """
    if len(marker_ids) == 1:
        return single_tag_pose(frame_detections, marker_ids[0], camera_name)

    correspondences = stack_correspondences(frame_detections, marker_ids)
    if correspondences is None:
        return None
    object_points, image_points_raw = correspondences
    model = camera_models[camera_name]
    image_points = cv2.fisheye.undistortPoints(
        image_points_raw.reshape(-1, 1, 2), model["K"], model["D"], P=model["K"]
    ).reshape(-1, 2)

    # A small tag subset is nearly planar and ITERATIVE can otherwise land on
    # its mirrored local solution. Seed it with the best single-tag pose from
    # this same image: geometric initialization, not temporal filtering.
    initial = seed_pose(frame_detections, marker_ids, camera_name)
    if initial is None:
        return None
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        model["K"],
        None,
        initial["rvec"].reshape(3, 1).copy(),
        initial["tvec"].reshape(3, 1).copy(),
        True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success or float(tvec.reshape(3)[2]) <= 0:
        return None
    return {
        "rvec": np.asarray(rvec).reshape(3),
        "tvec": np.asarray(tvec).reshape(3),
        "rmse_px": raw_reprojection_rmse(
            object_points, image_points_raw, rvec, tvec, camera_name
        ),
    }


def stereo_board_pose(frame0, frame1, marker_ids):
    """One pose minimizing corner reprojection in both fisheye cameras at once.

    The cross-baseline constraint is what tightens depth, so the two images are
    fitted together rather than triangulating two independent single-camera
    poses.
    """
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
        rvec1, tvec1 = cv2.composeRT(rvec0, tvec0, RVEC_STEREO, T_STEREO.reshape(3, 1))[
            :2
        ]
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

    result = least_squares(
        residual,
        np.concatenate([initial["rvec"], initial["tvec"]]),
        # Matches the validated stereo PnP in
        # trunkpose/dual_notebooks/verification_dual_camera.py. The cam0-only
        # seed can have a large cam1 residual, for which a robust loss would
        # suppress the very measurements needed to refine depth across the
        # baseline.
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


# %% Conditions and shared plotting helpers
# The same eight estimators 06 reports, named identically so its per-frame
# errors join straight onto the phase computed below.
CONDITIONS = {
    f"C{camera_count}-M{tag_count}": {
        "camera_count": camera_count,
        "tag_count": tag_count,
    }
    for camera_count in (1, 2)
    for tag_count in range(1, MAX_TAGS + 1)
}


def plain_log_ticks(target, minor_subs=(0.2, 0.3, 0.5)):
    """Label a log axis in millimetres rather than powers of ten."""
    target.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,)))
    target.set_minor_locator(ticker.LogLocator(base=10.0, subs=minor_subs))
    formatter = ticker.FuncFormatter(lambda value, _: f"{value:g}")
    target.set_major_formatter(formatter)
    target.set_minor_formatter(formatter)


# %% Per-frame inter-camera phase
# Recomputed here from the sensor timestamps, the same way 01_phase_check.py
# does, rather than read from phase_log.msgpack: the recorder samples that about
# once a second, and this needs a value against every frame.
cam0_sensor_ns = cam0["metadata"]["sensor_ns"]
cam1_sensor_ns = cam1["metadata"]["sensor_ns"]
frame_period_us = float(np.median(np.diff(cam0_sensor_ns)) / 1000.0)


def wrap_to_half_period(delta_us, period_us):
    """Phase is defined modulo the frame period; a whole period is lockstep."""
    return ((delta_us + period_us / 2.0) % period_us) - period_us / 2.0


paired = cam1_pair >= 0
phase_us = np.full(len(cam0_sensor_ns), np.nan)
phase_us[paired] = wrap_to_half_period(
    (cam1_sensor_ns[cam1_pair[paired]] - cam0_sensor_ns[paired]) / 1000.0,
    frame_period_us,
)
print(
    f"Frame period {frame_period_us:.0f} us; per-frame phase "
    f"{np.nanmin(phase_us):+.0f} to {np.nanmax(phase_us):+.0f} us, "
    f"median {np.nanmedian(phase_us):+.0f} us"
)


# %% Join 06's per-frame errors onto the phase
if not MOVEMENT_SAMPLES_CSV.exists():
    raise FileNotFoundError(
        f"Missing {MOVEMENT_SAMPLES_CSV}. Run jitter_model/06_movement_error.py "
        "on this recording first."
    )
samples = pd.read_csv(MOVEMENT_SAMPLES_CSV)
samples["phase_us"] = phase_us[samples["frame"].to_numpy()]
samples["error_mm"] = 1000.0 * samples["error_m"]
# The displacement the two views genuinely disagree by, which is what a timing
# effect would have to act through.
samples["predicted_mm"] = (
    1000.0 * np.abs(samples["phase_us"]) * 1e-6 * samples["speed_ms"]
)
samples = samples.loc[
    np.isfinite(samples["phase_us"]) & (samples["error_mm"] < 1000.0 * GROSS_ERROR_M)
].reset_index(drop=True)

# The direction of travel, for the signed test. Latency moves the estimate along
# the motion, so its signature is a shift on this axis, not a larger magnitude.
mocap_velocity = np.zeros((len(mocap_times), 3))
mocap_velocity[1:] = np.diff(mocap_positions, axis=0) / np.maximum(
    np.diff(mocap_times)[:, None], 1e-9
)
velocity_at = np.column_stack(
    [
        np.interp(samples["time_s"], mocap_times, mocap_velocity[:, axis])
        for axis in range(3)
    ]
)
speed_at = np.linalg.norm(velocity_at, axis=1)
direction = velocity_at / np.maximum(speed_at[:, None], 1e-9)
error_vector = samples[["error_x_m", "error_y_m", "error_z_m"]].to_numpy()
samples["along_track_mm"] = 1000.0 * np.einsum("ij,ij->i", error_vector, direction)
samples["cross_track_mm"] = np.sqrt(
    np.maximum(samples["error_mm"] ** 2 - samples["along_track_mm"] ** 2, 0.0)
)
print(
    f"Samples: {len(samples)} across {samples['condition'].nunique()} conditions; "
    f"predicted displacement median {samples['predicted_mm'].median():.3f} mm, "
    f"p95 {samples['predicted_mm'].quantile(0.95):.3f} mm"
)


# %% Tests 1-4: what the recording alone can say
def partial_correlation(target, predictor, controls):
    """Correlation between target and predictor once controls are removed.

    Both are regressed on the controls and their residuals correlated, so what
    is left is the part of the predictor that speed and distance do not already
    account for. Phase drifts monotonically through a take while the sweep moves
    the board nearer and faster, so without this the three are indistinguishable.
    """
    design = np.column_stack([np.ones(len(target)), controls])
    target_residual = target - design @ np.linalg.lstsq(design, target, rcond=None)[0]
    predictor_residual = (
        predictor - design @ np.linalg.lstsq(design, predictor, rcond=None)[0]
    )
    if target_residual.std() < 1e-12 or predictor_residual.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(target_residual, predictor_residual)[0, 1])


observational_rows = []
for name, group in samples.groupby("condition"):
    if len(group) < 100:
        continue
    spec = CONDITIONS[name]
    error = group["error_mm"].to_numpy()
    predicted = group["predicted_mm"].to_numpy()
    controls = group[["speed_ms", "distance_m"]].to_numpy()
    signed_phase_displacement = (
        1000.0 * group["phase_us"].to_numpy() * 1e-6 * group["speed_ms"].to_numpy()
    )
    observational_rows.append(
        {
            "condition": name,
            "cameras": spec["camera_count"],
            "tags": spec["tag_count"],
            "samples": len(group),
            "r_error_predicted": float(np.corrcoef(error, predicted)[0, 1]),
            "r_error_speed": float(np.corrcoef(error, group["speed_ms"])[0, 1]),
            "partial_r_error_predicted": partial_correlation(
                error, predicted, controls
            ),
            # The signed test: along-track error against signed phase * speed.
            "r_alongtrack_signed": float(
                np.corrcoef(group["along_track_mm"], signed_phase_displacement)[0, 1]
            ),
            "slope_alongtrack_mm_per_mm": float(
                np.polyfit(signed_phase_displacement, group["along_track_mm"], 1)[0]
            ),
            "median_alongtrack_mm": float(group["along_track_mm"].median()),
            "median_crosstrack_mm": float(group["cross_track_mm"].median()),
        }
    )

observational = (
    pd.DataFrame(observational_rows)
    .sort_values(["cameras", "tags"])
    .reset_index(drop=True)
)

# Test 4: the same stereo condition minus its single-camera twin, frame by
# frame. The systematic floor is common to both and cancels, so whatever timing
# contributes to the stereo solve alone is left behind undiluted.
paired_rows = []
for tag_count in sorted({spec["tag_count"] for spec in CONDITIONS.values()}):
    mono = samples.loc[samples["condition"] == f"C1-M{tag_count}"].set_index("frame")
    stereo = samples.loc[samples["condition"] == f"C2-M{tag_count}"].set_index("frame")
    shared = mono.index.intersection(stereo.index)
    if len(shared) < 100:
        continue
    difference = (
        stereo.loc[shared, "along_track_mm"] - mono.loc[shared, "along_track_mm"]
    ).to_numpy()
    signed = (
        1000.0
        * stereo.loc[shared, "phase_us"].to_numpy()
        * 1e-6
        * stereo.loc[shared, "speed_ms"].to_numpy()
    )
    paired_rows.append(
        {
            "tags": tag_count,
            "samples": len(shared),
            "r_difference_signed": float(np.corrcoef(difference, signed)[0, 1]),
            "slope_mm_per_mm": float(np.polyfit(signed, difference, 1)[0]),
            "median_difference_mm": float(np.median(difference)),
        }
    )
paired_difference = pd.DataFrame(paired_rows)

observational.to_csv(OBSERVED_CSV, index=False)
print("\nObservational tests (a real timing effect needs slope ~ +1):")
print(
    observational[
        [
            "condition",
            "samples",
            "r_error_predicted",
            "partial_r_error_predicted",
            "r_alongtrack_signed",
            "slope_alongtrack_mm_per_mm",
        ]
    ]
    .round(3)
    .to_string(index=False)
)
print("\nStereo minus mono, paired by frame (the floor cancels):")
print(paired_difference.round(3).to_string(index=False))


# %% Test 5: inject a known timing offset and watch what it does
# Everything above is observational, and the real phase is so small that a null
# result could equally mean "no effect" or "effect below the noise". This is an
# intervention: cam1 is paired one and two frames away from where it belongs,
# which is a timing error of a known size, and the stereo pose is re-solved. The
# displacement that produces gives the sensitivity in millimetres per
# millisecond, against which the real phase can be placed.
def quad_area(corners):
    x, y = np.asarray(corners)[:, 0], np.asarray(corners)[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def select_frame_tags(frame0, frame1, tag_count):
    """The ``tag_count`` largest tags visible to both cameras in this frame."""
    visible = {m: c for m, c in frame0.items() if m in frame1}
    if len(visible) < tag_count:
        return None
    ranked = sorted(visible, key=lambda m: -quad_area(visible[m]))
    return tuple(sorted(ranked[:tag_count]))


injection_rows = []
reference_positions = {}
for offset in INJECTION_FRAME_OFFSETS:
    solved = 0
    for index in tqdm(
        range(len(cam0["detections"])), desc=f"cam1 shifted {offset:+d} frames"
    ):
        base_match = int(cam1_pair[index])
        if base_match < 0:
            continue
        shifted = base_match + offset
        if not 0 <= shifted < len(cam1["detections"]):
            continue
        frame0 = cam0["detections"][index]
        # Tags are chosen on the correctly paired frame, so every offset solves
        # the same board and only the timing changes.
        marker_ids = select_frame_tags(
            frame0, cam1["detections"][base_match], INJECTION_CONDITION_TAGS
        )
        if marker_ids is None or marker_ids != select_frame_tags(
            frame0, cam1["detections"][shifted], INJECTION_CONDITION_TAGS
        ):
            continue
        pose = stereo_board_pose(frame0, cam1["detections"][shifted], marker_ids)
        if pose is None:
            continue
        # The true offset is cam1's own clock difference, not the nominal frame
        # period: frames are dropped here and there, so it is read off the
        # timestamps rather than assumed.
        offset_ms = (
            int(cam1_sensor_ns[shifted]) - int(cam1_sensor_ns[base_match])
        ) / 1e6
        solved += 1
        if offset == 0:
            reference_positions[index] = pose["tvec"]
        injection_rows.append(
            {
                "frame": index,
                "offset_frames": offset,
                "offset_ms": offset_ms,
                "x": pose["tvec"][0],
                "y": pose["tvec"][1],
                "z": pose["tvec"][2],
                "speed_ms": float(
                    np.interp(camera_time_s[index], mocap_times, mocap_speed)
                ),
            }
        )
    print(f"  cam1 {offset:+d} frames: {solved} poses")

injection = pd.DataFrame(injection_rows)
reference = pd.DataFrame(
    [
        {"frame": k, "ref_x": v[0], "ref_y": v[1], "ref_z": v[2]}
        for k, v in reference_positions.items()
    ]
)
injection = injection.merge(reference, on="frame", how="inner")
injection["shift_mm"] = 1000.0 * np.linalg.norm(
    injection[["x", "y", "z"]].to_numpy()
    - injection[["ref_x", "ref_y", "ref_z"]].to_numpy(),
    axis=1,
)
# What a perfectly rigid "the board simply moved" model predicts.
injection["expected_mm"] = (
    1000.0 * np.abs(injection["offset_ms"]) * 1e-3 * injection["speed_ms"]
)
injection.to_csv(INJECTION_CSV, index=False)

moved = injection.loc[injection["offset_frames"] != 0]
sensitivity = float(np.polyfit(moved["expected_mm"], moved["shift_mm"], 1)[0])
print(
    f"\nInjection: {len(moved)} shifted poses, "
    f"{moved['offset_ms'].abs().min():.1f}-{moved['offset_ms'].abs().max():.1f} ms offsets"
)
print(
    f"  measured shift tracks the expected board travel with slope {sensitivity:.2f} "
    "(1.0 would mean the solve follows the motion exactly)"
)
print(
    injection.groupby("offset_frames")
    .agg(
        offset_ms=("offset_ms", "median"),
        expected_mm=("expected_mm", "median"),
        measured_mm=("shift_mm", "median"),
    )
    .round(3)
    .to_string()
)

# Scale that sensitivity down to the phase the recording actually ran at.
real_phase_ms = float(np.nanmedian(np.abs(phase_us))) / 1000.0
typical_speed = float(samples["speed_ms"].median())
implied_mm = sensitivity * real_phase_ms * typical_speed
observed_floor = float(
    samples.loc[
        samples["condition"] == f"C2-M{INJECTION_CONDITION_TAGS}", "error_mm"
    ].median()
)
print(
    f"\nAt the phase this take actually ran at ({1000 * real_phase_ms:.0f} us) and a "
    f"typical {typical_speed:.2f} m/s, that sensitivity implies {implied_mm:.3f} mm "
    f"of error, against an observed median of {observed_floor:.2f} mm "
    f"({100 * implied_mm / observed_floor:.1f}% of it)."
)


# %% Figures
CONDITION_COLOURS = {
    name: plt.get_cmap("viridis")((spec["tag_count"] - 1) / max(1, MAX_TAGS - 1))
    for name, spec in CONDITIONS.items()
}


def condition_label(name):
    spec = CONDITIONS[name]
    cameras = "cams" if spec["camera_count"] > 1 else "cam"
    tags = "tags" if spec["tag_count"] > 1 else "tag"
    return f"{spec['camera_count']} {cameras}, {spec['tag_count']} {tags}"


STEREO_CONDITIONS = [n for n in CONDITIONS if CONDITIONS[n]["camera_count"] == 2]
MONO_CONDITIONS = [n for n in CONDITIONS if CONDITIONS[n]["camera_count"] == 1]


# %% Figure 1: phase, speed, and their product over the take
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
axes[0].plot(
    camera_time_s[paired], phase_us[paired], color="tab:blue", linewidth=0.7, alpha=0.4
)
axes[0].plot(
    camera_time_s[paired],
    pd.Series(phase_us[paired]).rolling(31, center=True, min_periods=1).median(),
    color="tab:blue",
    linewidth=1.6,
    label="phase (1 s rolling median)",
)
axes[0].axhline(0, color="0.7", linewidth=0.8)
axes[0].set_ylabel("Inter-camera phase [us]")
axes[0].set_title("Phase drifts monotonically through the take")
axes[0].grid(True, alpha=0.25)
axes[0].legend(loc="upper left", fontsize=9)

axes[1].plot(mocap_times, mocap_speed, color="tab:red", linewidth=0.9)
axes[1].set_ylabel("Speed [m/s]")
axes[1].set_title("Speed varies with the sweep, on its own schedule")
axes[1].grid(True, alpha=0.25)

for name in STEREO_CONDITIONS:
    group = samples.loc[samples["condition"] == name]
    axes[2].plot(
        group["time_s"],
        group["predicted_mm"],
        linewidth=0.8,
        color=CONDITION_COLOURS[name],
        label=condition_label(name),
    )
axes[2].set_ylabel("Predicted displacement [mm]")
axes[2].set_xlabel("Time after GPIO sync pulse [s]")
axes[2].set_title("|phase| x speed: the most a timing effect could contribute")
axes[2].grid(True, alpha=0.25)
axes[2].legend(loc="upper left", ncols=4, fontsize=8)
fig.suptitle(f"Phase, speed, and their product - {TAKE_NAME}")
fig.savefig(PHASE_FIGURE, dpi=180, bbox_inches="tight")
print(f"Saved {PHASE_FIGURE}")


# %% Figure 2: error against predicted displacement, with the control beside it
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True, constrained_layout=True)
for axis, names, title in (
    (axes[0], STEREO_CONDITIONS, "Two cameras (timing can act here)"),
    (axes[1], MONO_CONDITIONS, "One camera (control: never reads cam1)"),
):
    for name in names:
        group = samples.loc[samples["condition"] == name]
        axis.scatter(
            group["predicted_mm"],
            group["error_mm"],
            s=4,
            alpha=0.18,
            color=CONDITION_COLOURS[name],
            label=condition_label(name),
        )
        fit = np.polyfit(group["predicted_mm"], group["error_mm"], 1)
        span = np.linspace(0, group["predicted_mm"].max(), 50)
        axis.plot(
            span, np.polyval(fit, span), color=CONDITION_COLOURS[name], linewidth=1.6
        )
    axis.set_xlabel("Predicted displacement |phase| x speed [mm]")
    axis.set_title(title)
    axis.set_yscale("log")
    plain_log_ticks(axis.yaxis)
    axis.grid(True, alpha=0.25, which="both")
    axis.legend(loc="upper right", fontsize=8, markerscale=3)
axes[0].set_ylabel("Position error [mm]")
fig.suptitle(
    f"Error against what timing could explain - {TAKE_NAME}\n"
    "if timing mattered, the left panel would rise and the right would not"
)
fig.savefig(SCATTER_FIGURE, dpi=180, bbox_inches="tight")
print(f"Saved {SCATTER_FIGURE}")


# %% Figure 3: the signed, directional test
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
for name in STEREO_CONDITIONS:
    group = samples.loc[samples["condition"] == name]
    signed = 1000.0 * group["phase_us"] * 1e-6 * group["speed_ms"]
    axes[0].scatter(
        signed,
        group["along_track_mm"],
        s=4,
        alpha=0.18,
        color=CONDITION_COLOURS[name],
        label=condition_label(name),
    )
    fit = np.polyfit(signed, group["along_track_mm"], 1)
    span = np.linspace(signed.min(), signed.max(), 50)
    axes[0].plot(
        span, np.polyval(fit, span), color=CONDITION_COLOURS[name], linewidth=1.6
    )
diagonal = np.linspace(
    float(samples["predicted_mm"].max()) * -1.0,
    float(samples["predicted_mm"].max()),
    50,
)
axes[0].plot(
    diagonal,
    diagonal,
    color="black",
    linestyle="--",
    linewidth=1.4,
    label="slope 1 (pure latency)",
)
axes[0].set_xlabel("Signed phase x speed [mm]")
axes[0].set_ylabel("Along-track error [mm]")
axes[0].set_title("Latency would displace the estimate along the motion")
axes[0].grid(True, alpha=0.25)
axes[0].legend(loc="upper left", fontsize=8, markerscale=3)

stereo_signed = observational.loc[
    observational["cameras"] == 2, "r_alongtrack_signed"
].to_numpy()
width = 0.35
positions = np.arange(len(paired_difference))
axes[1].bar(
    positions - width / 2,
    paired_difference["r_difference_signed"],
    width,
    color="tab:blue",
    label="stereo - mono, paired by frame",
)
axes[1].bar(
    positions + width / 2,
    stereo_signed[: len(paired_difference)],
    width,
    color="tab:orange",
    label="stereo along-track alone",
)
axes[1].axhline(0, color="0.5", linewidth=0.9)
axes[1].set_xticks(positions)
axes[1].set_xticklabels(
    [f"{n} tags" if n > 1 else "1 tag" for n in paired_difference["tags"]]
)
axes[1].set_ylabel("Correlation with signed phase x speed")
axes[1].set_ylim(-1, 1)
axes[1].set_title("A real effect would sit near +1")
axes[1].grid(True, alpha=0.25, axis="y")
axes[1].legend(loc="upper right", fontsize=8)
fig.suptitle(f"Signed directional test - {TAKE_NAME}")
fig.savefig(DIRECTIONAL_FIGURE, dpi=180, bbox_inches="tight")
print(f"Saved {DIRECTIONAL_FIGURE}")


# %% Figure 4: the injection test
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
axes[0].scatter(
    moved["expected_mm"],
    moved["shift_mm"],
    s=5,
    alpha=0.2,
    color="tab:blue",
    label=f"cam1 shifted 1 and 2 frames ({INJECTION_CONDITION_TAGS} tags)",
)
limit = float(np.percentile(moved["expected_mm"], 99))
axes[0].plot(
    [0, limit],
    [0, limit],
    color="black",
    linestyle="--",
    linewidth=1.4,
    label="slope 1",
)
axes[0].plot(
    [0, limit],
    [0, sensitivity * limit],
    color="tab:red",
    linewidth=1.6,
    label=f"measured slope {sensitivity:.2f}",
)
axes[0].set_xlim(0, limit)
axes[0].set_ylim(0, limit)
axes[0].set_xlabel("Board travel over the injected offset [mm]")
axes[0].set_ylabel("Measured pose shift [mm]")
axes[0].set_title("A known timing error does move the solve")
axes[0].grid(True, alpha=0.25)
axes[0].legend(loc="upper left", fontsize=8, markerscale=3)

by_offset = injection.groupby("offset_frames").agg(
    offset_ms=("offset_ms", "median"), measured_mm=("shift_mm", "median")
)
# The zero-offset row is identically zero by construction and has no place on a
# log axis, so the injected points are the shifted ones only.
by_offset = by_offset.loc[by_offset.index != 0].assign(
    offset_ms=lambda frame: frame["offset_ms"].abs()
)
axes[1].plot(
    by_offset["offset_ms"],
    by_offset["measured_mm"],
    marker="o",
    linestyle="none",
    markersize=8,
    color="tab:blue",
    label="injected offsets",
)
# The line the sensitivity implies, carried down to the real phase.
span = np.logspace(
    np.log10(real_phase_ms) - 0.3, np.log10(by_offset["offset_ms"].max()), 50
)
axes[1].plot(
    span,
    sensitivity * span * 1e-3 * typical_speed * 1000.0,
    color="tab:blue",
    linestyle="-",
    linewidth=1.2,
    alpha=0.7,
    label=f"sensitivity {sensitivity:.2f} extrapolated",
)
axes[1].axvline(
    real_phase_ms,
    color="tab:red",
    linestyle=":",
    linewidth=1.5,
    label=f"this take's actual phase ({real_phase_ms:.3f} ms)",
)
axes[1].axhline(
    observed_floor,
    color="0.4",
    linestyle="--",
    linewidth=1.2,
    label=f"observed error floor ({observed_floor:.2f} mm)",
)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
plain_log_ticks(axes[1].yaxis)
axes[1].set_xlabel("Injected cam1 offset magnitude [ms]")
axes[1].set_ylabel("Median pose shift [mm]")
axes[1].set_title("Scaled back down, the real phase sits far below the floor")
axes[1].grid(True, alpha=0.25, which="both")
axes[1].legend(loc="upper left", fontsize=8)
fig.suptitle(
    f"Injection test - {TAKE_NAME}\n"
    "a deliberate offset of known size, to calibrate what the real one is worth"
)
fig.savefig(INJECTION_FIGURE, dpi=180, bbox_inches="tight")
print(f"Saved {INJECTION_FIGURE}")

print(f"\nSaved {OBSERVED_CSV}")
print(f"Saved {INJECTION_CSV}")

plt.show()
