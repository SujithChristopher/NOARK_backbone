# %% [markdown]
# # Dynamic movement accuracy against Motive
#
# The random-movement takes sweep the dome continuously, so what they measure is
# accuracy through real motion: how far the AprilTag pose sits from the mocap
# body, and how that error grows with distance and with speed.
#
# Jitter is deliberately absent. Precision needs a stationary target to separate
# estimator noise from real movement, and `dome_static_burst_sep18_26` exists for
# exactly that, so `05_static_jitter.py` owns it. Trying to recover it here would
# mean carving pseudo-static windows out of continuous motion, which is the
# compromise the burst recording was made to avoid.
#
# Eight estimators are compared, one camera and two, each with 1 to 4 tags. Tags
# are chosen per frame from whatever is visible rather than fixed in advance, so
# a condition is limited by how many tags are up, not by which ones.
#
# Timing comes off the GPIO pulse. Unlike the burst take, these recordings carry
# both a rising and a falling edge, and the high interval matches the Motive take
# length to within 0.2 s, so Motive is gated by the pulse and mocap seconds map
# straight onto time since the rise. That agreement is checked at runtime rather
# than assumed.
#
# The mocap-to-camera transform and the tag mounting offset are fitted on the
# first half and the error is reported on the held-out second half, so the
# alignment cannot flatter itself.
#
# Run from the repository root:
#
# ```powershell
# uv run python jitter_model/06_movement_error.py
# uv run python jitter_model/06_movement_error.py --recording data/dome/sep18_26/dome_random_movement_100hz_sep18_26
# ```

# %% Imports
import argparse
import itertools
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
OUTPUT_SUBDIR = "movement_error"
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

SUMMARY_CSV = OUTPUT_DIR / "movement_error_summary.csv"
SAMPLES_CSV = OUTPUT_DIR / "movement_error_samples.csv"
DISTANCE_CSV = OUTPUT_DIR / "movement_error_vs_distance.csv"
SPEED_CSV = OUTPUT_DIR / "movement_error_vs_speed.csv"
TRAJECTORY_FIGURE = OUTPUT_DIR / "movement_trajectory.png"
ERROR_TIME_FIGURE = OUTPUT_DIR / "movement_error_over_time.png"
DISTANCE_FIGURE = OUTPUT_DIR / "movement_error_vs_distance.png"
SPEED_FIGURE = OUTPUT_DIR / "movement_error_vs_speed.png"
HEATMAP_FIGURE = OUTPUT_DIR / "movement_error_heatmap.png"
HISTOGRAM_FIGURE = OUTPUT_DIR / "movement_error_histograms.png"

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


# %% Per-frame tag subsets and the eight conditions
def quad_area(corners):
    """Shoelace area of a detected tag quadrilateral, in square pixels."""
    x, y = np.asarray(corners)[:, 0], np.asarray(corners)[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def select_frame_tags(frame0, frame1, tag_count):
    """The ``tag_count`` largest tags visible in this frame, or None.

    Chosen per frame rather than per take, so a condition is limited by how many
    tags are up rather than by which ones. There is no static window to hold a
    subset across here, unlike the burst recording, so the subset can change
    between frames; how often it does is reported below, since each change is a
    small discontinuity in the trajectory.
    """
    visible = (
        frame0 if frame1 is None else {m: c for m, c in frame0.items() if m in frame1}
    )
    if len(visible) < tag_count:
        return None
    ranked = sorted(visible, key=lambda m: -quad_area(visible[m]))
    return tuple(sorted(ranked[:tag_count]))


CONDITIONS = {
    f"C{camera_count}-M{tag_count}": {
        "camera_count": camera_count,
        "tag_count": tag_count,
    }
    for camera_count in (1, 2)
    for tag_count in range(1, MAX_TAGS + 1)
}
TAG_COUNTS = sorted({spec["tag_count"] for spec in CONDITIONS.values()})
CAMERA_COUNTS = (1, 2)


def solve_condition(spec):
    """Per-frame board pose for one condition across the whole take."""
    stereo_pair = spec["camera_count"] == 2
    rows = []
    previous_ids = None
    switches = 0
    for index in range(len(cam0["detections"])):
        frame0 = cam0["detections"][index]
        match = int(cam1_pair[index]) if stereo_pair else -1
        if stereo_pair and match < 0:
            continue
        frame1 = cam1["detections"][match] if stereo_pair else None
        marker_ids = select_frame_tags(frame0, frame1, spec["tag_count"])
        if marker_ids is None:
            continue
        if previous_ids is not None and marker_ids != previous_ids:
            switches += 1
        previous_ids = marker_ids
        pose = (
            stereo_board_pose(frame0, frame1, marker_ids)
            if stereo_pair
            else mono_board_pose(frame0, marker_ids, "cam0")
        )
        if pose is None:
            continue
        rows.append(
            {
                "frame": index,
                "time_s": float(camera_time_s[index]),
                "marker_ids": "+".join(map(str, marker_ids)),
                "cam_x": pose["tvec"][0],
                "cam_y": pose["tvec"][1],
                "cam_z": pose["tvec"][2],
                "rvec_x": pose["rvec"][0],
                "rvec_y": pose["rvec"][1],
                "rvec_z": pose["rvec"][2],
                "reprojection_px": pose["rmse_px"],
            }
        )
    return pd.DataFrame(rows), switches


condition_tables = {}
for name, spec in tqdm(CONDITIONS.items(), desc="Solving conditions"):
    table, switches = solve_condition(spec)
    if table.empty:
        raise RuntimeError(f"Condition {name} solved no frames")
    # Only frames inside the Motive take can be compared against it.
    table = table.loc[table["time_s"].between(mocap_times[0], mocap_times[-1])]
    table = table.sort_values("time_s").reset_index(drop=True)
    condition_tables[name] = table
    print(
        f"  {name}: {len(table)} poses, {table['marker_ids'].nunique()} distinct "
        f"subsets, subset changed {switches} times"
    )


# %% Offset-aware mocap alignment, fitted on the first half only
# Two unknowns sit between the two systems: the rigid transform from Motive's
# world into cam0, and the constant offset from the mocap body origin to the tag
# board's reference point. They are fitted together, because neither is
# observable without the other, and only on the first half of the take. The
# numbers that get reported come from the second half, so the alignment cannot
# absorb the very error it is being used to measure.
def fit_rigid_transform(source_points, target_points):
    """Least-squares rotation and translation taking source onto target."""
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    covariance = (source_points - source_center).T @ (target_points - target_center)
    u, _s, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    return rotation, target_center - rotation @ source_center


def align_condition(table):
    """Fit world->cam0 and the board offset, then residuals for every sample."""
    times = table["time_s"].to_numpy()
    camera_xyz = table[["cam_x", "cam_y", "cam_z"]].to_numpy()
    body_positions, body_rotations = interpolate_mocap(times)
    body_matrices = body_rotations.as_matrix()

    fit_count = max(20, int(ALIGNMENT_FIT_FRACTION * len(table)))
    fit_count = min(fit_count, len(table) - 10)
    if fit_count < 20:
        return None
    fit = slice(0, fit_count)

    def predicted(parameters):
        offset = parameters[:3]
        rotation = Rotation.from_rotvec(parameters[3:6]).as_matrix()
        translation = parameters[6:9]
        world = body_positions + np.einsum("nij,j->ni", body_matrices, offset)
        return world @ rotation.T + translation

    # Seed the rigid part from a zero offset, which is close enough for a
    # least-squares polish and avoids starting the rotation from identity.
    seed_rotation, seed_translation = fit_rigid_transform(
        body_positions[fit], camera_xyz[fit]
    )
    initial = np.concatenate(
        [
            np.zeros(3),
            Rotation.from_matrix(seed_rotation).as_rotvec(),
            seed_translation,
        ]
    )

    def residual(parameters):
        return (predicted(parameters)[fit] - camera_xyz[fit]).ravel()

    solution = least_squares(residual, initial, method="lm", max_nfev=400)
    aligned = predicted(solution.x)
    result = table.copy()
    result[["mocap_x", "mocap_y", "mocap_z"]] = aligned
    result["error_m"] = np.linalg.norm(camera_xyz - aligned, axis=1)
    result[["error_x_m", "error_y_m", "error_z_m"]] = camera_xyz - aligned
    result["distance_m"] = np.linalg.norm(aligned, axis=1)
    result["speed_ms"] = np.interp(times, mocap_times, mocap_speed)
    result["heldout"] = np.arange(len(result)) >= fit_count
    result["board_offset_mm"] = 1000.0 * np.linalg.norm(solution.x[:3])
    return result


aligned_tables = {}
for name, table in condition_tables.items():
    aligned = align_condition(table)
    if aligned is None:
        raise RuntimeError(f"Condition {name} had too few samples to align")
    aligned_tables[name] = aligned

# Compare on frames every condition solved, so a harder condition is not
# flattered by having skipped the frames where the tags were marginal.
common_frames = set.intersection(
    *(set(table["frame"]) for table in aligned_tables.values())
)
print(f"Frames common to all {len(CONDITIONS)} conditions: {len(common_frames)}")
for name, table in aligned_tables.items():
    aligned_tables[name] = table.loc[table["frame"].isin(common_frames)].reset_index(
        drop=True
    )


# %% Summary on the held-out half
def geometric_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.exp(np.mean(np.log(values)))) if len(values) else np.nan


summary_rows = []
sample_frames = []
for name, table in aligned_tables.items():
    spec = CONDITIONS[name]
    heldout = table.loc[table["heldout"]]
    error_mm = 1000.0 * heldout["error_m"].to_numpy()
    # A pose flip lands metres away and is a failure, not an accuracy figure;
    # counting those separately keeps them out of the RMSE they would dominate.
    gross = error_mm > 1000.0 * GROSS_ERROR_M
    clean = error_mm[~gross]
    summary_rows.append(
        {
            "condition": name,
            "cameras": spec["camera_count"],
            "tags": spec["tag_count"],
            "samples": len(heldout),
            "distinct_tag_subsets": int(heldout["marker_ids"].nunique()),
            "gross_failures": int(gross.sum()),
            "gross_percent": 100.0 * gross.mean() if len(gross) else np.nan,
            "rmse_mm": float(np.sqrt(np.mean(clean**2))) if len(clean) else np.nan,
            "median_mm": float(np.median(clean)) if len(clean) else np.nan,
            "p95_mm": float(np.percentile(clean, 95)) if len(clean) else np.nan,
            # p99 is the honest answer to "how bad does it get" while still
            # averaging over a dozen frames; max is one frame and moves with how
            # long the take ran, so it is reported but never used to rank.
            "p99_mm": float(np.percentile(clean, 99)) if len(clean) else np.nan,
            "max_mm": float(clean.max()) if len(clean) else np.nan,
            "bias_x_mm": 1000.0 * float(heldout["error_x_m"].median()),
            "bias_y_mm": 1000.0 * float(heldout["error_y_m"].median()),
            "bias_z_mm": 1000.0 * float(heldout["error_z_m"].median()),
            "board_offset_mm": float(table["board_offset_mm"].iloc[0]),
            "reprojection_px": float(heldout["reprojection_px"].median()),
        }
    )
    sample = heldout.copy()
    sample.insert(0, "condition", name)
    sample_frames.append(sample)

summary = (
    pd.DataFrame(summary_rows).sort_values(["cameras", "tags"]).reset_index(drop=True)
)
summary.to_csv(SUMMARY_CSV, index=False)
samples = pd.concat(sample_frames, ignore_index=True)
samples.to_csv(SAMPLES_CSV, index=False)

print("\nHeld-out movement accuracy [mm]:")
print(
    summary[
        [
            "condition",
            "samples",
            "rmse_mm",
            "median_mm",
            "p95_mm",
            "p99_mm",
            "max_mm",
            "gross_failures",
            "board_offset_mm",
            "reprojection_px",
        ]
    ]
    .round(3)
    .to_string(index=False)
)


# %% Error against distance and against speed
def bin_metric(table, column, edges):
    rows = []
    for low, high in itertools.pairwise(edges):
        in_bin = table.loc[(table[column] >= low) & (table[column] < high)]
        error_mm = 1000.0 * in_bin["error_m"].to_numpy()
        error_mm = error_mm[error_mm <= 1000.0 * GROSS_ERROR_M]
        if len(error_mm) < MIN_SAMPLES_PER_BIN:
            continue
        rows.append(
            {
                "low": low,
                "high": high,
                "center": 0.5 * (low + high),
                "samples": len(error_mm),
                "rmse_mm": float(np.sqrt(np.mean(error_mm**2))),
                "median_mm": float(np.median(error_mm)),
                "p95_mm": float(np.percentile(error_mm, 95)),
            }
        )
    return rows


distance_rows, speed_rows = [], []
for name, table in aligned_tables.items():
    spec = CONDITIONS[name]
    heldout = table.loc[table["heldout"]]
    for row in bin_metric(heldout, "distance_m", DISTANCE_EDGES_M):
        distance_rows.append({"condition": name, **spec, **row})
    for row in bin_metric(heldout, "speed_ms", SPEED_EDGES_MS):
        speed_rows.append({"condition": name, **spec, **row})

distance_summary = pd.DataFrame(distance_rows)
speed_summary = pd.DataFrame(speed_rows)
distance_summary.to_csv(DISTANCE_CSV, index=False)
speed_summary.to_csv(SPEED_CSV, index=False)
print(
    f"\nDistance bins: {distance_summary['center'].nunique()}, "
    f"speed bins: {speed_summary['center'].nunique()}"
)


# %% Plotting helpers
CONDITION_COLOURS = {
    name: plt.get_cmap("viridis")((spec["tag_count"] - 1) / max(1, len(TAG_COUNTS) - 1))
    for name, spec in CONDITIONS.items()
}
CONDITION_STYLE = {1: "--", 2: "-"}
CONDITION_MARKER = {1: "o", 2: "s"}


def condition_label(name):
    spec = CONDITIONS[name]
    return (
        f"{spec['camera_count']} cam{'s' if spec['camera_count'] > 1 else ''}, "
        f"{spec['tag_count']} tag{'s' if spec['tag_count'] > 1 else ''}"
    )


def ordered_conditions():
    return sorted(
        CONDITIONS,
        key=lambda n: (CONDITIONS[n]["camera_count"], CONDITIONS[n]["tag_count"]),
    )


def plain_log_ticks(target, minor_subs=(0.2, 0.3, 0.5)):
    """Label a log axis in millimetres rather than powers of ten.

    ``minor_subs`` thins the within-decade labels. The default suits an axis
    spanning one or two decades; pass something sparser when the range is wider,
    or the labels run into each other at the bottom end.
    """
    target.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,)))
    target.set_minor_locator(ticker.LogLocator(base=10.0, subs=minor_subs))
    formatter = ticker.FuncFormatter(lambda value, _: f"{value:g}")
    target.set_major_formatter(formatter)
    target.set_minor_formatter(formatter)


def draw_binned(axis, frame, column="rmse_mm"):
    for name in ordered_conditions():
        subset = frame.loc[frame["condition"] == name]
        if subset.empty:
            continue
        spec = CONDITIONS[name]
        axis.plot(
            subset["center"],
            subset[column],
            linestyle=CONDITION_STYLE[spec["camera_count"]],
            marker=CONDITION_MARKER[spec["camera_count"]],
            markersize=4,
            linewidth=1.4,
            color=CONDITION_COLOURS[name],
            label=condition_label(name),
        )
    axis.set_yscale("log")
    plain_log_ticks(axis.yaxis)
    axis.grid(True, alpha=0.25, which="both")


BEST_CONDITION = min(
    summary["condition"], key=lambda n: summary.set_index("condition").loc[n, "rmse_mm"]
)


# %% Figure 1: trajectory against mocap
best = aligned_tables[BEST_CONDITION]
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
for axis, label in zip(axes, "xyz"):
    axis.plot(
        best["time_s"],
        best[f"cam_{label}"],
        color="tab:blue",
        linewidth=1.1,
        label=f"AprilTag board ({condition_label(BEST_CONDITION)})",
    )
    axis.plot(
        best["time_s"],
        best[f"mocap_{label}"],
        color="black",
        linewidth=1.0,
        alpha=0.7,
        label="Mocap body (aligned)",
    )
    axis.axvline(
        best.loc[~best["heldout"], "time_s"].max(),
        color="0.55",
        linestyle=":",
        linewidth=1,
        label="alignment fit / held-out split" if label == "x" else None,
    )
    axis.set_ylabel(f"{label} [m]")
    axis.grid(True, alpha=0.25)
axes[0].legend(loc="upper right", ncols=3, fontsize=9)
axes[-1].set_xlabel("Time after GPIO sync pulse [s]")
fig.suptitle(
    f"Board trajectory against Motive — {TAKE_NAME}\n"
    "alignment fitted on the first half only; everything right of the dashed line is held out"
)
fig.savefig(TRAJECTORY_FIGURE, dpi=180, bbox_inches="tight")
print(f"Saved {TRAJECTORY_FIGURE}")


# %% Figure 2: error over time, with the speed that drives it
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
for name in ordered_conditions():
    table = aligned_tables[name]
    spec = CONDITIONS[name]
    axes[0].plot(
        table["time_s"],
        1000.0 * table["error_m"],
        linestyle=CONDITION_STYLE[spec["camera_count"]],
        linewidth=0.8,
        alpha=0.8,
        color=CONDITION_COLOURS[name],
        label=condition_label(name),
    )
axes[0].set_yscale("log")
plain_log_ticks(axes[0].yaxis)
axes[0].axvline(
    best.loc[~best["heldout"], "time_s"].max(), color="0.55", linestyle=":", linewidth=1
)
axes[0].set_ylabel("Position error [mm]")
axes[0].set_title("Error against aligned mocap")
axes[0].grid(True, alpha=0.25, which="both")
axes[0].legend(loc="upper left", ncols=4, fontsize=8)

axes[1].plot(mocap_times, mocap_speed, color="tab:red", linewidth=0.9)
axes[1].set_ylabel("Mocap speed [m/s]")
axes[1].set_xlabel("Time after GPIO sync pulse [s]")
axes[1].set_title("Body speed, for reading the error trace against")
axes[1].grid(True, alpha=0.25)
fig.suptitle(f"Movement error over time — {TAKE_NAME}")
fig.savefig(ERROR_TIME_FIGURE, dpi=180, bbox_inches="tight")
print(f"Saved {ERROR_TIME_FIGURE}")


# %% Figures 3 and 4: error against distance and against speed
for figure_path, frame, xlabel, title in (
    (
        DISTANCE_FIGURE,
        distance_summary,
        "Camera-to-board distance [m]",
        "Movement error versus distance",
    ),
    (
        SPEED_FIGURE,
        speed_summary,
        "Mocap body speed [m/s]",
        "Movement error versus speed (a residual latency shows as a rising line)",
    ),
):
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5.5), sharey=True, constrained_layout=True
    )
    for axis, column, name in zip(
        axes, ("rmse_mm", "p95_mm"), ("RMSE", "95th percentile")
    ):
        draw_binned(axis, frame, column)
        axis.set_xlabel(xlabel)
        axis.set_title(name)
    axes[0].set_ylabel("Position error [mm]")
    axes[1].legend(loc="upper left", ncols=2, fontsize=8)
    fig.suptitle(f"{title} — {TAKE_NAME}, held-out half")
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    print(f"Saved {figure_path}")


# %% Figure 5: error distributions, one panel per condition
# Shared log-spaced bins across every panel, so the panels are directly
# comparable and a wider distribution looks wider rather than merely differently
# binned. The axis has to be logarithmic: pooled errors run from 0.08 mm to
# 145 mm, and on a linear axis every multi-tag condition would collapse into a
# single bar against the single-tag tail.
pooled_error_mm = 1000.0 * samples["error_m"].to_numpy()
pooled_error_mm = pooled_error_mm[pooled_error_mm > 0]
histogram_bins = np.logspace(
    np.log10(pooled_error_mm.min()), np.log10(pooled_error_mm.max()), HISTOGRAM_BINS
)

fig, axes = plt.subplots(
    len(CAMERA_COUNTS),
    len(TAG_COUNTS),
    figsize=(3.4 * len(TAG_COUNTS), 3.1 * len(CAMERA_COUNTS)),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
axes = np.atleast_2d(axes)
for row, camera_count in enumerate(CAMERA_COUNTS):
    for column, tag_count in enumerate(TAG_COUNTS):
        axis = axes[row, column]
        name = f"C{camera_count}-M{tag_count}"
        error_mm = (
            1000.0
            * aligned_tables[name]
            .loc[aligned_tables[name]["heldout"], "error_m"]
            .to_numpy()
        )
        error_mm = error_mm[error_mm > 0]
        axis.hist(
            error_mm, bins=histogram_bins, color=CONDITION_COLOURS[name], alpha=0.85
        )
        median = float(np.median(error_mm))
        p95 = float(np.percentile(error_mm, 95))
        axis.axvline(median, color="black", linestyle="--", linewidth=1.1)
        axis.axvline(p95, color="tab:red", linestyle="--", linewidth=1.1)
        axis.set_title(
            f"{condition_label(name)}\nmedian {median:.1f}, p95 {p95:.1f} mm",
            fontsize=10,
        )
        axis.set_xscale("log")
        plain_log_ticks(axis.xaxis, minor_subs=(0.3,))
        axis.grid(True, alpha=0.2, which="both")
for axis in axes[-1]:
    axis.set_xlabel("Position error [mm]")
for axis in axes[:, 0]:
    axis.set_ylabel("Frames")
fig.suptitle(
    f"Held-out error distributions — {TAKE_NAME}\n"
    "shared log-spaced bins; dashed black is the median, dashed red the 95th percentile"
)
fig.savefig(HISTOGRAM_FIGURE, dpi=180, bbox_inches="tight")
print(f"Saved {HISTOGRAM_FIGURE}")


# %% Figure 6: the condition grid
grid = np.full((len(TAG_COUNTS), len(CAMERA_COUNTS)), np.nan)
lookup = summary.set_index(["cameras", "tags"])["rmse_mm"]
for row, tag_count in enumerate(TAG_COUNTS):
    for column, camera_count in enumerate(CAMERA_COUNTS):
        if (camera_count, tag_count) in lookup.index:
            grid[row, column] = lookup.loc[(camera_count, tag_count)]

fig, ax = plt.subplots(figsize=(6.2, 6.4), constrained_layout=True)
image = ax.imshow(grid, cmap=ERROR_COLORMAP, aspect="auto")
ax.set_xticks(range(len(CAMERA_COUNTS)))
ax.set_xticklabels([f"{c} camera{'s' if c > 1 else ''}" for c in CAMERA_COUNTS])
ax.set_yticks(range(len(TAG_COUNTS)))
ax.set_yticklabels([f"{n} tag{'s' if n > 1 else ''}" for n in TAG_COUNTS])
for row in range(len(TAG_COUNTS)):
    for column in range(len(CAMERA_COUNTS)):
        value = grid[row, column]
        if not np.isfinite(value):
            continue
        shade = (value - np.nanmin(grid)) / max(np.ptp(grid[np.isfinite(grid)]), 1e-9)
        red, green, blue, _ = plt.get_cmap(ERROR_COLORMAP)(shade)
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        ax.text(
            column,
            row,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=15,
            color="white" if luminance < 0.55 else "black",
        )
fig.colorbar(image, ax=ax, label="Held-out RMSE [mm]")
ax.set_title(
    f"Movement accuracy by condition\n{len(common_frames)} frames common to every "
    "condition, held-out half"
)
fig.savefig(HEATMAP_FIGURE, dpi=180, bbox_inches="tight")
print(f"Saved {HEATMAP_FIGURE}")

print(f"\nSaved {SUMMARY_CSV}")
print(f"Saved {SAMPLES_CSV}")
print(f"Saved {DISTANCE_CSV}")
print(f"Saved {SPEED_CSV}")

plt.show()
