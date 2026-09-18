# %% [markdown]
# # Static jitter versus camera count, tag count, and distance
#
# `dome_static_burst_sep18_26` holds 50 bursts of 50 frames. The dome is held
# still for each burst and moved between them, so every burst is a repeat
# measurement of one pose and the spread inside it is the estimator's jitter,
# with nothing to unmix it from real motion.
#
# That is what separates this script from `04_jitter_model.py`. There the take
# was a continuous movement, so "static" had to be carved out of it by finding
# near-stationary frame pairs, and jitter came from the successive-difference
# estimator `std(diff(residual)) / sqrt(2)` against interpolated mocap. Here
# jitter is simply the standard deviation of the estimated position inside a
# burst. The two numbers are not directly comparable.
#
# Eight conditions are compared, each one camera count crossed with one tag
# count:
#
# - one camera (cam0) with 1, 2, 3, or 4 rigidly connected AprilTags;
# - two cameras with the same 1, 2, 3, or 4 tags.
#
# Tag geometry comes from `02_rigidbody_calib.py`, run on the separate
# `dome_rb_def` take. Multi-tag estimates use one joint board PnP over every
# visible corner, never an average of independent `tvec`s, and the two-camera
# estimates minimize corner reprojection in both fisheye cameras at once. No
# temporal filtering anywhere: filtering would trade the very jitter being
# measured for lag.
#
# Poses are reported in the **reference-burst basis**, not in camera
# coordinates. Burst 1's median pose defines the origin and axes, and every
# later frame is expressed relative to it, so the numbers describe the dome's
# own frame rather than an arbitrary camera frame.
#
# Run from the repository root:
#
# ```powershell
# uv run python jitter_model/05_static_jitter.py
# ```

# %% Imports
import collections
import itertools
import json
import pickle
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
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata
from scipy.optimize import least_squares
from tqdm.auto import tqdm

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
MOCAP_CSV = RECORDING_DIR / f"{RECORDING_DIR.name}.csv"
BURSTS_JSON = RECORDING_DIR / "bursts.json"

STEREO_TOML = (
    PROJECT_ROOT
    / "data"
    / "calibration"
    / "dual_160"
    / "calib_radxa_sep18_26"
    / "stereo_calibration.toml"
)
# The tag geometry is a property of the dome, not of this take, so it is read
# from the dedicated rigid-body definition recording rather than from beside
# these frames.
RIGIDBODY_TOML = (
    PROJECT_ROOT
    / "data"
    / "dome"
    / "sep18_26"
    / "dome_rb_def"
    / "rigidbody_calibration.toml"
)

OUTPUT_DIR = RECORDING_DIR / "static_jitter"
DETECTION_CACHE = RECORDING_DIR / "jitter_detections.pkl"

SUMMARY_CSV = OUTPUT_DIR / "static_jitter_summary.csv"
BURST_CSV = OUTPUT_DIR / "static_jitter_bursts.csv"
DISTANCE_CSV = OUTPUT_DIR / "static_jitter_vs_distance.csv"
HEATMAP_FIGURE = OUTPUT_DIR / "static_jitter_heatmap.png"
DISTANCE_FIGURE = OUTPUT_DIR / "static_jitter_vs_distance.png"
AXES_REFERENCE_FIGURE = OUTPUT_DIR / "static_jitter_axes_reference.png"
AXES_CAMERA_FIGURE = OUTPUT_DIR / "static_jitter_axes_camera.png"
BURST_FIGURE = OUTPUT_DIR / "static_jitter_bursts.png"
SPATIAL_FIGURE = OUTPUT_DIR / "static_jitter_spatial.png"

CAMERA_NAMES = ("cam0", "cam1")
REBUILD_DETECTION_CACHE = False
MAX_FRAMES = None

# Nested tag sets, so a 2-tag number is the 1-tag set plus one more and the
# counts stay comparable. Left as None the sets are chosen from this take's own
# co-visibility ranking, printed below; set a tuple of four ids to pin them.
MAX_TAGS = 4
# A tag must appear in at least this fraction of a burst's frames to be
# eligible for that burst's subset.
TAG_PRESENCE_FRACTION = 0.9

# Motive body frame from the labelled markers: x = m4 - m3, z = m4 - m1,
# origin = m4. The labelled markers are direct measurements, unlike Motive's
# solved rigid body, which can re-lock at a ghost pose after an occlusion.
MOCAP_BODY_X_FROM = ("m4", "m3")
MOCAP_BODY_Z_FROM = ("m4", "m1")
MOCAP_BODY_ORIGIN = "m4"

# A burst is only usable if the dome really was still and enough frames solved.
MIN_FRAMES_PER_BURST = 10
MAX_BURST_MOCAP_TRAVEL_M = 0.002
# Corner sub-pixel refinement, matching 02_rigidbody_calib.py.
SUBPIX_WINDOW = (5, 5)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
# Pair the two free-running cameras no further apart than this much of a frame.
MAX_PAIR_FRACTION_OF_FRAME = 0.55
# Distance bin edges; the observed range is printed so these can be retuned.
DISTANCE_EDGES_M = np.asarray([0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.95])
MIN_BURSTS_PER_BIN = 3
# Spatial map grid resolution, in samples across the working area.
SPATIAL_GRID = 220
# Error colormap: blue for low, yellow through the middle, red for high.
ERROR_COLORMAP = "RdYlBu_r"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Recording : {RECORDING_DIR}")
print(f"Outputs   : {OUTPUT_DIR}")


# %% Calibration, tag model, and detector
for required_path in (STEREO_TOML, RIGIDBODY_TOML, MOCAP_CSV, BURSTS_JSON):
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

# Prefer the stereo extrinsic that notebook 02 self-calibrated from the rigid
# board; the calibration TOML supplies the intrinsics and a fallback extrinsic.
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

# Each tag's four corners in the reference tag's frame, so any subset of tags
# is one rigid point cloud that a single PnP can solve against.
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
    """Load the per-frame timestamp columns, including this take's burst index.

    The burst recorder appends a sixth column holding the 1-based burst number,
    which is what groups frames into repeats of one pose. Older recordings stop
    at five columns, so it is read only when present.
    """
    with path.open("rb") as stream:
        records = list(msgpack.Unpacker(stream, object_hook=mpn.decode))
    metadata = {
        "sync": np.asarray([int(record[0]) for record in records], dtype=np.uint8),
        "monotonic_ns": np.asarray(
            [int(record[2]) for record in records], dtype=np.int64
        ),
        "sensor_ns": np.asarray([int(record[3]) for record in records], dtype=np.int64),
    }
    if records and len(records[0]) > 5:
        metadata["burst"] = np.asarray(
            [int(record[5]) for record in records], dtype=np.int32
        )
    else:
        metadata["burst"] = np.zeros(len(records), dtype=np.int32)
    return metadata


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
    """Index of cam1's frame closest in time, or None if nothing is close enough.

    The sensors free-run against each other, so frames are matched on the clock
    rather than by index; a gap over half a frame period means the pair would
    straddle different exposures and is rejected.
    """
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


# %% Joint tag-board pose estimators
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


# %% Burst grouping
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


# %% Per-burst tag subsets
def quad_area(corners):
    """Shoelace area of a detected tag quadrilateral, in square pixels.

    A proxy for how much signal the tag carries: a tag that is near, large and
    square-on is both easier to localize and better conditioned for pose than
    one that is far away or steeply foreshortened.
    """
    x, y = np.asarray(corners)[:, 0], np.asarray(corners)[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def select_burst_tags(frames0, frames1, tag_count):
    """The ``tag_count`` best tags for one burst, or None if there are not enough.

    Any combination is allowed rather than one fixed set for the whole take, so
    a burst is only lost when fewer than ``tag_count`` tags are usable in it at
    all, instead of whenever one nominated tag happens to be hidden.

    The subset is chosen once per burst and held for every frame in it. Choosing
    per frame would let the estimator change inside a burst, and that switching
    would land in the standard deviation as if it were jitter.

    Candidates must appear in nearly every frame of the burst, so the chosen set
    keeps most of the burst's frames usable; among those, the largest tags win.
    """
    frame_total = len(frames0)
    if frame_total == 0:
        return None
    presence = collections.Counter()
    areas = collections.defaultdict(list)
    for index, frame0 in enumerate(frames0):
        frame1 = None if frames1 is None else frames1[index]
        for marker_id, corners in frame0.items():
            if frames1 is not None and (frame1 is None or marker_id not in frame1):
                continue
            presence[marker_id] += 1
            areas[marker_id].append(quad_area(corners))
    candidates = [
        marker_id
        for marker_id, count in presence.items()
        if count >= TAG_PRESENCE_FRACTION * frame_total
    ]
    if len(candidates) < tag_count:
        return None
    candidates.sort(key=lambda m: -float(np.median(areas[m])))
    # Sorted by id only so the stored label is stable; the solve does not care.
    return tuple(sorted(candidates[:tag_count]))


CONDITIONS = {
    f"C{camera_count}-M{tag_count}": {
        "camera_count": camera_count,
        "tag_count": tag_count,
    }
    for camera_count in (1, 2)
    for tag_count in range(1, MAX_TAGS + 1)
}
TAG_COUNTS = sorted({spec["tag_count"] for spec in CONDITIONS.values()})


# %% Solve every condition, choosing each burst's tags as it goes
def solve_condition(spec):
    """Per-frame board pose for one condition, burst by burst.

    The tag subset is picked once per burst from whatever that burst actually
    shows, so a condition is limited by how many tags are visible rather than
    by which ones.
    """
    stereo = spec["camera_count"] == 2
    rows = []
    for burst in burst_indices:
        burst = int(burst)
        indices = burst_frames[burst]
        if stereo:
            indices = np.asarray([i for i in indices if cam1_pair[i] >= 0])
        if len(indices) < MIN_FRAMES_PER_BURST:
            continue
        frames0 = [cam0["detections"][i] for i in indices]
        frames1 = (
            [cam1["detections"][int(cam1_pair[i])] for i in indices] if stereo else None
        )
        marker_ids = select_burst_tags(frames0, frames1, spec["tag_count"])
        if marker_ids is None:
            continue
        label = "+".join(map(str, marker_ids))
        for position, index in enumerate(indices):
            frame0 = frames0[position]
            if stereo:
                pose = stereo_board_pose(frame0, frames1[position], marker_ids)
            else:
                pose = mono_board_pose(frame0, marker_ids, "cam0")
            if pose is None:
                continue
            rows.append(
                {
                    "frame": int(index),
                    "burst": burst,
                    "marker_ids": label,
                    "monotonic_ns": int(cam0_monotonic_ns[index]),
                    "cam_x": pose["tvec"][0],
                    "cam_y": pose["tvec"][1],
                    "cam_z": pose["tvec"][2],
                    "rvec_x": pose["rvec"][0],
                    "rvec_y": pose["rvec"][1],
                    "rvec_z": pose["rvec"][2],
                    "reprojection_px": pose["rmse_px"],
                }
            )
    return pd.DataFrame(rows)


condition_tables = {}
for name, spec in tqdm(CONDITIONS.items(), desc="Solving conditions"):
    table = solve_condition(spec)
    if table.empty:
        raise RuntimeError(f"Condition {name} solved no frames")
    condition_tables[name] = table
    print(
        f"  {name}: {len(table)} poses over {table['burst'].nunique()} bursts, "
        f"{table['marker_ids'].nunique()} distinct tag subsets"
    )


# %% Reference-burst basis
# One shared basis for all eight conditions, taken from the richest estimator
# (most cameras, most tags) over burst 1. Sharing it keeps the per-axis numbers
# comparable between conditions; letting each condition define its own would
# leave them in eight slightly different frames. The 3D jitter magnitude is a
# rotation invariant and does not depend on this choice at all.
REFERENCE_CONDITION = max(
    CONDITIONS,
    key=lambda name: (CONDITIONS[name]["camera_count"], CONDITIONS[name]["tag_count"]),
)
REFERENCE_BURST = int(burst_indices[0])

reference_rows = condition_tables[REFERENCE_CONDITION]
reference_rows = reference_rows.loc[reference_rows["burst"] == REFERENCE_BURST]
if len(reference_rows) < MIN_FRAMES_PER_BURST:
    raise RuntimeError(
        f"Reference burst {REFERENCE_BURST} has only {len(reference_rows)} poses "
        f"from {REFERENCE_CONDITION}"
    )
REFERENCE_TRANSLATION = (
    reference_rows[["cam_x", "cam_y", "cam_z"]].to_numpy().mean(axis=0)
)
REFERENCE_ROTATION = cv2.Rodrigues(
    reference_rows[["rvec_x", "rvec_y", "rvec_z"]].to_numpy().mean(axis=0)
)[0]
print(
    f"Reference basis: burst {REFERENCE_BURST} of {REFERENCE_CONDITION}, "
    f"{len(reference_rows)} poses, origin at "
    f"{np.round(REFERENCE_TRANSLATION, 4).tolist()} m in cam0"
)


def to_reference_basis(camera_positions):
    """Camera-frame positions expressed in the reference burst's own frame."""
    return (np.asarray(camera_positions) - REFERENCE_TRANSLATION) @ REFERENCE_ROTATION


for table in condition_tables.values():
    reference_xyz = to_reference_basis(table[["cam_x", "cam_y", "cam_z"]].to_numpy())
    table["ref_x"], table["ref_y"], table["ref_z"] = reference_xyz.T
    table["distance_m"] = np.linalg.norm(
        table[["cam_x", "cam_y", "cam_z"]].to_numpy(), axis=1
    )


# %% Mocap: fit the single camera-to-Motive time offset, then use it as a check
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


# %% Per-burst jitter for every condition
burst_rows = []
for name, table in condition_tables.items():
    spec = CONDITIONS[name]
    for burst, group in table.groupby("burst"):
        burst = int(burst)
        if burst not in static_bursts or len(group) < MIN_FRAMES_PER_BURST:
            continue
        reference_xyz = group[["ref_x", "ref_y", "ref_z"]].to_numpy()
        camera_xyz = group[["cam_x", "cam_y", "cam_z"]].to_numpy()
        # The dome is still for the whole burst, so the spread of the estimate
        # inside it is the estimator's jitter with nothing else mixed in. No
        # successive differencing and no mocap residual are needed, unlike in
        # 04_jitter_model.py where stillness had to be carved out of movement.
        reference_std_mm = 1000.0 * np.std(reference_xyz, axis=0, ddof=1)
        camera_std_mm = 1000.0 * np.std(camera_xyz, axis=0, ddof=1)
        reference_mean = reference_xyz.mean(axis=0)
        burst_rows.append(
            {
                "condition": name,
                "cameras": spec["camera_count"],
                "tags": spec["tag_count"],
                "marker_ids": str(group["marker_ids"].iloc[0]),
                "burst": burst,
                "frames": len(group),
                "distance_m": float(np.median(group["distance_m"])),
                "pos_x_m": reference_mean[0],
                "pos_y_m": reference_mean[1],
                "pos_z_m": reference_mean[2],
                "reprojection_px": float(np.median(group["reprojection_px"])),
                "jitter_ref_x_mm": reference_std_mm[0],
                "jitter_ref_y_mm": reference_std_mm[1],
                "jitter_ref_z_mm": reference_std_mm[2],
                "jitter_cam_x_mm": camera_std_mm[0],
                "jitter_cam_y_mm": camera_std_mm[1],
                "jitter_cam_z_mm": camera_std_mm[2],
                # A rotation invariant, so it reads the same in either basis.
                "jitter_3d_mm": float(np.linalg.norm(reference_std_mm)),
                "mocap_jitter_3d_mm": mocap_burst.get(burst, {}).get(
                    "mocap_jitter_3d_mm", np.nan
                ),
            }
        )

burst_table = pd.DataFrame(burst_rows)
if burst_table.empty:
    raise RuntimeError("No burst survived the frame-count and stillness filters")

# Compare the conditions only on bursts every one of them solved, so a harder
# condition is not flattered by having quietly skipped the difficult positions.
common_bursts = set.intersection(
    *(set(group["burst"]) for _, group in burst_table.groupby("condition"))
)
print(f"Bursts common to all {len(CONDITIONS)} conditions: {len(common_bursts)}")
burst_table["common"] = burst_table["burst"].isin(common_bursts)
burst_table = burst_table.sort_values(["cameras", "tags", "burst"]).reset_index(
    drop=True
)
burst_table.to_csv(BURST_CSV, index=False)

common_table = burst_table.loc[burst_table["common"]]
print(
    f"Distance over common bursts: {common_table['distance_m'].min():.3f}"
    f"-{common_table['distance_m'].max():.3f} m"
)


# %% Condition summary
# Per-burst jitter spans two orders of magnitude across this take's positions
# and is strongly right-skewed, so each condition is aggregated by geometric
# mean and compared against the baseline burst by burst. A median of each
# condition's own marginal distribution is an order statistic landing on a
# different burst per condition, which can invert the paired relationship: on
# this take it made the 4-tag single-camera condition look worse than 1 tag
# when it is in fact better on the very same bursts.
BASELINE_CONDITION = "C1-M1"


def geometric_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.exp(np.mean(np.log(values)))) if len(values) else np.nan


paired = common_table.pivot_table(
    index="burst", columns="condition", values="jitter_3d_mm"
)

AXIS_COLUMNS = [
    ("jitter_ref_x_mm", "jitter_ref_y_mm", "jitter_ref_z_mm"),
    ("jitter_cam_x_mm", "jitter_cam_y_mm", "jitter_cam_z_mm"),
]
summary_rows = []
for name, group in common_table.groupby("condition"):
    spec = CONDITIONS[name]
    row = {
        "condition": name,
        "cameras": spec["camera_count"],
        "tags": spec["tag_count"],
        "distinct_tag_subsets": int(group["marker_ids"].nunique()),
        "bursts": len(group),
        "frames": int(group["frames"].sum()),
        "coverage_percent": 100.0
        * len(condition_tables[name])
        / len(cam0["detections"]),
        "jitter_3d_geomean_mm": geometric_mean(group["jitter_3d_mm"]),
        "jitter_3d_median_mm": float(group["jitter_3d_mm"].median()),
        "jitter_3d_p95_mm": float(group["jitter_3d_mm"].quantile(0.95)),
        "paired_ratio_vs_baseline": geometric_mean(
            paired[name] / paired[BASELINE_CONDITION]
        ),
        "bursts_better_than_baseline": int(
            (paired[name] < paired[BASELINE_CONDITION]).sum()
        ),
        "reprojection_px": float(group["reprojection_px"].median()),
    }
    for columns in AXIS_COLUMNS:
        for column in columns:
            row[column.replace("_mm", "_median_mm")] = float(group[column].median())
    summary_rows.append(row)

summary = (
    pd.DataFrame(summary_rows).sort_values(["cameras", "tags"]).reset_index(drop=True)
)
summary.to_csv(SUMMARY_CSV, index=False)
print("\nStatic jitter over bursts common to every condition [mm]:")
print(
    summary[
        [
            "condition",
            "distinct_tag_subsets",
            "bursts",
            "jitter_3d_geomean_mm",
            "paired_ratio_vs_baseline",
            "bursts_better_than_baseline",
            "jitter_3d_median_mm",
            "jitter_cam_z_median_mm",
            "reprojection_px",
            "coverage_percent",
        ]
    ]
    .round(3)
    .to_string(index=False)
)


# %% Jitter versus distance
distance_rows = []
for name, group in common_table.groupby("condition"):
    spec = CONDITIONS[name]
    for low, high in itertools.pairwise(DISTANCE_EDGES_M):
        in_bin = group.loc[(group["distance_m"] >= low) & (group["distance_m"] < high)]
        if len(in_bin) < MIN_BURSTS_PER_BIN:
            continue
        row = {
            "condition": name,
            "cameras": spec["camera_count"],
            "tags": spec["tag_count"],
            "distance_low_m": low,
            "distance_high_m": high,
            "distance_center_m": 0.5 * (low + high),
            "bursts": len(in_bin),
            "jitter_3d_mm": geometric_mean(in_bin["jitter_3d_mm"]),
            "mocap_jitter_3d_mm": float(in_bin["mocap_jitter_3d_mm"].median()),
        }
        for columns in AXIS_COLUMNS:
            for column in columns:
                row[column] = geometric_mean(in_bin[column])
        distance_rows.append(row)

distance_summary = pd.DataFrame(distance_rows)
if distance_summary.empty:
    raise RuntimeError("No distance bin had enough bursts; retune DISTANCE_EDGES_M")
distance_summary.to_csv(DISTANCE_CSV, index=False)


# %% Shared plotting helpers
CAMERA_COUNTS = (1, 2)
CONDITION_COLOURS = {
    name: plt.get_cmap("viridis")((spec["tag_count"] - 1) / max(1, len(TAG_COUNTS) - 1))
    for name, spec in CONDITIONS.items()
}
CONDITION_STYLE = {1: "--", 2: "-"}
CONDITION_MARKER = {1: "o", 2: "s"}


def condition_label(name):
    spec = CONDITIONS[name]
    plural = "s" if spec["tag_count"] > 1 else ""
    return (
        f"{spec['camera_count']} cam{'s' if spec['camera_count'] > 1 else ''}, "
        f"{spec['tag_count']} tag{plural}"
    )


def ordered_conditions():
    return sorted(
        CONDITIONS,
        key=lambda n: (CONDITIONS[n]["camera_count"], CONDITIONS[n]["tag_count"]),
    )


def draw_distance_series(axis, column):
    for name in ordered_conditions():
        subset = distance_summary.loc[distance_summary["condition"] == name]
        if subset.empty:
            continue
        spec = CONDITIONS[name]
        axis.plot(
            subset["distance_center_m"],
            subset[column],
            linestyle=CONDITION_STYLE[spec["camera_count"]],
            marker=CONDITION_MARKER[spec["camera_count"]],
            markersize=4,
            linewidth=1.4,
            color=CONDITION_COLOURS[name],
            label=condition_label(name),
        )
    if np.isfinite(MOCAP_FLOOR_MM):
        axis.axhline(
            MOCAP_FLOOR_MM,
            color="0.4",
            linestyle=":",
            linewidth=1.2,
            label=f"mocap floor {MOCAP_FLOOR_MM:.2f} mm",
        )
    axis.set_yscale("log")
    axis.set_xlabel("Camera-to-dome distance [m]")
    axis.grid(True, alpha=0.25, which="both")


# %% Figure 1: the condition grid
grid = np.full((len(TAG_COUNTS), len(CAMERA_COUNTS)), np.nan)
lookup = summary.set_index(["cameras", "tags"])["jitter_3d_geomean_mm"]
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
        # This colormap is light in the middle and dark at both ends, so the
        # readable text colour follows the cell's luminance rather than its
        # position along the scale.
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
fig.colorbar(image, ax=ax, label="Geometric-mean 3D jitter [mm]")
ax.set_title(
    f"Static jitter by condition (geometric mean)\n{len(common_bursts)} bursts common to every "
    f"condition, {bursts_meta['frames_per_burst']} frames each"
)
fig.savefig(HEATMAP_FIGURE, dpi=180, bbox_inches="tight")
print(f"Saved {HEATMAP_FIGURE}")


# %% Figure 2: jitter versus distance
fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
draw_distance_series(ax, "jitter_3d_mm")
ax.set_ylabel("3D jitter [mm]")
ax.set_title("Static jitter versus camera-to-dome distance")
ax.legend(loc="upper left", ncols=2, fontsize=9)
fig.savefig(DISTANCE_FIGURE, dpi=180, bbox_inches="tight")
print(f"Saved {DISTANCE_FIGURE}")


# %% Figures 3 and 4: per-axis jitter versus distance
for figure_path, columns, basis_name, basis_note in (
    (
        AXES_REFERENCE_FIGURE,
        AXIS_COLUMNS[0],
        f"reference basis (burst {REFERENCE_BURST})",
        "Axes are the dome's own axes at the reference burst.",
    ),
    (
        AXES_CAMERA_FIGURE,
        AXIS_COLUMNS[1],
        "cam0 basis",
        "z is camera depth, where a single camera is weakest and the baseline helps most.",
    ),
):
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5), sharey=True, constrained_layout=True
    )
    for axis, column, label in zip(axes, columns, "xyz"):
        draw_distance_series(axis, column)
        axis.set_title(f"{label} axis")
    axes[0].set_ylabel("Axis jitter [mm]")
    axes[-1].legend(loc="upper left", ncols=1, fontsize=8)
    fig.suptitle(f"Per-axis static jitter versus distance — {basis_name}\n{basis_note}")
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    print(f"Saved {figure_path}")


# %% Figure 5: where in the working volume each condition jitters
# One map per condition, over the x-z plane of the reference basis. The dome
# stays within a narrow band in y (printed below), so a plane loses very little
# and a plane can actually be read.
#
# Each panel uses every burst that condition solved rather than the 24 common to
# all of them: requiring four tags in both cameras costs most of the take, and a
# map drawn only on the intersection would show a fraction of the volume that
# the single-tag conditions cover perfectly well. The sampled bursts are drawn
# on top so the differing support is visible rather than implied.
#
# Interpolation is linear and in log space, because jitter spans two orders of
# magnitude across the volume and a linear interpolation between 0.02 mm and
# 2 mm would be dominated by the large end. griddata leaves everything outside
# the convex hull of the samples as NaN, so nothing is drawn where nothing was
# measured.
spatial_y_span = burst_table["pos_y_m"].max() - burst_table["pos_y_m"].min()
print(
    f"Spatial maps: x-z plane, dome spans {1000 * spatial_y_span:.0f} mm in y "
    "(collapsed)"
)

grid_x = np.linspace(
    burst_table["pos_x_m"].min(), burst_table["pos_x_m"].max(), SPATIAL_GRID
)
grid_z = np.linspace(
    burst_table["pos_z_m"].min(), burst_table["pos_z_m"].max(), SPATIAL_GRID
)
mesh_x, mesh_z = np.meshgrid(grid_x, grid_z)

spatial_values = burst_table["jitter_3d_mm"]
spatial_values = spatial_values[np.isfinite(spatial_values) & (spatial_values > 0)]
spatial_norm = LogNorm(vmin=spatial_values.min(), vmax=spatial_values.max())

fig, axes = plt.subplots(
    len(CAMERA_COUNTS),
    len(TAG_COUNTS),
    figsize=(4.1 * len(TAG_COUNTS), 4.3 * len(CAMERA_COUNTS)),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
axes = np.atleast_2d(axes)
mesh = None
for row, camera_count in enumerate(CAMERA_COUNTS):
    for column, tag_count in enumerate(TAG_COUNTS):
        axis = axes[row, column]
        name = f"C{camera_count}-M{tag_count}"
        subset = burst_table.loc[burst_table["condition"] == name]
        subset = subset.loc[
            np.isfinite(subset["jitter_3d_mm"]) & (subset["jitter_3d_mm"] > 0)
        ]
        axis.set_title(f"{condition_label(name)}  ({len(subset)} bursts)", fontsize=10)
        if len(subset) >= 4:
            points = subset[["pos_x_m", "pos_z_m"]].to_numpy()
            surface = griddata(
                points,
                np.log10(subset["jitter_3d_mm"].to_numpy()),
                (mesh_x, mesh_z),
                method="linear",
            )
            mesh = axis.pcolormesh(
                mesh_x,
                mesh_z,
                10.0**surface,
                norm=spatial_norm,
                cmap=ERROR_COLORMAP,
                shading="auto",
            )
        # Dark rings and a white-filled star: the colormap now runs dark blue
        # through pale yellow to dark red, so white markers disappear in its
        # middle and red ones disappear at its top.
        axis.scatter(
            subset["pos_x_m"],
            subset["pos_z_m"],
            s=16,
            facecolor="none",
            edgecolor="0.15",
            linewidth=0.7,
            zorder=3,
        )
        axis.scatter(
            [0],
            [0],
            marker="*",
            s=170,
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
            zorder=4,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.2)
for axis in axes[-1]:
    axis.set_xlabel("Reference-basis x [m]")
for axis in axes[:, 0]:
    axis.set_ylabel("Reference-basis z [m]")
if mesh is not None:
    fig.colorbar(mesh, ax=axes, label="3D jitter [mm]", shrink=0.85)
fig.suptitle(
    f"Static jitter across the working area — {RECORDING_DIR.name}\n"
    "linear interpolation in log space between the sampled bursts (rings); "
    "star is the reference burst"
)
fig.savefig(SPATIAL_FIGURE, dpi=180, bbox_inches="tight")
print(f"Saved {SPATIAL_FIGURE}")


# %% Figure 6: what the bursts actually sampled
fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

# Every burst any condition could solve, not just the intersection: the
# intersection is throttled by four tags in both cameras, which the take
# satisfies for only about half its positions, and showing only those would
# misrepresent what was actually recorded. Bursts no condition solved are
# marked separately so the gaps in the sweep stay visible.
solved_positions = (
    burst_table.groupby("burst")[["pos_x_m", "pos_z_m", "distance_m"]]
    .mean()
    .reindex(burst_indices)
)
solved = solved_positions.dropna()
unsolved = solved_positions.index.difference(solved.index)

scatter = axes[0].scatter(
    solved["pos_x_m"],
    solved["pos_z_m"],
    c=solved["distance_m"],
    cmap="plasma",
    s=45,
    label=f"solved ({len(solved)})",
)
common_positions = solved.loc[solved.index.isin(common_bursts)]
axes[0].scatter(
    common_positions["pos_x_m"],
    common_positions["pos_z_m"],
    facecolor="none",
    edgecolor="black",
    linewidth=1.1,
    s=130,
    label=f"common to all 8 ({len(common_positions)})",
)
axes[0].scatter(
    [0], [0], marker="*", s=220, color="tab:red", label="reference burst", zorder=3
)
fig.colorbar(scatter, ax=axes[0], label="Distance [m]")
axes[0].set_xlabel("Reference-basis x [m]")
axes[0].set_ylabel("Reference-basis z [m]")
axes[0].set_title(
    f"Burst positions ({len(burst_indices)} recorded, "
    f"{len(unsolved)} unsolved by every condition)"
)
axes[0].set_aspect("equal", adjustable="datalim")
axes[0].grid(True, alpha=0.25)
axes[0].legend(loc="best", fontsize=9)

for name in ordered_conditions():
    subset = common_table.loc[common_table["condition"] == name].sort_values("burst")
    spec = CONDITIONS[name]
    axes[1].plot(
        subset["burst"],
        subset["jitter_3d_mm"],
        linestyle=CONDITION_STYLE[spec["camera_count"]],
        linewidth=1.2,
        color=CONDITION_COLOURS[name],
        label=condition_label(name),
    )
axes[1].set_yscale("log")
axes[1].set_xlabel("Burst index")
axes[1].set_ylabel("3D jitter [mm]")
axes[1].set_title("Jitter per burst")
axes[1].grid(True, alpha=0.25, which="both")
axes[1].legend(loc="upper left", ncols=2, fontsize=8)

coverage = (
    burst_table.groupby("condition")["burst"].nunique().reindex(ordered_conditions())
)
axes[2].bar(
    range(len(coverage)),
    coverage.to_numpy(),
    color=[CONDITION_COLOURS[name] for name in coverage.index],
)
axes[2].axhline(
    len(common_bursts),
    color="tab:red",
    linestyle="--",
    linewidth=1.2,
    label=f"common to all ({len(common_bursts)})",
)
axes[2].set_xticks(range(len(coverage)))
axes[2].set_xticklabels(
    [condition_label(n) for n in coverage.index], rotation=45, ha="right", fontsize=8
)
axes[2].set_ylabel("Bursts solved")
axes[2].set_title("Coverage: bursts each condition could solve")
axes[2].grid(True, alpha=0.25, axis="y")
axes[2].legend(loc="lower left", fontsize=9)

fig.suptitle(f"Burst sampling and per-burst jitter — {RECORDING_DIR.name}")
fig.savefig(BURST_FIGURE, dpi=180, bbox_inches="tight")
print(f"Saved {BURST_FIGURE}")

print(f"\nSaved {SUMMARY_CSV}")
print(f"Saved {BURST_CSV}")
print(f"Saved {DISTANCE_CSV}")

plt.show()
