# %% [markdown]
# # AprilTag 12 vs mocap rigid-body trajectory
#
# This notebook detects a 50 mm AprilTag in the cam0 recording, synchronizes the
# camera and Motive timelines using the GPIO pulse, jointly estimates the
# camera/mocap transform and tag mounting offset, and compares X/Y/Z over time.
#
# Run from the repository root, cell-by-cell or as a script:
#
# ```powershell
# uv run python jitter_model/01_plot_data.py
# ```

# %% Imports
from datetime import datetime
from pathlib import Path
import sys
import warnings

import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import msgpack
import msgpack_numpy as mpn
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
import toml
from tqdm.auto import tqdm


# %% Paths and settings
try:
    NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:  # running cells in an interactive kernel
    NOTEBOOK_DIR = Path.cwd()
    if NOTEBOOK_DIR.name != "jitter_model":
        NOTEBOOK_DIR = NOTEBOOK_DIR / "jitter_model"

PROJECT_ROOT = NOTEBOOK_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

RECORDING_DIR = PROJECT_ROOT / "data" / "radxa" / "jitter_test"
MOCAP_CSV = RECORDING_DIR / "jitter_test.csv"
CALIBRATION_TOML = (
    PROJECT_ROOT
    / "data"
    / "calibration"
    / "dual_160"
    / "radxa_calib_parallel"
    / "stereo_calibration.toml"
)

CAMERA_NAME = "cam0"
TAG_ID = 12
TAG_SIZE_M = 0.05

# Use None for the complete recording. A small integer is useful while editing.
MAX_FRAMES = None

# Fit the mocap-world->camera transform and the tag's rigid-body mounting offset
# on the first half of the paired samples, then report fit and held-out errors
# separately. Increase this toward 1.0 if the goal is visualization only rather
# than an honest held-out diagnostic.
ALIGNMENT_FIT_FRACTION = 0.5

# Set to None for notebook-only display.
SAVE_FIGURE = NOTEBOOK_DIR / "apriltag12_vs_mocap_xyz.png"

# False plots the aligned positions directly. True median-centers each system so
# small jitter is easier to see independent of any remaining alignment bias.
PLOT_AS_DISPLACEMENT = False


# %% Motive CSV loader
def read_rigid_body_csv(path):
    """Read a Motive rigid-body CSV using this repository's export layout."""
    raw_header = pd.read_csv(path, nrows=0, dtype=str).columns.tolist()
    try:
        start_index = raw_header.index("Capture Start Time")
    except ValueError as exc:
        raise ValueError(f"No 'Capture Start Time' field in {path}") from exc
    capture_start = datetime.strptime(
        raw_header[start_index + 1], "%Y-%m-%d %I.%M.%S.%f %p"
    )

    # Six metadata/header lines precede the final column-name row in this Motive
    # 1.23 export, so that seventh line remains pandas' header.
    table = pd.read_csv(path, skiprows=6)
    rigid_body_columns = {
        "Frame": "frame",
        "Time (Seconds)": "seconds",
        "X": "rb_ang_x",
        "Y": "rb_ang_y",
        "Z": "rb_ang_z",
        "W": "rb_ang_w",
        "X.1": "rb_pos_x",
        "Y.1": "rb_pos_y",
        "Z.1": "rb_pos_z",
    }
    missing = [name for name in rigid_body_columns if name not in table.columns]
    if missing:
        raise ValueError(f"Unexpected Motive CSV layout; missing columns: {missing}")

    rigid_body = table[list(rigid_body_columns)].rename(columns=rigid_body_columns)
    rigid_body = rigid_body.apply(pd.to_numeric, errors="coerce")
    return rigid_body, capture_start


# %% Camera recording loaders
def load_timestamp_records(path):
    """Return GPIO flags and wall-clock timestamps from a camera msgpack file."""
    with path.open("rb") as stream:
        records = list(msgpack.Unpacker(stream, object_hook=mpn.decode))
    if not records:
        raise ValueError(f"No timestamp records in {path}")

    sync = np.asarray([int(record[0]) for record in records], dtype=np.uint8)
    timestamps = np.asarray(
        [datetime.fromisoformat(record[1]) for record in records],
        dtype="datetime64[us]",
    )
    return sync, timestamps


def iter_frames(path, max_frames=None):
    """Yield `(frame_index, image)` without loading the recording into memory."""
    with path.open("rb") as stream:
        unpacker = msgpack.Unpacker(stream, object_hook=mpn.decode)
        for frame_index, frame in enumerate(unpacker):
            if max_frames is not None and frame_index >= max_frames:
                break
            yield frame_index, frame


def first_sync_rise(sync):
    """First high GPIO sample; the recorder is low before the actual rising edge."""
    high = np.flatnonzero(np.asarray(sync) == 1)
    if not len(high):
        raise ValueError("No GPIO sync pulse in the camera timestamp recording")
    return int(high[0])


# %% Calibration and AprilTag pose estimation
calibration = toml.load(CALIBRATION_TOML)
camera_calibration = calibration[CAMERA_NAME]
K = np.asarray(camera_calibration["camera_matrix"], dtype=np.float64)
D = np.asarray(camera_calibration["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
CALIBRATION_RESOLUTION = tuple(camera_calibration["resolution"])

dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
detector_parameters = aruco.DetectorParameters()
detector_parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
detector = aruco.ArucoDetector(dictionary, detector_parameters)

half_size = TAG_SIZE_M / 2.0
# Required point order for SOLVEPNP_IPPE_SQUARE: top-left, top-right,
# bottom-right, bottom-left in the marker coordinate frame.
TAG_OBJECT_POINTS = np.asarray(
    [
        [-half_size, +half_size, 0.0],
        [+half_size, +half_size, 0.0],
        [+half_size, -half_size, 0.0],
        [-half_size, -half_size, 0.0],
    ],
    dtype=np.float64,
)


def estimate_tag_pose(frame):
    """Return `(tvec, rvec, reprojection_rmse)` for TAG_ID, or None.

    The supplied calibration is OpenCV fisheye calibration (four distortion
    coefficients). Corners are therefore fisheye-undistorted first, followed by
    solvePnP with zero distortion; ArUco's standard pose helper assumes the
    pinhole distortion model and must not consume these coefficients directly.
    """
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return None

    matches = np.flatnonzero(ids.ravel() == TAG_ID)
    if not len(matches):
        return None
    distorted_corners = np.asarray(corners[int(matches[0])], dtype=np.float64)

    undistorted_corners = cv2.fisheye.undistortPoints(
        distorted_corners.reshape(-1, 1, 2), K, D, P=K
    ).reshape(-1, 2)
    success, rvec, tvec = cv2.solvePnP(
        TAG_OBJECT_POINTS,
        undistorted_corners,
        K,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success or float(tvec.reshape(3)[2]) <= 0:
        return None

    projected, _ = cv2.projectPoints(
        TAG_OBJECT_POINTS,
        rvec,
        tvec,
        K,
        np.zeros((4, 1), dtype=np.float64),
    )
    residual = projected.reshape(-1, 2) - undistorted_corners
    reprojection_rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    return tvec.reshape(3), rvec.reshape(3), reprojection_rmse


# %% Detect AprilTag 12 in cam0
timestamp_path = RECORDING_DIR / f"{CAMERA_NAME}_timestamp.msgpack"
frame_path = RECORDING_DIR / f"{CAMERA_NAME}_frame.msgpack"

sync_flags, camera_timestamps = load_timestamp_records(timestamp_path)
sync_frame = first_sync_rise(sync_flags)
camera_time_s = (
    (camera_timestamps - camera_timestamps[sync_frame]) / np.timedelta64(1, "s")
).astype(np.float64)

tag_rows = []
frame_limit = len(camera_timestamps) if MAX_FRAMES is None else MAX_FRAMES
for frame_index, frame in tqdm(
    iter_frames(frame_path, MAX_FRAMES),
    total=min(len(camera_timestamps), frame_limit),
    desc=f"Detecting AprilTag {TAG_ID} in {CAMERA_NAME}",
):
    if frame_index >= len(camera_timestamps):
        warnings.warn("More images than timestamps; ignoring unmatched images")
        break
    if tuple(frame.shape[1::-1]) != CALIBRATION_RESOLUTION:
        raise ValueError(
            f"Frame resolution {frame.shape[1::-1]} does not match calibration "
            f"{CALIBRATION_RESOLUTION}"
        )
    pose = estimate_tag_pose(frame)
    if pose is None:
        continue
    tvec, rvec, reprojection_rmse = pose
    tag_rows.append(
        {
            "frame": frame_index,
            "time_s": camera_time_s[frame_index],
            "tag_x": tvec[0],
            "tag_y": tvec[1],
            "tag_z": tvec[2],
            "tag_rx": rvec[0],
            "tag_ry": rvec[1],
            "tag_rz": rvec[2],
            "reprojection_rmse_px": reprojection_rmse,
        }
    )

tag_df = pd.DataFrame(tag_rows)
if tag_df.empty:
    raise RuntimeError(
        f"AprilTag {TAG_ID} ({TAG_SIZE_M:.3f} m, DICT_APRILTAG_36h11) was not detected"
    )

print(
    f"Detected tag {TAG_ID} in {len(tag_df)}/{min(len(camera_timestamps), frame_limit)} "
    f"frames ({100 * len(tag_df) / min(len(camera_timestamps), frame_limit):.1f}%)."
)
print(f"GPIO sync rise: {CAMERA_NAME} frame {sync_frame} (camera time = 0 s).")
print(
    "Median corner reprojection RMSE: "
    f"{tag_df['reprojection_rmse_px'].median():.3f} px"
)


# %% Load mocap and interpolate its rigid-body origin at camera detection times
mocap_df, mocap_capture_start = read_rigid_body_csv(MOCAP_CSV)
mocap_columns = ["rb_pos_x", "rb_pos_y", "rb_pos_z"]
quaternion_columns = ["rb_ang_x", "rb_ang_y", "rb_ang_z", "rb_ang_w"]
mocap_valid = np.isfinite(
    mocap_df[["seconds", *mocap_columns, *quaternion_columns]]
).all(axis=1)
mocap_samples = mocap_df.loc[
    mocap_valid, ["seconds", *mocap_columns, *quaternion_columns]
].copy()

# Hardware synchronization, following 07_icp_trunk.py / 06_trunk_axis.py:
# Motive seconds=0 is the first high GPIO sample in the camera timestamp stream.
overlap = tag_df["time_s"].between(
    float(mocap_samples["seconds"].min()),
    float(mocap_samples["seconds"].max()),
)
paired = tag_df.loc[overlap].reset_index(drop=True).copy()
if len(paired) < 3:
    raise RuntimeError("Fewer than three synchronized AprilTag/mocap samples")

for source, destination in zip(mocap_columns, ("mocap_x", "mocap_y", "mocap_z")):
    paired[destination] = np.interp(
        paired["time_s"], mocap_samples["seconds"], mocap_samples[source]
    )

mocap_quaternions = mocap_samples[quaternion_columns].to_numpy(
    dtype=np.float64, copy=True
)
mocap_quaternions /= np.linalg.norm(mocap_quaternions, axis=1, keepdims=True)
mocap_orientation = Slerp(
    mocap_samples["seconds"].to_numpy(dtype=np.float64),
    Rotation.from_quat(mocap_quaternions),
)(paired["time_s"].to_numpy(dtype=np.float64))

print(
    f"Synchronized overlap: {len(paired)} tag detections from "
    f"{paired['time_s'].iloc[0]:.3f} to {paired['time_s'].iloc[-1]:.3f} s."
)
print(f"Motive header capture time (not used for sync): {mocap_capture_start}")


# %% Align mocap coordinates and solve the rigid-body-to-tag mounting offset
def fit_rigid_transform(source_points, target_points):
    """Kabsch fit of target = source @ R.T + t, with no scale or reflection."""
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    if source_points.shape != target_points.shape or source_points.shape[1] != 3:
        raise ValueError("source_points and target_points must both have shape (N, 3)")

    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    covariance = (source_points - source_center).T @ (
        target_points - target_center
    )
    U, singular_values, Vt = np.linalg.svd(covariance)
    handedness = np.sign(np.linalg.det(Vt.T @ U.T))
    rotation = Vt.T @ np.diag([1.0, 1.0, handedness]) @ U.T
    translation = target_center - rotation @ source_center
    return rotation, translation, singular_values


def fit_offset_aware_alignment(
    rigid_body_positions,
    rigid_body_orientations,
    tag_positions,
):
    """Estimate world->camera extrinsics and a constant tag offset on the body.

    Model:

        tag_camera = R_camera_world @
            (rigid_body_world + R_world_body @ tag_offset_body) + t_camera_world

    Motive exports `R_world_body` through its rigid-body quaternion. This extra
    offset term is essential whenever the tag center and Motive rigid-body origin
    are not coincident: a constant world-space translation cannot model the arc
    traced by the tag as the rigid body rotates.
    """
    initial_rotation, initial_translation, singular_values = fit_rigid_transform(
        rigid_body_positions, tag_positions
    )
    initial = np.concatenate(
        [
            Rotation.from_matrix(initial_rotation).as_rotvec(),
            initial_translation,
            np.zeros(3, dtype=np.float64),
        ]
    )

    def residual(parameters):
        camera_from_world = Rotation.from_rotvec(parameters[:3])
        camera_translation = parameters[3:6]
        tag_offset_body = parameters[6:9]
        tag_world = rigid_body_positions + rigid_body_orientations.apply(
            tag_offset_body
        )
        prediction = camera_from_world.apply(tag_world) + camera_translation
        return (prediction - tag_positions).ravel()

    result = least_squares(
        residual,
        initial,
        loss="soft_l1",
        f_scale=0.003,  # 3 mm: down-weight occasional pose outliers
        max_nfev=2000,
    )
    if not result.success:
        warnings.warn(f"Alignment optimizer did not converge: {result.message}")

    return (
        Rotation.from_rotvec(result.x[:3]),
        result.x[3:6],
        result.x[6:9],
        singular_values,
    )


tag_points = paired[["tag_x", "tag_y", "tag_z"]].to_numpy(dtype=np.float64)
mocap_points = paired[["mocap_x", "mocap_y", "mocap_z"]].to_numpy(
    dtype=np.float64
)

fit_count = max(3, int(len(paired) * ALIGNMENT_FIT_FRACTION))
fit_count = min(fit_count, len(paired))
(
    camera_from_mocap,
    t_mocap_to_camera,
    tag_offset_body,
    motion_singular_values,
) = fit_offset_aware_alignment(
    mocap_points[:fit_count],
    mocap_orientation[:fit_count],
    tag_points[:fit_count],
)
R_mocap_to_camera = camera_from_mocap.as_matrix()
tag_points_mocap_world = mocap_points + mocap_orientation.apply(tag_offset_body)
aligned_mocap = camera_from_mocap.apply(tag_points_mocap_world) + t_mocap_to_camera
paired[["mocap_aligned_x", "mocap_aligned_y", "mocap_aligned_z"]] = aligned_mocap

alignment_residual = aligned_mocap - tag_points
alignment_error_mm = 1000.0 * np.linalg.norm(alignment_residual, axis=1)
fit_error_mm = alignment_error_mm[:fit_count]
test_error_mm = alignment_error_mm[fit_count:]

# A static jitter recording cannot strongly constrain all three rotation axes.
# Report that limitation rather than silently treating the fitted matrix as a
# reusable extrinsic calibration.
motion_condition = (
    np.inf
    if motion_singular_values[-1] <= np.finfo(float).eps
    else motion_singular_values[0] / motion_singular_values[-1]
)
if not np.isfinite(motion_condition) or motion_condition > 1e4:
    warnings.warn(
        "The fitted motion is nearly planar/static, so the rotation matrix is "
        "poorly constrained. It is suitable for overlaying this jitter take, but "
        "a moved calibration take is required for reusable camera-mocap extrinsics."
    )

print("R_mocap_to_camera =\n", np.array_str(R_mocap_to_camera, precision=7))
print("t_mocap_to_camera [m] =", np.round(t_mocap_to_camera, 7))
print("tag_offset_in_rigid_body [m] =", np.round(tag_offset_body, 7))
print(f"det(R) = {np.linalg.det(R_mocap_to_camera):.6f}")
print("motion singular values =", np.array_str(motion_singular_values, precision=5))
print(f"fit position RMSE = {np.sqrt(np.mean(fit_error_mm**2)):.3f} mm")
if len(test_error_mm):
    print(f"held-out position RMSE = {np.sqrt(np.mean(test_error_mm**2)):.3f} mm")


# %% Compare per-axis jitter numerically
axis_rows = []
for axis in "xyz":
    tag_axis = paired[f"tag_{axis}"].to_numpy()
    mocap_axis = paired[f"mocap_aligned_{axis}"].to_numpy()
    axis_rows.append(
        {
            "axis": axis.upper(),
            "apriltag_std_mm": 1000.0 * np.std(tag_axis, ddof=1),
            "mocap_std_mm": 1000.0 * np.std(mocap_axis, ddof=1),
            "difference_rmse_mm": 1000.0
            * np.sqrt(np.mean((tag_axis - mocap_axis) ** 2)),
            "correlation": np.corrcoef(tag_axis, mocap_axis)[0, 1],
        }
    )

jitter_summary = pd.DataFrame(axis_rows).set_index("axis")
print(jitter_summary.round(4))


# %% Plot AprilTag vs aligned mocap in X, Y, and Z
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
axis_colors = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}

for axis, ax in zip("xyz", axes):
    tag_mm = 1000.0 * paired[f"tag_{axis}"].to_numpy()
    mocap_mm = 1000.0 * paired[f"mocap_aligned_{axis}"].to_numpy()
    if PLOT_AS_DISPLACEMENT:
        tag_plot = tag_mm - np.median(tag_mm[:fit_count])
        mocap_plot = mocap_mm - np.median(mocap_mm[:fit_count])
        ylabel = f"$\\Delta${axis.upper()} [mm]"
    else:
        tag_plot = tag_mm
        mocap_plot = mocap_mm
        ylabel = f"{axis.upper()} [mm]"

    ax.plot(
        paired["time_s"],
        tag_plot,
        color=axis_colors[axis],
        linewidth=1.2,
        label=f"AprilTag {TAG_ID}",
    )
    ax.plot(
        paired["time_s"],
        mocap_plot,
        color="black",
        linewidth=1.0,
        alpha=0.75,
        label="Mocap rigid body (aligned)",
    )
    ax.axvline(
        paired["time_s"].iloc[fit_count - 1],
        color="0.55",
        linestyle=":",
        linewidth=1,
        label="alignment fit/test split" if axis == "x" else None,
    )
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncols=3)

axes[-1].set_xlabel("Time after GPIO sync pulse [s]")
fig.suptitle(
    f"AprilTag {TAG_ID} vs mocap rigid-body trajectory — {CAMERA_NAME}\n"
    + (
        "Offset-aware mocap alignment; each system median-centered for jitter"
        if PLOT_AS_DISPLACEMENT
        else "Offset-aware mocap alignment in camera coordinates"
    )
)

if SAVE_FIGURE is not None:
    save_path = Path(SAVE_FIGURE)
    if not save_path.is_absolute():
        save_path = PROJECT_ROOT / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    print(f"Saved plot -> {save_path}")

plt.show()


# %% Optional: inspect the synchronized, aligned table
paired.head()
