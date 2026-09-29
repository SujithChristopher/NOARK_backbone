# %% [markdown]
# # Detection video with pose axes
#
# Renders one camera's recording to an mp4 with every cached tag detection
# drawn on it: the outline, the ID, and the tag's pose axes (x red, y green,
# z blue). The outlines are the cached sub-pixel corners. The axes come from an
# IPPE pose on the undistorted corners, projected back through the fisheye
# model so they sit correctly on the raw image.
#
# ```powershell
# uv run python jitter_model/render_detection_video.py
# ```

# %% Imports
from pathlib import Path
import pickle

import cv2
import msgpack
import msgpack_numpy as mpn
import numpy as np
import toml
from tqdm.auto import tqdm

# %% Settings
try:
    NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:  # running cell-by-cell in an interactive kernel
    NOTEBOOK_DIR = Path.cwd()
    if NOTEBOOK_DIR.name != "jitter_model":
        NOTEBOOK_DIR = NOTEBOOK_DIR / "jitter_model"
PROJECT_ROOT = NOTEBOOK_DIR.parent
RECORDING_DIR = PROJECT_ROOT / "data" / "dome" / "sep18_26" / "dome_rb_def"
DETECTION_CACHE = RECORDING_DIR / "jitter_detections.pkl"
CALIBRATION_TOML = (
    PROJECT_ROOT
    / "data"
    / "calibration"
    / "dual_160"
    / "calib_radxa_sep18_26"
    / "stereo_calibration.toml"
)

CAMERA = "cam0"
OUTPUT_VIDEO = RECORDING_DIR / f"{CAMERA}_detections.mp4"

TAG_SIZE_M = 0.05
AXIS_LENGTH_M = 0.03
PLAYBACK_SPEED = 1.0  # 1.0 = real time; 0.5 = half speed
MAX_FRAMES = None  # e.g. 200 for a quick test

OUTLINE_COLOR = (0, 220, 0)  # BGR
AXIS_COLORS = ((0, 0, 255), (0, 255, 0), (255, 0, 0))  # x red, y green, z blue

TAG_CORNERS = np.asarray(
    [
        [-TAG_SIZE_M / 2, +TAG_SIZE_M / 2, 0.0],
        [+TAG_SIZE_M / 2, +TAG_SIZE_M / 2, 0.0],
        [+TAG_SIZE_M / 2, -TAG_SIZE_M / 2, 0.0],
        [-TAG_SIZE_M / 2, -TAG_SIZE_M / 2, 0.0],
    ],
    dtype=np.float64,
)
AXIS_POINTS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [AXIS_LENGTH_M, 0.0, 0.0],
        [0.0, AXIS_LENGTH_M, 0.0],
        [0.0, 0.0, AXIS_LENGTH_M],
    ],
    dtype=np.float64,
)


# %% Pose and drawing
def estimate_marker_pose(corners, K, D):
    """Lowest-reprojection-error positive-depth IPPE pose of one square tag."""
    undistorted = cv2.fisheye.undistortPoints(
        np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2), K, D, P=K
    ).reshape(4, 2)
    ok, rvecs, tvecs, _err = cv2.solvePnPGeneric(
        TAG_CORNERS, undistorted, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    best = None
    for rvec, tvec in zip(rvecs, tvecs):
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        if tvec[2] <= 0:
            continue
        projected, _ = cv2.projectPoints(TAG_CORNERS, rvec, tvec, K, None)
        rmse = float(
            np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - undistorted) ** 2, axis=1)))
        )
        if best is None or rmse < best[2]:
            best = (rvec.reshape(3, 1), tvec.reshape(3, 1), rmse)
    return best


def draw_tag(canvas, marker_id, corners, K, D):
    points = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [points], True, OUTLINE_COLOR, 1)
    # ID above the tag, with a dark halo so it reads over any background.
    label_origin = (int(corners[:, 0].mean()) - 8, int(corners[:, 1].min()) - 8)
    for color, thickness in (((0, 0, 0), 5), (OUTLINE_COLOR, 2)):
        cv2.putText(
            canvas,
            str(marker_id),
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            thickness,
            cv2.LINE_AA,
        )

    pose = estimate_marker_pose(corners, K, D)
    if pose is None:
        return
    rvec, tvec, _rmse = pose
    axis_pixels, _ = cv2.fisheye.projectPoints(
        AXIS_POINTS.reshape(-1, 1, 3), rvec, tvec, K, D
    )
    axis_pixels = axis_pixels.reshape(-1, 2)
    origin = tuple(int(v) for v in axis_pixels[0])
    for tip, color in zip(axis_pixels[1:], AXIS_COLORS):
        cv2.line(canvas, origin, tuple(int(v) for v in tip), color, 2, cv2.LINE_AA)


# %% Load calibration, detections and timing
calibration = toml.load(CALIBRATION_TOML)[CAMERA]
K = np.asarray(calibration["camera_matrix"], dtype=np.float64)
D = np.asarray(calibration["dist_coeffs"], dtype=np.float64).reshape(-1, 1)

with DETECTION_CACHE.open("rb") as stream:
    camera_cache = pickle.load(stream)["cameras"][CAMERA]  # noqa: S301
all_detections = camera_cache["detections"]
monotonic_ns = camera_cache["metadata"]["monotonic_ns"]

frame_period_s = float(np.median(np.diff(monotonic_ns))) / 1e9
fps = 1.0 / frame_period_s * PLAYBACK_SPEED
print(f"{CAMERA}: {len(all_detections)} frames, {1.0 / frame_period_s:.1f} fps recorded")


# %% Render
frame_path = RECORDING_DIR / f"{CAMERA}_frame.msgpack"
limit = len(all_detections) if MAX_FRAMES is None else min(MAX_FRAMES, len(all_detections))
writer = None

with frame_path.open("rb") as stream:
    unpacker = msgpack.Unpacker(stream, object_hook=mpn.decode)
    for frame_index, frame in enumerate(
        tqdm(unpacker, total=limit, desc=f"Rendering {CAMERA}")
    ):
        if frame_index >= limit:
            break
        canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()

        for marker_id, corners in all_detections[frame_index].items():
            draw_tag(canvas, marker_id, corners, K, D)

        cv2.putText(
            canvas,
            f"{CAMERA}  frame {frame_index}  tags {sorted(all_detections[frame_index])}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if writer is None:
            height, width = canvas.shape[:2]
            writer = cv2.VideoWriter(
                str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
        writer.write(canvas)

if writer is not None:
    writer.release()
print(f"Saved video -> {OUTPUT_VIDEO}")
