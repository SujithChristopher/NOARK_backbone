# %% 11_camera_segmentation_trunk_axis_figure.py
"""Create a dual-camera segmentation and trunk-pose writing figure.

Layout (the same recording frame is used in all four panels)::

    camera 1 torso segmentation | camera 2 torso segmentation
    camera 1 trunk pose         | camera 2 trunk pose

The top row uses the YOLO torso/chest segmentation model and mask post-processing
from ``06_trunk_axis.py``.  Different overlay colours make the two camera views
easy to distinguish.  The bottom row extends
``10_camera_shoulder_axis_figure.py``: cached triangulated shoulders and the cached
trunk coordinate frame are reprojected into both fisheye cameras.  No stereo,
MediaPipe, or trunk-plane geometry is recomputed.

Run cell-by-cell (VSCode/Jupyter interactive) or::

    uv run python trunkpose/dual_notebooks/11_camera_segmentation_trunk_axis_figure.py

Environment variables:

``TRUNK_GEOM_CACHE``
    Geometry cache path (default: ``RECORDING_DIR/geom_cache.pkl``).
``TRUNK_FIG_FRAME``
    Frame index to render (default: middle frame with a valid trunk pose).
``TRUNK_SEG_WEIGHTS``
    YOLO segmentation weights; inherited by ``06_trunk_axis.py``.
``TRUNK_SEG_ALPHA``
    Segmentation overlay opacity in ``[0, 1]`` (default: ``0.45``).
``TRUNK_SEG_POSE_FIG``
    Output PNG path (default: recording directory).
"""

import importlib
import os
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from ultralytics import YOLO  # noqa: E402

try:
    NB_DIR = Path(__file__).resolve().parent
except NameError:  # running cell-by-cell in an interactive/Jupyter kernel
    NB_DIR = Path.cwd()
    if NB_DIR.name != "dual_notebooks":
        NB_DIR = NB_DIR / "trunkpose" / "dual_notebooks"
sys.path.insert(0, str(NB_DIR))
m6 = importlib.import_module("06_trunk_axis")


# BGR so the same values can be passed directly to OpenCV.
CAMERA_SEG_COLORS_BGR = {
    "camera1": (70, 210, 80),    # green
    "camera2": (40, 120, 255),   # orange
}
SHOULDER_COLORS_BGR = {
    "L_shoulder": (0, 220, 80),
    "R_shoulder": (255, 150, 30),
}


def nth_pair(idx):
    """Return the zero-based paired recording frame at *idx*."""
    for i, pair in enumerate(m6.iter_frame_pairs(m6.CAM0_FRAMES, m6.CAM1_FRAMES)):
        if i == idx:
            return pair
    raise IndexError(f"Frame {idx} is outside the paired recording.")


def as_gray(frame):
    """Normalise stored grayscale/RGB frames to one-channel uint8 images."""
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def project(points_cam, camera_matrix, dist_coeffs):
    """Project camera-frame 3D points through the calibrated fisheye model."""
    points = np.asarray(points_cam, dtype=np.float64).reshape(-1, 1, 3)
    image_points, _ = cv2.fisheye.projectPoints(
        points,
        np.zeros(3),
        np.zeros(3),
        camera_matrix,
        dist_coeffs,
    )
    return np.rint(image_points.reshape(-1, 2)).astype(int)


def segmentation_panel(gray, mask, color_bgr, alpha):
    """Overlay a solid camera-specific colour and a white mask boundary."""
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    inside = mask > 0
    if inside.any():
        color = np.empty_like(image)
        color[:] = color_bgr
        blended = cv2.addWeighted(image, 1.0 - alpha, color, alpha, 0.0)
        image[inside] = blended[inside]

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, (245, 245, 245), 2, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def trunk_pose_panel(
    gray,
    shoulder_points_board,
    axis_points_board,
    camera_matrix,
    dist_coeffs,
    to_camera,
):
    """Draw triangulated shoulders and the 3D trunk frame in one camera view."""
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    shoulder_px = project(
        [to_camera(point) for point in shoulder_points_board.values()],
        camera_matrix,
        dist_coeffs,
    )
    for (name, _), pixel in zip(shoulder_points_board.items(), shoulder_px):
        cv2.circle(
            image,
            tuple(pixel),
            8,
            SHOULDER_COLORS_BGR[name],
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(image, tuple(pixel), 9, (255, 255, 255), 1, cv2.LINE_AA)

    axis_px = project(
        [to_camera(point) for point in axis_points_board],
        camera_matrix,
        dist_coeffs,
    )
    for endpoint, name in zip((1, 2, 3), ("lateral", "up", "anterior")):
        cv2.line(
            image,
            tuple(axis_px[0]),
            tuple(axis_px[endpoint]),
            m6.AXIS_COLORS_BGR[name],
            4,
            cv2.LINE_AA,
        )
    cv2.circle(image, tuple(axis_px[0]), 5, (255, 255, 255), -1, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def angle_text(value):
    return "--" if not np.isfinite(value) else f"{value:+.1f}°"


def main():
    cache_path = Path(
        os.environ.get("TRUNK_GEOM_CACHE", str(m6.RECORDING_DIR / "geom_cache.pkl"))
    )
    output_path = Path(
        os.environ.get(
            "TRUNK_SEG_POSE_FIG",
            str(m6.RECORDING_DIR / "camera_segmentation_trunk_axis_figure.png"),
        )
    )
    alpha = float(os.environ.get("TRUNK_SEG_ALPHA", "0.45"))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("TRUNK_SEG_ALPHA must be between 0 and 1.")

    with cache_path.open("rb") as cache_file:
        frames = pickle.load(cache_file)
    print(f"Loaded {len(frames)} cached frames from {cache_path}")

    R0_c, t0_c, _Rm_c, _tm_c = m6.load_charuco_basis(m6.CHARUCO_TOML)
    K0, D0, K1, D1, R, T_mm = m6.load_stereo(m6.STEREO_TOML)
    T_m = (T_mm / 1000.0).ravel()

    R_t_list, _R_neu, flexion, lateral, axial, _all_points = m6.assemble(frames)
    candidates = [
        i
        for i, (frame, R_t) in enumerate(zip(frames, R_t_list))
        if R_t is not None
        and frame["cam_pts"]["L_shoulder"] is not None
        and frame["cam_pts"]["R_shoulder"] is not None
    ]
    if not candidates:
        raise SystemExit("No frame with both shoulders and a valid trunk frame.")

    requested = os.environ.get("TRUNK_FIG_FRAME")
    frame_idx = int(requested) if requested is not None else candidates[len(candidates) // 2]
    if frame_idx not in candidates:
        raise ValueError(
            f"TRUNK_FIG_FRAME={frame_idx} does not have both shoulders and a valid "
            "trunk frame."
        )
    frame = frames[frame_idx]
    R_t = R_t_list[frame_idx]
    print(f"Using frame {frame_idx}/{len(frames) - 1}")

    raw0, raw1 = nth_pair(frame_idx)
    gray0, gray1 = as_gray(raw0), as_gray(raw1)

    print(f"Loading torso segmentation weights from {m6.SEG_WEIGHTS}")
    segmenter = YOLO(str(m6.SEG_WEIGHTS))
    mask0 = m6.torso_mask(segmenter, gray0, (gray0.shape[1], gray0.shape[0]))
    mask1 = m6.torso_mask(segmenter, gray1, (gray1.shape[1], gray1.shape[0]))
    if not mask0.any():
        print("Warning: no torso segmentation detected in camera1.")
    if not mask1.any():
        print("Warning: no torso segmentation detected in camera2.")

    left_shoulder = frame["cam_pts"]["L_shoulder"]
    right_shoulder = frame["cam_pts"]["R_shoulder"]
    shoulder_points = {
        "L_shoulder": left_shoulder,
        "R_shoulder": right_shoulder,
    }
    origin_board = m6.trunk_origin(left_shoulder, right_shoulder, R_t)
    axis_points = [origin_board] + [
        origin_board + R_t[:, column] * m6.AXIS_LEN_M for column in (0, 1, 2)
    ]

    def to_cam0(point):
        return m6.board_to_cam0(point, R0_c, t0_c)

    def to_cam1(point):
        return R @ to_cam0(point) + T_m

    panels = {
        "seg0": segmentation_panel(
            gray0, mask0, CAMERA_SEG_COLORS_BGR["camera1"], alpha
        ),
        "seg1": segmentation_panel(
            gray1, mask1, CAMERA_SEG_COLORS_BGR["camera2"], alpha
        ),
        "pose0": trunk_pose_panel(
            gray0, shoulder_points, axis_points, K0, D0, to_cam0
        ),
        "pose1": trunk_pose_panel(
            gray1, shoulder_points, axis_points, K1, D1, to_cam1
        ),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    specifications = (
        (axes[0, 0], panels["seg0"], "Camera 1 — torso segmentation"),
        (axes[0, 1], panels["seg1"], "Camera 2 — torso segmentation"),
        (axes[1, 0], panels["pose0"], "Camera 1 — trunk pose"),
        (axes[1, 1], panels["pose1"], "Camera 2 — trunk pose"),
    )
    for axis, panel, title in specifications:
        axis.imshow(panel)
        axis.set_title(title, fontsize=12)
        axis.axis("off")

    angles = (
        f"flexion {angle_text(flexion[frame_idx])}   |   "
        f"lateral bending {angle_text(lateral[frame_idx])}   |   "
        f"axial rotation {angle_text(axial[frame_idx])}"
    )
    fig.suptitle(f"Dual-camera torso segmentation and trunk pose — frame {frame_idx}\n{angles}")

    axis_handles = [
        Line2D([0], [0], color=m6.AXIS_COLORS_MPL[name], lw=3, label=name)
        for name in ("lateral", "up", "anterior")
    ]
    shoulder_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=np.array(SHOULDER_COLORS_BGR[name][::-1]) / 255.0,
            markeredgecolor="white",
            markersize=8,
            label=name.replace("_", " "),
        )
        for name in ("L_shoulder", "R_shoulder")
    ]
    fig.legend(
        handles=axis_handles + shoulder_handles,
        loc="outside lower center",
        ncol=5,
        frameon=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved -> {output_path}")

    if "ipykernel" in sys.modules:
        from IPython.display import display

        display(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()

