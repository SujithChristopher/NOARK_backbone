# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ar_support import *
from pd_support import *
from pathlib import Path
from tqdm.auto import tqdm
import sys
import toml
import cv2
from cv2 import aruco
import msgpack as mp
import msgpack_numpy as mnp


try:
    NB_DIR = Path(__file__).resolve().parent
except NameError:  # running cell-by-cell in an interactive/Jupyter kernel
    NB_DIR = Path.cwd()
    if NB_DIR.name != "dual_notebooks":
        NB_DIR = NB_DIR / "trunkpose" / "dual_notebooks"
sys.path.insert(0, str(NB_DIR))
from ar_support import calculate_rotmat  # noqa: E402
from pd_support import get_marker_name, read_rigid_body_csv, get_rb_marker_name  # noqa: E402



# %% defining paths
PROJECT_ROOT = NB_DIR.resolve().parents[1]
SCRIPT_DIR = NB_DIR.resolve().parent

RECORDING_NAME = "dual_160_tframe_july1"
RECORDING_DIR = PROJECT_ROOT / "data" / "trunk_july1_2026" / RECORDING_NAME
STEREO_TOML = (
    PROJECT_ROOT
    / "data" / "calibration" / "dual_160"
    / "calib_cz30_dual_v2" / "stereo_calibration.toml"
)
CHARUCO_TOML = (
    PROJECT_ROOT
    / "data" / "trunk_july1_2026"
    / "dual_160_tframe_july1" / "charuco_basis.toml"
)


# %% Defining marker indices

# E:\CMC\pyprojects\programs_rpi\NOARK_backbone\data\calibration\dual_160\calib_cz30_dual_v2
ORIGIN_MARKER = 4
XDIR_MARKER = 1
ZDIR_MARKER = 3

AXIS_LENGTH_M = 0.08
# Known mounting offset: XD is directly above the ChArUco origin.
XD_MARKER_CHARUCO_M = np.array([0.0, 0.15, 0.0], dtype=np.float64)

mocap_df, st = read_rigid_body_csv(RECORDING_DIR / f"{RECORDING_NAME}.csv")

om = get_rb_marker_name(ORIGIN_MARKER)
xm = get_rb_marker_name(XDIR_MARKER)
zm = get_rb_marker_name(ZDIR_MARKER)

marker_columns = [
    om["x"], om["y"], om["z"],
    xm["x"], xm["y"], xm["z"],
    zm["x"], zm["y"], zm["z"],
]
marker_samples = mocap_df[marker_columns].to_numpy(dtype=np.float64)
marker_samples = marker_samples[np.isfinite(marker_samples).all(axis=1)]
if len(marker_samples) == 0:
    raise RuntimeError("No mocap samples contain all three ChArUco basis markers")

ov = np.median(marker_samples[:, 0:3], axis=0).reshape(3, 1)
xd = np.median(marker_samples[:, 3:6], axis=0).reshape(3, 1)
zd = np.median(marker_samples[:, 6:9], axis=0).reshape(3, 1)

# calculate_rotmat expects (x-direction point, z-direction point, origin).
# Its columns are the mocap frame's X, Y, Z axes expressed in mocap coordinates.
mc_rmat = calculate_rotmat(xd, zd, ov)
mc_rmat

# %% Load stereo calibration (fisheye K, D per camera)
calib = toml.load(STEREO_TOML)
K0 = np.array(calib["cam0"]["camera_matrix"])
D0 = np.array(calib["cam0"]["dist_coeffs"][0]).reshape(-1, 1)
K1 = np.array(calib["cam1"]["camera_matrix"])
D1 = np.array(calib["cam1"]["dist_coeffs"][0]).reshape(-1, 1)
R_st = np.array(calib["stereo"]["R"])
T_st = np.array(calib["stereo"]["T"]).reshape(3, 1) / 1000.0  # mm -> m

# %%

cam0_file_name = RECORDING_DIR / "cam0_frame.msgpack"
cam0_unpacker = mp.Unpacker(open(cam0_file_name, "rb"), object_hook=mnp.decode)
cam0_first_frame = next(cam0_unpacker)
# print(cam0_first_frame)

# %%
charuco_basis = toml.load(CHARUCO_TOML)
R_charuco_to_cam0 = np.asarray(
    charuco_basis["cam0"]["rotation"], dtype=np.float64
)
t_charuco_in_cam0 = np.asarray(
    charuco_basis["cam0"]["origin"], dtype=np.float64
)


def make_axis_points(origin, length=AXIS_LENGTH_M):
    """Return origin followed by the +X, +Y, and +Z axis endpoints."""
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    return np.vstack((origin, origin + np.eye(3) * length))


def project_charuco_points_cam0(points_charuco):
    """Project ChArUco-frame points into the distorted cam0 image."""
    points_charuco = np.asarray(points_charuco, dtype=np.float64).reshape(-1, 3)
    points_cam0 = points_charuco @ R_charuco_to_cam0.T + t_charuco_in_cam0
    if np.any(points_cam0[:, 2] <= 0):
        raise ValueError("At least one axis point lies behind cam0")
    image_points, _ = cv2.fisheye.projectPoints(
        points_cam0.reshape(-1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        K0,
        D0,
    )
    return image_points.reshape(-1, 2)


# mc_rmat maps L-frame coordinates -> mocap-world coordinates, so its transpose
# maps mocap-world directions into the aligned ChArUco axes. Translation is then
# determined by requiring the measured XD marker to land at its known board offset.
R_mocap_to_charuco = mc_rmat.T
t_mocap_to_charuco = (
    XD_MARKER_CHARUCO_M
    - (R_mocap_to_charuco @ xd).ravel()
)


def mocap_to_charuco_points(points_mocap):
    """Transform mocap-world points into the aligned ChArUco coordinate frame."""
    points_mocap = np.asarray(points_mocap, dtype=np.float64).reshape(-1, 3)
    return points_mocap @ R_mocap_to_charuco.T + t_mocap_to_charuco


T_mocap_to_charuco = np.eye(4, dtype=np.float64)
T_mocap_to_charuco[:3, :3] = R_mocap_to_charuco
T_mocap_to_charuco[:3, 3] = t_mocap_to_charuco

xd_in_lframe = (R_mocap_to_charuco @ (xd - ov)).ravel()
mocap_axis_world = np.vstack(
    (
        ov.ravel(),
        *(ov.ravel() + mc_rmat[:, axis] * AXIS_LENGTH_M for axis in range(3)),
    )
)
mocap_axis_charuco = mocap_to_charuco_points(mocap_axis_world)
mocap_origin_charuco = mocap_axis_charuco[0]
xd_marker_charuco = mocap_to_charuco_points(xd.ravel())[0]

charuco_axis_px = project_charuco_points_cam0(make_axis_points(np.zeros(3)))
mocap_axis_px = project_charuco_points_cam0(mocap_axis_charuco)
xd_marker_px = project_charuco_points_cam0(
    xd_marker_charuco.reshape(1, 3)
)[0]

axis_colors = ("tab:red", "tab:green", "tab:blue")
axis_names = ("X", "Y", "Z")

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(cam0_first_frame, cmap="gray" if cam0_first_frame.ndim == 2 else None)

for axis_index, (axis_name, color) in enumerate(zip(axis_names, axis_colors), start=1):
    ax.plot(
        charuco_axis_px[[0, axis_index], 0],
        charuco_axis_px[[0, axis_index], 1],
        color=color,
        linewidth=3,
        label=f"ChArUco {axis_name}",
    )
    ax.plot(
        mocap_axis_px[[0, axis_index], 0],
        mocap_axis_px[[0, axis_index], 1],
        color=color,
        linewidth=2,
        linestyle="--",
        label=f"Mocap {axis_name}",
    )

ax.scatter(*charuco_axis_px[0], color="white", edgecolor="black", s=55, zorder=5)
ax.scatter(
    *mocap_axis_px[0],
    color="yellow",
    edgecolor="black",
    s=55,
    zorder=5,
)
ax.scatter(
    *xd_marker_px,
    color="cyan",
    edgecolor="black",
    marker="D",
    s=55,
    zorder=6,
)
ax.annotate(
    "ChArUco origin",
    charuco_axis_px[0],
    xytext=(7, 7),
    textcoords="offset points",
    color="white",
    fontsize=10,
)
ax.annotate(
    "Calculated L-frame origin",
    mocap_axis_px[0],
    xytext=(7, 7),
    textcoords="offset points",
    color="yellow",
    fontsize=10,
)
ax.annotate(
    f"XD marker: {np.round(xd_marker_charuco, 3)} m",
    xd_marker_px,
    xytext=(7, -15),
    textcoords="offset points",
    color="cyan",
    fontsize=10,
)
ax.set_title(
    "cam0: ChArUco frame (solid) and assumed mocap frame (dashed)\n"
    "Mocap->ChArUco transform anchored by XD at [0, +0.15, 0] m"
)
ax.legend(loc="upper right", ncols=2, fontsize=8)
ax.axis("off")
fig.tight_layout()

output_path = RECORDING_DIR / "cam0_charuco_mocap_axes.png"
fig.savefig(output_path, dpi=180, bbox_inches="tight")
print(f"XD in calculated L-frame [m]: {xd_in_lframe.round(4)}")
print(f"Calculated L-frame origin in ChArUco [m]: {mocap_origin_charuco.round(4)}")
print(f"Transformed XD marker in ChArUco [m]: {xd_marker_charuco.round(4)}")
print("T_mocap_to_charuco =\n", np.array_str(T_mocap_to_charuco, precision=6))
print(f"Saved axis overlay -> {output_path}")
plt.show()
