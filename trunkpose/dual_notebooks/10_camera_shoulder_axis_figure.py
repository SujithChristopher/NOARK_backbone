# %% 10_camera_shoulder_axis_figure.py
# Writing figure: one raw frame from camera1 (cam0) + camera2 (cam1), side by side,
# each with the triangulated shoulder points and the trunk coordinate axis (lateral/
# up/anterior) reprojected onto it. No angle plots -- just the two annotated images.
#
# Reuses 06_trunk_axis's geometry cache (TRUNK_GEOM_CACHE / geom_cache.pkl) -- no live
# MediaPipe/YOLO/stereo recompute needed. Shoulder + axis points are the same
# board-frame 3D points reprojected into each camera (via fisheye.projectPoints +
# the cam0<->cam1 stereo extrinsics), so both panels show a geometrically consistent
# reconstruction rather than two independent per-camera detections.
#
# Run cell-by-cell (VSCode/Jupyter interactive) or:
#   uv run python trunkpose/dual_notebooks/10_camera_shoulder_axis_figure.py
#
# Env:
#   TRUNK_GEOM_CACHE   path to geom_cache.pkl (default: RECORDING_DIR/geom_cache.pkl)
#   TRUNK_FIG_FRAME    force a specific frame index (default: middle of valid frames)

import importlib
import os
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    NB_DIR = Path(__file__).resolve().parent
except NameError:  # running cell-by-cell in an interactive/Jupyter kernel
    NB_DIR = Path.cwd()
    if NB_DIR.name != "dual_notebooks":
        NB_DIR = NB_DIR / "trunkpose" / "dual_notebooks"
sys.path.insert(0, str(NB_DIR))
m6 = importlib.import_module("06_trunk_axis")  # side effect: matplotlib.use("Agg")

if "ipykernel" not in sys.modules:
    plt.switch_backend("TkAgg")

OUT_PNG = m6.RECORDING_DIR / "camera_shoulder_axis_figure.png"

# %% Load geometry cache + basis/stereo calib
cache = Path(os.environ.get("TRUNK_GEOM_CACHE", str(m6.RECORDING_DIR / "geom_cache.pkl")))
with open(cache, "rb") as f:
    frames = pickle.load(f)
print(f"Loaded {len(frames)} cached frames from {cache}")

R0_c, t0_c, Rm_c, tm_c = m6.load_charuco_basis(m6.CHARUCO_TOML)
K0, D0, K1, D1, R, T_mm = m6.load_stereo(m6.STEREO_TOML)
T_m = (T_mm / 1000.0).ravel()

R_t_list, _R_neu, _flex, _lat, _axi, _all_pts = m6.assemble(frames)
for fr, R_t in zip(frames, R_t_list):
    fr["R_t"] = R_t

# %% Pick a representative frame: both shoulders present + a valid trunk frame
candidates = [
    i for i, fr in enumerate(frames)
    if fr["R_t"] is not None
    and fr["cam_pts"]["L_shoulder"] is not None
    and fr["cam_pts"]["R_shoulder"] is not None]
if not candidates:
    raise SystemExit("No frame with both shoulders + a valid trunk frame.")
FRAME_IDX = int(os.environ.get("TRUNK_FIG_FRAME", candidates[len(candidates) // 2]))
fr = frames[FRAME_IDX]
print(f"Using frame {FRAME_IDX}/{len(frames)}")

# %% Raw images for that frame index
def nth_pair(idx):
    for i, (f0, f1) in enumerate(m6.iter_frame_pairs(m6.CAM0_FRAMES, m6.CAM1_FRAMES)):
        if i == idx:
            return f0, f1
    raise IndexError(idx)

g0, g1 = nth_pair(FRAME_IDX)
g0 = g0 if g0.ndim == 2 else cv2.cvtColor(g0, cv2.COLOR_RGB2GRAY)
g1 = g1 if g1.ndim == 2 else cv2.cvtColor(g1, cv2.COLOR_RGB2GRAY)

# %% Board-frame points to draw: shoulders + trunk axis endpoints
R_t = fr["R_t"]
ls, rs = fr["cam_pts"]["L_shoulder"], fr["cam_pts"]["R_shoulder"]
origin_b = m6.trunk_origin(ls, rs, R_t)
axis_ends_b = [origin_b] + [origin_b + R_t[:, c] * m6.AXIS_LEN_M for c in (0, 1, 2)]
shoulder_pts_b = [ls, rs]


def to_cam0(p):
    return m6.board_to_cam0(p, R0_c, t0_c)


def to_cam1(p):
    return R @ to_cam0(p) + T_m


def project(pts_cam, K, D):
    arr = np.array(pts_cam, dtype=np.float64).reshape(-1, 1, 3)
    ip, _ = cv2.fisheye.projectPoints(arr, np.zeros(3), np.zeros(3), K, D)
    return ip.reshape(-1, 2).astype(int)


def draw_panel(gray, K, D, to_cam):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    sh_ip = project([to_cam(p) for p in shoulder_pts_b], K, D)
    for px in sh_ip:
        cv2.circle(img, tuple(px), 8, (0, 255, 0), -1)
    axis_ip = project([to_cam(p) for p in axis_ends_b], K, D)
    for c, name in zip((1, 2, 3), ("lateral", "up", "anterior")):
        cv2.line(img, tuple(axis_ip[0]), tuple(axis_ip[c]), m6.AXIS_COLORS_BGR[name], 3)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


panel0 = draw_panel(g0, K0, D0, to_cam0)
panel1 = draw_panel(g1, K1, D1, to_cam1)

# %% 1x2 figure
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
for ax, panel, label in ((ax0, panel0, "camera1"), (ax1, panel1, "camera2")):
    ax.imshow(panel)
    ax.set_title(label)
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print(f"Saved -> {OUT_PNG}")

if "ipykernel" in sys.modules:
    import io
    from IPython.display import Image, display
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    display(Image(buf.getvalue()))
else:
    plt.show()
