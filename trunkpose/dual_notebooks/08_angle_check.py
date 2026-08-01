# %% 08_angle_check.py
# Plot-only REPL notebook for GPIO-synchronized ICP and mocap trunk angles.
#
# Unlike 07_icp_trunk.py / 08_icp_video.py, mocap rotations are mapped directly
# into the ChArUco frame using the calibrated L-frame rotation. No residual
# camera-vs-mocap rotation S is fitted, and no video frames are generated.
#
# Run cell-by-cell in VS Code/Jupyter, or:
#   uv run python trunkpose/dual_notebooks/08_angle_check.py

import importlib
import io
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import toml
from IPython.display import Image, display
from scipy.spatial.transform import Rotation

try:
    NB_DIR = Path(__file__).resolve().parent
except NameError:  # running cell-by-cell in an interactive/Jupyter kernel
    NB_DIR = Path.cwd()
    if NB_DIR.name != "dual_notebooks":
        NB_DIR = NB_DIR / "trunkpose" / "dual_notebooks"
sys.path.insert(0, str(NB_DIR))

m6 = importlib.import_module("06_trunk_axis")
m7 = importlib.import_module("07_icp_trunk")

if "ipykernel" not in sys.modules:
    try:
        plt.switch_backend("TkAgg")
    except ImportError:
        pass

OUT_PNG = m6.RECORDING_DIR / "angle_check.png"

# Reuse the existing cloud cache automatically when the caller did not select one.
if not os.environ.get("TRUNK_GEOM_CACHE"):
    for candidate in (
        m6.RECORDING_DIR / "geom_cache_v2.pkl",
        m6.RECORDING_DIR / "geom_cache.pkl",
    ):
        if candidate.exists():
            os.environ["TRUNK_GEOM_CACHE"] = str(candidate)
            break


# %% L-frame rotation: mocap world -> ChArUco coordinates
charuco_basis = toml.load(m6.CHARUCO_TOML)
R_lframe_to_mocap = np.asarray(
    charuco_basis["mocap"]["rotation"], dtype=np.float64
)
R_mocap_to_charuco = R_lframe_to_mocap.T

print("R_mocap_to_charuco =")
print(np.array_str(R_mocap_to_charuco, precision=6, suppress_small=True))
print(f"det={np.linalg.det(R_mocap_to_charuco):+.6f}")


# %% Camera geometry + ICP rotations
inp = m7.load_inputs()
frames, n, ts0 = inp["frames"], inp["n"], inp["ts0"]

_R_plane, R_neu, _flex_plane, _lat_plane, _axi_plane, _ = m6.assemble(frames)
icp = m7.run_icp_pass(frames, R_neu, n)
Rb_icp = icp["Rb_icp"]


# %% GPIO synchronization and clean mocap segment
mocap_df = inp["mocap_df"]
m1, m4, m2 = m6.load_trunk_rb(
    str(m6.RECORDING_DIR / f"{m6.RECORDING_NAME}.csv")
)

sync = m6.load_sync_flags(m6.RECORDING_DIR / "cam0_timestamp.msgpack")[:n]
if not (sync == 1).any():
    raise RuntimeError("No GPIO synchronization pulse in cam0 timestamps")

rise = int(np.argmax(sync == 1))
mocap_time = ts0[rise] + (
    mocap_df["seconds"].to_numpy(dtype=np.float64) * 1e6
).astype("timedelta64[us]")

valid_m1 = np.isfinite(m1).all(axis=1)
valid_idx = np.where(valid_m1)[0]
if len(valid_idx) < 2:
    raise RuntimeError("Too few valid mocap trunk-marker samples")

jumps = np.where(
    np.linalg.norm(np.diff(m1[valid_idx], axis=0), axis=1) > m7.SEG_JUMP_M
)[0]
bounds = np.concatenate(([0], jumps + 1, [len(valid_idx)]))
segments = [
    (valid_idx[a], valid_idx[b - 1])
    for a, b in zip(bounds[:-1], bounds[1:])
    if b > a
]
s0, s1 = max(
    segments,
    key=lambda segment: mocap_time[segment[1]] - mocap_time[segment[0]],
)
in_window = (ts0 >= mocap_time[s0]) & (ts0 <= mocap_time[s1])
mocap_index = np.array([m6.nearest_index(mocap_time, t) for t in ts0])

print(
    f"GPIO rise at camera frame {rise}; clean mocap segment "
    f"{(mocap_time[s0] - mocap_time[0]) / np.timedelta64(1, 's'):.2f}-"
    f"{(mocap_time[s1] - mocap_time[0]) / np.timedelta64(1, 's'):.2f}s"
)


# %% Mocap trunk frames expressed directly in ChArUco axes
R_mocap_charuco = [None] * n
for i, j in enumerate(mocap_index):
    if not in_window[i]:
        continue
    R_world = m6.mocap_trunk_frame(m1[j], m4[j], m2[j])
    if R_world is not None:
        R_mocap_charuco[i] = R_mocap_to_charuco @ R_world

common = [
    i
    for i in range(n)
    if Rb_icp[i] is not None and R_mocap_charuco[i] is not None
]
if len(common) < m6.N_NEUTRAL:
    raise RuntimeError("Too few common ICP/mocap frames for neutral zeroing")

neutral_idx = common[: m6.N_NEUTRAL]
R0_icp = Rotation.from_matrix(
    np.asarray([Rb_icp[i] for i in neutral_idx])
).mean().as_matrix()
R0_mocap = Rotation.from_matrix(
    np.asarray([R_mocap_charuco[i] for i in neutral_idx])
).mean().as_matrix()


# %% Relative rotations -> anatomical trunk angles
def empty_series():
    return tuple(np.full(n, np.nan, dtype=np.float64) for _ in range(4))


icp_flex, icp_lat, icp_axi, icp_mag = empty_series()
moc_flex, moc_lat, moc_axi, moc_mag = empty_series()

for i in range(n):
    if not in_window[i]:
        continue

    if Rb_icp[i] is not None:
        Rrel = Rb_icp[i] @ R0_icp.T
        icp_flex[i], icp_lat[i], icp_axi[i] = m6.trunk_angles(
            Rrel @ R_neu, R_neu
        )
        icp_mag[i] = np.degrees(
            np.linalg.norm(Rotation.from_matrix(Rrel).as_rotvec())
        )

    if R_mocap_charuco[i] is not None:
        Rrel = R_mocap_charuco[i] @ R0_mocap.T
        moc_flex[i], moc_lat[i], moc_axi[i] = m6.trunk_angles(
            Rrel @ R_neu, R_neu
        )
        moc_mag[i] = np.degrees(
            np.linalg.norm(Rotation.from_matrix(Rrel).as_rotvec())
        )


def smooth_without_bridging_gaps(values):
    smoothed = m6.smooth_series(values)
    smoothed[~np.isfinite(values)] = np.nan
    return smoothed


icp_angles = tuple(
    smooth_without_bridging_gaps(values)
    for values in (icp_flex, icp_lat, icp_axi)
)
mocap_angles = (moc_flex, moc_lat, moc_axi)

t_seconds = (
    (ts0 - ts0[neutral_idx[0]]) / np.timedelta64(1, "s")
).astype(np.float64)


# %% Agreement summary
def agreement(camera, mocap):
    mask = np.isfinite(camera) & np.isfinite(mocap) & in_window
    if mask.sum() < 3:
        return np.nan, np.nan, int(mask.sum())
    correlation = float(np.corrcoef(camera[mask], mocap[mask])[0, 1])
    rmse = float(np.sqrt(np.mean((camera[mask] - mocap[mask]) ** 2)))
    return correlation, rmse, int(mask.sum())


angle_names = ("flexion", "lateral bending", "axial rotation")
stats = [
    agreement(icp_angles[k], mocap_angles[k])
    for k in range(3)
]

print("\nDirect ChArUco-frame ICP vs mocap agreement (no fitted S):")
for name, (correlation, rmse, count) in zip(angle_names, stats):
    print(f"  {name:16s} r={correlation:+.3f}  RMSE={rmse:6.2f} deg  n={count}")


# %% Plot only -- no video generation
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for k, ax in enumerate(axes):
    correlation, rmse, _count = stats[k]
    ax.plot(
        t_seconds,
        mocap_angles[k],
        color="black",
        linestyle="--",
        linewidth=1.1,
        alpha=0.85,
        label="mocap mapped by L-frame rotation",
    )
    ax.plot(
        t_seconds,
        icp_angles[k],
        color=m6.ANGLE_COLORS[("flexion", "lateral", "axial")[k]],
        linewidth=1.3,
        label="camera ICP",
    )
    ax.axvspan(
        t_seconds[neutral_idx[0]],
        t_seconds[neutral_idx[-1]],
        color="gray",
        alpha=0.12,
        label="neutral window" if k == 0 else None,
    )
    ax.set_ylabel(f"{angle_names[k]}\n(deg)")
    ax.set_title(f"r={correlation:+.3f}, RMSE={rmse:.2f}°", loc="right", fontsize=9)
    ax.grid(alpha=0.3)

axes[0].legend(loc="upper right", fontsize=9)
axes[-1].set_xlabel("time from neutral-window start (s)")
fig.suptitle(
    "Trunk angles: ICP vs mocap directly aligned to the ChArUco frame\n"
    "(Gram-Schmidt L-frame rotation; no fitted residual rotation)"
)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print(f"\nSaved plot -> {OUT_PNG}")

if "ipykernel" in sys.modules:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    display(Image(buffer.getvalue()))
else:
    plt.show()
