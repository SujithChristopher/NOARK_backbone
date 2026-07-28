# %% 09_mocap_angle.py
# Quick REPL check: mocap-only trunk angle, no camera/ICP involved.
# Uses Motive's own solved rigid-body orientation (quaternion columns rb_ang_*)
# directly -- NOT the marker-reconstructed frame 06/07 build for anatomical axes.
# Angle at frame i = rotation from a neutral pose: R_neu.T @ R_i.
#
# Run cell-by-cell (VSCode/Jupyter interactive) or:
#   uv run python trunkpose/dual_notebooks/09_mocap_angle.py

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

try:
    NB_DIR = Path(__file__).resolve().parent
except NameError:  # running cell-by-cell in an interactive/Jupyter kernel
    NB_DIR = Path.cwd()
    if NB_DIR.name != "dual_notebooks":
        NB_DIR = NB_DIR / "trunkpose" / "dual_notebooks"
sys.path.insert(0, str(NB_DIR))
m6 = importlib.import_module("06_trunk_axis")  # side effect: matplotlib.use("Agg")

if "ipykernel" not in sys.modules:
    plt.switch_backend("TkAgg")  # 06's Agg import only matters for a plain script run

N_NEUTRAL = m6.N_NEUTRAL

# %% Load mocap rigid body track
mocap_df, st_time = m6.read_rigid_body_csv(
    str(m6.RECORDING_DIR / f"{m6.RECORDING_NAME}.csv"))
t = mocap_df["seconds"].to_numpy(dtype=np.float64)

# rb_ang_x/y/z/w repeats once per rigid body in the CSV (trunk, left_ew, right_ew
# all share the column names) -- "trunk" is the first rigid body block, so take the
# first occurrence of each name, not a plain name lookup (which would hand back all
# three bodies' columns).
cols = list(mocap_df.columns)
quat_idx = [cols.index(c) for c in ("rb_ang_x", "rb_ang_y", "rb_ang_z", "rb_ang_w")]
quat = mocap_df.iloc[:, quat_idx].to_numpy(dtype=np.float64)
valid = np.isfinite(quat).all(axis=1)
print(f"{valid.sum()}/{len(quat)} frames with a solved rigid body")

R_i = np.full((len(quat), 3, 3), np.nan)
R_i[valid] = Rotation.from_quat(quat[valid]).as_matrix()

# %% Neutral pose + relative angle (R_neu.T @ R_i)
neu_idx = np.where(valid)[0][:N_NEUTRAL]
R_neu = Rotation.from_matrix(R_i[neu_idx]).mean().as_matrix()

angle_deg = np.full(len(quat), np.nan)
rotvec_deg = np.full((len(quat), 3), np.nan)
for i in np.where(valid)[0]:
    Rrel = R_neu.T @ R_i[i]
    rv = Rotation.from_matrix(Rrel).as_rotvec(degrees=True)
    rotvec_deg[i] = rv
    angle_deg[i] = np.linalg.norm(rv)

# %% Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax1.plot(t, angle_deg, color="orange", lw=1.2)
ax1.set_ylabel("|angle| (deg)")
ax1.set_title("mocap trunk rigid body: rotation from neutral (R_neu.T @ R_i)")
ax1.grid(alpha=0.3)

for k, (lab, col) in enumerate(zip(("x", "y", "z"), ("tab:red", "tab:green", "tab:blue"))):
    ax2.plot(t, rotvec_deg[:, k], color=col, lw=1.0, label=lab)
ax2.set_ylabel("rotvec (deg)")
ax2.set_xlabel("time (s)")
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(alpha=0.3)

fig.tight_layout()
if "ipykernel" in sys.modules:
    # bypass backend_inline entirely (it needs %matplotlib inline's display-hook
    # setup to give a figure _repr_png_; switch_backend alone doesn't do that,
    # which is why display(fig) was falling back to plain repr) -- render straight
    # to PNG bytes and show those instead.
    import io
    from IPython.display import Image, display
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    display(Image(buf.getvalue()))
else:
    plt.show()
