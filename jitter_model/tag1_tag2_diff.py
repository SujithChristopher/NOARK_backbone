# %% [markdown]
# # Tag 1 -> tag 2 offset, rigid body vs CAD
#
# Prints the centre-to-centre vector from the apex tag (1) to the first ring-30
# tag (2), as solved by `02_rigidbody_calib.py` and as given by the CAD dome.
# No alignment is applied: the solved rig puts tag 1 at the origin, and so does
# CAD, so the two vectors are directly comparable. The distance is also
# rotation-free.
#
# ```powershell
# uv run python jitter_model/tag1_tag2_diff.py
# ```

# %% Imports
from pathlib import Path

import numpy as np
import toml

from jitter_model.common import build_tag_rig

# %% Settings
try:
    NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:  # running cell-by-cell in an interactive kernel
    NOTEBOOK_DIR = Path.cwd()
    if NOTEBOOK_DIR.name != "jitter_model":
        NOTEBOOK_DIR = NOTEBOOK_DIR / "jitter_model"
PROJECT_ROOT = NOTEBOOK_DIR.parent
RIGIDBODY_TOML = (
    PROJECT_ROOT / "data" / "dome" / "sep18_26" / "dome_rb_def"
    / "rigidbody_calibration.toml"
)

TAG_A, TAG_B = 1, 2
MM = 1000.0

# CAD tag 1 -> tag 2 offset, taken straight off the model (mm): tag 2 is on the
# +x axis of the 30 degree ring, below the apex along -z.
CAD_RADIAL_MM = 56.61
CAD_DEPTH_MM = 15.17


# %% Compare
rig = build_tag_rig(toml.load(RIGIDBODY_TOML))
solved_centers = rig.centers()

solved = solved_centers[TAG_B] - solved_centers[TAG_A]
cad = np.array([CAD_RADIAL_MM, 0.0, -CAD_DEPTH_MM]) / MM
diff = solved - cad

print(f"Rigid body: {RIGIDBODY_TOML}")
print(f"Vector tag {TAG_A} -> tag {TAG_B} (mm)")
print(f"{'':>8} {'x':>9} {'y':>9} {'z':>9} {'|v|':>9} {'radial':>9} {'depth':>9}")
for label, vec in (("CAD", cad), ("solved", solved), ("solved-CAD", diff)):
    print(
        f"{label:>10} {MM * vec[0]:9.3f} {MM * vec[1]:9.3f} {MM * vec[2]:9.3f} "
        f"{MM * np.linalg.norm(vec):9.3f} {MM * np.hypot(vec[0], vec[1]):9.3f} "
        f"{-MM * vec[2]:9.3f}"
    )

distance_error = MM * (np.linalg.norm(solved) - np.linalg.norm(cad))
angle = np.degrees(
    np.arccos(
        np.clip(
            solved @ cad / (np.linalg.norm(solved) * np.linalg.norm(cad)), -1.0, 1.0
        )
    )
)
print(f"\nDistance error (solved - CAD): {distance_error:+.3f} mm")
print(f"Angle between the two vectors: {angle:.3f} deg")

# %%
