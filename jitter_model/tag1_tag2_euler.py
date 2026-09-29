# %% [markdown]
# # Tag 1 -> tag 2 relative rotation, rigid body vs CAD
#
# Relative rotation of tag 2 in tag 1's frame, `R_12 = R_1.T @ R_2`, from the
# solved rigid body and from CAD, as Euler angles, plus the rotation between the
# two, `R_err = R_cad.T @ R_solved`.
#
# CAD only fixes each facet normal, not the in-plane roll. Tag 2 is taken as tag
# 1 tilted 30 degrees about y (x along the tag's radial direction), so a roll
# about z in the result means the solved tag's in-plane axes differ from that
# assumption.
#
# ```powershell
# uv run python jitter_model/tag1_tag2_euler.py
# ```

# %% Imports
from pathlib import Path

import numpy as np
import toml
from scipy.spatial.transform import Rotation

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
CAD_TILT_DEG = 30.0
EULER_ORDER = "xyz"


# %% Compare
rig = build_tag_rig(toml.load(RIGIDBODY_TOML))
rot_a = rig.marker_to_reference[TAG_A][0]
rot_b = rig.marker_to_reference[TAG_B][0]

solved_rel = Rotation.from_matrix(rot_a.T @ rot_b)
cad_rel = Rotation.from_euler("y", CAD_TILT_DEG, degrees=True)
error = cad_rel.inv() * solved_rel

print(f"Euler order '{EULER_ORDER}' (deg), tag {TAG_B} in tag {TAG_A} frame")
print(f"{'':>10} {'rx':>8} {'ry':>8} {'rz':>8} {'angle':>8}")
for label, rot in (("CAD", cad_rel), ("solved", solved_rel), ("error", error)):
    rx, ry, rz = rot.as_euler(EULER_ORDER, degrees=True)
    print(f"{label:>10} {rx:8.3f} {ry:8.3f} {rz:8.3f} {np.degrees(rot.magnitude()):8.3f}")
print("\nsolved R_12:\n", np.round(solved_rel.as_matrix(), 4))

# %%
