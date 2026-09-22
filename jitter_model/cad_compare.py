# %% [markdown]
# # CAD model vs solved rigid body
#
# The dome's tag centres come from a CAD model: a hemisphere of radius
# 113.228 mm with tag rings at polar angles 30, 60, and 90 degrees. That single
# radius reproduces all three CAD measurements taken off the model,
#
# | ring | depth (mm) | radial (mm) |
# |------|-----------:|------------:|
# | 30   |      15.17 |       56.61 |
# | 60   |      56.61 |      98.058 |
# | 90   |    113.228 |     113.228 |
#
# since `R = (radial**2 + depth**2) / (2 * depth)` gives 113.2 for each.
#
# This script compares that ideal geometry against the rigid body solved by
# `02_rigidbody_calib.py`, three ways:
#
# 1. Kabsch rigid alignment, plus a separate scale-fitting pass that reports
#    whether the assumed 50 mm tag size is right.
# 2. Depth / radial decomposition, matching how the CAD numbers were measured.
# 3. All 210 pairwise centre distances, which need no alignment at all.
#
# Every printed table is also written next to the recording as a CSV.
#
# Diagnostic only - nothing downstream reads its outputs.
#
# Run from the repository root:
#
# ```powershell
# uv run python jitter_model/cad_compare.py
# ```

# %% Imports
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml

from jitter_model.common import build_tag_rig

# %% Paths and settings
try:
    NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:  # running cell-by-cell in an interactive kernel
    NOTEBOOK_DIR = Path.cwd()
    if NOTEBOOK_DIR.name != "jitter_model":
        NOTEBOOK_DIR = NOTEBOOK_DIR / "jitter_model"

PROJECT_ROOT = NOTEBOOK_DIR.parent
RECORDING_DIR = PROJECT_ROOT / "data" / "dome" / "sep18_26" / "dome_rb_def"
RIGIDBODY_TOML = RECORDING_DIR / "rigidbody_calibration.toml"
OUTPUT_FIGURE = RECORDING_DIR / "cad_vs_rigidbody.png"
MARKER_TABLE_CSV = RECORDING_DIR / "cad_vs_rigidbody_markers.csv"
PAIR_TABLE_CSV = RECORDING_DIR / "cad_vs_rigidbody_pairs.csv"

# Sphere the CAD dome is built on.
CAD_RADIUS_M = 0.113228

# Which tag sits where. The polar angle is the ring; the azimuth is read off
# the physical dome, with 0 degrees along +x and 90 degrees along +y in the
# reference tag's frame. Printed below so it can be checked against the model.
CAD_LAYOUT = {
    1: (0.0, 0.0),
    2: (30.0, 0.0),
    5: (30.0, 90.0),
    3: (30.0, 180.0),
    4: (30.0, 270.0),
    8: (60.0, 0.0),
    7: (60.0, 45.0),
    6: (60.0, 90.0),
    13: (60.0, 135.0),
    12: (60.0, 180.0),
    11: (60.0, 225.0),
    10: (60.0, 270.0),
    9: (60.0, 315.0),
    16: (90.0, 0.0),
    15: (90.0, 45.0),
    14: (90.0, 90.0),
    21: (90.0, 135.0),
    20: (90.0, 180.0),
    19: (90.0, 225.0),
    18: (90.0, 270.0),
    17: (90.0, 315.0),
}

# Tags whose facet is not radial to the dome sphere, so their normal has to be
# given separately: the diagonals of the 60 degree ring are tilted 67.792
# degrees off the apex tag in CAD. Every other tag's normal is its position
# polar angle.
CAD_NORMAL_POLAR = {
    7: 67.792,
    9: 67.792,
    11: 67.792,
    13: 67.792,
}

# Centres that are not on the dome sphere either, as (radial_mm, depth_mm).
# `None` keeps whatever the sphere gives. CAD quotes these facets as an equal x
# and y offset, which on a 45 degree diagonal is a radial offset of v * sqrt(2).
RING60_DIAGONAL_XY_MM = 64.146
RING90_DIAGONAL_XY_MM = 71.740
CAD_POSITION_OVERRIDE = {
    7: (RING60_DIAGONAL_XY_MM * np.sqrt(2.0), None),
    9: (RING60_DIAGONAL_XY_MM * np.sqrt(2.0), None),
    11: (RING60_DIAGONAL_XY_MM * np.sqrt(2.0), None),
    13: (RING60_DIAGONAL_XY_MM * np.sqrt(2.0), None),
    15: (RING90_DIAGONAL_XY_MM * np.sqrt(2.0), None),
    17: (RING90_DIAGONAL_XY_MM * np.sqrt(2.0), None),
    19: (RING90_DIAGONAL_XY_MM * np.sqrt(2.0), None),
    21: (RING90_DIAGONAL_XY_MM * np.sqrt(2.0), None),
}

MM = 1000.0


# %% CAD geometry
def cad_centers(
    radius_m=CAD_RADIUS_M, layout=CAD_LAYOUT, override=CAD_POSITION_OVERRIDE
):
    """Ideal tag centres on the CAD hemisphere, in the reference tag's frame.

    The apex tag sits at the origin and the dome falls away along -z, which is
    the convention `02_rigidbody_calib.py` solves in. `override` replaces the
    radial offset, the depth, or both for facets CAD does not put on the sphere.
    """
    centers = {}
    for marker_id, (polar_deg, azimuth_deg) in layout.items():
        polar = np.radians(polar_deg)
        azimuth = np.radians(azimuth_deg)
        radial = radius_m * np.sin(polar)
        depth = radius_m * (1.0 - np.cos(polar))

        radial_override, depth_override = override.get(marker_id, (None, None))
        if radial_override is not None:
            radial = radial_override / MM
        if depth_override is not None:
            depth = depth_override / MM

        centers[marker_id] = np.array(
            [radial * np.cos(azimuth), radial * np.sin(azimuth), -depth]
        )
    return centers


def cad_normals(layout=CAD_LAYOUT, normal_polar=CAD_NORMAL_POLAR):
    """Each CAD facet's outward normal in the reference tag's frame.

    A tag that sits radially on the dome has a normal at its own position polar
    angle; `CAD_NORMAL_POLAR` overrides that for the facets that do not.
    """
    normals = {}
    for marker_id, (polar_deg, azimuth_deg) in layout.items():
        polar = np.radians(normal_polar.get(marker_id, polar_deg))
        azimuth = np.radians(azimuth_deg)
        normals[marker_id] = np.array(
            [
                np.sin(polar) * np.cos(azimuth),
                np.sin(polar) * np.sin(azimuth),
                np.cos(polar),
            ]
        )
    return normals


def outward_directions(points, radius_m=CAD_RADIUS_M):
    """The dome's outward radial direction at each centre.

    The sphere is centred a radius below the apex, so this points away from the
    dome's interior everywhere, including at the equator.
    """
    radial = points - np.array([0.0, 0.0, -radius_m])
    return radial / np.linalg.norm(radial, axis=1, keepdims=True)


def tilt_and_azimuth(vectors, reference, outward):
    """Each vector's angle off `reference`, and the azimuth of its tilt, in degrees.

    A tag is defined with either facing, so each normal is first flipped to face
    outward. Folding against `reference` instead would be a coin flip at the
    equator, where a normal is perpendicular to the apex axis.
    """
    signs = np.sign(np.sum(vectors * outward, axis=1))
    signs[signs == 0.0] = 1.0
    facing = vectors * signs[:, None]
    tilt = np.degrees(np.arccos(np.clip(facing @ reference, -1.0, 1.0)))
    azimuth = np.degrees(np.arctan2(facing[:, 1], facing[:, 0])) % 360.0
    return tilt, azimuth


# %% Alignment
def kabsch(source, target, with_scale=False):
    """Rigid, or optionally similarity, transform taking `source` onto `target`."""
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centred = source - source_centroid
    target_centred = target - target_centroid

    u, singular, vt = np.linalg.svd(source_centred.T @ target_centred)
    reflection = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = (u @ reflection @ vt).T

    scale = 1.0
    if with_scale:
        scale = float(np.sum(singular * np.diag(reflection))) / float(
            np.sum(source_centred**2)
        )
    translation = target_centroid - scale * rotation @ source_centroid
    return rotation, translation, scale


def apply_transform(points, rotation, translation, scale=1.0):
    return scale * points @ rotation.T + translation


# %% Load the solved rigid body
rigidbody = toml.load(RIGIDBODY_TOML)
rig = build_tag_rig(rigidbody)
solved_centers = rig.centers()

cad = cad_centers()
marker_ids = sorted(set(cad) & set(solved_centers))
missing = sorted(set(cad) ^ set(solved_centers))
if missing:
    print(f"Tags in only one of CAD/solve, skipped: {missing}")

cad_points = np.array([cad[m] for m in marker_ids])
solved_points = np.array([solved_centers[m] for m in marker_ids])

print(f"Rigid body:   {RIGIDBODY_TOML}")
print(f"Reference id: {rig.reference_id}   tag size: {rig.tag_size_m * MM:.1f} mm")
print(f"CAD radius:   {CAD_RADIUS_M * MM:.3f} mm   tags compared: {len(marker_ids)}")
print("\nCAD layout (azimuth degrees per ring):")
for polar in sorted({p for p, _a in CAD_LAYOUT.values()}):
    ring = sorted((a, m) for m, (p, a) in CAD_LAYOUT.items() if p == polar)
    joined = "  ".join(f"{m}@{a:.0f}" for a, m in ring)
    print(f"  theta={polar:5.1f}:  {joined}")


# %% Rigid alignment and per-marker residuals
rotation, translation, _scale = kabsch(solved_points, cad_points)
aligned = apply_transform(solved_points, rotation, translation)
residuals = aligned - cad_points
residual_norms = np.linalg.norm(residuals, axis=1)
rigid_rmse_mm = MM * float(np.sqrt(np.mean(residual_norms**2)))

_r_scaled, _t_scaled, fitted_scale = kabsch(solved_points, cad_points, with_scale=True)

print("\n=== Rigid (Kabsch) alignment, solved -> CAD ===")
print(f"RMSE: {rigid_rmse_mm:.3f} mm   max: {MM * residual_norms.max():.3f} mm")
print(
    f"Similarity fit scale: {fitted_scale:.5f} "
    f"(implies effective tag size {rig.tag_size_m * fitted_scale * MM:.2f} mm)"
)
print(f"\n{'tag':>4} {'dx':>8} {'dy':>8} {'dz':>8} {'|d|':>8}   (mm)")
for marker_id, residual, norm in zip(marker_ids, residuals, residual_norms):
    print(
        f"{marker_id:>4} {MM * residual[0]:8.2f} {MM * residual[1]:8.2f} "
        f"{MM * residual[2]:8.2f} {MM * norm:8.2f}"
    )


# %% Depth and radial decomposition
def depth_radial(points):
    return -points[:, 2], np.hypot(points[:, 0], points[:, 1])


cad_depth, cad_radial = depth_radial(cad_points)
# Compared in the aligned frame so the dome axis is shared: without alignment a
# small tilt in the solved reference tag leaks into both columns.
solved_depth, solved_radial = depth_radial(aligned)

print("\n=== Depth / radial vs CAD (mm) ===")
print(
    f"{'tag':>4} {'theta':>6} {'depth_cad':>10} {'depth_rb':>9} {'d_depth':>8} "
    f"{'rad_cad':>8} {'rad_rb':>8} {'d_rad':>8}"
)
for index, marker_id in enumerate(marker_ids):
    polar = CAD_LAYOUT[marker_id][0]
    print(
        f"{marker_id:>4} {polar:6.1f} {MM * cad_depth[index]:10.3f} "
        f"{MM * solved_depth[index]:9.3f} "
        f"{MM * (solved_depth[index] - cad_depth[index]):8.3f} "
        f"{MM * cad_radial[index]:8.3f} {MM * solved_radial[index]:8.3f} "
        f"{MM * (solved_radial[index] - cad_radial[index]):8.3f}"
    )


# %% Pairwise centre distances
pairs = list(itertools.combinations(range(len(marker_ids)), 2))
cad_distances = np.array(
    [np.linalg.norm(cad_points[a] - cad_points[b]) for a, b in pairs]
)
solved_distances = np.array(
    [np.linalg.norm(solved_points[a] - solved_points[b]) for a, b in pairs]
)
distance_errors = MM * (solved_distances - cad_distances)

print(f"\n=== Pairwise centre distances, {len(pairs)} pairs (alignment-free) ===")
print(
    f"mean {distance_errors.mean():+.3f} mm   "
    f"RMSE {np.sqrt(np.mean(distance_errors**2)):.3f} mm   "
    f"max |err| {np.abs(distance_errors).max():.3f} mm"
)
print(f"\n{'pair':>9} {'cad':>9} {'solved':>9} {'error':>8}   (mm)")
for rank in np.argsort(-np.abs(distance_errors))[:10]:
    a, b = pairs[rank]
    print(
        f"{marker_ids[a]:>4}-{marker_ids[b]:<4} {MM * cad_distances[rank]:9.3f} "
        f"{MM * solved_distances[rank]:9.3f} {distance_errors[rank]:8.3f}"
    )


# %% Implied sphere radius per tag
# A tag at depth d and radial offset r sits on a sphere of radius
# (r**2 + d**2) / (2 * d) through the apex. Grouping by azimuth family shows
# whether every tag shares the one CAD sphere or the dome has nested surfaces.
AXIS_FAMILY = "axis"
DIAGONAL_FAMILY = "diagonal"


def azimuth_family(marker_id):
    azimuth = CAD_LAYOUT[marker_id][1]
    return AXIS_FAMILY if azimuth % 90.0 == 0.0 else DIAGONAL_FAMILY


families = np.array([azimuth_family(m) for m in marker_ids])
# The apex has zero depth, so its implied radius is undefined.
with np.errstate(divide="ignore", invalid="ignore"):
    implied_radius = (solved_radial**2 + solved_depth**2) / (2.0 * solved_depth)

print("\n=== Implied sphere radius, solved (mm) ===")
for family in (AXIS_FAMILY, DIAGONAL_FAMILY):
    selected = (
        (families == family) & np.isfinite(implied_radius) & (solved_depth > 1e-3)
    )
    radii = MM * implied_radius[selected]
    print(
        f"{family:>8}: n={selected.sum():2d}  "
        f"mean {radii.mean():8.3f}  sd {radii.std(ddof=1):6.3f}  "
        f"range {radii.min():8.3f} .. {radii.max():8.3f}"
    )
print(f"{'CAD':>8}: {CAD_RADIUS_M * MM:8.3f}")

# The similarity scale above is dominated by the diagonals; the axis tags alone
# say whether the 50 mm tag size itself is right.
axis_mask = families == AXIS_FAMILY
_r_axis, _t_axis, axis_scale = kabsch(
    solved_points[axis_mask], cad_points[axis_mask], with_scale=True
)
axis_rotation, axis_translation, _s = kabsch(
    solved_points[axis_mask], cad_points[axis_mask]
)
axis_residuals = (
    apply_transform(solved_points[axis_mask], axis_rotation, axis_translation)
    - cad_points[axis_mask]
)
axis_rmse_mm = MM * float(np.sqrt(np.mean(np.sum(axis_residuals**2, axis=1))))
print(
    f"\nAxis tags only: rigid RMSE {axis_rmse_mm:.3f} mm, "
    f"similarity scale {axis_scale:.5f} "
    f"(effective tag size {rig.tag_size_m * axis_scale * MM:.2f} mm)"
)


# %% Facet orientation
# Rotated into the CAD frame by the same rigid alignment used for the centres,
# so tilt and azimuth are measured against one dome axis.
solved_normals_rig = np.array([rig.normals()[m] for m in marker_ids])
solved_normals = solved_normals_rig @ rotation.T
cad_normal_map = cad_normals()
cad_normal_vectors = np.array([cad_normal_map[m] for m in marker_ids])

apex_axis = np.array([0.0, 0.0, 1.0])
cad_outward = outward_directions(cad_points)
cad_tilt, cad_normal_azimuth = tilt_and_azimuth(
    cad_normal_vectors, apex_axis, cad_outward
)
rb_tilt, rb_normal_azimuth = tilt_and_azimuth(solved_normals, apex_axis, cad_outward)
tilt_error = rb_tilt - cad_tilt
azimuth_error = (rb_normal_azimuth - cad_normal_azimuth + 180.0) % 360.0 - 180.0
# The apex tag defines the axis, so its normal azimuth is undefined.
azimuth_error[np.asarray(marker_ids) == rig.reference_id] = np.nan

# Where the centre actually sits on the CAD sphere, which is what the facet
# normal would be if the tag were radial. Differing from the measured tilt is
# what marks a facet as non-radial.
position_polar = np.degrees(np.arctan2(solved_radial, CAD_RADIUS_M - solved_depth))

print("\n=== Facet orientation vs CAD (degrees) ===")
print(
    f"{'tag':>4} {'tilt_cad':>9} {'tilt_rb':>8} {'d_tilt':>7} "
    f"{'az_cad':>7} {'az_rb':>7} {'d_az':>7} {'pos_polar':>10}"
)
for index, marker_id in enumerate(marker_ids):
    print(
        f"{marker_id:>4} {cad_tilt[index]:9.3f} {rb_tilt[index]:8.3f} "
        f"{tilt_error[index]:7.3f} {cad_normal_azimuth[index]:7.1f} "
        f"{rb_normal_azimuth[index]:7.1f} {azimuth_error[index]:7.1f} "
        f"{position_polar[index]:10.3f}"
    )

for family in (AXIS_FAMILY, DIAGONAL_FAMILY):
    selected = families == family
    print(
        f"{family:>8}: tilt error mean {tilt_error[selected].mean():+7.3f} "
        f"sd {tilt_error[selected].std(ddof=1):6.3f} "
        f"max |err| {np.abs(tilt_error[selected]).max():6.3f}"
    )


# %% Tables on disk
# The printed blocks above are one table each, split for readability; these two
# CSVs carry the same numbers in full so they can be read back or pasted into a
# report without re-running the script.
marker_table = pd.DataFrame(
    {
        "tag": marker_ids,
        "theta_deg": [CAD_LAYOUT[m][0] for m in marker_ids],
        "azimuth_deg": [CAD_LAYOUT[m][1] for m in marker_ids],
        "family": families,
        "cad_x_mm": MM * cad_points[:, 0],
        "cad_y_mm": MM * cad_points[:, 1],
        "cad_z_mm": MM * cad_points[:, 2],
        "rb_x_mm": MM * aligned[:, 0],
        "rb_y_mm": MM * aligned[:, 1],
        "rb_z_mm": MM * aligned[:, 2],
        "dx_mm": MM * residuals[:, 0],
        "dy_mm": MM * residuals[:, 1],
        "dz_mm": MM * residuals[:, 2],
        "residual_mm": MM * residual_norms,
        "depth_cad_mm": MM * cad_depth,
        "depth_rb_mm": MM * solved_depth,
        "depth_error_mm": MM * (solved_depth - cad_depth),
        "radial_cad_mm": MM * cad_radial,
        "radial_rb_mm": MM * solved_radial,
        "radial_error_mm": MM * (solved_radial - cad_radial),
        "implied_sphere_mm": MM * implied_radius,
        "position_polar_deg": position_polar,
        "tilt_cad_deg": cad_tilt,
        "tilt_rb_deg": rb_tilt,
        "tilt_error_deg": tilt_error,
        "normal_azimuth_cad_deg": cad_normal_azimuth,
        "normal_azimuth_rb_deg": rb_normal_azimuth,
        "normal_azimuth_error_deg": azimuth_error,
    }
)
marker_table.to_csv(MARKER_TABLE_CSV, index=False, float_format="%.4f")

pair_table = pd.DataFrame(
    {
        "tag_a": [marker_ids[a] for a, _b in pairs],
        "tag_b": [marker_ids[b] for _a, b in pairs],
        "family_a": [families[a] for a, _b in pairs],
        "family_b": [families[b] for _a, b in pairs],
        "cad_mm": MM * cad_distances,
        "rb_mm": MM * solved_distances,
        "error_mm": distance_errors,
    }
)
pair_table.to_csv(PAIR_TABLE_CSV, index=False, float_format="%.4f")

print(f"\nSaved marker table -> {MARKER_TABLE_CSV}")
print(f"Saved pair table   -> {PAIR_TABLE_CSV}")


# %% Figure
figure = plt.figure(figsize=(15, 10))

axis_3d = figure.add_subplot(2, 2, 1, projection="3d")
axis_3d.scatter(
    MM * cad_points[:, 0],
    MM * cad_points[:, 1],
    MM * cad_points[:, 2],
    facecolors="none",
    edgecolors="tab:blue",
    s=45,
    label="CAD",
)
axis_3d.scatter(
    MM * aligned[:, 0],
    MM * aligned[:, 1],
    MM * aligned[:, 2],
    color="tab:red",
    s=18,
    label="solved (aligned)",
)
for cad_point, aligned_point in zip(cad_points, aligned):
    segment = MM * np.vstack([cad_point, aligned_point])
    axis_3d.plot(
        segment[:, 0], segment[:, 1], segment[:, 2], color="0.4", linewidth=0.8
    )
for marker_id, cad_point in zip(marker_ids, cad_points):
    axis_3d.text(*(MM * cad_point), str(marker_id), fontsize=7, color="0.3")
axis_3d.set_xlabel("x (mm)")
axis_3d.set_ylabel("y (mm)")
axis_3d.set_zlabel("z (mm)")
axis_3d.set_title(f"CAD vs solved, RMSE {rigid_rmse_mm:.2f} mm")
axis_3d.legend(loc="upper left", fontsize=8)

axis_bars = figure.add_subplot(2, 2, 2)
ring_colors = {0.0: "0.5", 30.0: "tab:green", 60.0: "tab:orange", 90.0: "tab:purple"}
axis_bars.bar(
    range(len(marker_ids)),
    MM * residual_norms,
    color=[ring_colors[CAD_LAYOUT[m][0]] for m in marker_ids],
)
axis_bars.set_xticks(range(len(marker_ids)))
axis_bars.set_xticklabels([str(m) for m in marker_ids], fontsize=7)
axis_bars.set_xlabel("tag id")
axis_bars.set_ylabel("|residual| (mm)")
axis_bars.set_title("Per-tag error after rigid alignment")
axis_bars.grid(axis="y", alpha=0.3)
axis_bars.legend(
    handles=[
        plt.Line2D([], [], color=color, linewidth=6, label=f"theta={polar:.0f}")
        for polar, color in sorted(ring_colors.items())
    ],
    fontsize=8,
)

axis_tilt = figure.add_subplot(2, 2, 3)
axis_tilt.bar(
    range(len(marker_ids)),
    tilt_error,
    color=[ring_colors[CAD_LAYOUT[m][0]] for m in marker_ids],
)
axis_tilt.axhline(0.0, color="0.3", linewidth=1)
axis_tilt.set_xticks(range(len(marker_ids)))
axis_tilt.set_xticklabels([str(m) for m in marker_ids], fontsize=7)
axis_tilt.set_xlabel("tag id")
axis_tilt.set_ylabel("tilt error (deg)")
axis_tilt.set_title("Facet normal tilt, solved - CAD")
axis_tilt.grid(axis="y", alpha=0.3)

axis_hist = figure.add_subplot(2, 2, 4)
axis_hist.hist(distance_errors, bins=40, color="tab:blue", alpha=0.8)
axis_hist.axvline(0.0, color="0.3", linewidth=1)
axis_hist.set_xlabel("solved - CAD distance (mm)")
axis_hist.set_ylabel("pairs")
axis_hist.set_title(f"Pairwise distance error, {len(pairs)} pairs")
axis_hist.grid(axis="y", alpha=0.3)

figure.tight_layout()
figure.savefig(OUTPUT_FIGURE, dpi=150)
print(f"\nSaved figure -> {OUTPUT_FIGURE}")
plt.show()
