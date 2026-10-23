# %% [markdown]
# # Static jitter across the working area, one row per corner refinement
#
# The spatial heatmap of `05b_static_jitter_multitag.py`, redrawn so the corner
# refinements compare directly: one row per refinement, one column per camera
# count, every panel at the same tag count and on one shared colour scale.
#
# Reads the `static_jitter_bursts.csv` that 05b writes for each refinement mode;
# nothing is re-solved here. Run 05b for every mode first:
#
# ```powershell
# foreach ($m in "NONE","CONTOUR","SUBPIX","APRILTAG","EDGE") {
#     $env:CORNER_REFINEMENT = $m; uv run python jitter_model/05b_static_jitter_multitag.py
# }
# uv run python jitter_model/06b_refinement_spatial.py
# ```
#
# `TAG_COUNT` picks how many tags each burst's board fit combines (env
# `TAG_COUNT` overrides it). Left as None it is the largest count every mode
# solved; 05b's own `MAX_TAGS` (env `MAX_TAGS`) caps what is available.

# %% Imports and settings
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects, ticker
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECORDING_DIR = (
    PROJECT_ROOT / "data" / "dome" / "sep18_26" / "dome_static_burst_sep18_26"
)
# Row title, folder tag; ordered from no refinement to the edge fit.
MODES = (
    ("No refinement", "no_refine"),
    ("Contour", "contour_refine"),
    ("Subpix", "subpix_refine"),
    ("AprilTag", "april_refine"),
    ("Edge (Diwakar)", "edge_refine"),
)
CAMERA_COUNTS = (1, 2)
TAG_COUNT = int(os.environ["TAG_COUNT"]) if "TAG_COUNT" in os.environ else None
# Burst positions for the map come from this mode; the reference basis differs
# between modes by well under a millimetre, so one layout serves every panel.
POSITION_MODE = "april_refine"
SPATIAL_GRID = 220
ERROR_COLORMAP = "turbo"
SPATIAL_LABEL_BURSTS = True


def geometric_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.exp(np.mean(np.log(values)))) if len(values) else np.nan


def in_plane_rotation(degrees):
    angle = np.radians(degrees)
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])


# %% Load every mode's per-burst jitter
tables = {}
for title, tag in MODES:
    path = RECORDING_DIR / f"static_jitter_{tag}_multitag" / "static_jitter_bursts.csv"
    if not path.exists():
        print(f"Skipping {title}: no {path.relative_to(PROJECT_ROOT)}")
        continue
    tables[title] = pd.read_csv(path)
if not tables:
    raise FileNotFoundError("No 05b burst tables found; run 05b first")

if TAG_COUNT is None:
    TAG_COUNT = min(int(table["tags"].max()) for table in tables.values())
print(f"Tag count per board fit: {TAG_COUNT}")

# %% Shared squared-up layout, as in 05b
position_title = next(t for t, tag in MODES if tag == POSITION_MODE and t in tables)
burst_positions = (
    tables[position_title].groupby("burst")[["pos_x_m", "pos_z_m"]].mean()
)
centred = burst_positions.to_numpy() - burst_positions.to_numpy().mean(axis=0)
principal = np.linalg.svd(centred, full_matrices=False)[2][0]
rotation_deg = -np.degrees(np.arctan2(principal[1], principal[0]))
trial = burst_positions.to_numpy() @ in_plane_rotation(rotation_deg)
if (0 - trial[:, 1].min()) > 0.5 * np.ptp(trial[:, 1]):
    rotation_deg -= 180.0
plane = pd.DataFrame(
    1000.0 * (burst_positions.to_numpy() @ in_plane_rotation(rotation_deg)),
    index=burst_positions.index,
    columns=["x", "z"],
)

grid_x = np.linspace(plane["x"].min(), plane["x"].max(), SPATIAL_GRID)
grid_z = np.linspace(plane["z"].min(), plane["z"].max(), SPATIAL_GRID)
mesh_x, mesh_z = np.meshgrid(grid_x, grid_z)

panels = {}
for title, table in tables.items():
    for camera_count in CAMERA_COUNTS:
        subset = table.loc[
            (table["cameras"] == camera_count)
            & (table["tags"] == TAG_COUNT)
            & np.isfinite(table["jitter_3d_mm"])
            & (table["jitter_3d_mm"] > 0)
            & table["burst"].isin(plane.index)
        ]
        panels[title, camera_count] = subset
all_values = pd.concat([p["jitter_3d_mm"] for p in panels.values()])
norm = LogNorm(vmin=all_values.min(), vmax=all_values.max())
mocap_floor = np.nanmedian(
    np.concatenate([t["mocap_jitter_3d_mm"].to_numpy() for t in tables.values()])
)

# %% Figure
panel_width = 4.6
panel_height = panel_width * np.ptp(grid_z) / np.ptp(grid_x)
fig, axes = plt.subplots(
    len(tables),
    len(CAMERA_COUNTS),
    figsize=(panel_width * len(CAMERA_COUNTS) + 1.2, panel_height * len(tables) + 1.0),
    sharex=True,
    sharey=True,
    constrained_layout=True,
    squeeze=False,
)
mesh = None
for row, title in enumerate(tables):
    for column, camera_count in enumerate(CAMERA_COUNTS):
        axis = axes[row, column]
        subset = panels[title, camera_count]
        points = plane.loc[subset["burst"]].to_numpy()
        if len(subset) >= 4:
            surface = griddata(
                points,
                np.log10(subset["jitter_3d_mm"].to_numpy()),
                (mesh_x, mesh_z),
                method="linear",
            )
            mesh = axis.pcolormesh(
                mesh_x, mesh_z, 10.0**surface, norm=norm, cmap=ERROR_COLORMAP, shading="auto"
            )
        axis.scatter(
            points[:, 0], points[:, 1], s=16, facecolor="none", edgecolor="0.15", linewidth=0.7, zorder=3
        )
        if SPATIAL_LABEL_BURSTS:
            for (x, z), burst in zip(points, subset["burst"]):
                axis.annotate(
                    str(int(burst)),
                    (x, z),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                    color="black",
                    path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")],
                    zorder=5,
                )
        axis.scatter([0], [0], marker="*", s=150, facecolor="white", edgecolor="black", linewidth=1.0, zorder=4)
        axis.set_xlim(grid_x[0], grid_x[-1])
        axis.set_ylim(grid_z[0], grid_z[-1])
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.2)
        cams = f"{camera_count} cam{'s' if camera_count > 1 else ''}"
        axis.set_title(
            f"{title} — {cams}, {TAG_COUNT} tags\n"
            f"geo-mean {geometric_mean(subset['jitter_3d_mm']):.3f} mm, {len(subset)} bursts",
            fontsize=9.5,
        )
for axis in axes[-1]:
    axis.set_xlabel("Table across [mm]")
for axis in axes[:, 0]:
    axis.set_ylabel("Away from reference [mm]")
if mesh is not None:
    colorbar = fig.colorbar(mesh, ax=axes, label="3D jitter [mm]", shrink=0.6)
    colorbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    colorbar.ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    colorbar.ax.axhline(mocap_floor, color="white", linewidth=2)
    colorbar.ax.axhline(mocap_floor, color="black", linewidth=1, linestyle="--")
fig.suptitle(
    f"Static jitter across the working area by corner refinement — {RECORDING_DIR.name}\n"
    f"{TAG_COUNT}-tag board fit per burst; log-space interpolation between bursts (rings); "
    f"star = reference burst; dashed on colour bar = mocap floor {mocap_floor:.3f} mm"
)
OUTPUT = RECORDING_DIR / f"static_jitter_refinement_spatial_M{TAG_COUNT}.png"
fig.savefig(OUTPUT, dpi=170, bbox_inches="tight")
print(f"Saved {OUTPUT}")

# Same numbers as a table: geometric mean per refinement and camera count, on the
# bursts every panel shares so no row is flattered by skipping hard positions.
common = set.intersection(*(set(p["burst"]) for p in panels.values()))
summary = pd.DataFrame(
    {
        f"{c} cam": {
            title: geometric_mean(panels[title, c].loc[panels[title, c]["burst"].isin(common), "jitter_3d_mm"])
            for title in tables
        }
        for c in CAMERA_COUNTS
    }
)
print(f"\nGeometric-mean 3D jitter [mm] on the {len(common)} common bursts, {TAG_COUNT} tags:")
print(summary.round(3).to_string())

# %%
plt.show()
