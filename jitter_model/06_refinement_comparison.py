# %% [markdown]
# # Static jitter versus distance, one panel per corner refinement
#
# Reads the `static_jitter_vs_distance.csv` written by `05_static_jitter.py` or
# `05b_static_jitter_multitag.py` for each refinement mode and draws them side by
# side with a shared y axis, so the panels compare directly. Nothing is
# recomputed here; run the 05 scripts for every mode first.
#
# ```powershell
# uv run python jitter_model/06_refinement_comparison.py
# ```

# %% Imports and settings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECORDING_DIR = (
    PROJECT_ROOT / "data" / "dome" / "sep18_26" / "dome_static_burst_sep18_26"
)
# "" reads 05_static_jitter.py output (1-4 tags); "_multitag" reads
# 05b_static_jitter_multitag.py output (1-7 tags).
SUFFIX = "_multitag"
# Panel title, folder tag; ordered from no refinement to the slowest mode.
MODES = (
    ("No refinement", "no_refine"),
    ("Contour", "contour_refine"),
    ("Subpix", "subpix_refine"),
    ("AprilTag", "april_refine"),
)
OUTPUT = RECORDING_DIR / f"static_jitter_refinement_comparison{SUFFIX}.png"
STYLE = {1: "--", 2: "-"}
MARKER = {1: "o", 2: "s"}

# %% Figure
tables = {
    title: pd.read_csv(
        RECORDING_DIR / f"static_jitter_{tag}{SUFFIX}" / "static_jitter_vs_distance.csv"
    )
    for title, tag in MODES
}
max_tags = max(int(table["tags"].max()) for table in tables.values())
colours = plt.get_cmap("viridis")

fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), sharex=True, sharey=True, constrained_layout=True)
for axis, (title, table) in zip(axes.ravel(), tables.items()):
    for name, group in table.groupby("condition"):
        cameras, tags = int(group["cameras"].iloc[0]), int(group["tags"].iloc[0])
        group = group.sort_values("distance_center_m")
        axis.plot(
            group["distance_center_m"],
            group["jitter_3d_mm"],
            linestyle=STYLE[cameras],
            marker=MARKER[cameras],
            markersize=4,
            linewidth=1.4,
            color=colours((tags - 1) / max(1, max_tags - 1)),
            label=f"{cameras} cam{'s' if cameras > 1 else ''}, {tags} tag{'s' if tags > 1 else ''}",
        )
    floor = table["mocap_jitter_3d_mm"].median()
    axis.axhline(floor, color="0.4", linestyle=":", linewidth=1.2, label=f"mocap floor {floor:.2f} mm")
    axis.set_yscale("log")
    axis.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,)))
    axis.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2, 0.3, 0.5)))
    formatter = ticker.FuncFormatter(lambda value, _: f"{value:g}")
    axis.yaxis.set_major_formatter(formatter)
    axis.yaxis.set_minor_formatter(formatter)
    axis.grid(True, alpha=0.25, which="both")
    axis.set_title(title)
for axis in axes[-1]:
    axis.set_xlabel("Camera-to-dome distance [m]")
for axis in axes[:, 0]:
    axis.set_ylabel("3D jitter [mm]")
axes[0, 0].legend(loc="upper left", ncols=2, fontsize=7.5)
fig.suptitle(
    f"Static jitter versus distance by corner refinement — {RECORDING_DIR.name}\n"
    "geometric mean per distance bin, bursts common to every condition"
)
fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
print(f"Saved {OUTPUT}")
