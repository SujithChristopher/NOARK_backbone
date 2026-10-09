"""Figure: T1 hold errors (pooled registration) per method, X and Z, plus error maps."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_T1 import METHODS, apply, fit
from common import OUT

df = pd.read_csv(OUT / "T1_holds.csv")
res = pd.read_csv(OUT / "T1_errors.csv")
fig, ax = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
for j, m in enumerate(METHODS):
    cw, mw = apply(df, m, fit(df, m))
    e = (cw - mw) * 1000
    a = ax[0, j]
    a.quiver(mw[:, 0] * 1000, mw[:, 2] * 1000, e[:, 0], e[:, 2], np.hypot(e[:, 0], e[:, 2]), angles="xy",
             scale_units="xy", scale=0.05, cmap="viridis")
    a.set_title(f"{m}: error x20 (arrows)")
    a.set_xlabel("mocap X (mm)")
    a.set_ylabel("mocap Z (mm)")
    a.set_aspect("equal")
    b = ax[1, j]
    b.scatter(mw[:, 2] * 1000, e[:, 2], s=10, label="Z error vs Z")
    b.scatter(mw[:, 0] * 1000, e[:, 0], s=10, label="X error vs X")
    b.axhline(0, c="k", lw=0.5)
    ho = res[(res.method == m) & (res.kind == "held-out")]
    b.set_title(f"held-out RMSE X {ho.x_rmse_mm.mean():.2f}  Z {ho.z_rmse_mm.mean():.2f} mm")
    b.set_xlabel("mocap position (mm)")
    b.set_ylabel("error (mm)")
    b.set_ylim(-5, 5)
    b.legend(fontsize=8)
fig.suptitle("T1 grid holds: camera grip point vs mocap (registration pooled r1+r2), T0 jig except joint_logged")
fig.savefig(OUT / "T1_errors.png", dpi=110)
