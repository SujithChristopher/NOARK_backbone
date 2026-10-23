"""Per-sample board pose + grip point (cam0 frame) for cam0-only, cam1-only and joint stereo, T0 jig.

Writes out/poses/<source>/<take>.csv. Usage: python compute_poses.py SOURCE [take ...]
SOURCE: a VARIANTS key = (corner source, jig). Default takes: all non-T0.
"""

import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import DATA, OUT, Rotation, shipped_jig_path, group_obs, load_jig, load_rig, load_samples, solve_joint, solve_single

METHODS = ("cam0", "cam1", "joint")
# label -> (corner source, jig). Corner-source comparison: each source gets the jig bundle-adjusted
# from its own T0 corners (10-03 T0 takes). Jig comparison: logged corners with different rigid bodies.
VARIANTS = {
    "logged": ("logged", OUT / "jig_T0_logged10_03.json"),
    "none": ("none", OUT / "jig_T0_none.json"),
    "contour": ("contour", OUT / "jig_T0_contour.json"),
    "april": ("april", OUT / "jig_T0_april.json"),
    "tag12only": ("logged", OUT / "jig_tag12_only.json"),
    "april_tag12only": ("april", OUT / "jig_tag12_only.json"),
    "none_tag12only": ("none", OUT / "jig_tag12_only.json"),
    "shipped": ("logged", shipped_jig_path()),
    "T0all": ("logged", OUT / "jig_T0.json"),
}
# undistort-image ("ud") vs undistort-corners ("raw") study, corners from detect_paths.py
for _p in ("ud", "raw"):
    for _r in ("none", "contour", "edge", "april"):
        VARIANTS[f"{_p}_{_r}"] = (f"{_p}_{_r}", OUT / f"jig_T0_{_p}_{_r}.json")
        VARIANTS[f"{_p}_{_r}_tag12"] = (f"{_p}_{_r}", OUT / "jig_tag12_only.json")
# refinement layers on edge refinement, corners from detect_grid.py
for _s in ("ud_edge_trk", "ud_grid", "ud_grid_trk"):
    VARIANTS[_s] = (_s, OUT / f"jig_T0_{_s}.json")
# failed edge refinements left out; same jig as the source they come from, so only those tags differ
VARIANTS["ud_edge_drop"] = ("ud_edge_drop", OUT / "jig_T0_ud_edge.json")
VARIANTS["ud_grid_drop"] = ("ud_grid_drop", OUT / "jig_T0_ud_grid.json")
# jig fitted on T1 run 1 (build_jig_T0.py with JIG_TAKES), scored on run 2: how much a better jig buys
for _s in ("ud_edge", "ud_grid_trk"):
    VARIANTS[f"{_s}_jT1r1"] = (_s, OUT / f"jig_T1r1_{_s}.json")


def run(args):
    name, source = args
    corners, jig_path = VARIANTS[source]
    rig = load_rig(name)
    jig = load_jig(jig_path)
    s = load_samples(name).set_index("sample")
    obs = group_obs(name, corners)
    rows = []
    for k, ob in obs.items():
        row = {"sample": k, "t_cam0_s": s.at[k, "t_cam0_s"], "t_cam1_s": s.at[k, "t_cam1_s"]}
        res = {"cam0": solve_single(ob, jig, rig, 0), "cam1": solve_single(ob, jig, rig, 1)}
        res["joint"] = solve_joint(ob, jig, rig)
        for m in METHODS:
            r = res[m]
            if r is None:
                continue
            R, t, n, e = r
            g = R @ jig["grip"] + t
            q = Rotation.from_matrix(R).as_quat()
            row.update({f"{m}_n": n, f"{m}_px": e, f"{m}_gx": g[0], f"{m}_gy": g[1], f"{m}_gz": g[2],
                        f"{m}_tx": t[0], f"{m}_ty": t[1], f"{m}_tz": t[2],
                        f"{m}_qx": q[0], f"{m}_qy": q[1], f"{m}_qz": q[2], f"{m}_qw": q[3]})
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("sample")
    df.to_csv(index=False, path_or_buf=OUT / "poses" / source / f"{name}.csv")
    return name, len(df), {m: float(df[f"{m}_gx"].notna().mean()) for m in METHODS}


if __name__ == "__main__":
    source = sys.argv[1]
    (OUT / "poses" / source).mkdir(parents=True, exist_ok=True)
    takes = sys.argv[2:] or sorted(p.name for p in DATA.glob("2026-*") if "_T0_" not in p.name)
    with Pool(min(8, len(takes))) as pool:
        for name, n, cov in pool.imap_unordered(run, [(t, source) for t in takes]):
            print(name, n, {k: round(v, 3) for k, v in cov.items()}, flush=True)
