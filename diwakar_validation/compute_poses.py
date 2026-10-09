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
    "shipped": ("logged", shipped_jig_path()),
    "T0all": ("logged", OUT / "jig_T0.json"),
}


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
