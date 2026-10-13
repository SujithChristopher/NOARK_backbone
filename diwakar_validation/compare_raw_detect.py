"""Detect on the RAW fisheye crops and undistort the corner points, vs Diwakar's pipeline (undistort the
image, detect, edge_refine = the logged corners.csv). Judged in his own metric.

Sources: logged (his pipeline) and the raw-detect corners of detect_corners.py (none, contour, april), all in
the undistorted pinhole image. Per sample with corners: his joint pose solve (board.estimate_board_pose_dual_gn,
Huber 1 px, shipped jig, started from the logged pose), then his 13-parameter mocap fit (reproduce_diwakar.py,
W_ROT 200 mm/rad) in-sample per run, and fitted on one run / applied to the other.
Usage: python compare_raw_detect.py <bkd_tracker_dir>
"""

import json
import sys

import numpy as np
import pandas as pd

import reproduce_diwakar as rd
from analyze_T1 import load_holds
from common import (
    MOCAP,
    UV_COLS,
    Rotation,
    corners_path,
    load_rig,
    load_samples,
    shipped_jig_path,
)

sys.path.insert(0, sys.argv[1])
from board import BoardGeometry, estimate_board_pose_dual_gn  # noqa: E402

RUNS = ["2026-10-03_T1_grid_r1", "2026-10-03_T1_grid_r2"]
SOURCES = ["logged", "none", "contour", "april"]
board = BoardGeometry.load(str(shipped_jig_path()))


def tag_side(uv):
    return np.linalg.norm(uv - np.roll(uv, -1, axis=1), axis=2).mean(1)


def corner_stats(name, src):
    key = ["sample", "cam", "marker_id"]
    a = pd.read_csv(corners_path(name, "logged"))
    b = pd.read_csv(corners_path(name, src))
    j = b.merge(a, on=key, suffixes=("", "_log"))
    ub = j[UV_COLS].to_numpy().reshape(-1, 4, 2)
    ua = j[[c + "_log" for c in UV_COLS]].to_numpy().reshape(-1, 4, 2)
    d = np.linalg.norm(ub - ua, axis=2)
    return (
        len(j),
        np.median(d),
        np.percentile(d, 95),
        100 * np.median(tag_side(ub) / tag_side(ua) - 1),
    )


def poses(name, src, samples_in_holds):
    rig = load_rig(name)
    s = load_samples(name).set_index("sample")
    c = pd.read_csv(corners_path(name, src))
    c = c[c["sample"].isin(samples_in_holds)]
    rows = []
    for smp, g in c.groupby("sample"):
        r = s.loc[smp]
        if not np.isfinite(r.qw):
            continue
        args = []
        for cam in (0, 1):
            x = g[g.cam == cam]
            uv = x[UV_COLS].to_numpy(float).reshape(-1, 1, 4, 2)
            args += [list(uv) or None, x.marker_id.to_numpy() if len(x) else None]
        guess = (
            Rotation.from_quat([r.qx, r.qy, r.qz, r.qw]).as_rotvec(),
            np.array([r.tx_m, r.ty_m, r.tz_m]),
        )
        out = estimate_board_pose_dual_gn(
            board,
            *args,
            rig["K"][0],
            rig["K"][1],
            rig["Rx"],
            rig["tx"],
            guess,
            max_iter=30,
            robust_px=1.0,
        )
        if out is None or not out[3]:
            continue
        rv, t = out[0], out[1]
        q = Rotation.from_rotvec(rv).as_quat()
        rows.append((r.t_cam0_s, *t, *q))
    return pd.DataFrame(
        rows, columns=["t_cam0_s", "tx_m", "ty_m", "tz_m", "qx", "qy", "qz", "qw"]
    )


def rmse(x, H, T):
    p, _ = rd.predict(x, H)
    e = (p - np.array([h["p_m"] for h in H])) @ T.T
    return np.sqrt((e**2).mean(0))


def main():
    T = np.array(
        json.loads((MOCAP / RUNS[0] / "mocap_fit.json").read_text())["table_frame"]
    )
    x0 = np.array(
        json.loads((MOCAP / RUNS[0] / "mocap_fit.json").read_text())["free"]["params"]
    )
    print(
        "corners vs logged (matched tags): n, median px, p95 px, tag side vs logged %"
    )
    for name in RUNS:
        for src in SOURCES[1:]:
            n, med, p95, side = corner_stats(name, src)
            print(
                f"  {name[-2:]} {src:8s} n {n:6d}  med {med:.3f}  p95 {p95:.3f}  side {side:+.2f} %"
            )

    H = {}
    for name in RUNS:
        s = load_samples(name)
        hs = load_holds(name)
        inh = np.zeros(len(s), bool)
        for h in hs.itertuples():
            inh |= (s.t_cam0_s >= h.t0).to_numpy() & (s.t_cam0_s <= h.t1).to_numpy()
        ids = s["sample"][inh].to_numpy()
        for src in SOURCES:
            H[name, src] = rd.holds(name, poses(name, src, ids))
            print(f"  {name[-2:]} {src}: {len(H[name, src])} holds", flush=True)

    print("\nmocap fit RMSE across / depth / height mm (his metric)")
    print(
        f"{'source':8s} {'scale':>6s} | {'r1 in-sample':>20s} | {'r2 in-sample':>20s} | {'fit r1 -> r2':>20s} | {'fit r2 -> r1':>20s}"
    )
    for fix in (False, True):
        for src in SOURCES:
            xs = [rd.fit(H[n, src], x0, fix) for n in RUNS]
            cells = [
                rmse(xs[0], H[RUNS[0], src], T),
                rmse(xs[1], H[RUNS[1], src], T),
                rmse(xs[0], H[RUNS[1], src], T),
                rmse(xs[1], H[RUNS[0], src], T),
            ]
            sc = "1" if fix else f"{xs[0][12]:.4f}"
            print(
                f"{src:8s} {sc:>6s} | "
                + " | ".join(" ".join(f"{v:6.3f}" for v in c) for c in cells)
            )


if __name__ == "__main__":
    main()
