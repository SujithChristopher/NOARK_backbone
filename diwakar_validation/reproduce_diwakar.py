"""Reproduce Diwakar's T1 mocap comparison (compare_T1.csv, mocap_fit.json in the mocap folder).

His fit (13 params): camera->mocap rotation (rotvec) + translation (mm), board->mocap-body rotation
(rotvec), mocap-body origin in the board frame (mm), and a scale on the camera translation.
Model per hold i (device = logged joint pose from samples.csv, shipped jig):
    p_body_i = R_A (s t_cb_i + R_cb_i o) + t_A          R_body_i = R_A R_cb_i R_B
Fit in-sample on one run, residuals in mm and rad (weight W_ROT = 200 mm per rad).
Device = mean over the whole hold, mocap = mean over the hold trimmed TRIM_S each side.
Errors reported in his table frame (rows: across, depth, height) as device - mocap.
"""

import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from analyze_T1 import TRIM_S, load_holds
from common import MOCAP, Rotation, load_mocap, load_samples, mocap_frame_of

W_ROT = 200.0  # mm per rad; 200 matches his table (scan 2026-10-12); argv[1] overrides when run as a script


def holds(name, s=None):
    """s: samples with t_cam0_s, tx_m.., qx..qw (default: the logged samples.csv)."""
    s = load_samples(name) if s is None else s
    s = s[
        np.linalg.norm(s[["qx", "qy", "qz", "qw"]].fillna(0).to_numpy(), axis=1) > 0.5
    ]
    mc = load_mocap(name)
    rows = []
    for i, h in load_holds(name).iterrows():
        a, b = h.t0 + TRIM_S, h.t1 - TRIM_S
        w = s[
            (s.t_cam0_s >= h.t0) & (s.t_cam0_s <= h.t1)
        ]  # device: whole hold; mocap: trimmed
        f0, f1 = mocap_frame_of(name, np.array([a, b]))
        fr = np.arange(int(np.ceil(f0)), int(np.floor(f1)) + 1)
        fr = fr[mc["valid"][fr]]
        if not len(w):
            continue
        rows.append(
            {
                "hold": i,
                "t_cb": w[["tx_m", "ty_m", "tz_m"]].mean().to_numpy() * 1000,
                "R_cb": Rotation.from_quat(w[["qx", "qy", "qz", "qw"]].to_numpy())
                .mean()
                .as_matrix(),
                "p_m": mc["p"][fr].mean(0) * 1000,
                "R_m": Rotation.from_matrix(mc["R"][fr]).mean().as_matrix(),
                "dev_sd": w[["tx_m", "ty_m", "tz_m"]].std().to_numpy() * 1000,
            }
        )
    return rows


def predict(x, H):
    RA = Rotation.from_rotvec(x[:3]).as_matrix()
    RB = Rotation.from_rotvec(x[6:9]).as_matrix()
    o, s = x[9:12], x[12]
    p = np.array([RA @ (s * h["t_cb"] + h["R_cb"] @ o) + x[3:6] for h in H])
    R = np.array([RA @ h["R_cb"] @ RB for h in H])
    return p, R


def residuals(x, H, fix_scale):
    if fix_scale:
        x = np.r_[x, 1.0]
    p, R = predict(x, H)
    rp = (p - np.array([h["p_m"] for h in H])).ravel()
    rr = np.concatenate(
        [
            Rotation.from_matrix(Rm.T @ Rd).as_rotvec()
            for Rd, Rm in zip(R, [h["R_m"] for h in H])
        ]
    )
    return np.r_[rp, W_ROT * rr]


def fit(H, x0, fix_scale):
    sol = least_squares(residuals, x0[:12] if fix_scale else x0, args=(H, fix_scale))
    return np.r_[sol.x, 1.0] if fix_scale else sol.x


def main():
    global W_ROT
    if len(sys.argv) > 1:
        W_ROT = float(sys.argv[1])
    for name in ["2026-10-03_T1_grid_r1", "2026-10-03_T1_grid_r2"]:
        ref = json.loads((MOCAP / name / "mocap_fit.json").read_text())
        comp = pd.read_csv(MOCAP / name / "compare_T1.csv")
        T = np.array(ref["table_frame"])
        H = holds(name)
        print(f"\n== {name}: {len(H)} holds (his {len(comp)})")
        for key, fix in (("free", False), ("s1", True)):
            his = np.array(ref[key]["params"])
            x0 = his.copy()
            x0[:12] += np.r_[
                0.05, -0.05, 0.05, 30, -30, 30, 0.02, -0.02, 0.02, 10, -10, 10
            ]  # start away from his answer
            x = fit(H, x0, fix)
            print(f"  {key}: params mine vs his")
            print(
                "    mine",
                np.array2string(
                    x, precision=4, suppress_small=True, max_line_width=200
                ),
            )
            print(
                "    his ",
                np.array2string(
                    his, precision=4, suppress_small=True, max_line_width=200
                ),
            )
            p, _ = predict(x, H)
            e = (p - np.array([h["p_m"] for h in H])) @ T.T
            ph, _ = predict(his, H)
            eh = (ph - np.array([h["p_m"] for h in H])) @ T.T
            for lab, ee in (("mine", e), ("his params on my holds", eh)):
                print(
                    f"    {lab:24s} RMSE across {np.sqrt((ee[:, 0] ** 2).mean()):.3f}  depth {np.sqrt((ee[:, 1] ** 2).mean()):.3f}"
                    f"  height {np.sqrt((ee[:, 2] ** 2).mean()):.3f} mm"
                )
        c = comp[["err_across_mm", "err_depth_mm", "err_height_mm"]].to_numpy()
        print(
            f"  his compare_T1.csv        RMSE across {np.sqrt((c[:, 0] ** 2).mean()):.3f}  depth {np.sqrt((c[:, 1] ** 2).mean()):.3f}"
            f"  height {np.sqrt((c[:, 2] ** 2).mean()):.3f} mm"
        )


if __name__ == "__main__":
    main()
