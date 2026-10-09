"""T1 grid holds: camera grip point vs mocap rigid body, error in mocap X and Z (horizontal plane).

Per hold (marks.csv hold_start..hold_end, trimmed TRIM_S each side) take the median camera grip point
(cam0 frame) per method and the mean mocap rigid-body pose over the GPIO-synced frames.
Registration per method: p_mocap_i = R_wc @ g_cam_i + t_wc - R_body_i @ g_body   (R_wc, t_wc, g_body free).
Fit on one T1 run, evaluate on the other (held-out) and in-sample. Saves out/T1_holds.csv, out/T1_registration.json.
"""

import json

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from common import OUT, Rotation, load_jig, load_mocap, load_samples, mocap_frame_of, shipped_jig_path, take_dir

TAKES = ["2026-10-03_T1_grid_r1", "2026-10-03_T1_grid_r2"]
METHODS = ["cam0", "cam1", "joint", "joint_logged"]
TRIM_S = 0.5


def load_holds(name):
    m = pd.read_csv(take_dir(name) / "marks.csv")
    m[["kind", "idx", "tx", "ty"]] = m["label"].str.split(" ", expand=True)
    m["idx"] = m["idx"].astype(int)
    st = m[m.kind == "hold_start"].set_index("idx")
    en = m[m.kind == "hold_end"].set_index("idx")
    return pd.DataFrame({"t0": st.t_s, "t1": en.t_s, "target_x": st.tx.astype(float), "target_y": st.ty.astype(float)})


def hold_table(name):
    poses = pd.read_csv(OUT / "poses" / "logged" / f"{name}.csv")
    # logged joint pose (shipped jig) -> grip point with shipped grip
    s = load_samples(name)
    s = s[np.linalg.norm(s[["qx", "qy", "qz", "qw"]].fillna(0).to_numpy(), axis=1) > 0.5]
    gs = load_jig(shipped_jig_path())["grip"]
    R = Rotation.from_quat(s[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()
    g = np.einsum("nij,j->ni", R, gs) + s[["tx_m", "ty_m", "tz_m"]].to_numpy()
    poses = poses.merge(pd.DataFrame({"sample": s["sample"], "joint_logged_gx": g[:, 0], "joint_logged_gy": g[:, 1],
                                      "joint_logged_gz": g[:, 2]}), on="sample", how="left")
    mc = load_mocap(name)
    holds = load_holds(name)
    rows = []
    for i, h in holds.iterrows():
        a, b = h.t0 + TRIM_S, h.t1 - TRIM_S
        w = poses[(poses.t_cam0_s >= a) & (poses.t_cam0_s <= b)]
        f0, f1 = mocap_frame_of(name, np.array([a, b]))
        fr = np.arange(int(np.ceil(f0)), int(np.floor(f1)) + 1)
        fr = fr[mc["valid"][fr]]
        row = {"take": name, "hold": i, "target_x": h.target_x, "target_y": h.target_y, "n_cam": len(w), "n_mocap": len(fr)}
        for m in METHODS:
            v = w[[f"{m}_gx", f"{m}_gy", f"{m}_gz"]].dropna().to_numpy()
            row.update({f"{m}_{ax}": val for ax, val in zip("xyz", np.median(v, 0))})
            row[f"{m}_sd_mm"] = float(np.linalg.norm(v.std(0)) * 1000)
        p = mc["p"][fr]
        Rm = Rotation.from_matrix(mc["R"][fr]).mean()
        row.update({"mc_x": p[:, 0].mean(), "mc_y": p[:, 1].mean(), "mc_z": p[:, 2].mean(), "mc_sd_mm": float(np.linalg.norm(p.std(0)) * 1000)})
        row.update({f"mc_q{c}": v for c, v in zip("xyzw", Rm.as_quat())})
        rows.append(row)
    return pd.DataFrame(rows)


def fit(df, m):
    gc = df[[f"{m}_x", f"{m}_y", f"{m}_z"]].to_numpy()
    pm = df[["mc_x", "mc_y", "mc_z"]].to_numpy()
    Rb = Rotation.from_quat(df[["mc_qx", "mc_qy", "mc_qz", "mc_qw"]].to_numpy()).as_matrix()
    # Kabsch init
    A, B = gc - gc.mean(0), pm - pm.mean(0)
    U, _, Vt = np.linalg.svd(A.T @ B)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R0 = Vt.T @ D @ U.T
    x0 = np.r_[Rotation.from_matrix(R0).as_rotvec(), pm.mean(0) - R0 @ gc.mean(0), 0, 0, 0]

    def res(x):
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        return (gc @ R.T + x[3:6] - (pm + np.einsum("nij,j->ni", Rb, x[6:9]))).ravel()

    sol = least_squares(res, x0)
    return sol.x


def apply(df, m, x):
    gc = df[[f"{m}_x", f"{m}_y", f"{m}_z"]].to_numpy()
    pm = df[["mc_x", "mc_y", "mc_z"]].to_numpy()
    Rb = Rotation.from_quat(df[["mc_qx", "mc_qy", "mc_qz", "mc_qw"]].to_numpy()).as_matrix()
    R = Rotation.from_rotvec(x[:3]).as_matrix()
    cam_w = gc @ R.T + x[3:6]
    mc_w = pm + np.einsum("nij,j->ni", Rb, x[6:9])
    return cam_w, mc_w


def stats(e):
    return {"bias_mm": float(e.mean() * 1000), "rmse_mm": float(np.sqrt((e**2).mean()) * 1000),
            "mae_mm": float(np.abs(e).mean() * 1000), "max_mm": float(np.abs(e).max() * 1000)}


if __name__ == "__main__":
    df = pd.concat([hold_table(n) for n in TAKES], ignore_index=True)
    print("holds per take", df.groupby("take").size().to_dict())
    print("min n_cam / n_mocap", df.n_cam.min(), df.n_mocap.min(), " max mocap sd mm", df.mc_sd_mm.max().round(2))
    yaw = Rotation.from_quat(df[["mc_qx", "mc_qy", "mc_qz", "mc_qw"]].to_numpy())
    print("mocap body rotation spread (deg from mean):", np.degrees((yaw.mean().inv() * yaw).magnitude()).max().round(1))
    print("within-hold camera spread mm (median):", {m: round(df[f"{m}_sd_mm"].median(), 3) for m in METHODS})

    regs, out_rows = {}, []
    for m in METHODS:
        for fit_on in TAKES + ["both"]:
            fit_df = df if fit_on == "both" else df[df["take"] == fit_on]
            x = fit(fit_df, m)
            regs[f"{m}|{fit_on}"] = x.tolist()
            for ev in TAKES:
                cw, mw = apply(df[df["take"] == ev], m, x)
                e = cw - mw
                kind = "in-sample" if fit_on in (ev, "both") else "held-out"
                if fit_on == "both":
                    kind = "pooled"
                r = {"method": m, "fit_on": fit_on[-2:] if fit_on != "both" else "both", "eval": ev[-2:], "kind": kind}
                for ax, k in (("x", 0), ("z", 2)):
                    r.update({f"{ax}_{s}": v for s, v in stats(e[:, k]).items()})
                r["xz_rmse_mm"] = float(np.sqrt((e[:, [0, 2]] ** 2).sum(1).mean()) * 1000)
                out_rows.append(r)
                if fit_on == "both":
                    sub = df.index[df["take"] == ev]
                    df.loc[sub, f"{m}_ex_mm"] = e[:, 0] * 1000
                    df.loc[sub, f"{m}_ez_mm"] = e[:, 2] * 1000
    res = pd.DataFrame(out_rows)
    df.to_csv(OUT / "T1_holds.csv", index=False)
    res.to_csv(OUT / "T1_errors.csv", index=False)
    (OUT / "T1_registration.json").write_text(json.dumps(regs, indent=1))
    pd.set_option("display.width", 250)
    cols = ["method", "fit_on", "eval", "kind", "x_bias_mm", "x_rmse_mm", "x_max_mm", "z_bias_mm", "z_rmse_mm", "z_max_mm", "xz_rmse_mm"]
    print(res[res.kind == "held-out"][cols].round(2).to_string(index=False))
    print(res[res.kind == "pooled"][cols].round(2).to_string(index=False))
    for m in METHODS:
        print(m, "g_body (mm):", (np.array(regs[f"{m}|both"][6:9]) * 1000).round(1))
