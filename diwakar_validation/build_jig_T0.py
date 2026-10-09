"""Rebuild the tag jig (tag -> tag-12 transforms) from the T0 takes by stereo bundle adjustment.

Fit on alternating 5 s blocks of all T0 takes, hold out the other blocks. Marker length, aspect,
intrinsics and stereo extrinsics stay fixed (shipped). Grip point is copied from the shipped
geometry (it is defined in the tag-12 frame, which stays the reference).
Writes out/jig_T0.json in board_geometry.json format.

Usage: python build_jig_T0.py [SOURCE]   SOURCE = logged (default, all 3 T0 takes -> out/jig_T0.json)
       or logged10_03 | none | contour (10-03 T0 takes only, which have images -> out/jig_T0_<SOURCE>.json)
"""

import sys

import json
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from common import (OUT, Rotation, group_obs, load_jig, load_rig, load_samples, project,
                    shipped_jig_path, solve_joint, solve_single)

SRC = sys.argv[1] if len(sys.argv) > 1 else "logged"
T0 = ["2026-10-03_T0_test_r1", "2026-10-03_T0_test_r2"] + ([] if SRC != "logged" else ["2026-10-05_T0_test_r1"])
CORNERS = "logged" if SRC.startswith("logged") else SRC
JIG_OUT = OUT / ("jig_T0.json" if SRC == "logged" else f"jig_T0_{SRC}.json")
REF = 12
STEP = 5  # every 5th sample (20 Hz) of fit blocks
BLOCK_S = 5.0

shipped = load_jig(shipped_jig_path())
rig = load_rig(T0[0])
local = shipped["local"]


def split_blocks(name):
    s = load_samples(name)
    t = s["t_cam0_s"].to_numpy()
    blk = ((t - t[0]) // BLOCK_S).astype(int)
    return dict(zip(s["sample"].to_numpy(), blk % 2 == 0))


# ---------- init: per-tag IPPE poses in each camera, chain relative poses to tag 12 ----------
frames = []  # (take, sample, obs, is_fit)
for name in T0:
    fit = split_blocks(name)
    for k, obs in group_obs(name, CORNERS).items():
        frames.append((name, k, obs, fit.get(k, False)))
print("frames", len(frames), "fit", sum(f[3] for f in frames))

import cv2  # noqa: E402

pair_rel = defaultdict(list)  # (a, b) -> list of T_a<-b (4x4) estimates
for i, (_, _, obs, is_fit) in enumerate(frames):
    if not is_fit or i % 3:
        continue
    for cam in (0, 1):
        poses = {}
        for c, mid, uv in obs:
            if c != cam:
                continue
            ok, rv, tv = cv2.solvePnP(local, uv, rig["K"][cam], None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                T = np.eye(4)
                T[:3, :3] = cv2.Rodrigues(rv)[0]
                T[:3, 3] = tv.ravel()
                poses[mid] = T
        ids = sorted(poses)
        for a in ids:
            for b in ids:
                if a != b:
                    pair_rel[(a, b)].append(np.linalg.inv(poses[a]) @ poses[b])


def robust_mean(Ts):
    Ts = np.array(Ts)
    R = Rotation.from_matrix(Ts[:, :3, :3])
    Rm = R.mean()
    ang = (Rm.inv() * R).magnitude()
    keep = ang < np.percentile(ang, 70)
    T = np.eye(4)
    T[:3, :3] = R[keep].mean().as_matrix()
    T[:3, 3] = np.median(Ts[keep, :3, 3], axis=0)
    return T


tags_all = sorted({m for (a, b) in pair_rel for m in (a, b)})
init = {REF: np.eye(4)}
frontier = [REF]
while frontier:  # BFS on the co-visibility graph, best-supported edges first
    nxt = []
    for a in frontier:
        cand = sorted(((len(v), b) for (x, b), v in pair_rel.items() if x == a and b not in init), reverse=True)
        for n, b in cand:
            if b not in init and n >= 20:
                init[b] = init[a] @ robust_mean(pair_rel[(a, b)])
                nxt.append(b)
    frontier = nxt
print("init tags", sorted(init), "pair counts to 12:", {b: len(pair_rel[(REF, b)]) for b in tags_all if b != REF})


def jig_from(Tdict):
    g = json.loads(json.dumps(shipped["raw"]))
    g["markers"] = {str(k): {"rotation": T[:3, :3].tolist(), "translation": T[:3, 3].tolist()} for k, T in Tdict.items()}
    return g


def as_jig(g):
    p = OUT / f"_tmp_jig_{SRC}.json"
    p.write_text(json.dumps(g))
    return load_jig(p)


jig0 = as_jig(jig_from(init))

# ---------- bundle adjustment ----------
fit_frames = [f for i, f in enumerate(frames) if f[3] and i % STEP == 0]
tag_ids = [m for m in sorted(init) if m != REF]
tag_idx = {m: i for i, m in enumerate(tag_ids)}
obs_list = []  # (frame_i, cam, tag, uv)
fposes = []
for fi, (_, _, obs, _) in enumerate(fit_frames):
    r = solve_joint(obs, jig0, rig)
    fposes.append(np.r_[Rotation.from_matrix(r[0]).as_rotvec(), r[1]])
    for c, mid, uv in obs:
        if mid in init:
            obs_list.append((fi, c, mid, uv))
nT, nF = len(tag_ids), len(fit_frames)
x0 = np.concatenate([np.r_[Rotation.from_matrix(init[m][:3, :3]).as_rotvec(), init[m][:3, 3]] for m in tag_ids] + fposes)
o_f = np.array([o[0] for o in obs_list])
o_c = np.array([o[1] for o in obs_list])
o_t = np.array([tag_idx.get(o[2], -1) for o in obs_list])
o_uv = np.array([o[3] for o in obs_list])
print(f"BA: {nT} tags, {nF} frames, {len(obs_list)} tag observations")
Rx, tx = rig["Rx"], rig["tx"]


def residuals(x):
    tp = x[: 6 * nT].reshape(nT, 6)
    fp = x[6 * nT :].reshape(nF, 6)
    Rt = Rotation.from_rotvec(tp[:, :3]).as_matrix()
    Rf = Rotation.from_rotvec(fp[:, :3]).as_matrix()
    # tag corners in board frame
    Pb = np.repeat(local[None], len(obs_list), 0)
    m = o_t >= 0
    Pb[m] = np.einsum("nij,kj->nki", Rt[o_t[m]], local) + tp[o_t[m], None, 3:]
    Pc = np.einsum("nij,nkj->nki", Rf[o_f], Pb) + fp[o_f, None, 3:]
    out = np.empty((len(obs_list), 4, 2))
    for cam in (0, 1):
        mc = o_c == cam
        P = Pc[mc] if cam == 0 else (Pc[mc] - tx) @ Rx
        out[mc] = project(P, rig["K"][cam]) - o_uv[mc]
    return out.ravel()


A = lil_matrix((len(obs_list) * 8, len(x0)), dtype=int)
for i in range(len(obs_list)):
    rows = slice(8 * i, 8 * i + 8)
    if o_t[i] >= 0:
        A[rows, 6 * o_t[i] : 6 * o_t[i] + 6] = 1
    A[rows, 6 * nT + 6 * o_f[i] : 6 * nT + 6 * o_f[i] + 6] = 1
r0 = residuals(x0)
print("init reproj px median", np.median(np.linalg.norm(r0.reshape(-1, 2), axis=1)).round(3))
sol = least_squares(residuals, x0, jac_sparsity=A, loss="soft_l1", f_scale=1.0, x_scale="jac", verbose=1, max_nfev=200)
r1 = np.linalg.norm(sol.fun.reshape(-1, 2), axis=1)
print("BA reproj px median", np.median(r1).round(3), "p95", np.percentile(r1, 95).round(3))

tp = sol.x[: 6 * nT].reshape(nT, 6)
new = {REF: np.eye(4)}
for m, p in zip(tag_ids, tp):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(p[:3]).as_matrix()
    T[:3, 3] = p[3:]
    new[m] = T
g = jig_from(new)
g["meta"] = {"tool": "diwakar_validation/build_jig_T0.py", "takes": T0, "corners": CORNERS, "fit": "even 5 s blocks, every 5th sample",
             "frames_fitted": nF, "ba_median_px": float(np.median(r1))}
JIG_OUT.write_text(json.dumps(g, indent=2))
(OUT / f"_tmp_jig_{SRC}.json").unlink()
jig_new = load_jig(JIG_OUT)

# ---------- compare with shipped and evaluate on held-out blocks ----------
print("\ntag   d_trans_mm  d_rot_deg  (new vs shipped)")
for m in sorted(new):
    if m in shipped["tags"]:
        Rs, ts = shipped["tags"][m]
        dr = np.degrees(Rotation.from_matrix(new[m][:3, :3] @ Rs.T).magnitude())
        print(f"{m:4d}  {np.linalg.norm(new[m][:3, 3] - ts) * 1000:9.3f}  {dr:9.3f}")

for name in T0:
    hold = [f for f in frames if f[0] == name and not f[3]][::5]
    res = {"shipped": [], "T0": []}
    for _, _, obs, _ in hold:
        for lab, jg in (("shipped", shipped), ("T0", jig_new)):
            r = solve_joint(obs, jg, rig)
            if r is not None:
                res[lab].append(r[3])
    print(f"held-out {name}: mean reproj px  shipped {np.median(res['shipped']):.3f}  T0 {np.median(res['T0']):.3f}  (n={len(hold)})")
