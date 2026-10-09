"""Shared loaders and geometry for the Diwakar dual-camera mocap validation set.

Conventions (verified against samples.csv reprojection, 0.47 px):
- corners.csv u, v are in the undistorted pinhole image; K = meta['undistorted'], zero distortion.
- Tag corners in tag frame: TL(-h, h), TR(h, h), BR(h, -h), BL(-h, -h), z = 0.
- Tag -> board (tag 12 frame): P_board = R_tag @ c + t_tag.
- Stereo: p_cam0 = Rx @ p_cam1 + tx.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "diwakar" / "validaton_data"
MOCAP = DATA / "NOARK_mocap_validation_rec01_bk"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

UV_COLS = ["u0", "v0", "u1", "v1", "u2", "v2", "u3", "v3"]


def take_dir(name):
    return DATA / name


def load_rig(name):
    d = take_dir(name)
    meta = json.loads((d / "meta.json").read_text())
    st = json.loads((d / "calib" / "stereo_extrinsics.json").read_text())
    return {
        "K": [np.array(meta["undistorted"]["camera_matrix_cam0"]), np.array(meta["undistorted"]["camera_matrix_cam1"])],
        "Rx": np.array(st["Rx"]),
        "tx": np.array(st["tx"]),
    }


def load_jig(path):
    g = json.loads(Path(path).read_text())
    h = g["marker_length"] / 2
    a = g.get("marker_aspect", 1.0)
    local = np.array([[-h, h * a, 0], [h, h * a, 0], [h, -h * a, 0], [-h, -h * a, 0]])
    tags = {int(k): (np.array(v["rotation"]), np.array(v["translation"])) for k, v in g["markers"].items()}
    obj = {k: local @ R.T + t for k, (R, t) in tags.items()}
    return {"raw": g, "local": local, "tags": tags, "obj": obj, "grip": np.array(g["grip_point"])}


def shipped_jig_path(name="2026-10-03_T0_test_r1"):
    return take_dir(name) / "calib" / "board_geometry.json"


def corners_path(name, source="logged"):
    """logged = the tracker's corners.csv; none / contour = re-detected by detect_corners.py."""
    return take_dir(name) / "corners.csv" if source == "logged" else OUT / "corners" / f"{name}_{source}.csv"


def load_corners(name, source="logged"):
    c = pd.read_csv(corners_path(name, source))
    uv = c[UV_COLS].to_numpy(float).reshape(-1, 4, 2)
    return c["sample"].to_numpy(), c["cam"].to_numpy(), c["marker_id"].to_numpy(), uv


def load_samples(name):
    return pd.read_csv(take_dir(name) / "samples.csv")


def project(P, K):
    uv = P @ K.T
    return uv[..., :2] / uv[..., 2:3]


# ---------- per-frame pose solvers (board pose in cam0 frame) ----------


def _gather(obs, jig, cam):
    obj, img = [], []
    for c, mid, uv in obs:
        if c == cam and mid in jig["obj"]:
            obj.append(jig["obj"][mid])
            img.append(uv)
    if not obj:
        return None, None
    return np.concatenate(obj), np.concatenate(img)


def solve_single(obs, jig, rig, cam, min_tags=1):
    """PnP from one camera; returns (R, t) of board in cam0 frame, n_tags, reproj px."""
    obj, img = _gather(obs, jig, cam)
    if obj is None or len(obj) < 4 * min_tags:
        return None
    K = rig["K"][cam]
    if len(obj) == 4:
        ok, rv, tv = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    else:
        ok, rv, tv = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
        if ok:
            rv, tv = cv2.solvePnPRefineLM(obj, img, K, None, rv, tv)
    if not ok:
        return None
    R = cv2.Rodrigues(rv)[0]
    t = tv.ravel()
    err = np.linalg.norm(project(obj @ R.T + t, K) - img, axis=1).mean()
    if cam == 1:
        R, t = rig["Rx"] @ R, rig["Rx"] @ t + rig["tx"]
    return R, t, len(obj) // 4, err


def solve_joint(obs, jig, rig, init=None, iters=10):
    """Stereo Gauss-Newton over both cameras' corners, board pose in cam0 frame."""
    sets = []
    for cam in (0, 1):
        obj, img = _gather(obs, jig, cam)
        if obj is not None:
            sets.append((cam, obj, img))
    if not sets:
        return None
    if init is None:
        best = None
        for cam, _, _ in sets:
            r = solve_single(obs, jig, rig, cam)
            if r is not None and (best is None or r[2] > best[2]):
                best = r
        if best is None:
            return None
        init = best[:2]
    R, t = init
    Rx, tx = rig["Rx"], rig["tx"]
    for _ in range(iters):
        J_all, r_all = [], []
        for cam, obj, img in sets:
            Pc = obj @ R.T + t  # cam0 frame
            Pcam = Pc if cam == 0 else (Pc - tx) @ Rx  # Rx.T @ (Pc - tx)
            K = rig["K"][cam]
            fx, fy = K[0, 0], K[1, 1]
            X, Y, Z = Pcam.T
            res = (project(Pcam, K) - img).ravel()
            # d(uv)/d(Pcam)
            dpi = np.zeros((len(X), 2, 3))
            dpi[:, 0, 0] = fx / Z
            dpi[:, 0, 2] = -fx * X / Z**2
            dpi[:, 1, 1] = fy / Z
            dpi[:, 1, 2] = -fy * Y / Z**2
            M = np.eye(3) if cam == 0 else Rx.T
            # perturbation: R <- exp(w) R, t <- t + dt ; dPc/dw = -[Pc - t]x... use world-left perturb on Pc
            q = obj @ R.T  # rotated points
            dPc = np.zeros((len(X), 3, 6))
            dPc[:, :, :3] = -_skew(q)
            dPc[:, :, 3:] = np.eye(3)
            J = np.einsum("nij,jk,nkl->nil", dpi, M, dPc).reshape(-1, 6)
            J_all.append(J)
            r_all.append(res)
        J = np.concatenate(J_all)
        r = np.concatenate(r_all)
        dx = np.linalg.lstsq(J, -r, rcond=None)[0]
        R = Rotation.from_rotvec(dx[:3]).as_matrix() @ R
        t = t + dx[3:]
        if np.abs(dx).max() < 1e-9:
            break
    errs = []
    for cam, obj, img in sets:
        Pc = obj @ R.T + t
        Pcam = Pc if cam == 0 else (Pc - tx) @ Rx
        errs.append(np.linalg.norm(project(Pcam, rig["K"][cam]) - img, axis=1))
    ntags = sum(len(o) // 4 for _, o, _ in sets)
    return R, t, ntags, float(np.concatenate(errs).mean())


def _skew(v):
    S = np.zeros((len(v), 3, 3))
    S[:, 0, 1], S[:, 0, 2] = -v[:, 2], v[:, 1]
    S[:, 1, 0], S[:, 1, 2] = v[:, 2], -v[:, 0]
    S[:, 2, 0], S[:, 2, 1] = -v[:, 1], v[:, 0]
    return S


def group_obs(name, source="logged"):
    """Dict sample -> list of (cam, marker_id, uv[4,2])."""
    s, cam, mid, uv = load_corners(name, source)
    order = np.argsort(s, kind="stable")
    s, cam, mid, uv = s[order], cam[order], mid[order], uv[order]
    bounds = np.flatnonzero(np.diff(s)) + 1
    out = {}
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(s)]):
        out[int(s[a])] = [(int(cam[i]), int(mid[i]), uv[i]) for i in range(a, b)]
    return out


# ---------- mocap ----------


def load_mocap(name):
    """Rigid body 'noark' pose per mocap frame: frame, R (n,3,3), p (n,3) in metres, valid mask."""
    m = pd.read_csv(MOCAP / name / "mocap.csv", skiprows=7, header=None)
    frame = m[0].to_numpy(int)
    q = m[[2, 3, 4, 5]].to_numpy(float)
    p = m[[6, 7, 8]].to_numpy(float) / 1000.0
    valid = np.isfinite(q).all(1) & np.isfinite(p).all(1)
    R = np.full((len(m), 3, 3), np.nan)
    R[valid] = Rotation.from_quat(q[valid]).as_matrix()
    return {"frame": frame, "R": R, "p": p, "valid": valid}


def mocap_frame_of(name, t_mono):
    """GPIO sync: k-th rising edge in sync.csv = mocap frame k. Fractional frame index for monotonic times."""
    edges = pd.read_csv(take_dir(name) / "sync.csv")["t_s"].to_numpy()
    f = np.interp(t_mono, edges, np.arange(len(edges)), left=np.nan, right=np.nan)
    return f
