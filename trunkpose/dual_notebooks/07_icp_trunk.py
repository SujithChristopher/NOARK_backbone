# %% 07_icp_trunk.py
# Rigid-registration (ICP) trunk angles: register each frame's torso front-shell
# cloud to a NEUTRAL cloud (pooled first N frames) -> full 3-DoF trunk rotation,
# no landmarks, no plane. Compared per-axis against mocap AND against the existing
# plane+shoulder pipeline (06_trunk_axis.py), same sync + zeroing conventions.
#
# Requires the 06 geometry cache WITH per-frame clouds (cloud key). If the cache is
# missing/stale it recomputes geometry via 06's parallel pass and re-saves it.
#
#   $env:TRUNK_GEOM_CACHE=".../geom_cache.pkl"
#   uv run python trunkpose/dual_notebooks/07_icp_trunk.py
#
# Output: icp_trunk_angles.png (3-row angle comparison) + printed r/RMSE table.

import importlib
import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
m6 = importlib.import_module("06_trunk_axis")

NEUTRAL_MAX_PTS = 6000       # pooled neutral cloud size
ICP_ITERS = 20
ICP_DISTS = (0.06, 0.03, 0.02)   # trimming threshold annealing (m)
ICP_MIN_MATCH = 200
NORMAL_K = 15                 # neighbors for local-PCA normal estimation
# direct-to-neutral acceptance; on failure fall back to keyframe odometry.
# Very loose on purpose: wrong minima are prevented by the shoulder anchors, and
# odometry composition drifts (it twisted the whole series ~138 deg when it ran for
# 700+ frames), so direct should win whenever the registration converges at all.
DIRECT_RMS_M, DIRECT_FRAC = 0.030, 0.30
ODO_RMS_M, ODO_FRAC = 0.010, 0.50
KEY_ROT_DEG = 15.0           # spawn a new keyframe after rotating this far from it
# Anchor support kept for future trustworthy keypoints, but DISABLED for MediaPipe
# shoulders: tried it -- 2 jittery points at ~30% total weight steered the rotation
# axes wrong (S blew up to ~143 deg, flexion RMSE 21 deg). Not worth it.
ANCHOR_FRAC = 0.3
USE_SHOULDER_ANCHORS = False
SEG_JUMP_M = 0.10            # mocap displacement across a gap that flags a re-lock teleport
# Wrong-basin registrations look healthy in rms/frac, but a seated trunk never
# exceeds ~45 deg from neutral -> gate on physical plausibility instead.
ROT_MAX_DEG = 45.0
ODO_MAX_RUN = 30             # max consecutive odometry frames before declaring loss
ANGLE_PLOT_MODE = os.environ.get("TRUNK_ANGLE_PLOT_MODE", "all").lower()
if ANGLE_PLOT_MODE not in ("all", "icp_only", "mocap_only"):
    raise ValueError("TRUNK_ANGLE_PLOT_MODE must be all, icp_only, or mocap_only")
SHOW_ICP_ANGLES = ANGLE_PLOT_MODE in ("all", "icp_only")
SHOW_PLANE_ANGLES = ANGLE_PLOT_MODE == "all"
_plot_suffix = "" if ANGLE_PLOT_MODE == "all" else f"_{ANGLE_PLOT_MODE}"
OUT_PNG = m6.RECORDING_DIR / f"icp_trunk_angles{_plot_suffix}.png"


# %% ICP (trimmed, point-to-point Kabsch, warm-started per frame)
def kabsch(P, Q, w=None):
    """R, t minimizing weighted |R@P + t - Q| (rows are points)."""
    if w is None:
        w = np.ones(len(P))
    w = w / w.sum()
    cp, cq = w @ P, w @ Q
    H = (P - cp).T @ ((Q - cq) * w[:, None])
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))])
    R = Vt.T @ D @ U.T
    return R, cq - R @ cp


def estimate_normals(pts, k=NORMAL_K):
    """Local-PCA point normals, oriented outward from the cloud centroid (valid for
    a convex-ish front shell, which is what the torso cloud is)."""
    k = min(k, len(pts) - 1)
    if k < 3:
        return np.tile(np.array([0.0, 0.0, -1.0]), (len(pts), 1))
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k)
    neigh = pts[idx]
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k
    _, eigvecs = np.linalg.eigh(cov)   # ascending eigenvalues -> normal = first column
    normals = eigvecs[:, :, 0]
    outward = pts - pts.mean(axis=0)
    flip = np.einsum("ni,ni->n", normals, outward) < 0
    normals[flip] *= -1
    return normals


def solve_point_to_plane(rp, t, q, n, w):
    """Small-angle point-to-plane update: rotation vector d and translation dt
    minimizing weighted sum w * (n . (rp + t - q) + n . (d x rp) + n . dt)^2, where
    rp = R @ p (current rotation applied, no translation). Returns (d, dt)."""
    resid = np.einsum("ni,ni->n", n, rp + t - q)
    A = np.concatenate([np.cross(rp, n), n], axis=1)
    sw = np.sqrt(w)
    x, *_ = np.linalg.lstsq(A * sw[:, None], -resid * sw, rcond=None)
    return x[:3], x[3:]


def icp(src, dst, tree, R, t, dst_normals=None, anchors=None):
    """Refine (R, t) aligning src->dst. Point-to-plane when `dst_normals` is given
    (lower-noise, better-conditioned than point-to-point on a smooth torso shell);
    falls back to point-to-point Kabsch otherwise. `anchors` = (a_src, a_dst)
    known-identity correspondence pairs mixed in with total weight ANCHOR_FRAC of the
    cloud matches (point-to-point only -- unused while USE_SHOULDER_ANCHORS=False).
    Returns (R, t, rms, n_matched) or None."""
    rms, matched = np.nan, 0
    for dist in ICP_DISTS:
        for _ in range(ICP_ITERS):
            moved = src @ R.T + t
            d, idx = tree.query(moved, k=1)
            m = d < dist
            matched = int(m.sum())
            if matched < ICP_MIN_MATCH:
                return None
            P, Q = src[m], dst[idx[m]]
            w = np.ones(len(P))
            if dst_normals is not None:
                N = dst_normals[idx[m]]
                dvec, dt = solve_point_to_plane(P @ R.T, t, Q, N, w)
                Rn = Rotation.from_rotvec(dvec).as_matrix() @ R
                tn = t + dt
            else:
                if anchors is not None:
                    a_src, a_dst = anchors
                    wa = ANCHOR_FRAC * matched / len(a_src)
                    P = np.vstack([P, a_src]); Q = np.vstack([Q, a_dst])
                    w = np.concatenate([w, np.full(len(a_src), wa)])
                Rn, tn = kabsch(P, Q, w)
            delta = np.degrees(np.arccos(np.clip((np.trace(Rn @ R.T) - 1) / 2, -1, 1)))
            R, t = Rn, tn
            rms = float(np.sqrt((d[m] ** 2).mean()))
            if delta < 0.01:
                break
    return R, t, rms, matched


# %% Inputs
def load_inputs():
    """Basis + mocap CSV + camera timestamps + the (cached) 06 geometry pass."""
    print("Loading basis + mocap + timestamps...")
    R0_c, t0_c, Rm_c, tm_c = m6.load_charuco_basis(m6.CHARUCO_TOML)
    mocap_df, st_time = m6.read_rigid_body_csv(
        str(m6.RECORDING_DIR / f"{m6.RECORDING_NAME}.csv"))
    mocap_time = np.datetime64(st_time) + (
        mocap_df["seconds"].to_numpy() * 1e6).astype("timedelta64[us]")
    ts0 = m6.load_frame_times(m6.RECORDING_DIR / "cam0_timestamp.msgpack")
    ts1 = m6.load_frame_times(m6.RECORDING_DIR / "cam1_timestamp.msgpack")
    n = min(len(ts0), len(ts1))
    max_frames = int(os.environ.get("TRUNK_MAX_FRAMES", "0"))
    if max_frames:
        n = min(n, max_frames)
    ts0 = ts0[:n]

    cache = os.environ.get("TRUNK_GEOM_CACHE", "")
    frames = None
    if cache and Path(cache).exists() and not max_frames:
        with open(cache, "rb") as f:
            frames = pickle.load(f)
        if len(frames) != n or "cloud" not in frames[0]:
            print("Cache stale (no clouds / wrong length) -> recomputing geometry")
            frames = None
    if frames is None:
        print(f"Geometry pass over {n} frames (parallel, via 06)...")
        frames = m6.run_geometry(n)
        if cache and not max_frames:
            with open(cache, "wb") as f:
                pickle.dump(frames, f)
            print(f"Cached geometry -> {cache}")

    return dict(R0_c=R0_c, t0_c=t0_c, Rm_c=Rm_c, tm_c=tm_c, mocap_df=mocap_df,
                mocap_time=mocap_time, ts0=ts0, n=n, frames=frames)


# %% ICP pass
def run_icp_pass(frames, R_neu, n):
    """Register every frame's torso front-shell cloud to the neutral cloud.

    Returns the per-frame body rotation Rb (neutral -> current), the raw cur->neutral
    transforms A (kept for rendering), the neutral-zeroed angle series and the pooled
    neutral cloud."""
    # ---- neutral cloud from first N_NEUTRAL frames with clouds ----
    neu = [fr["cloud"] for fr in frames[: m6.N_NEUTRAL] if fr["cloud"] is not None]
    if not neu:
        raise SystemExit("No clouds in the first neutral window.")
    neu = m6.voxel_downsample(np.concatenate(neu).astype(np.float64),
                              max_pts=NEUTRAL_MAX_PTS)
    tree = cKDTree(neu)
    neu_normals = estimate_normals(neu)
    print(f"Neutral cloud: {len(neu)} pts from first {m6.N_NEUTRAL} frames")

    # ---- ICP per frame: direct-to-neutral, keyframe odometry as fallback ----
    # Direct registration fails when the trunk has turned far enough that the live
    # front shell no longer overlaps the neutral view (self-occlusion). Then we
    # register frame -> last-good keyframe instead and compose transforms; keyframes
    # roll forward every KEY_ROT_DEG so consecutive registrations stay easy.
    def rot_deg(R):
        return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))

    # neutral shoulder anchors (mean over the neutral window)
    neu_anch = {}
    for nm in ("L_shoulder", "R_shoulder"):
        pts = [fr["cam_pts"][nm] for fr in frames[: m6.N_NEUTRAL]
               if fr["cam_pts"][nm] is not None]
        if pts:
            neu_anch[nm] = np.mean(pts, axis=0)

    flex_i = np.full(n, np.nan); lat_i = np.full(n, np.nan); axi_i = np.full(n, np.nan)
    Rb_icp = [None] * n
    A_list = [None] * n             # accepted cur->neutral transform (for rendering)
    A_raw = [None] * n              # incl. poses the plausibility gate threw away
    gated = np.zeros(n, bool)       # converged but rejected as physically implausible
    rms_l, match_l = [], []
    A = (np.eye(3), np.zeros(3))   # warm start / current estimate: cur -> neutral
    key = None                     # (tree, cloud, A_k) fallback keyframe
    last_good = None               # (cloud, A) most recent accepted frame
    n_direct = n_odo = odo_run = 0
    for i, fr in enumerate(frames):
        if fr["cloud"] is None:
            continue
        src = fr["cloud"].astype(np.float64)
        anch = None
        if USE_SHOULDER_ANCHORS:
            pairs = [(fr["cam_pts"][nm], neu_anch[nm]) for nm in neu_anch
                     if fr["cam_pts"][nm] is not None]
            if pairs:
                anch = (np.array([p for p, _ in pairs]),
                        np.array([q for _, q in pairs]))
        got = None
        res = icp(src, neu, tree, *A, dst_normals=neu_normals, anchors=anch)
        if res and res[2] <= DIRECT_RMS_M and res[3] >= DIRECT_FRAC * len(src):
            got = (res[0], res[1]); key = None; odo_run = 0; n_direct += 1
            rms_l.append(res[2]); match_l.append(res[3])
        else:
            if key is None and last_good is not None:
                kc, kA = last_good
                key = (cKDTree(kc), kc, kA, estimate_normals(kc))
            if key is not None and odo_run < ODO_MAX_RUN:
                ktree, kc, (Rk, tk), kc_normals = key
                res2 = icp(src, kc, ktree,
                           Rk.T @ A[0], Rk.T @ (A[1] - tk),  # warm cur->key
                           dst_normals=kc_normals)
                if res2 and res2[2] <= ODO_RMS_M and res2[3] >= ODO_FRAC * len(src):
                    Rrel, trel = res2[0], res2[1]
                    got = (Rk @ Rrel, Rk @ trel + tk)
                    odo_run += 1; n_odo += 1
                    rms_l.append(res2[2]); match_l.append(res2[3])
                    if rot_deg(Rrel) > KEY_ROT_DEG:
                        key = (cKDTree(src), src, got, estimate_normals(src))
        if got is None:
            continue                       # keep previous estimate as warm start
        A_raw[i] = got                     # what ICP converged to, gate or no gate
        if rot_deg(got[0]) > ROT_MAX_DEG:
            gated[i] = True
            A = got                        # keep as warm start so we can re-converge,
            continue                       # but never report an implausible pose
        A = got
        last_good = (src, A)
        # A maps current -> neutral, so the body rotation from neutral is A[0].T
        Rb_icp[i] = A[0].T
        A_list[i] = A
        flex_i[i], lat_i[i], axi_i[i] = m6.trunk_angles(A[0].T @ R_neu, R_neu)
    print(f"ICP valid {int(np.isfinite(flex_i).sum())}/{n} "
          f"(direct {n_direct}, odometry {n_odo}, gated {int(gated.sum())})  "
          f"rms {np.mean(rms_l)*1000:.1f}mm  matched ~{int(np.mean(match_l))} pts")
    return dict(Rb_icp=Rb_icp, A_list=A_list, A_raw=A_raw, gated=gated, neutral=neu,
                flex=flex_i, lat=lat_i, axi=axi_i)


# %% Mocap comparison
def compare_mocap(inp, Rb_icp, R_neu):
    """GPIO-synced, teleport-split, convention-free comparison against mocap.

    Returns the re-zeroed angle series for both systems plus the pieces a renderer
    needs (zero rotations, residual frame rotation S, per-frame mocap markers)."""
    Rm_c, tm_c = inp["Rm_c"], inp["tm_c"]
    mocap_df, mocap_time = inp["mocap_df"], inp["mocap_time"]
    ts0, n = inp["ts0"], inp["n"]

    # ---- mocap: convention-free comparison on the longest clean segment ----
    # Mocap markers are mapped into the board frame (ChArUco mocap<->board basis) and
    # only the body rotation-from-zero R_f(t) @ R_f(z)^T is used, expressed in the
    # CAMERA anatomical axes (R_neu), so marker-frame conventions cancel. The residual
    # frame rotation S (basis rotation error) is fitted on the FIRST half of the
    # segment and validated on the held-out second half.
    mo, mx, mz = m6.load_trunk_rb(
        str(m6.RECORDING_DIR / f"{m6.RECORDING_NAME}.csv"))
    mob, mxb, mzb = ((m - tm_c) @ Rm_c for m in (mo, mx, mz))

    # hardware sync: the GPIO sync bit in the camera timestamp file is high while
    # mocap records, so mocap t=0 is the camera frame at the first rising edge.
    # No xcorr guessing -- the two timelines share a wire.
    sync = m6.load_sync_flags(m6.RECORDING_DIR / "cam0_timestamp.msgpack")[:n]
    if not (sync == 1).any():
        raise SystemExit("No GPIO sync pulse in cam0_timestamp.msgpack.")
    rise = int(np.argmax(sync == 1))
    mt = ts0[rise] + (mocap_df["seconds"].to_numpy() * 1e6).astype("timedelta64[us]")
    lag_s = float((mocap_time[0] - mt[0]) / np.timedelta64(1, "s"))
    print(f"GPIO sync: mocap t=0 at camera frame {rise} "
          f"(wall-clock offset vs Motive clock {lag_s:+.2f}s)")

    # Motive re-lock teleports: the solved rigid body can reappear at a ghost pose
    # after a short occlusion (this take: 326mm across 10ms and 267mm across 30ms).
    # Rotations on either side of a teleport carry different constant offsets, so
    # validation uses only the longest teleport-free segment.
    ok_m = np.isfinite(mo).all(axis=1)
    vidx = np.where(ok_m)[0]
    cut = np.where(np.linalg.norm(np.diff(mo[vidx], axis=0), axis=1) > SEG_JUMP_M)[0]
    bounds = np.concatenate(([0], cut + 1, [len(vidx)]))
    segs = [(vidx[a], vidx[b - 1]) for a, b in zip(bounds[:-1], bounds[1:])]
    s0, s1 = max(segs, key=lambda ab: mt[ab[1]] - mt[ab[0]])
    print("mocap segments (teleport-split): "
          + ", ".join(f"{(mt[a] - mt[0]) / np.timedelta64(1, 's'):.1f}-"
                      f"{(mt[b] - mt[0]) / np.timedelta64(1, 's'):.1f}s"
                      for a, b in segs)
          + f" -> using {(mt[s0] - mt[0]) / np.timedelta64(1, 's'):.1f}-"
            f"{(mt[s1] - mt[0]) / np.timedelta64(1, 's'):.1f}s")
    in_win = (ts0 >= mt[s0]) & (ts0 <= mt[s1])

    midx = np.array([m6.nearest_index(mt, t) for t in ts0])
    R_moc = [m6.mocap_trunk_frame(mob[j], mxb[j], mzb[j]) if in_win[i] else None
             for i, j in enumerate(midx)]

    # zero both systems over the first frames of the segment where both are valid
    both = [i for i in range(n) if R_moc[i] is not None and Rb_icp[i] is not None]
    if len(both) < m6.N_NEUTRAL:
        raise SystemExit("Too few common valid frames in the clean segment.")
    zwin = both[: m6.N_NEUTRAL]
    Rz_i = m6.Rotation.from_matrix(np.array([Rb_icp[i] for i in zwin])).mean().as_matrix()
    Rz_m = m6.Rotation.from_matrix(np.array([R_moc[i] for i in zwin])).mean().as_matrix()

    def cam_angles():
        f = np.full(n, np.nan); l = np.full(n, np.nan); a = np.full(n, np.nan)
        mg = np.full(n, np.nan)
        for i, Rb in enumerate(Rb_icp):
            if Rb is None or not in_win[i]:
                continue
            Rr = Rb @ Rz_i.T
            f[i], l[i], a[i] = m6.trunk_angles(Rr @ R_neu, R_neu)
            mg[i] = np.degrees(np.linalg.norm(m6.Rotation.from_matrix(Rr).as_rotvec()))
        return f, l, a, mg

    def moc_angles(S):
        f = np.full(n, np.nan); l = np.full(n, np.nan); a = np.full(n, np.nan)
        mg = np.full(n, np.nan)
        for i, R_f in enumerate(R_moc):
            if R_f is None:
                continue
            Rc = S @ (R_f @ Rz_m.T) @ S.T
            f[i], l[i], a[i] = m6.trunk_angles(Rc @ R_neu, R_neu)
            mg[i] = np.degrees(np.linalg.norm(m6.Rotation.from_matrix(Rc).as_rotvec()))
        return f, l, a, mg

    flex_iz, lat_iz, axi_iz, mag_i = cam_angles()

    # residual frame rotation S: Kabsch on rotation-vector pairs (|angle| > 5 deg),
    # fitted on the first half of the segment, validated on the second.
    pairs = []
    for i in both:
        ra = m6.Rotation.from_matrix(Rb_icp[i] @ Rz_i.T).as_rotvec()
        rm = m6.Rotation.from_matrix(R_moc[i] @ Rz_m.T).as_rotvec()
        if np.degrees(np.linalg.norm(ra)) > 5 and np.degrees(np.linalg.norm(rm)) > 5:
            pairs.append((i, ra, rm))
    if len(pairs) < 50:
        raise SystemExit("Too few well-excited rotation pairs to estimate S.")
    half = pairs[len(pairs) // 2][0]
    S_half, _ = kabsch(np.array([p[2] for p in pairs if p[0] < half]),
                       np.array([p[1] for p in pairs if p[0] < half]))
    S, _ = kabsch(np.array([p[2] for p in pairs]), np.array([p[1] for p in pairs]))
    s_deg = np.degrees(np.linalg.norm(m6.Rotation.from_matrix(S).as_rotvec()))
    dh = np.degrees(np.linalg.norm(m6.Rotation.from_matrix(S @ S_half.T).as_rotvec()))
    print(f"det(Rm_c)={np.linalg.det(Rm_c):+.3f}  basis residual rot S={s_deg:.1f}deg "
          f"({len(pairs)} pairs; half-fit differs {dh:.1f}deg)")

    # held-out score: S from the first half only, scored on the second half
    mfh, mlh, mah, _ = moc_angles(S_half)
    held = np.arange(n) >= half
    print("held-out (S fit on 1st half, scored on 2nd):")
    for lab, c, mo in (("flexion", flex_iz, mfh), ("lateral", lat_iz, mlh),
                       ("axial", axi_iz, mah)):
        cs = m6.smooth_series(c)
        m = np.isfinite(cs) & np.isfinite(mo) & held
        r = float(np.corrcoef(cs[m], mo[m])[0, 1])
        e = float(np.sqrt(np.mean((cs[m] - mo[m]) ** 2)))
        print(f"  {lab:8s}: r={r:+.2f} RMSE={e:4.1f}d")

    mflex, mlat, maxi, mag_m = moc_angles(S)
    return dict(flex_iz=flex_iz, lat_iz=lat_iz, axi_iz=axi_iz, mag_i=mag_i,
                mflex=mflex, mlat=mlat, maxi=maxi, mag_m=mag_m,
                in_win=in_win, zwin=zwin, Rz_i=Rz_i, Rz_m=Rz_m, S=S,
                mob=mob, mxb=mxb, mzb=mzb, midx=midx, R_moc=R_moc,
                # Compatibility aliases consumed by 08_icp_video.py. Their physical
                # meaning is now (origin, +X, +Z), not literal marker numbers.
                m1b=mob, m4b=mxb, m2b=mzb)


# %% Main
def main():
    inp = load_inputs()
    frames, n, ts0 = inp["frames"], inp["n"], inp["ts0"]

    # ---- baseline: plane+shoulder pipeline angles (unchanged 06 path) ----
    _R_t_list, R_neu, flex_p, lat_p, axi_p, _ = m6.assemble(frames)
    icp = run_icp_pass(frames, R_neu, n)
    cmp_ = compare_mocap(inp, icp["Rb_icp"], R_neu)
    flex_iz, lat_iz, axi_iz, mag_i = (cmp_["flex_iz"], cmp_["lat_iz"],
                                      cmp_["axi_iz"], cmp_["mag_i"])
    mflex, mlat, maxi, mag_m = (cmp_["mflex"], cmp_["mlat"],
                                cmp_["maxi"], cmp_["mag_m"])
    in_win, zwin = cmp_["in_win"], cmp_["zwin"]

    i0 = zwin[0]
    t_s = ((ts0 - ts0[i0]) / np.timedelta64(1, "s")).astype(np.float64)

    # plane baseline re-zeroed per axis over the same zero window (angles are not
    # composable like rotations, but the offset subtraction is fine for small zeros)
    series = dict(
        plane=(m6.smooth_series(flex_p), m6.smooth_series(lat_p), m6.smooth_series(axi_p)),
        icp=(m6.smooth_series(flex_iz), m6.smooth_series(lat_iz), m6.smooth_series(axi_iz)),
        mocap=(mflex, mlat, maxi))
    for arr in series["plane"]:
        off = np.nanmean(arr[zwin])
        if np.isfinite(off):
            arr -= off

    # ---- per-axis agreement over the clean segment ----
    def agree(a, b):
        m = np.isfinite(a) & np.isfinite(b) & in_win
        if m.sum() < 50:
            return np.nan, np.nan
        r = float(np.corrcoef(a[m], b[m])[0, 1])
        return r, float(np.sqrt(((a[m] - b[m]) ** 2).mean()))

    names = ("flexion", "lateral", "axial")
    rmag, emag = agree(m6.smooth_series(mag_i), mag_m)
    print(f"\nrotation magnitude (frame-invariant): ICP vs mocap r={rmag:.2f} "
          f"RMSE={emag:.1f}deg")
    print(f"{'axis':<10}{'plane r':>9}{'plane RMSE':>12}{'ICP r':>9}{'ICP RMSE':>12}")
    for k in range(3):
        rp, ep = agree(series["plane"][k], series["mocap"][k])
        ri, ei = agree(series["icp"][k], series["mocap"][k])
        print(f"{names[k]:<10}{rp:>9.2f}{ep:>11.1f}d{ri:>9.2f}{ei:>11.1f}d")

    # ---- plot ----
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for k, ax in enumerate(axes):
        ax.plot(t_s, series["mocap"][k], color="orange", lw=1.2, label="mocap")
        if SHOW_PLANE_ANGLES:
            ax.plot(t_s, series["plane"][k], color="tab:blue", lw=1.0, alpha=0.8,
                    label="camera plane+shoulder")
        if SHOW_ICP_ANGLES:
            ax.plot(t_s, series["icp"][k], color="tab:green", lw=1.2,
                    label="camera ICP")
        ax.set_ylabel(f"{names[k]} (deg)")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("time (s)")
    title = {
        "all": "Trunk angles: mocap vs plane+shoulder vs ICP rigid registration",
        "icp_only": "Trunk angles: mocap vs ICP rigid registration",
        "mocap_only": "Trunk angles: mocap only",
    }[ANGLE_PLOT_MODE]
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)
    print(f"\nPlot -> {OUT_PNG}")


if __name__ == "__main__":
    main()
