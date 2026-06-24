# %% [markdown]
# # Dual-camera stereo verification against motion capture
#
# Question being tested: does a 2nd camera reduce single-tag pose jitter?
#
# Per synced frame we estimate the tag pose 4 ways and compare each against the
# OptiTrack rigid body:
#   - cam0-only   : solvePnP on 4 undistorted corners (single-view planar PnP)
#   - cam1-only   : same, on the other camera
#   - stereo-tri  : triangulate 4 corners w/ baseline -> Kabsch-fit tag model
#                   (algebraic, 8 image pts -> 6-DOF pose, bad-corner rejection)
#   - stereo-pnp  : joint multi-view PnP, one pose minimising reprojection in
#                   BOTH cameras (maximum-likelihood estimate)
#
# Metrics per estimator:
#   - translation accuracy : RMS of (cam - mocap), Kabsch-aligned, lag-corrected
#   - translation jitter   : std of frame-to-frame residual during STATIC holds
#   - rotation accuracy    : geodesic angle vs mocap (relative to first frame)
#   - rotation jitter      : frame-to-frame angular noise during static holds
# Plus a jitter-vs-depth plot to show where stereo pulls ahead.

# %% Imports
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Mean of empty slice")

import cv2
import numpy as np
import toml
import matplotlib.pyplot as plt
from cv2 import aruco
from tqdm.auto import tqdm
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy, Slerp

sys.path.insert(1, os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.insert(1, os.path.dirname(os.getcwd()))
from pd_support import read_rigid_body_csv, get_rb_marker_name  # noqa: E402
import msgpack as mp  # noqa: E402
import msgpack_numpy as mpn  # noqa: E402

# %% Paths / config
PROJECT_ROOT = Path(__file__).parents[3]

RECORDING = PROJECT_ROOT / "data" / "dual_data" / "dual_cam_single_aprl_50mm_t0"
STEREO_TOML = (
    PROJECT_ROOT / "data" / "calibration" / "dual_160"
    / "calib_cz30_dual_v2" / "stereo_calibration.toml"
)

MARKER_LENGTH = 0.05          # 50 mm april tag, in metres
TARGET_ID = None              # None = auto-pick the most frequently seen id
WH = (1280, 800)

# rigid-body marker ids in the mocap csv that bound the tag (mean = tag centre)
RB_MARKERS = [5, 2, 4, 1]

# static-hold detection: a frame is "static" if mocap speed is below this
STATIC_SPEED_MM = 3.0         # mm per frame
SUBPIX_WIN = (5, 5)
SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)

# %% Load stereo calibration
calib = toml.load(STEREO_TOML)
K0 = np.array(calib["cam0"]["camera_matrix"])
D0 = np.array(calib["cam0"]["dist_coeffs"][0]).reshape(-1, 1)
K1 = np.array(calib["cam1"]["camera_matrix"])
D1 = np.array(calib["cam1"]["dist_coeffs"][0]).reshape(-1, 1)
R_st = np.array(calib["stereo"]["R"])
T_st = np.array(calib["stereo"]["T"]).reshape(3, 1) / 1000.0   # mm -> m
RVEC_ST = cv2.Rodrigues(R_st)[0]

print(f"baseline = {np.linalg.norm(T_st) * 1000:.1f} mm   "
      f"epipolar = {calib['stereo']['epipolar_err']:.3f} px")

# tag model (metres), corner order matches aruco: TL, TR, BR, BL
MARKER_PTS = np.array(
    [
        [-MARKER_LENGTH / 2,  MARKER_LENGTH / 2, 0],
        [ MARKER_LENGTH / 2,  MARKER_LENGTH / 2, 0],
        [ MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0],
        [-MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0],
    ],
    dtype=np.float64,
)
MARKER_PTS_PROJ = MARKER_PTS.reshape(-1, 1, 3)


# %% msgpack helpers
def load_meta(path):
    with open(path, "rb") as f:
        arr = np.array(list(mp.Unpacker(f, object_hook=mpn.decode)))
    sync = arr[:, 0].astype(int).astype(bool)
    ts = arr[:, 1].astype("datetime64[us]")
    return sync, ts


sync0, ts0 = load_meta(RECORDING / "cam0_timestamp.msgpack")
sync1, ts1 = load_meta(RECORDING / "cam1_timestamp.msgpack")
print(f"cam0 frames: {len(ts0)}   cam1 frames: {len(ts1)}")

# %% Detect tag corners (with sub-pixel refinement) in every frame
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
detector = aruco.ArucoDetector(ARUCO_DICT, aruco.DetectorParameters())


def detect_corners(frame_path, n_frames):
    """Return {frame_idx: {id: corners(4,2)}} with sub-pixel refined corners."""
    out = {}
    with open(frame_path, "rb") as f:
        for idx, frame in tqdm(enumerate(mp.Unpacker(f, object_hook=mpn.decode)),
                               total=n_frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None:
                continue
            refined = {}
            for i, c in zip(ids.ravel(), corners):
                cc = c.reshape(-1, 1, 2).astype(np.float32)
                cv2.cornerSubPix(gray, cc, SUBPIX_WIN, (-1, -1), SUBPIX_CRIT)
                refined[int(i)] = cc.reshape(4, 2).astype(np.float64)
            out[idx] = refined
    return out


det0 = detect_corners(RECORDING / "cam0_frame.msgpack", len(ts0))
det1 = detect_corners(RECORDING / "cam1_frame.msgpack", len(ts1))

# %% Pick target id (most frequently seen in cam0)
if TARGET_ID is None:
    from collections import Counter
    cnt = Counter(i for f in det0 for i in det0[f])
    TARGET_ID = cnt.most_common(1)[0][0]
print(f"TARGET_ID = {TARGET_ID}")

# %% Pair cam0 frames to nearest cam1 frame by timestamp (reject large gaps)
ts1_f = ts1.astype("int64").astype(float)
ts0_f = ts0.astype("int64").astype(float)
CAM_DT = float(np.median(np.diff(ts0_f)))            # microseconds/frame
MAX_PAIR_GAP = 0.5 * CAM_DT                           # half a frame period


def nearest_cam1(idx0):
    j = int(np.searchsorted(ts1_f, ts0_f[idx0]))
    cands = [k for k in (j - 1, j) if 0 <= k < len(ts1_f)]
    if not cands:
        return None
    k = min(cands, key=lambda k: abs(ts1_f[k] - ts0_f[idx0]))
    return k if abs(ts1_f[k] - ts0_f[idx0]) <= MAX_PAIR_GAP else None


# %% Pose estimators
def kabsch(src, dst):
    """R,t mapping src->dst (dst ~= R@src + t)."""
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    Rk = Vt.T @ U.T
    if np.linalg.det(Rk) < 0:
        Vt[2] *= -1
        Rk = Vt.T @ U.T
    return Rk, cd - Rk @ cs


NAN3 = np.full(3, np.nan)
NAN33 = np.full((3, 3), np.nan)


def pose_mono(corners, K, D):
    """solvePnP on undistorted corners -> (centre m, R) in that cam's frame."""
    und = cv2.fisheye.undistortPoints(corners.reshape(-1, 1, 2), K, D, P=K)
    ok, rvec, tvec = cv2.solvePnP(MARKER_PTS, und, K, None,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return NAN3, NAN33
    return tvec.ravel(), cv2.Rodrigues(rvec)[0]


def _triangulate(c0, c1):
    n0 = cv2.fisheye.undistortPoints(c0.reshape(-1, 1, 2), K0, D0).reshape(-1, 2).T
    n1 = cv2.fisheye.undistortPoints(c1.reshape(-1, 1, 2), K1, D1).reshape(-1, 2).T
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = np.hstack([R_st, T_st])
    h = cv2.triangulatePoints(P0, P1, n0, n1)
    return (h[:3] / h[3]).T                           # (4,3) m, cam0 frame


def pose_stereo_tri(c0, c1):
    """Triangulate -> Kabsch-fit model, dropping one bad corner if needed."""
    pts3d = _triangulate(c0, c1)
    Rk, tk = kabsch(MARKER_PTS, pts3d)
    res = np.linalg.norm((Rk @ MARKER_PTS.T).T + tk - pts3d, axis=1)
    if res.max() > 3 * np.median(res) + 1e-9:         # reject worst corner, refit
        keep = np.argsort(res)[:-1]
        Rk, tk = kabsch(MARKER_PTS[keep], pts3d[keep])
    return tk, Rk


def pose_stereo_pnp(c0, c1, rvec0, tvec0):
    """Joint multi-view PnP: one pose minimising reprojection in both cams."""
    def resid(p):
        rvec, tvec = p[:3].reshape(3, 1), p[3:].reshape(3, 1)
        proj0, _ = cv2.fisheye.projectPoints(MARKER_PTS_PROJ, rvec, tvec, K0, D0)
        rvec1, tvec1 = cv2.composeRT(rvec, tvec, RVEC_ST, T_st)[:2]
        proj1, _ = cv2.fisheye.projectPoints(MARKER_PTS_PROJ, rvec1, tvec1, K1, D1)
        return np.concatenate([(proj0.reshape(-1, 2) - c0).ravel(),
                               (proj1.reshape(-1, 2) - c1).ravel()])
    x0 = np.concatenate([rvec0.ravel(), tvec0.ravel()])
    sol = least_squares(resid, x0, method="lm")
    return sol.x[3:], cv2.Rodrigues(sol.x[:3].reshape(3, 1))[0]


# %% Run all estimators per frame
EST = ["cam0", "cam1", "stereo_tri", "stereo_pnp"]
pos = {e: [] for e in EST}
rot = {e: [] for e in EST}
ts_pair = []

for idx0 in tqdm(sorted(det0)):
    if TARGET_ID not in det0[idx0]:
        continue
    c0 = det0[idx0][TARGET_ID]
    k = nearest_cam1(idx0)

    p0, r0 = pose_mono(c0, K0, D0)
    pos["cam0"].append(p0); rot["cam0"].append(r0)

    if k is not None and k in det1 and TARGET_ID in det1[k]:
        c1 = det1[k][TARGET_ID]
        p1, r1 = pose_mono(c1, K1, D1)
        pt, rt = pose_stereo_tri(c0, c1)
        pp, rp = pose_stereo_pnp(c0, c1, cv2.Rodrigues(rt)[0], pt)
    else:
        p1 = pt = pp = NAN3
        r1 = rt = rp = NAN33
    pos["cam1"].append(p1); rot["cam1"].append(r1)
    pos["stereo_tri"].append(pt); rot["stereo_tri"].append(rt)
    pos["stereo_pnp"].append(pp); rot["stereo_pnp"].append(rp)
    ts_pair.append(ts0[idx0])

for e in EST:
    pos[e] = np.array(pos[e])
    rot[e] = np.array(rot[e])
ts_pair = np.array(ts_pair)
print(f"paired tag frames: {len(ts_pair)}")

# %% Sync window (cam0 sync pulse)
pair_idx = np.searchsorted(ts0_f, ts_pair.astype("int64").astype(float))
pair_sync = np.where(pair_idx < len(sync0), sync0[np.clip(pair_idx, 0, len(sync0) - 1)], False)
start = int(np.argmax(pair_sync))
after = pair_sync[start:]
end = start + int(np.argmin(after)) if (~after).any() else len(pair_sync)
print(f"sync window: {start} -> {end}")

sl = slice(start, end)
for e in EST:
    pos[e] = pos[e][sl]
    rot[e] = rot[e][sl]
ts_win = ts_pair[sl]
cam_t = ts_win.astype("int64").astype(float)

# %% Load mocap (position + orientation)
mocap_df, st_time = read_rigid_body_csv(str(RECORDING / f"{RECORDING.name}.csv"))
sec = mocap_df["seconds"].to_numpy().astype(float)
mocap_t = (np.datetime64(st_time).astype("int64") + sec * 1e6)   # us, float

cols = [get_rb_marker_name(m) for m in RB_MARKERS]
with np.errstate(invalid="ignore"):
    mc = np.stack([
        np.nanmean([mocap_df[c["x"]].to_numpy() for c in cols], axis=0),
        np.nanmean([mocap_df[c["y"]].to_numpy() for c in cols], axis=0),
        np.nanmean([mocap_df[c["z"]].to_numpy() for c in cols], axis=0),
    ], axis=1).astype(float)

quat = mocap_df[["rb_ang_x", "rb_ang_y", "rb_ang_z", "rb_ang_w"]].to_numpy().astype(float)
q_ok = np.isfinite(quat).all(1) & np.isfinite(mc).all(1)
mocap_t, mc, quat = mocap_t[q_ok], mc[q_ok], quat[q_ok]
moc_rot = R_scipy.from_quat(quat)


def interp_mocap(time_shift_us):
    """Interpolate mocap pos + orientation onto cam_t given a time shift."""
    mt = mocap_t + time_shift_us
    p = np.column_stack([interp1d(mt, mc[:, a], fill_value="extrapolate")(cam_t)
                         for a in range(3)])
    tq = np.clip(cam_t, mt[0], mt[-1])
    r = Slerp(mt, moc_rot)(tq)
    return p, r


# %% Sub-frame time-lag correction (cross-correlate speed signals)
dt0 = cam_t[0] - mocap_t[0]                            # first-sample alignment
mocap_ip, _ = interp_mocap(dt0)

ref = pos["stereo_tri"]
cam_speed = np.linalg.norm(np.diff(ref, axis=0), axis=1)
moc_speed = np.linalg.norm(np.diff(mocap_ip, axis=0), axis=1)
cs = np.nan_to_num(cam_speed - np.nanmean(cam_speed))
ms = np.nan_to_num(moc_speed - np.nanmean(moc_speed))
xc = np.correlate(cs, ms, mode="full")
lag = int(np.argmax(xc) - (len(cs) - 1))              # frames mocap leads cam
dt = dt0 + lag * CAM_DT
print(f"time-lag correction: {lag} frames ({lag * CAM_DT / 1000:.1f} ms)")

mocap_ip, moc_rot_ip = interp_mocap(dt)

# %% Static-hold mask (low mocap speed)
moc_speed = np.r_[0, np.linalg.norm(np.diff(mocap_ip, axis=0), axis=1)] * 1e3  # mm
static = moc_speed < STATIC_SPEED_MM
print(f"static frames: {static.sum()} / {len(static)} "
      f"({100 * static.mean():.0f}%)")


# %% Evaluation
def geodesic_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def evaluate(name):
    p, r = pos[name], rot[name]
    valid = np.isfinite(p).all(1) & np.isfinite(mocap_ip).all(1)
    cam, moc = p[valid], mocap_ip[valid]
    Rk, tk = kabsch(cam, moc)                          # cam -> mocap frame
    cam_a = (Rk @ cam.T).T + tk
    resid = cam_a - moc

    rms = np.sqrt(np.nanmean(resid ** 2, axis=0)) * 1e3   # mm

    # translation jitter: frame-to-frame residual change during static holds
    st = static[valid]
    dres = np.diff(resid, axis=0)
    st_pair = st[1:] & st[:-1]
    jit = (np.nanstd(dres[st_pair], axis=0) if st_pair.any()
           else np.nanstd(dres, axis=0)) * 1e3

    # --- rotation accuracy (frame-invariant) ---
    # cam & mocap live in different world/body frames: R_moc = G R_cam B, so
    # increments conjugate (dM = G dC G^T). Recover G by aligning the rotation
    # axes of the increments, then score the residual conjugation error.
    rvalid = valid & np.isfinite(r).all((1, 2))
    idx = np.where(rvalid)[0]
    ref = idx[0]
    dC = np.array([r[i] @ r[ref].T for i in idx])
    dM = np.array([moc_rot_ip[i].as_matrix() @ moc_rot_ip[ref].as_matrix().T
                   for i in idx])
    aC = R_scipy.from_matrix(dC).as_rotvec()
    aM = R_scipy.from_matrix(dM).as_rotvec()
    big = np.linalg.norm(aC, axis=1) > np.radians(5)      # ignore near-identity
    H = aC[big].T @ aM[big]
    U, _, Vt = np.linalg.svd(H)
    G = Vt.T @ U.T
    if np.linalg.det(G) < 0:
        Vt[2] *= -1
        G = Vt.T @ U.T
    rot_err = np.array([geodesic_deg(G @ dC[j] @ G.T, dM[j])
                        for j in range(len(idx))])
    rot_rms = np.sqrt(np.mean(rot_err ** 2))

    # rotation jitter: consecutive cam orientation change during static holds
    # (median is robust to occasional planar-flip outliers)
    st_idx = idx[static[idx]]
    consec = st_idx[np.r_[False, np.diff(st_idx) == 1]]
    ang = [geodesic_deg(r[i - 1], r[i]) for i in consec]
    rot_jit = np.median(ang) if len(ang) > 1 else np.nan

    print(f"\n[{name:>10}]  n={valid.sum()}")
    print(f"  trans RMS (mm)  x={rms[0]:5.1f} y={rms[1]:5.1f} z={rms[2]:5.1f}"
          f"  | tot {np.linalg.norm(rms):5.1f}")
    print(f"  trans jit (mm)  x={jit[0]:5.1f} y={jit[1]:5.1f} z={jit[2]:5.1f}"
          f"  | tot {np.linalg.norm(jit):5.1f}")
    print(f"  rot   RMS (deg) {rot_rms:5.2f}   jit (deg, median) {rot_jit:5.2f}")
    return dict(name=name, cam_a=cam_a, valid=valid, resid=resid,
                rms=rms, jit=jit, rot_rms=rot_rms, rot_jit=rot_jit)


print("\n" + "=" * 64)
print("RMS = accuracy vs mocap | jitter = static-hold frame-to-frame noise")
print("=" * 64)
results = {e: evaluate(e) for e in EST}

# %% Plot: per-axis trajectories
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
colors = {"cam0": "tab:blue", "cam1": "tab:green",
          "stereo_tri": "tab:red", "stereo_pnp": "tab:purple"}
for a, lbl in enumerate("XYZ"):
    for e in EST:
        axes[a].plot(pos[e][:, a], color=colors[e], lw=.8, alpha=.7, label=e)
    axes[a].set_ylabel(f"{lbl} (m)")
    axes[a].grid(True, alpha=.3)
axes[0].legend(loc="upper right", ncol=4)
axes[2].set_xlabel("frame")
plt.suptitle("Tag centre: 4 estimators (raw, before alignment)")
plt.tight_layout()
plt.savefig(RECORDING / "verify_trajectories.png", dpi=120)
plt.show()

# %% Plot: jitter + RMS bars
x = np.arange(len(EST))
w = 0.25
fig, (axj, axr) = plt.subplots(1, 2, figsize=(14, 5))
for a, lbl in enumerate("XYZ"):
    axj.bar(x + (a - 1) * w, [results[e]["jit"][a] for e in EST], w, label=lbl)
    axr.bar(x + (a - 1) * w, [results[e]["rms"][a] for e in EST], w, label=lbl)
for ax, title in [(axj, "Static jitter (mm)"), (axr, "RMS vs mocap (mm)")]:
    ax.set_xticks(x); ax.set_xticklabels(EST, rotation=15)
    ax.legend(); ax.grid(True, alpha=.3); ax.set_title(title)
plt.tight_layout()
plt.savefig(RECORDING / "verify_jitter_rms.png", dpi=120)
plt.show()

# %% Plot: jitter vs depth (range)
fig, ax = plt.subplots(figsize=(10, 6))
depth = np.abs(pos["stereo_tri"][:, 2])               # m, cam0 optical axis
bins = np.linspace(np.nanpercentile(depth, 2), np.nanpercentile(depth, 98), 7)
centres = 0.5 * (bins[:-1] + bins[1:])
for e in EST:
    resid_mag = np.full(len(depth), np.nan)
    res = results[e]
    resid_mag[res["valid"]] = np.linalg.norm(res["resid"], axis=1)
    dres = np.abs(np.r_[np.nan, np.diff(resid_mag)]) * 1e3
    jit_by_bin = [np.nanstd(dres[(depth >= bins[i]) & (depth < bins[i + 1])])
                  for i in range(len(bins) - 1)]
    ax.plot(centres, jit_by_bin, "o-", color=colors[e], label=e)
ax.set_xlabel("depth (m)"); ax.set_ylabel("jitter (mm)")
ax.set_title("Jitter vs range — where stereo pulls ahead")
ax.legend(); ax.grid(True, alpha=.3)
plt.tight_layout()
plt.savefig(RECORDING / "verify_jitter_vs_depth.png", dpi=120)
plt.show()

# %% Plot: XZ top-down trajectory, one subplot per estimator (vs mocap)
fig, axes = plt.subplots(2, 2, figsize=(12, 11), sharex=True, sharey=True)
for ax, e in zip(axes.ravel(), EST):
    res = results[e]
    cam = res["cam_a"]                         # aligned cam, valid frames, metres
    moc = mocap_ip[res["valid"]]
    ax.plot(moc[:, 0], moc[:, 2], color="tab:gray", lw=1.2, label="mocap")
    ax.plot(cam[:, 0], cam[:, 2], color=colors[e], lw=.9, alpha=.8, label=e)
    ax.set_title(f"{e}   (jit {np.linalg.norm(res['jit']):.1f} mm | "
                 f"rms {np.linalg.norm(res['rms']):.1f} mm)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=.3); ax.legend(loc="upper right")
plt.suptitle("Top-down XZ trajectory vs mocap (Kabsch-aligned)")
plt.tight_layout()
plt.savefig(RECORDING / "verify_xz_trajectory.png", dpi=120)
plt.show()
