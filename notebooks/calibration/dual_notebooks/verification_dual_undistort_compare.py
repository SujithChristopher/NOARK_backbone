# %% [markdown]
# # Dual-camera verification: corner-undistort vs full-image-undistort
#
# Same 4 pose estimators as `verification_dual_camera.py`, but run through TWO
# undistortion pipelines so they can be compared head-to-head:
#
#   corner   : detect tags on the RAW fisheye frame, undistort only the corner
#              POINTS (cv2.fisheye.undistortPoints). No image resampling.
#   fullimg  : remap the WHOLE image to a pinhole model (initUndistortRectifyMap
#              + remap), detect tags on the undistorted image. Detection sees
#              straight lines but pays a bilinear-resampling blur.
#
# Estimators (x both methods = 8 trajectories):
#   cam0 / cam1 (single-view PnP) | stereo_tri (triangulate+Kabsch) | stereo_pnp
#   (joint multi-view PnP). Metrics identical to the main script:
#   translation RMS + static-hold jitter, rotation RMS + jitter.

# %% Imports
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import toml
import matplotlib.pyplot as plt
from cv2 import aruco
from tqdm.auto import tqdm
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy, Slerp

warnings.filterwarnings("ignore", message="Mean of empty slice")

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

MARKER_LENGTH = 0.05
TARGET_ID = None
WH = (1280, 800)
RB_MARKERS = [5, 2, 4, 1]
STATIC_SPEED_MM = 3.0
BALANCE = 1.0                  # full-image undistort FOV (1.0 keeps all pixels)
SUBPIX_WIN = (5, 5)
SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)

METHODS = ["corner", "fullimg"]
ESTS = ["cam0", "cam1", "stereo_tri", "stereo_pnp"]

# %% Load stereo calibration
calib = toml.load(STEREO_TOML)
K0 = np.array(calib["cam0"]["camera_matrix"])
D0 = np.array(calib["cam0"]["dist_coeffs"][0]).reshape(-1, 1)
K1 = np.array(calib["cam1"]["camera_matrix"])
D1 = np.array(calib["cam1"]["dist_coeffs"][0]).reshape(-1, 1)
R_st = np.array(calib["stereo"]["R"])
T_st = np.array(calib["stereo"]["T"]).reshape(3, 1) / 1000.0
RVEC_ST = cv2.Rodrigues(R_st)[0]
print(f"baseline = {np.linalg.norm(T_st) * 1000:.1f} mm   "
      f"epipolar = {calib['stereo']['epipolar_err']:.3f} px")

# new (pinhole) camera matrices + remap tables for the full-image pipeline
ZERO_D = np.zeros((4, 1))
nK0 = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K0, D0, WH, np.eye(3), balance=BALANCE)
nK1 = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K1, D1, WH, np.eye(3), balance=BALANCE)
map0 = cv2.fisheye.initUndistortRectifyMap(K0, D0, np.eye(3), nK0, WH, cv2.CV_16SC2)
map1 = cv2.fisheye.initUndistortRectifyMap(K1, D1, np.eye(3), nK1, WH, cv2.CV_16SC2)

# per-method, per-cam intrinsics used by PnP / projection
INTR = {
    "corner":  {0: (K0, D0), 1: (K1, D1)},
    "fullimg": {0: (nK0, ZERO_D), 1: (nK1, ZERO_D)},
}

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
    return arr[:, 0].astype(int).astype(bool), arr[:, 1].astype("datetime64[us]")


sync0, ts0 = load_meta(RECORDING / "cam0_timestamp.msgpack")
sync1, ts1 = load_meta(RECORDING / "cam1_timestamp.msgpack")
print(f"cam0 frames: {len(ts0)}   cam1 frames: {len(ts1)}")

# %% Detect tags on BOTH raw and undistorted images, in one pass per camera
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
detector = aruco.ArucoDetector(ARUCO_DICT, aruco.DetectorParameters())


def _detect(gray):
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    out = {}
    for i, c in zip(ids.ravel(), corners):
        cc = c.reshape(-1, 1, 2).astype(np.float32)
        cv2.cornerSubPix(gray, cc, SUBPIX_WIN, (-1, -1), SUBPIX_CRIT)
        out[int(i)] = cc.reshape(4, 2).astype(np.float64)
    return out


def detect_both(frame_path, n_frames, remap):
    raw, und = {}, {}
    with open(frame_path, "rb") as f:
        for idx, frame in tqdm(enumerate(mp.Unpacker(f, object_hook=mpn.decode)),
                               total=n_frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
            r = _detect(gray)
            if r:
                raw[idx] = r
            u = _detect(cv2.remap(gray, remap[0], remap[1], cv2.INTER_LINEAR))
            if u:
                und[idx] = u
    return raw, und


raw0, und0 = detect_both(RECORDING / "cam0_frame.msgpack", len(ts0), map0)
raw1, und1 = detect_both(RECORDING / "cam1_frame.msgpack", len(ts1), map1)
DET = {"corner": (raw0, raw1), "fullimg": (und0, und1)}

# %% Target id (most frequent in raw cam0)
if TARGET_ID is None:
    from collections import Counter
    cnt = Counter(i for f in raw0 for i in raw0[f])
    TARGET_ID = cnt.most_common(1)[0][0]
print(f"TARGET_ID = {TARGET_ID}")

# %% Frame pairing (timestamp nearest, reject > half frame gap)
ts0_f = ts0.astype("int64").astype(float)
ts1_f = ts1.astype("int64").astype(float)
CAM_DT = float(np.median(np.diff(ts0_f)))
MAX_GAP = 0.5 * CAM_DT


def nearest_cam1(idx0):
    j = int(np.searchsorted(ts1_f, ts0_f[idx0]))
    cands = [k for k in (j - 1, j) if 0 <= k < len(ts1_f)]
    if not cands:
        return None
    k = min(cands, key=lambda k: abs(ts1_f[k] - ts0_f[idx0]))
    return k if abs(ts1_f[k] - ts0_f[idx0]) <= MAX_GAP else None


# %% Pose estimators (method-aware)
def kabsch(src, dst):
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    Rk = Vt.T @ U.T
    if np.linalg.det(Rk) < 0:
        Vt[2] *= -1
        Rk = Vt.T @ U.T
    return Rk, cd - Rk @ cs


NAN3, NAN33 = np.full(3, np.nan), np.full((3, 3), np.nan)


def normalize(pts, method, cam):
    """Pixel corners -> normalized image coords (z=1 plane)."""
    if method == "corner":
        K, D = INTR["corner"][cam]
        return cv2.fisheye.undistortPoints(pts.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    K = INTR["fullimg"][cam][0]
    h = np.c_[pts, np.ones(len(pts))]
    n = (np.linalg.inv(K) @ h.T).T
    return n[:, :2]


def pose_mono(corners, method, cam):
    K, D = INTR[method][cam]
    if method == "corner":
        pts = cv2.fisheye.undistortPoints(corners.reshape(-1, 1, 2), K, D, P=K)
    else:
        pts = corners.reshape(-1, 1, 2)            # already undistorted pixels
    ok, rvec, tvec = cv2.solvePnP(MARKER_PTS, pts, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return NAN3, NAN33
    return tvec.ravel(), cv2.Rodrigues(rvec)[0]


def pose_stereo_tri(c0, c1, method):
    n0 = normalize(c0, method, 0).T
    n1 = normalize(c1, method, 1).T
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = np.hstack([R_st, T_st])
    h = cv2.triangulatePoints(P0, P1, n0, n1)
    pts3d = (h[:3] / h[3]).T
    Rk, tk = kabsch(MARKER_PTS, pts3d)
    res = np.linalg.norm((Rk @ MARKER_PTS.T).T + tk - pts3d, axis=1)
    if res.max() > 3 * np.median(res) + 1e-9:
        keep = np.argsort(res)[:-1]
        Rk, tk = kabsch(MARKER_PTS[keep], pts3d[keep])
    return tk, Rk


def _project(rvec, tvec, method, cam):
    K, D = INTR[method][cam]
    if method == "corner":
        proj, _ = cv2.fisheye.projectPoints(MARKER_PTS_PROJ, rvec, tvec, K, D)
    else:
        proj, _ = cv2.projectPoints(MARKER_PTS_PROJ, rvec, tvec, K, ZERO_D)
    return proj.reshape(-1, 2)


def pose_stereo_pnp(c0, c1, method, rvec0, tvec0):
    def resid(p):
        rvec, tvec = p[:3].reshape(3, 1), p[3:].reshape(3, 1)
        r0 = _project(rvec, tvec, method, 0) - c0
        rvec1, tvec1 = cv2.composeRT(rvec, tvec, RVEC_ST, T_st)[:2]
        r1 = _project(rvec1, tvec1, method, 1) - c1
        return np.concatenate([r0.ravel(), r1.ravel()])
    sol = least_squares(resid, np.concatenate([rvec0.ravel(), tvec0.ravel()]),
                        method="lm")
    return sol.x[3:], cv2.Rodrigues(sol.x[:3].reshape(3, 1))[0]


# %% Run every method x estimator
pos = {f"{m}:{e}": [] for m in METHODS for e in ESTS}
rot = {f"{m}:{e}": [] for m in METHODS for e in ESTS}
ts_pair = []

common0 = sorted(set(raw0) | set(und0))
for idx0 in tqdm(common0):
    if not any(TARGET_ID in DET[m][0].get(idx0, {}) for m in METHODS):
        continue
    k = nearest_cam1(idx0)
    for m in METHODS:
        d0, d1 = DET[m]
        c0 = d0.get(idx0, {}).get(TARGET_ID)
        if c0 is None:
            for e in ESTS:
                pos[f"{m}:{e}"].append(NAN3); rot[f"{m}:{e}"].append(NAN33)
            continue
        p0, r0 = pose_mono(c0, m, 0)
        c1 = d1.get(k, {}).get(TARGET_ID) if k is not None else None
        if c1 is not None:
            p1, r1 = pose_mono(c1, m, 1)
            pt, rt = pose_stereo_tri(c0, c1, m)
            pp, rp = pose_stereo_pnp(c0, c1, m, cv2.Rodrigues(rt)[0], pt)
        else:
            p1 = pt = pp = NAN3
            r1 = rt = rp = NAN33
        for e, (pv, rv) in zip(ESTS, [(p0, r0), (p1, r1), (pt, rt), (pp, rp)]):
            pos[f"{m}:{e}"].append(pv); rot[f"{m}:{e}"].append(rv)
    ts_pair.append(ts0[idx0])

for key in pos:
    pos[key] = np.array(pos[key])
    rot[key] = np.array(rot[key])
ts_pair = np.array(ts_pair)
print(f"paired tag frames: {len(ts_pair)}")

# %% Sync window
pair_idx = np.searchsorted(ts0_f, ts_pair.astype("int64").astype(float))
pair_sync = np.where(pair_idx < len(sync0), sync0[np.clip(pair_idx, 0, len(sync0) - 1)], False)
start = int(np.argmax(pair_sync))
after = pair_sync[start:]
end = start + int(np.argmin(after)) if (~after).any() else len(pair_sync)
print(f"sync window: {start} -> {end}")
sl = slice(start, end)
for key in pos:
    pos[key] = pos[key][sl]
    rot[key] = rot[key][sl]
cam_t = ts_pair[sl].astype("int64").astype(float)

# %% Mocap load + interpolation
mocap_df, st_time = read_rigid_body_csv(str(RECORDING / f"{RECORDING.name}.csv"))
sec = mocap_df["seconds"].to_numpy().astype(float)
mocap_t = np.datetime64(st_time).astype("int64") + sec * 1e6
cols = [get_rb_marker_name(m) for m in RB_MARKERS]
with np.errstate(invalid="ignore"):
    mc = np.stack([
        np.nanmean([mocap_df[c["x"]].to_numpy() for c in cols], axis=0),
        np.nanmean([mocap_df[c["y"]].to_numpy() for c in cols], axis=0),
        np.nanmean([mocap_df[c["z"]].to_numpy() for c in cols], axis=0),
    ], axis=1).astype(float)
quat = mocap_df[["rb_ang_x", "rb_ang_y", "rb_ang_z", "rb_ang_w"]].to_numpy().astype(float)
ok = np.isfinite(quat).all(1) & np.isfinite(mc).all(1)
mocap_t, mc, quat = mocap_t[ok], mc[ok], quat[ok]
moc_rot = R_scipy.from_quat(quat)


def interp_mocap(shift):
    mt = mocap_t + shift
    p = np.column_stack([interp1d(mt, mc[:, a], fill_value="extrapolate")(cam_t)
                         for a in range(3)])
    r = Slerp(mt, moc_rot)(np.clip(cam_t, mt[0], mt[-1]))
    return p, r


# %% Time-lag correction (ref = corner:stereo_pnp, NaN-interpolated for xcorr)
dt0 = cam_t[0] - mocap_t[0]
mocap_ip, _ = interp_mocap(dt0)


def _fill_nan(a):
    a = a.copy()
    idx = np.arange(len(a))
    for c in range(a.shape[1]):
        good = np.isfinite(a[:, c])
        a[~good, c] = np.interp(idx[~good], idx[good], a[good, c])
    return a


ref = _fill_nan(pos["corner:stereo_pnp"])
cspeed = np.linalg.norm(np.diff(ref, axis=0), axis=1)
mspeed = np.linalg.norm(np.diff(mocap_ip, axis=0), axis=1)
cs = cspeed - cspeed.mean()
ms = mspeed - mspeed.mean()
lag = int(np.argmax(np.correlate(cs, ms, "full")) - (len(cs) - 1))
dt = dt0 + lag * CAM_DT
print(f"time-lag correction: {lag} frames ({lag * CAM_DT / 1000:.1f} ms)")
mocap_ip, moc_rot_ip = interp_mocap(dt)
moc_speed = np.r_[0, np.linalg.norm(np.diff(mocap_ip, axis=0), axis=1)] * 1e3
static = moc_speed < STATIC_SPEED_MM
print(f"static frames: {static.sum()} / {len(static)} ({100 * static.mean():.0f}%)")


# %% Evaluate
def geodesic_deg(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def evaluate(key):
    p, r = pos[key], rot[key]
    valid = np.isfinite(p).all(1) & np.isfinite(mocap_ip).all(1)
    cam, moc = p[valid], mocap_ip[valid]
    Rk, tk = kabsch(cam, moc)
    resid = (Rk @ cam.T).T + tk - moc
    rms = np.sqrt(np.nanmean(resid ** 2, axis=0)) * 1e3
    st = static[valid]
    dres = np.diff(resid, axis=0)
    sp = st[1:] & st[:-1]
    jit = (np.nanstd(dres[sp], axis=0) if sp.any() else np.nanstd(dres, axis=0)) * 1e3

    rvalid = valid & np.isfinite(r).all((1, 2))
    idx = np.where(rvalid)[0]
    refi = idx[0]
    dC = np.array([r[i] @ r[refi].T for i in idx])
    dM = np.array([moc_rot_ip[i].as_matrix() @ moc_rot_ip[refi].as_matrix().T for i in idx])
    aC = R_scipy.from_matrix(dC).as_rotvec()
    aM = R_scipy.from_matrix(dM).as_rotvec()
    big = np.linalg.norm(aC, axis=1) > np.radians(5)
    U, _, Vt = np.linalg.svd(aC[big].T @ aM[big])
    G = Vt.T @ U.T
    if np.linalg.det(G) < 0:
        Vt[2] *= -1
        G = Vt.T @ U.T
    rot_rms = np.sqrt(np.mean([geodesic_deg(G @ dC[j] @ G.T, dM[j])
                               for j in range(len(idx))]) ** 2)
    st_idx = idx[static[idx]]
    consec = st_idx[np.r_[False, np.diff(st_idx) == 1]]
    ang = [geodesic_deg(r[i - 1], r[i]) for i in consec]
    rot_jit = np.median(ang) if len(ang) > 1 else np.nan
    return dict(rms=rms, jit=jit, rot_rms=rot_rms, rot_jit=rot_jit, n=int(valid.sum()))


results = {f"{m}:{e}": evaluate(f"{m}:{e}") for m in METHODS for e in ESTS}

# %% Print comparison table
print("\n" + "=" * 78)
print(f"{'method:estimator':22} {'transRMS':>9} {'transJIT':>9} "
      f"{'Zjit':>6} {'rotRMS':>7} {'rotJIT':>7}")
print("-" * 78)
for e in ESTS:
    for m in METHODS:
        r = results[f"{m}:{e}"]
        print(f"{m + ':' + e:22} {np.linalg.norm(r['rms']):8.1f}  "
              f"{np.linalg.norm(r['jit']):8.1f}  {r['jit'][2]:5.1f}  "
              f"{r['rot_rms']:6.1f}  {r['rot_jit']:6.2f}")
    print()

# %% Plot: corner vs fullimg per estimator
x = np.arange(len(ESTS))
w = 0.38
mcol = {"corner": "tab:blue", "fullimg": "tab:orange"}
fig, (aj, ar) = plt.subplots(1, 2, figsize=(14, 5))
for j, m in enumerate(METHODS):
    aj.bar(x + (j - .5) * w, [np.linalg.norm(results[f"{m}:{e}"]["jit"]) for e in ESTS],
           w, color=mcol[m], label=m)
    ar.bar(x + (j - .5) * w, [np.linalg.norm(results[f"{m}:{e}"]["rms"]) for e in ESTS],
           w, color=mcol[m], label=m)
for ax, title in [(aj, "Static jitter total (mm)"), (ar, "RMS total vs mocap (mm)")]:
    ax.set_xticks(x); ax.set_xticklabels(ESTS, rotation=15)
    ax.legend(); ax.grid(True, alpha=.3); ax.set_title(title)
plt.suptitle("Corner-undistort vs full-image-undistort")
plt.tight_layout()
plt.savefig(RECORDING / "verify_undistort_compare.png", dpi=120)
plt.show()
