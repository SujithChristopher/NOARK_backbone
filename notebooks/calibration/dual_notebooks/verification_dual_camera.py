# %% [markdown]
# # Dual-camera stereo verification against motion capture
#
# Question being tested: does a 2nd camera reduce single-tag pose jitter?
#
# For every synced frame we estimate the tag centre 3 ways and compare each
# against the OptiTrack rigid body:
#   - cam0-only  : solvePnP on 4 undistorted corners (single-view planar PnP)
#   - cam1-only  : same, on the other camera
#   - stereo     : triangulate the 4 corners with the calibrated baseline
#                  (8 image points -> 4 metric 3D points -> centroid)
#
# Two metrics per estimator:
#   - accuracy : RMS of (cam - mocap) per axis, after Kabsch alignment
#   - jitter   : std of the frame-to-frame difference of the residual
#                (alignment-independent high-frequency noise -- the real
#                 "does it shake" number)

# %% Imports
import os
import sys
import pickle  # noqa: F401  (kept for parity with sibling notebooks)
from pathlib import Path

import cv2
import numpy as np
import toml
import polars as pl
import matplotlib.pyplot as plt
from cv2 import aruco
from tqdm.auto import tqdm
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R_scipy

sys.path.insert(1, os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.insert(1, os.path.dirname(os.getcwd()))
from pd_support import read_rigid_body_csv, get_rb_marker_name  # noqa: E402

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

# %% Load stereo calibration
calib = toml.load(STEREO_TOML)
K0 = np.array(calib["cam0"]["camera_matrix"])
D0 = np.array(calib["cam0"]["dist_coeffs"][0]).reshape(-1, 1)
K1 = np.array(calib["cam1"]["camera_matrix"])
D1 = np.array(calib["cam1"]["dist_coeffs"][0]).reshape(-1, 1)
R_st = np.array(calib["stereo"]["R"])
T_st = np.array(calib["stereo"]["T"]).reshape(3, 1) / 1000.0   # mm -> m

print(f"baseline = {np.linalg.norm(T_st) * 1000:.1f} mm   epipolar = {calib['stereo']['epipolar_err']:.3f} px")

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


# %% msgpack helpers
import msgpack as mp  # noqa: E402
import msgpack_numpy as mpn  # noqa: E402


def load_meta(path):
    with open(path, "rb") as f:
        arr = np.array(list(mp.Unpacker(f, object_hook=mpn.decode)))
    sync = arr[:, 0].astype(int).astype(bool)
    ts = arr[:, 1].astype("datetime64[us]")
    return sync, ts


sync0, ts0 = load_meta(RECORDING / "cam0_timestamp.msgpack")
sync1, ts1 = load_meta(RECORDING / "cam1_timestamp.msgpack")
print(f"cam0 frames: {len(ts0)}   cam1 frames: {len(ts1)}")

# %% Detect tag corners in every frame of each camera
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
detector = aruco.ArucoDetector(ARUCO_DICT, aruco.DetectorParameters())


def detect_corners(frame_path, n_frames):
    """Return {frame_idx: {id: corners(4,2)}} for all detected markers."""
    out = {}
    with open(frame_path, "rb") as f:
        for idx, frame in tqdm(enumerate(mp.Unpacker(f, object_hook=mpn.decode)), total=n_frames):
            corners, ids, _ = detector.detectMarkers(frame)
            if ids is None:
                continue
            out[idx] = {
                int(i): c.reshape(4, 2).astype(np.float64)
                for i, c in zip(ids.ravel(), corners)
            }
    return out


det0 = detect_corners(RECORDING / "cam0_frame.msgpack", len(ts0))
det1 = detect_corners(RECORDING / "cam1_frame.msgpack", len(ts1))

# %% Pick target id (most frequently co-visible)
if TARGET_ID is None:
    from collections import Counter
    cnt = Counter()
    for f in det0:
        for i in det0[f]:
            cnt[i] += 1
    TARGET_ID = cnt.most_common(1)[0][0]
print(f"TARGET_ID = {TARGET_ID}")

# %% Pair cam0 frames to nearest cam1 frame by timestamp
ts1_f = ts1.astype("int64").astype(float)
ts0_f = ts0.astype("int64").astype(float)


def nearest_cam1(idx0):
    j = int(np.searchsorted(ts1_f, ts0_f[idx0]))
    cands = [k for k in (j - 1, j) if 0 <= k < len(ts1_f)]
    if not cands:
        return None
    k = min(cands, key=lambda k: abs(ts1_f[k] - ts0_f[idx0]))
    # reject if temporal gap > half a frame period (~camera dt)
    return k


# %% Pose estimation per frame
def pose_mono(corners, K, D):
    """solvePnP on undistorted corners -> tag centre (cam frame, metres)."""
    und = cv2.fisheye.undistortPoints(corners.reshape(-1, 1, 2), K, D, P=K)
    ok, rvec, tvec = cv2.solvePnP(MARKER_PTS, und, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    return tvec.ravel() if ok else np.full(3, np.nan)


def pose_stereo(c0, c1):
    """Triangulate 4 corners with the baseline -> centroid (cam0 frame, metres)."""
    n0 = cv2.fisheye.undistortPoints(c0.reshape(-1, 1, 2), K0, D0).reshape(-1, 2).T
    n1 = cv2.fisheye.undistortPoints(c1.reshape(-1, 1, 2), K1, D1).reshape(-1, 2).T
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = np.hstack([R_st, T_st])
    h = cv2.triangulatePoints(P0, P1, n0, n1)   # 4x4
    pts3d = (h[:3] / h[3]).T                     # (4,3) metres, cam0 frame
    return pts3d.mean(axis=0)


traj0, traj1, trajS, ts_pair = [], [], [], []
for idx0 in sorted(det0):
    if TARGET_ID not in det0[idx0]:
        continue
    k = nearest_cam1(idx0)
    c0 = det0[idx0][TARGET_ID]

    traj0.append(pose_mono(c0, K0, D0))

    if k is not None and k in det1 and TARGET_ID in det1[k]:
        c1 = det1[k][TARGET_ID]
        traj1.append(pose_mono(c1, K1, D1))
        trajS.append(pose_stereo(c0, c1))
    else:
        traj1.append(np.full(3, np.nan))
        trajS.append(np.full(3, np.nan))
    ts_pair.append(ts0[idx0])

traj0 = np.array(traj0)
traj1 = np.array(traj1)
trajS = np.array(trajS)
ts_pair = np.array(ts_pair)
print(f"paired tag frames: {len(ts_pair)}")

# %% Sync window (cam0 sync pulse) -> trim to the active capture segment
# map paired frames back to cam0 sync via timestamp
pair_sync = np.array([sync0[np.searchsorted(ts0_f, t.astype("int64").astype(float))]
                      if np.searchsorted(ts0_f, t.astype("int64").astype(float)) < len(sync0)
                      else False for t in ts_pair])
start = int(np.argmax(pair_sync))                      # first True
after = pair_sync[start:]
end = start + int(np.argmin(after)) if (~after).any() else len(pair_sync)
print(f"sync window: {start} -> {end}")

sl = slice(start, end)
traj0, traj1, trajS, ts_win = traj0[sl], traj1[sl], trajS[sl], ts_pair[sl]

# %% Load + interpolate mocap onto the camera timeline
mocap_df, st_time = read_rigid_body_csv(str(RECORDING / f"{RECORDING.name}.csv"))
sec = mocap_df["seconds"].to_numpy().astype(float)
mocap_t = (np.datetime64(st_time) + (sec * 1e6).astype("timedelta64[us]"))

cols = [get_rb_marker_name(m) for m in RB_MARKERS]
with np.errstate(invalid="ignore"):
    mc = np.stack([
        np.nanmean([mocap_df[c["x"]].to_numpy() for c in cols], axis=0),
        np.nanmean([mocap_df[c["y"]].to_numpy() for c in cols], axis=0),
        np.nanmean([mocap_df[c["z"]].to_numpy() for c in cols], axis=0),
    ], axis=1).astype(float)

# align timelines: shift mocap so its first sample matches first cam frame
dt = ts_win[0].astype("int64") - mocap_t[0].astype("int64")
mocap_t_shift = (mocap_t.astype("int64") + dt).astype(float)
cam_t = ts_win.astype("int64").astype(float)

mocap_ip = np.column_stack([
    interp1d(mocap_t_shift, mc[:, a], fill_value="extrapolate")(cam_t)
    for a in range(3)
])

# %% Kabsch alignment + metrics
def kabsch(src, dst):
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    Rk = Vt.T @ U.T
    if np.linalg.det(Rk) < 0:
        Vt[2] *= -1
        Rk = Vt.T @ U.T
    return Rk, cd - Rk @ cs


def evaluate(traj, name):
    valid = ~np.isnan(traj).any(1) & ~np.isnan(mocap_ip).any(1)
    cam, moc = traj[valid], mocap_ip[valid]
    Rk, tk = kabsch(cam, moc)            # cam -> mocap frame
    cam_a = (Rk @ cam.T).T + tk
    resid = cam_a - moc                  # metres
    rms = np.sqrt(np.nanmean(resid ** 2, axis=0))
    jitter = np.nanstd(np.diff(resid, axis=0), axis=0)
    print(f"\n[{name}]  n={valid.sum()}")
    print(f"  RMS   (mm)  x={rms[0]*1e3:6.1f}  y={rms[1]*1e3:6.1f}  z={rms[2]*1e3:6.1f}"
          f"   |  total={np.linalg.norm(rms)*1e3:6.1f}")
    print(f"  jitter(mm)  x={jitter[0]*1e3:6.1f}  y={jitter[1]*1e3:6.1f}  z={jitter[2]*1e3:6.1f}"
          f"   |  total={np.linalg.norm(jitter)*1e3:6.1f}")
    return cam_a, moc, valid, rms, jitter


print("\n" + "=" * 60)
print("RMS = accuracy vs mocap   |   jitter = frame-to-frame noise")
print("=" * 60)
res0 = evaluate(traj0, "cam0-only  (4 pts)")
res1 = evaluate(traj1, "cam1-only  (4 pts)")
resS = evaluate(trajS, "stereo     (8 pts)")

# %% Plot: per-axis trajectories, all estimators vs mocap
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
labels = ["X", "Y", "Z"]
for a in range(3):
    axes[a].plot(traj0[:, a], color="tab:blue", alpha=.6, lw=.8, label="cam0 (4pt)")
    axes[a].plot(traj1[:, a], color="tab:green", alpha=.6, lw=.8, label="cam1 (4pt)")
    axes[a].plot(trajS[:, a], color="tab:red", lw=1.0, label="stereo (8pt)")
    axes[a].set_ylabel(f"{labels[a]} (m)")
    axes[a].grid(True, alpha=.3)
axes[0].legend(loc="upper right", ncol=3)
axes[2].set_xlabel("frame")
plt.suptitle("Tag centre: cam0 vs cam1 vs stereo (raw, before alignment)")
plt.tight_layout()
plt.savefig(RECORDING / "verify_trajectories.png", dpi=120)
plt.show()

# %% Plot: jitter comparison bar chart
names = ["cam0\n(4pt)", "cam1\n(4pt)", "stereo\n(8pt)"]
jit = np.array([res0[4], res1[4], resS[4]]) * 1e3   # mm
rmsv = np.array([res0[3], res1[3], resS[3]]) * 1e3
x = np.arange(3)
w = 0.25
fig, (axj, axr) = plt.subplots(1, 2, figsize=(13, 5))
for a, lbl in enumerate(labels):
    axj.bar(x + (a - 1) * w, jit[:, a], w, label=lbl)
    axr.bar(x + (a - 1) * w, rmsv[:, a], w, label=lbl)
for ax, title in [(axj, "Jitter (frame-to-frame std, mm)"), (axr, "RMS error vs mocap (mm)")]:
    ax.set_xticks(x); ax.set_xticklabels(names); ax.legend(); ax.grid(True, alpha=.3)
    ax.set_title(title)
plt.tight_layout()
plt.savefig(RECORDING / "verify_jitter_rms.png", dpi=120)
plt.show()
