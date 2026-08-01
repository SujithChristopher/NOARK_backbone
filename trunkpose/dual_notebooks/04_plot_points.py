# %% Imports

import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import mediapipe as mp
import msgpack
import msgpack_numpy as mpn
import numpy as np
import toml
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pd_support import read_rigid_body_csv  # noqa: E402

# %% Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

RECORDING_NAME = "dual_160_trunk_ragav"
RECORDING_DIR = PROJECT_ROOT / "data" / "trunk_july1_2026" / RECORDING_NAME
STEREO_TOML = (
    PROJECT_ROOT
    / "data" / "calibration" / "dual_160"
    / "calib_cz30_dual_v2" / "stereo_calibration.toml"
)
CHARUCO_TOML = (
    PROJECT_ROOT
    / "data" / "trunk_july1_2026"
    / "dual_160_tframe_july1" / "charuco_basis.toml"
)
POSE_MODEL = SCRIPT_DIR / "pose_landmarker_full.task"
OUT_VIDEO = RECORDING_DIR / "upperlimb_3d_charuco.mp4"

# %% Config
CAM_SIZE = (1280, 800)  # (W, H) -- both OV9281
VISIBILITY_THRESH = 0.5
PANEL = 720  # square video panel, px
AXIS_PAD_M = 0.3
Z_MAX_M = 5.0  # reject triangulated points beyond this depth in cam0 -- misdetections
# (e.g. background clutter mistaken for a joint, or near-parallel rays from a bad
# stereo correspondence) can triangulate to wildly wrong depth; a plausible person
# is within a few metres of the rig.

# BlazePose landmark indices (MediaPipe PoseLandmarker, 33-point)
UPPER_LIMB = {
    "L_shoulder": 11, "R_shoulder": 12,
    "L_elbow": 13, "R_elbow": 14,
    "L_wrist": 15, "R_wrist": 16,
}
BONES = [
    ("L_shoulder", "R_shoulder"),
    ("L_shoulder", "L_elbow"), ("L_elbow", "L_wrist"),
    ("R_shoulder", "R_elbow"), ("R_elbow", "R_wrist"),
]
# optitrack "body:xx" free markers -> matching UPPER_LIMB joint (see pd_support
# .read_rigid_body_csv: "body:re" becomes column prefix "mre", etc)
MOCAP_MARKERS = {
    "L_shoulder": "mls", "R_shoulder": "mrs",
    "L_elbow": "mle", "R_elbow": "mre",
    "L_wrist": "mlw", "R_wrist": "mrw",
}

CAM_JOINT_COLORS = {
    "L_shoulder": "limegreen", "R_shoulder": "dodgerblue",
    "L_elbow": "green", "R_elbow": "royalblue",
    "L_wrist": "darkgreen", "R_wrist": "navy",
}
CAM_BONE_COLOR = "white"
MOCAP_COLOR = "orange"

FALLBACK_LIMITS = {"x": (-1, 1), "y": (-1, 1), "z": (-1, 1)}


# %% Calibration / reference frame
def load_stereo(path):
    d = toml.load(path)
    K0 = np.array(d["cam0"]["camera_matrix"])
    D0 = np.array(d["cam0"]["dist_coeffs"][0]).reshape(-1, 1)
    K1 = np.array(d["cam1"]["camera_matrix"])
    D1 = np.array(d["cam1"]["dist_coeffs"][0]).reshape(-1, 1)
    R = np.array(d["stereo"]["R"])
    T = np.array(d["stereo"]["T"]).reshape(3, 1) / 1000.0  # mm -> m
    return K0, D0, K1, D1, R, T


def load_charuco_basis(path):
    d = toml.load(path)
    R0 = np.array(d["cam0"]["rotation"])
    t0 = np.array(d["cam0"]["origin"])
    Rm = np.array(d["mocap"]["rotation"])
    tm = np.array(d["mocap"]["origin"])
    return R0, t0, Rm, tm


def to_board(p_cam0, R0, t0):
    """cam0-frame point (m) -> charuco board frame."""
    return R0.T @ (p_cam0 - t0)


def mocap_to_board(p_world, Rm, tm):
    """optitrack-world point (m) -> charuco board frame."""
    return Rm.T @ (p_world - tm)


# %% Frame / timestamp loading
def iter_frame_pairs(path0, path1):
    with open(path0, "rb") as f0, open(path1, "rb") as f1:
        u0 = msgpack.Unpacker(f0, object_hook=mpn.decode)
        u1 = msgpack.Unpacker(f1, object_hook=mpn.decode)
        yield from zip(u0, u1)


def load_frame_times(path):
    with open(path, "rb") as f:
        recs = list(msgpack.Unpacker(f, object_hook=mpn.decode))
    return np.array(
        [datetime.fromisoformat(r[1]) for r in recs], dtype="datetime64[us]"
    )


def load_sync_flags(path):
    """GPIO pin-17 bit per frame (column 0 of the timestamp msgpack), HIGH while Motive
    records. Same helper as 06_trunk_axis.load_sync_flags."""
    with open(path, "rb") as f:
        recs = list(msgpack.Unpacker(f, object_hook=mpn.decode))
    return np.array([int(r[0]) for r in recs])


def gpio_mocap_time(ts0, seconds, sync):
    """Mocap timeline on the CAMERA clock. The first rising edge of the GPIO bit is
    mocap seconds=0 -- the two systems share a wire, so this is exact. Motive's own
    'Capture Start Time' header disagrees by seconds (this take: -2.38 s), and xcorr
    sync was worse still, so neither is used here."""
    if not (sync == 1).any():
        raise SystemExit("No GPIO sync pulse in cam0_timestamp.msgpack.")
    rise = int(np.argmax(sync == 1))
    return ts0[rise] + (seconds * 1e6).astype("timedelta64[us]"), rise


def nearest_index(times, t):
    j = int(np.searchsorted(times, t))
    cands = [k for k in (j - 1, j) if 0 <= k < len(times)]
    return min(cands, key=lambda k: abs(times[k] - t))


# %% Stereo triangulation (fisheye, meters, cam0 frame)
def triangulate(p0_px, p1_px, K0, D0, K1, D1, R, T):
    n0 = cv2.fisheye.undistortPoints(
        np.array([[p0_px]], dtype=np.float64), K0, D0
    ).reshape(-1, 2).T
    n1 = cv2.fisheye.undistortPoints(
        np.array([[p1_px]], dtype=np.float64), K1, D1
    ).reshape(-1, 2).T
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = np.hstack([R, T])
    h = cv2.triangulatePoints(P0, P1, n0, n1)
    if abs(h[3, 0]) < 1e-9:
        return None
    p_cam0 = (h[:3] / h[3]).ravel()
    if not (0 < p_cam0[2] < Z_MAX_M):
        return None
    return p_cam0


# %% MediaPipe PoseLandmarker
def make_pose_landmarker():
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def landmark_px(lms, idx, W, H):
    if lms is None:
        return None
    lm = lms[idx]
    if lm.visibility < VISIBILITY_THRESH:
        return None
    return (float(np.clip(lm.x * W, 0, W - 1)), float(np.clip(lm.y * H, 0, H - 1)))


# %% Rendering
def render_frame(fig, ax, cam_pts, mocap_pts, limits):
    ax.cla()
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")
    ax.set_xlim(*limits["x"])
    ax.set_ylim(*limits["y"])
    ax.set_zlim(*limits["z"])
    ax.set_xlabel("X (m)", color="gray", fontsize=8)
    ax.set_ylabel("Y (m)", color="gray", fontsize=8)
    ax.set_zlabel("Z (m)", color="gray", fontsize=8)
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")

    for a, b in BONES:
        pa, pb = cam_pts.get(a), cam_pts.get(b)
        if pa is not None and pb is not None:
            pts = np.stack([pa, pb])
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=CAM_BONE_COLOR, lw=2.5)
        ma, mb = mocap_pts.get(a), mocap_pts.get(b)
        if ma is not None and mb is not None:
            pts = np.stack([ma, mb])
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=MOCAP_COLOR, lw=1.5, ls="--")

    for name, xyz in cam_pts.items():
        if xyz is not None:
            ax.scatter(*xyz, c=CAM_JOINT_COLORS[name], s=50, zorder=5, depthshade=False)
    for name, xyz in mocap_pts.items():
        if xyz is not None:
            ax.scatter(*xyz, c=MOCAP_COLOR, marker="x", s=40, zorder=5, depthshade=False)

    # charuco board origin, for reference
    ax.scatter(0, 0, 0, c="red", marker="^", s=80, zorder=6)
    ax.text(0, 0, 0, "board", color="red", fontsize=7)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
    return cv2.cvtColor(buf[..., 1:], cv2.COLOR_RGB2BGR)  # ARGB -> RGB -> BGR


def compute_limits(all_pts, pad=AXIS_PAD_M):
    """1st/99th percentile bounds -- robust to the rare stray triangulated point."""
    pts = np.array(all_pts)
    lo = np.percentile(pts, 1, axis=0)
    hi = np.percentile(pts, 99, axis=0)
    return {
        "x": (lo[0] - pad, hi[0] + pad),
        "y": (lo[1] - pad, hi[1] + pad),
        "z": (lo[2] - pad, hi[2] + pad),
    }


# %% Mocap<->camera alignment
# The [mocap] block of charuco_basis.toml does NOT put mocap points in the board frame:
# 03_get_charuco_basis.py assumes the 4 "tframe" markers sit on the board face, but they
# are on the stand's base -- their plane normal is Motive +Y (0.014, 0.994, 0.108) while
# the board normal is nearly horizontal, i.e. the two planes are ~90 deg apart, and the
# marker-to-pattern offset was never measured. Using the TOML basis leaves the mocap
# skeleton 671 mm and 160 deg off the camera one.
#
# So the display transform is FITTED from the data (Kabsch on paired joints), fitted on
# the first half of the overlap and scored on the held-out second half. This is a
# visualisation aid, not a calibration: it must never be written back into
# charuco_basis.toml (a fitted "basis fix" was already reverted once for being circular).
def kabsch(A, B):
    """Rigid (R, t) mapping points A onto points B, no scaling."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cb - R @ ca


def fit_mocap_alignment(cam_frames, mocap_frames):
    """Fit on the first half of frames that have pairs, report on the second half."""
    idx = [i for i, (c, m) in enumerate(zip(cam_frames, mocap_frames))
           if any(c[k] is not None and m[k] is not None for k in UPPER_LIMB)]
    if len(idx) < 20:
        raise SystemExit("Too few paired camera/mocap frames to fit an alignment.")
    split = idx[len(idx) // 2]

    def pairs(lo, hi):
        A, B = [], []
        for i in range(lo, hi):
            for k in UPPER_LIMB:
                c, m = cam_frames[i][k], mocap_frames[i][k]
                if c is not None and m is not None:
                    A.append(m)
                    B.append(c)
        return np.array(A), np.array(B)

    A_fit, B_fit = pairs(idx[0], split)
    R, t = kabsch(A_fit, B_fit)
    train = np.linalg.norm((A_fit @ R.T + t) - B_fit, axis=1)
    A_val, B_val = pairs(split, idx[-1] + 1)
    val = np.linalg.norm((A_val @ R.T + t) - B_val, axis=1)
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"  fitted alignment: rotation {ang:.1f} deg, translation "
          f"{np.linalg.norm(t) * 1000:.0f} mm")
    print(f"  train (n={len(train)}): mean {train.mean() * 1000:.0f} mm, "
          f"median {np.median(train) * 1000:.0f} mm")
    print(f"  HELD-OUT (n={len(val)}): mean {val.mean() * 1000:.0f} mm, "
          f"median {np.median(val) * 1000:.0f} mm")
    return R, t


# %% Main
def main():
    print("Loading calibration + reference frame...")
    K0, D0, K1, D1, R_st, T_st = load_stereo(STEREO_TOML)
    # only the cam0 half of the basis is usable -- see the alignment note above
    R0_c, t0_c, _Rm_unused, _tm_unused = load_charuco_basis(CHARUCO_TOML)

    print("Loading mocap...")
    mocap_df, st_time = read_rigid_body_csv(str(RECORDING_DIR / f"{RECORDING_NAME}.csv"))

    print("Loading camera timestamps...")
    ts0 = load_frame_times(RECORDING_DIR / "cam0_timestamp.msgpack")
    ts1 = load_frame_times(RECORDING_DIR / "cam1_timestamp.msgpack")
    n = min(len(ts0), len(ts1))
    t_ms = ((ts0[:n] - ts0[0]) / np.timedelta64(1, "ms")).astype(np.int64)

    sync = load_sync_flags(RECORDING_DIR / "cam0_timestamp.msgpack")[:n]
    mocap_time, rise = gpio_mocap_time(
        ts0[:n], mocap_df["seconds"].to_numpy(), sync)
    wall = np.datetime64(st_time)
    print(f"  GPIO sync: mocap t=0 at camera frame {rise}; Motive header clock is "
          f"{(wall - mocap_time[0]) / np.timedelta64(1, 's'):+.2f} s off")
    in_win = (ts0[:n] >= mocap_time[0]) & (ts0[:n] <= mocap_time[-1])
    print(f"  mocap covers {int(in_win.sum())}/{n} camera frames")

    print(f"Pass 1/2: pose detection + triangulation ({n} paired frames)...")
    lmk0 = make_pose_landmarker()
    lmk1 = make_pose_landmarker()
    W, H = CAM_SIZE

    cam_frames = []    # list[dict name -> xyz(board frame) or None]
    mocap_frames = []  # list[dict name -> xyz(board frame) or None]
    all_pts = []

    frame_pairs = iter_frame_pairs(
        RECORDING_DIR / "cam0_frame.msgpack", RECORDING_DIR / "cam1_frame.msgpack"
    )
    for i, (f0, f1) in enumerate(tqdm(frame_pairs, total=n, desc="detect")):
        if i >= n:
            break
        f0_rgb = cv2.cvtColor(f0, cv2.COLOR_GRAY2RGB) if f0.ndim == 2 else f0
        f1_rgb = cv2.cvtColor(f1, cv2.COLOR_GRAY2RGB) if f1.ndim == 2 else f1
        res0 = lmk0.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=f0_rgb), int(t_ms[i])
        )
        res1 = lmk1.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=f1_rgb), int(t_ms[i])
        )
        lms0 = res0.pose_landmarks[0] if res0.pose_landmarks else None
        lms1 = res1.pose_landmarks[0] if res1.pose_landmarks else None

        cam_pts = {}
        for name, idx in UPPER_LIMB.items():
            p0 = landmark_px(lms0, idx, W, H)
            p1 = landmark_px(lms1, idx, W, H)
            xyz = None
            if p0 is not None and p1 is not None:
                p_cam0 = triangulate(p0, p1, K0, D0, K1, D1, R_st, T_st)
                if p_cam0 is not None:
                    xyz = to_board(p_cam0, R0_c, t0_c)
                    all_pts.append(xyz)
            cam_pts[name] = xyz
        cam_frames.append(cam_pts)

        # mocap markers stay in Motive world coords for now -- the alignment that maps
        # them into the board frame is fitted below, once all pairs are collected.
        mocap_pts = {}
        if in_win[i]:
            row = mocap_df.iloc[nearest_index(mocap_time, ts0[i])]
            for name, prefix in MOCAP_MARKERS.items():
                p_world = np.array(
                    [row[f"{prefix}_x"], row[f"{prefix}_y"], row[f"{prefix}_z"]],
                    dtype=float,
                )
                mocap_pts[name] = p_world if np.isfinite(p_world).all() else None
        else:
            mocap_pts = dict.fromkeys(MOCAP_MARKERS)
        mocap_frames.append(mocap_pts)

    lmk0.__exit__(None, None, None)
    lmk1.__exit__(None, None, None)

    print("Fitting mocap -> board alignment (held-out validated)...")
    R_al, t_al = fit_mocap_alignment(cam_frames, mocap_frames)
    for mocap_pts in mocap_frames:
        for name, p_world in mocap_pts.items():
            if p_world is not None:
                xyz = R_al @ p_world + t_al
                mocap_pts[name] = xyz
                all_pts.append(xyz)

    limits = compute_limits(all_pts) if all_pts else FALLBACK_LIMITS
    print(f"Axis limits (board frame, m): {limits}")

    dt = np.diff(ts0[:n]).astype("timedelta64[us]").astype(np.float64) / 1e6
    fps = float(1.0 / np.median(dt[dt > 0])) if (dt > 0).any() else 15.0
    print(f"fps ~= {fps:.2f}")

    print("Pass 2/2: rendering...")
    fig = plt.figure(figsize=(PANEL / 100, PANEL / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_VIDEO), fourcc, fps, (PANEL, PANEL))

    for cam_pts, mocap_pts in tqdm(list(zip(cam_frames, mocap_frames)), desc="render"):
        frame = render_frame(fig, ax, cam_pts, mocap_pts, limits)
        if frame.shape[1::-1] != (PANEL, PANEL):
            frame = cv2.resize(frame, (PANEL, PANEL))
        # the overlay is only as meaningful as the alignment -- say so on every frame
        cv2.putText(frame, "mocap: fitted alignment (not charuco_basis.toml)",
                    (10, PANEL - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 165, 255), 1, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    plt.close(fig)
    print(f"Done -> {OUT_VIDEO}")


if __name__ == "__main__":
    main()
