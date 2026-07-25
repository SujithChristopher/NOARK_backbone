# %% 08_basis_check.py
# Independent check of the ChArUco mocap->board basis (charuco_basis.toml [mocap]).
#
# 07_icp_trunk.py found a ~136 deg residual frame rotation S between camera-ICP and
# mocap body ROTATIONS. That estimate could in principle be blamed on either side.
# Here we test the basis with POSITIONS instead: the camera shoulder-mid point and
# mocap trunk Marker1 ride on the same body, so after mapping both into the board
# frame their VELOCITY vectors must be parallel. The best-fit rotation M with
# v_cam ~ M @ v_moc (Kabsch over time) measures the true mocap->board frame error:
#   M ~ identity  -> basis fine, S came from something else
#   M ~ S         -> confirms the basis rotation is wrong; patch Rm_c -> Rm_c @ M...
# (patching is done by 09_fix_basis.py after reviewing this output)
#
# Uses only the geometry cache (cam_pts) -- fast, no stereo/MediaPipe.

import importlib
import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
m6 = importlib.import_module("06_trunk_axis")

OUT_PNG = m6.RECORDING_DIR / "basis_check.png"
MIN_SPEED = 0.02  # m/s: only use samples where the body actually moves


def main():
    R0_c, t0_c, Rm_c, tm_c = m6.load_charuco_basis(m6.CHARUCO_TOML)
    mocap_df, st_time = m6.read_rigid_body_csv(
        str(m6.RECORDING_DIR / f"{m6.RECORDING_NAME}.csv"))
    mocap_time = np.datetime64(st_time) + (
        mocap_df["seconds"].to_numpy() * 1e6).astype("timedelta64[us]")
    ts0 = m6.load_frame_times(m6.RECORDING_DIR / "cam0_timestamp.msgpack")

    cache = os.environ.get("TRUNK_GEOM_CACHE", "")
    if not (cache and Path(cache).exists()):
        raise SystemExit("Set TRUNK_GEOM_CACHE to the 06 geometry cache.")
    with open(cache, "rb") as f:
        frames = pickle.load(f)
    n = min(len(frames), len(ts0))
    frames, ts0 = frames[:n], ts0[:n]

    cam = np.array([
        (fr["cam_pts"]["L_shoulder"] + fr["cam_pts"]["R_shoulder"]) / 2
        if fr["cam_pts"]["L_shoulder"] is not None
        and fr["cam_pts"]["R_shoulder"] is not None else [np.nan] * 3
        for fr in frames])
    m1, _, _ = m6.load_trunk_rb(str(m6.RECORDING_DIR / f"{m6.RECORDING_NAME}.csv"))
    m1b = (m1 - tm_c) @ Rm_c

    # coarse sync (speed xcorr), then sample mocap at camera times
    period = float(np.median(np.diff(ts0) / np.timedelta64(1, "s")))
    moc0 = np.array([m1b[m6.nearest_index(mocap_time, t)] for t in ts0])
    corr, lagL = m6.best_lag(m6.series_speed(cam), m6.series_speed(moc0),
                             max_lag=int(round(m6.MAX_SYNC_LAG_S / period)))
    mt = mocap_time - np.timedelta64(int(round(lagL * period * 1e6)), "us")
    moc = np.array([m1b[m6.nearest_index(mt, t)] for t in ts0])
    in_win = (ts0 >= mt[0]) & (ts0 <= mt[-1])

    # smoothed velocities where both tracks are valid and moving
    def vel(p):
        s = np.stack([m6.smooth_series(p[:, c], window=15) for c in range(3)], axis=1)
        return np.gradient(s, axis=0)

    vc, vm = vel(cam), vel(moc)
    ok = (np.isfinite(vc).all(1) & np.isfinite(vm).all(1) & in_win
          & (np.linalg.norm(vc, axis=1) / period > MIN_SPEED)
          & (np.linalg.norm(vm, axis=1) / period > MIN_SPEED))
    print(f"sync lag {lagL * period:+.2f}s (r={corr:.2f}); {ok.sum()} moving samples")
    if ok.sum() < 100:
        raise SystemExit("Not enough moving samples for a rotation fit.")

    # Kabsch: v_cam ~ M @ v_moc
    H = vm[ok].T @ vc[ok]
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))])
    M = Vt.T @ D @ U.T
    ang = np.degrees(np.linalg.norm(m6.Rotation.from_matrix(M).as_rotvec()))
    resid = vc[ok] - vm[ok] @ M.T
    fit = 1 - (resid ** 2).sum() / ((vc[ok] - vc[ok].mean(0)) ** 2).sum()
    print(f"velocity-fit rotation M: {ang:.1f} deg (R^2={fit:.2f})")
    print("M =\n", np.array_str(M, precision=3, suppress_small=True))
    print("suggested corrected mocap rotation Rm_c @ M.T =\n",
          np.array_str(Rm_c @ M.T, precision=3, suppress_small=True))

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    t_s = ((ts0 - ts0[0]) / np.timedelta64(1, "s")).astype(float)
    moc_fixed = (moc - np.nanmean(moc, axis=0)) @ M.T + np.nanmean(cam[in_win], axis=0)
    for k, lab in enumerate("XYZ"):
        axes[k].plot(t_s, cam[:, k], color="tab:blue", lw=1, label="camera shoulder-mid")
        axes[k].plot(t_s, moc[:, k] - np.nanmean(moc[:, k]) + np.nanmean(cam[in_win, k]),
                     color="orange", lw=1, label="mocap m1 (current basis, centered)")
        axes[k].plot(t_s, moc_fixed[:, k], color="tab:green", lw=1,
                     label="mocap m1 (M-corrected, centered)")
        axes[k].set_ylabel(f"{lab} (m)")
        axes[k].grid(alpha=0.3)
        if k == 0:
            axes[k].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Board-frame position tracks -- velocity-fit rotation {ang:.1f} deg")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)
    print(f"Plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()
