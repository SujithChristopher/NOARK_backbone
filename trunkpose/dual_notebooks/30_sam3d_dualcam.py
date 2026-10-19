# %% 30_sam3d_dualcam.py
# Second-camera spike for the SAM 3D Body teacher. A distilled student cannot beat its
# teacher, so the question is whether the teacher itself gets closer to mocap when it
# sees the subject from both cameras of the stereo pair and the two answers are fused.
#
# Needs 19 + 20 run twice on the take: as usual (<rec>/sam3d) and with --cam 1
# (<rec>/sam3d_cam1). Both exports map into the same board frame, so per frame:
#   cam0   Rb0(i)   the reference camera -- what 21 scores
#   cam1   Rb1(i)   the other camera of the pair
#   fused  chordal mean of Rb0(i) and A Rb1(i), with A = mean_i Rb0(i) Rb1(i)^T the
#          constant offset between the two cameras' answers (no mocap in it)
# Each source goes through 07's compare_mocap unchanged (own zeroing, own fitted S). The
# table uses runs/minisam3d/*_octtest/compare.py's cell -- median |err| / bias+-SD -- on
# frames where ICP and all three sources solved, so the cam0 column is the teacher
# column there (up to the common-frame rule).
#
# The view-agreement line says how far apart the two cameras' answers are frame to
# frame; the error-correlation line says how much of the teacher's error both views
# share. Shared error is what fusing cannot remove.
#
#   uv run python trunkpose/dual_notebooks/30_sam3d_dualcam.py \
#       --session data/trunk_take_oct/oct9_26 --subject s3_lee_wc
#
# Writes <rec>/sam3d_cam1/dualcam_vs_mocap<take>.npz and .png.

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SESSION = PROJECT_ROOT / "data" / "sep23_26"
sys.path.insert(0, str(SCRIPT_DIR))

AX = ("flexion", "lateral", "axial", "magnitude")
KEYS = ("flex_iz", "lat_iz", "axi_iz", "mag_i")
MKEYS = ("mflex", "mlat", "maxi", "mag_m")
SOURCES = ("cam0", "cam1", "fused")
COLORS = {"mocap": "#1a1a19", "cam0": "#2a78d6", "cam1": "#eb6834", "fused": "#1baf7a"}


def chordal_mean(Rs):
    """Rotation nearest (Frobenius) to the element-wise mean of the matrices."""
    U, _, Vt = np.linalg.svd(np.mean(Rs, axis=0))
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt


def angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def cell(e):
    return f"{np.median(np.abs(e)):4.1f} / {e.mean():+4.1f}±{e.std():3.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default=str(DEFAULT_SESSION))
    ap.add_argument("--subject", required=True)
    ap.add_argument("--game", action="store_true", help="NOARK game take")
    ap.add_argument("--stride", type=int, default=None,
                    help="must match 19's (default 3 for --game, 1 otherwise)")
    a = ap.parse_args()

    m15 = importlib.import_module("15_subject_icp")
    session = Path(a.session).resolve()
    rec, csv, _ = m15.resolve_recording(session, a.subject, a.game)
    stride = a.stride or (3 if a.game else 1)
    os.environ.update(TRUNK_RECORDING_NAME=a.subject, TRUNK_RECORDING_DIR=str(rec),
                      TRUNK_MOCAP_CSV=str(csv), TRUNK_FRAME_STRIDE=str(stride),
                      **m15.take_env(session, a.subject))
    m6 = importlib.import_module("06_trunk_axis")
    m7 = importlib.import_module("07_icp_trunk")
    m21 = importlib.import_module("21_sam3d_score")
    os.environ.setdefault("TRUNK_GEOM_CACHE", str(m6.default_geom_cache()))
    if a.game or m15.game_take(session, a.subject):
        m6.NEUTRAL_START = m6.sync_rise(m6.load_sync_flags(m6.CAM0_TIMES))

    inp = m7.load_inputs()
    frames, n = inp["frames"], inp["n"]
    R_neu = m6.assemble(frames)[1]
    Rb = {}
    for src, sub in (("cam0", "sam3d"), ("cam1", "sam3d_cam1")):
        meta = np.load(rec / sub / "meta.npz")
        if int(meta["stride"]) != stride:
            raise SystemExit(f"{sub}: 19 exported at stride {int(meta['stride'])}, "
                             f"scoring at {stride}")
        Rb[src] = m21.board_rotations(np.load(rec / sub / "sam3d_out.npz"),
                                      meta["R_model_to_board"], n)

    both = [i for i in range(n) if Rb["cam0"][i] is not None and Rb["cam1"][i] is not None]
    A = chordal_mean([Rb["cam0"][i] @ Rb["cam1"][i].T for i in both])
    Rb["fused"] = [None] * n
    for i in both:
        Rb["fused"][i] = chordal_mean([Rb["cam0"][i], A @ Rb["cam1"][i]])
    raw = np.array([angle_deg(Rb["cam0"][i] @ Rb["cam1"][i].T) for i in both])
    aligned = np.array([angle_deg(Rb["cam0"][i] @ (A @ Rb["cam1"][i]).T) for i in both])

    res = {}
    for src in SOURCES:
        print(f"\n--- {src} ---")
        res[src] = m7.compare_mocap(inp, Rb[src], R_neu)
    in_win = res["cam0"]["in_win"]
    cam = {s: {ax: m6.smooth_series(res[s][k]) for ax, k in zip(AX, KEYS)} for s in SOURCES}
    moc = {s: {ax: res[s][k] for ax, k in zip(AX, MKEYS)} for s in SOURCES}

    icp = np.load(rec / f"icp_vs_mocap{m6.RESULT_SUFFIX}{m6.TAKE_SUFFIX}.npz",
                  allow_pickle=True)
    if len(icp["in_win"]) != n:
        raise SystemExit(f"ICP result has {len(icp['in_win'])} frames, this run {n}: "
                         f"run 15 at --stride {stride} too")

    print(f"\n### {a.subject}: SAM 3D Body on both cameras vs mocap")
    print(f"  cam1->cam0 answer offset A = {angle_deg(A):.1f} deg; per-frame view "
          f"disagreement median/p95 {np.median(raw):.1f}/{np.percentile(raw, 95):.1f} deg "
          f"raw, {np.median(aligned):.1f}/{np.percentile(aligned, 95):.1f} after A "
          f"({len(both)} frames both solved)")
    print("  median |err| / bias±SD deg, frames where ICP and every source solved")
    print(f"  {'axis':<10}" + "".join(f"{s:>20}" for s in (*SOURCES, "ICP"))
          + f"{'r(err0,err1)':>14}{'n':>6}")
    keep = {}
    for ax in AX:
        m = (icp["valid_" + ax] & np.isfinite(icp["icp_" + ax]) & icp["in_win"]
             & np.isfinite(icp["moc_" + ax]) & in_win)
        for s in SOURCES:
            m &= np.isfinite(cam[s][ax]) & np.isfinite(moc[s][ax])
        err = {s: (cam[s][ax] - moc[s][ax])[m] for s in SOURCES}
        err["ICP"] = (icp["icp_" + ax] - icp["moc_" + ax])[m]
        r01 = np.corrcoef(err["cam0"], err["cam1"])[0, 1]
        print(f"  {ax:<10}" + "".join(f"{cell(err[s]):>20}" for s in (*SOURCES, "ICP"))
              + f"{r01:>+14.2f}{int(m.sum()):>6}")
        keep[ax] = m

    out = rec / "sam3d_cam1"
    dst = out / f"dualcam_vs_mocap{m6.TAKE_SUFFIX}.npz"
    np.savez_compressed(dst, subject=a.subject, in_win=in_win, A=A,
                        view_raw=raw, view_aligned=aligned,
                        **{f"{s}_{ax}": cam[s][ax] for s in SOURCES for ax in AX},
                        **{f"moc_{s}_{ax}": moc[s][ax] for s in SOURCES for ax in AX},
                        **{f"common_{ax}": keep[ax] for ax in AX})
    print(f"  -> {dst}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = (inp["ts0"] - inp["ts0"][np.argmax(in_win)]) / np.timedelta64(1, "s")
    fig, axs = plt.subplots(len(AX), 1, figsize=(12, 9), sharex=True)
    for axp, ax in zip(axs, AX):
        axp.plot(t, np.where(in_win, moc["cam0"][ax], np.nan), color=COLORS["mocap"],
                 lw=1.5, label="mocap (cam0's S)")
        for s in SOURCES:
            axp.plot(t, np.where(in_win, cam[s][ax], np.nan), color=COLORS[s], lw=1.0,
                     label=s)
        axp.set_ylabel(f"{ax} (deg)")
        axp.grid(alpha=0.25, lw=0.5)
        for side in ("top", "right"):
            axp.spines[side].set_visible(False)
    axs[0].legend(ncol=4, frameon=False, loc="upper left")
    axs[0].set_title(f"{a.subject}: SAM 3D Body trunk rotation, cam0 / cam1 / fused vs mocap")
    axs[-1].set_xlabel("time from clean-segment start (s)")
    t_in = t[in_win]
    axs[-1].set_xlim(t_in.min(), t_in.max())
    fig.tight_layout()
    fig.savefig(dst.with_suffix(".png"), dpi=130)
    print(f"  -> {dst.with_suffix('.png')}")


if __name__ == "__main__":
    main()
