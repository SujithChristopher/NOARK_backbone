# %% 21_sam3d_score.py
# Stage C of the SAM 3D Body trunk test: SAM 3D Body trunk rotation vs mocap, through
# 07's comparison unchanged (GPIO sync, teleport split, zeroing, fitted S), side by
# side with the ICP result from 15 on the same frames.
#
# Trunk rotation = global rotation of MHR joint c_spine3 (index 37, upper thoracic --
# where the sternum rigid body sits). The hips are behind the table and hallucinated
# by the model, so nothing here uses them.
#
# Axis conventions (checked on gokul, see SAM3D_BODY.md):
#   * the 3D keypoints + cam_t are in OpenCV camera axes -- they reproject onto the
#     model's own 2D keypoints at 0.0 px (the y/z-flipped alternative: 55-62 px);
#   * the per-joint global rotations are NOT: they are in the body model's y-up axes
#     and need conjugating by diag(1,-1,-1). Without it the fitted S sits 145-152 deg
#     from the geometric one; with it 62-71 deg, the same gap ICP has on the protocol
#     take (72). --no-rotflip reproduces the unflipped run. Magnitude is unaffected.
#
#   uv run python trunkpose/dual_notebooks/21_sam3d_score.py --subject gokul --game
#   uv run python trunkpose/dual_notebooks/21_sam3d_score.py --subject gokul_woc
#   ... --session data/trunk_take_oct/oct10_26 --subject s3_arjun_game_wtc --stride 6
#
# Needs <recording>/sam3d/{meta,sam3d_out}.npz (19, 20) and icp_vs_mocap<sfx>.npz (15).
# Writes <recording>/sam3d/sam3d_vs_mocap.npz -- with _<take> for a take-qualified game
# name (15.game_take), since one export/inference covers every take of the recording.

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
SHELL_BINS = [(0, 200), (200, 400), (400, 800), (800, 10 ** 9)]
C_SPINE3 = 37


def stat(x, y, m):
    """median / p95 |error|, r and n over mask m, as one table cell."""
    if m.sum() < 30:
        return f"{'n<30':>22}"
    e = np.abs(x[m] - y[m])
    r = np.corrcoef(x[m], y[m])[0, 1]
    return f"{np.median(e):5.1f}/{np.percentile(e, 95):5.1f} r{r:+.2f} n{int(m.sum()):>5}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default=str(DEFAULT_SESSION))
    ap.add_argument("--subject", required=True)
    ap.add_argument("--game", action="store_true", help="NOARK game take")
    ap.add_argument("--stride", type=int, default=None,
                    help="must match 19's (default 3 for --game, 1 otherwise)")
    ap.add_argument("--joint", type=int, default=C_SPINE3,
                    help="MHR joint whose global rotation is the trunk")
    ap.add_argument("--no-rotflip", dest="rotflip", action="store_false",
                    help="use the joint rotations as-is (y-up model axes)")
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
    os.environ.setdefault("TRUNK_GEOM_CACHE", str(m6.default_geom_cache()))
    if a.game or m15.game_take(session, a.subject):
        m6.NEUTRAL_START = m6.sync_rise(m6.load_sync_flags(m6.CAM0_TIMES))

    meta = np.load(rec / "sam3d" / "meta.npz")
    out = np.load(rec / "sam3d" / "sam3d_out.npz")
    if int(meta["stride"]) != stride:
        raise SystemExit(f"19 exported at stride {int(meta['stride'])}, scoring at {stride}")
    K = meta["K"]

    # ---- keypoint camera convention, verified against the model's own 2D output ----
    kp3, kp2, ct, good = (out["pred_keypoints_3d"], out["pred_keypoints_2d"],
                          out["pred_cam_t"], out["ok"])
    errs = {}
    for nm, C in (("opencv", np.eye(3)), ("flip_yz", np.diag([1., -1., -1.]))):
        p = (kp3[good][:, :21] + ct[good][:, None]) @ C.T
        uv = p[..., :2] / p[..., 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        errs[nm] = float(np.nanmedian(np.linalg.norm(uv - kp2[good][:, :21], axis=-1)))
    print(f"\n### {a.subject}: keypoint reprojection px "
          + ", ".join(f"{k} {v:.1f}" for k, v in errs.items()))
    if errs["opencv"] > 2.0:
        print("  !! keypoints are not in OpenCV camera axes -- recheck the convention")

    inp = m7.load_inputs()
    frames, n = inp["frames"], inp["n"]
    R_neu = m6.assemble(frames)[1]
    M = meta["R_model_to_board"]
    Cr = np.diag([1., -1., -1.]) if a.rotflip else np.eye(3)
    Rb = [None] * n
    for j, i in enumerate(out["idx"]):
        G = out["pred_global_rots"][j, a.joint]
        if i < n and good[j] and np.isfinite(G).all():
            # Cr G Cr^T: joint rotation in OpenCV camera axes. 07 only ever uses
            # Rb(i) Rb(z)^T, so M (.) is the conjugation into the board frame.
            Rb[i] = M @ Cr @ G @ Cr.T
    cmp_ = m7.compare_mocap(inp, Rb, R_neu)
    S_diff = float(np.degrees(np.linalg.norm(m6.Rotation.from_matrix(
        cmp_["S"] @ (R_neu @ cmp_["Rz_m"].T).T).as_rotvec())))

    in_win = cmp_["in_win"]
    sam = {ax: m6.smooth_series(cmp_[k]) for ax, k in zip(AX, KEYS)}
    val = {ax: np.isfinite(cmp_[k]) & in_win for ax, k in zip(AX, KEYS)}
    moc = {ax: cmp_[k] for ax, k in zip(AX, MKEYS)}

    icp = np.load(rec / f"icp_vs_mocap{m6.RESULT_SUFFIX}{m6.TAKE_SUFFIX}.npz",
                  allow_pickle=True)
    if len(icp["in_win"]) != n:
        raise SystemExit(f"ICP result has {len(icp['in_win'])} frames, this run {n}: "
                         f"run 15 at --stride {stride} too")
    icp_ok = {ax: icp["valid_" + ax] & np.isfinite(icp["icp_" + ax]) & in_win
              for ax in AX}
    npts = np.array([0 if fr["cloud"] is None else len(fr["cloud"]) for fr in frames])
    icp_sd = float(icp["S_diff_deg"]) if "S_diff_deg" in icp.files else float("nan")

    print(f"  SAM solved {int(val['magnitude'].sum())}/{int(in_win.sum())} in-window "
          f"frames; ICP solved {int(icp_ok['magnitude'].sum())}.  "
          f"S fitted vs geometric: SAM {S_diff:.0f} deg, ICP {icp_sd:.0f} deg")
    print("  |error| median/p95 deg, r, n")
    print(f"  {'frames':<16}{'axis':<10}{'SAM':>30}{'ICP':>30}")
    for name in ("all in-window", "ICP solved", "ICP failed"):
        for ax in AX:
            ms = val[ax] & np.isfinite(moc[ax])
            mi = icp_ok[ax] & np.isfinite(icp["moc_" + ax])
            if name == "ICP solved":
                ms = ms & icp_ok[ax]
            elif name == "ICP failed":
                ms, mi = ms & ~icp_ok[ax], np.zeros_like(mi)
            icp_cell = stat(icp["icp_" + ax], icp["moc_" + ax], mi) if mi.any() else ""
            print(f"  {name:<16}{ax:<10}{stat(sam[ax], moc[ax], ms):>30}{icp_cell:>30}")
    print("  magnitude by torso shell size (SAM | ICP):")
    for lo, hi in SHELL_BINS:
        sel = (npts >= lo) & (npts < hi)
        ms = val["magnitude"] & np.isfinite(moc["magnitude"]) & sel
        mi = icp_ok["magnitude"] & np.isfinite(icp["moc_magnitude"]) & sel
        icp_cell = stat(icp["icp_magnitude"], icp["moc_magnitude"], mi) if mi.any() else ""
        lab = f"{lo}-{hi if hi < 10 ** 8 else '+'}"
        print(f"    {lab:<10}{stat(sam['magnitude'], moc['magnitude'], ms):>30}{icp_cell:>30}")

    dst = rec / "sam3d" / f"sam3d_vs_mocap{m6.TAKE_SUFFIX}.npz"
    np.savez_compressed(dst, subject=a.subject, joint=a.joint, rotflip=a.rotflip,
                        in_win=in_win, npts=npts, S_diff=S_diff,
                        **{f"sam_{ax}": sam[ax] for ax in AX},
                        **{f"valid_{ax}": val[ax] for ax in AX},
                        **{f"moc_{ax}": moc[ax] for ax in AX})
    print(f"  -> {dst}")


if __name__ == "__main__":
    main()
