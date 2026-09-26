# %% 15_subject_icp.py
# ICP-vs-mocap trunk angles for a multi-subject session (data/sep23_26).
#
# Per subject it runs exactly the validated 07 path -- 06 geometry (cached per
# recording) -> ICP to the neutral cloud -> GPIO sync -> teleport-split mocap ->
# residual frame rotation S -> per-axis agreement -- and writes:
#
#   <subject>/geom_cache.pkl            cached geometry pass (clouds + shoulders)
#   <subject>/icp_vs_mocap.npz          the two angle series + agreement stats
#   <output>/<subject>_icp_vs_mocap.png 3-row ICP vs mocap trace
#
# 16_subject_agreement.py then pools the npz files into the group figure.
#
# The session's rig facts (half-res sensor, swapped cameras, upside-down mounting,
# Motive rigid-body naming) live in data/sep23_26/session.toml, which 06 reads.
#
#   uv run python trunkpose/dual_notebooks/15_subject_icp.py --all
#   uv run python trunkpose/dual_notebooks/15_subject_icp.py --subject ram_wc
#
# --game switches to the NOARK game recordings (data/sep23_26/noark_game): frames in
# <subject>/Session-*/Video/rec_*/ as 900-frame chunks, one mocap folder for the whole
# session, and a camera that rolls for a minute before Motive starts. Those takes run
# strided (--stride, default 3 -> 10 Hz) and zero both systems on the GPIO rise
# instead of camera frame 0.
#
#   uv run python trunkpose/dual_notebooks/15_subject_icp.py --game --all
#
# --all spawns one subprocess per subject: 06's configuration is resolved at import
# time from the recording directory, so a single interpreter cannot switch subjects.

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT.parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SESSION = PROJECT_ROOT / "data" / "sep23_26"


# %% Recording layout
def resolve_recording(session_dir, subject, game):
    """(recording dir, mocap CSV, output dir) for this subject under either layout.

    A trunk-protocol take is one folder holding the frames and its own Motive export.
    A game take lives in the recorder's own tree and shares the session's mocap
    folder, so the CSV has to be named separately -- 06 reads it from TRUNK_MOCAP_CSV.
    """
    if not game:
        rec = session_dir / subject
        return rec, rec / f"{subject}.csv", session_dir / "output"
    root = session_dir / "noark_game"
    vids = sorted((root / "data" / subject).glob("Session-*/Video/rec_*"))
    if len(vids) != 1:
        raise SystemExit(
            f"{subject}: expected exactly one rec_* folder under "
            f"{root / 'data' / subject}, found {len(vids)}")
    return vids[0], root / "mocap" / f"{subject}_game.csv", root / "output"


def game_subjects(session_dir):
    """Subjects with both a game recording and a game mocap export."""
    root = session_dir / "noark_game"
    return sorted(q.stem[: -len("_game")] for q in (root / "mocap").glob("*_game.csv")
                  if (root / "data" / q.stem[: -len("_game")]).is_dir())


# %% Driver (one subprocess per subject)
def run_all(session_dir, subjects, workers, extra_env, game=False, stride=None):
    import toml

    if not subjects:
        subjects = (game_subjects(session_dir) if game else
                    toml.load(session_dir / "session.toml").get("subjects", {}).get("wc", []))
    if not subjects:
        raise SystemExit(f"No subjects given and none listed in {session_dir}/session.toml")
    out_dir = resolve_recording(session_dir, subjects[0], game)[2]
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for s in subjects:
        print(f"\n{'=' * 70}\n== {s}\n{'=' * 70}", flush=True)
        env = {**os.environ, **extra_env, "TRUNK_WORKERS": str(workers)}
        cmd = [sys.executable, str(SCRIPT), "--subject", s, "--session", str(session_dir)]
        if game:
            cmd.append("--game")
        if stride:
            cmd += ["--stride", str(stride)]
        r = subprocess.run(cmd, env=env, check=False)
        if r.returncode != 0:
            failed.append(s)
            print(f"!! {s} failed (exit {r.returncode}) -- continuing", flush=True)
    print(f"\nDone. {len(subjects) - len(failed)}/{len(subjects)} subjects ok"
          + (f", failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


# %% Per-subject run
def run_subject(session_dir, subject, game=False, stride=None):
    """Import 06/07 against this subject's recording and write its results."""
    rec_dir, mocap_csv, out_dir = resolve_recording(session_dir, subject, game)
    if not rec_dir.exists():
        raise SystemExit(f"No such recording: {rec_dir}")
    if not mocap_csv.exists():
        raise SystemExit(f"No mocap export for {subject}: {mocap_csv}")
    # 06 resolves paths at import time, so the environment has to be set first.
    os.environ["TRUNK_RECORDING_NAME"] = subject
    os.environ["TRUNK_RECORDING_DIR"] = str(rec_dir)
    os.environ["TRUNK_MOCAP_CSV"] = str(mocap_csv)
    if stride:
        os.environ["TRUNK_FRAME_STRIDE"] = str(stride)
    os.environ.setdefault("TRUNK_GEOM_CACHE", str(rec_dir / "geom_cache.pkl"))

    import importlib

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, str(SCRIPT_DIR))
    m6 = importlib.import_module("06_trunk_axis")
    m7 = importlib.import_module("07_icp_trunk")
    m12 = importlib.import_module("12_poster_figures")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{subject}] session={m6.SESSION_TOML} cam={m6.CAM_SIZE} "
          f"swap={m6.SWAP_CAMS} rot180={m6.ROT180} seg={m6.SEG_WEIGHTS.parent.parent.name} "
          f"stride={m6.FRAME_STRIDE} chunks={len(m6.CAM0_FRAMES)}")

    if game:
        # The camera rolls for ~40-70 s before Motive starts, so the opening frames are
        # setup, not a neutral pose. Anchor the neutral window on the GPIO rise, which
        # IS mocap t=0 -- the same instant 18_game_mocap_rom.py zeroes the mocap on, so
        # the two sides are zeroed together and the angles are comparable to it.
        sync = m6.load_sync_flags(m6.CAM0_TIMES)
        if not (sync == 1).any():
            raise SystemExit(f"{subject}: no GPIO pulse in the camera timestamps")
        m6.NEUTRAL_START = int(np.argmax(sync == 1))
        print(f"[{subject}] neutral window starts at frame {m6.NEUTRAL_START} "
              f"(GPIO rise, {m6.NEUTRAL_START * m6.FRAME_STRIDE / 30:.0f}s into the take)")

    inp = m7.load_inputs()
    frames, n, ts0 = inp["frames"], inp["n"], inp["ts0"]
    R_neu = m6.assemble(frames)[1]
    icp = m7.run_icp_pass(frames, R_neu, n)
    cmp_ = m7.compare_mocap(inp, icp["Rb_icp"], R_neu)

    in_win, zwin = cmp_["in_win"], cmp_["zwin"]
    t_s = ((ts0 - ts0[zwin[0]]) / np.timedelta64(1, "s")).astype(np.float64)
    # 06's Savitzky-Golay on the camera side only; mocap is already 100 Hz and clean
    series_icp = [m6.smooth_series(cmp_[k]) for k in ("flex_iz", "lat_iz", "axi_iz")]
    series_moc = [cmp_[k] for k in ("mflex", "mlat", "maxi")]

    # Frame-invariant cross-check. The per-axis split depends on S, the residual
    # camera<->mocap rotation that 07 fits from the rotation pairs it then scores; the
    # total rotation magnitude needs no S at all, so it says whether the camera tracked
    # the trunk independently of how the axes were assigned.
    names = ("flexion", "lateral", "axial", "magnitude")
    series_icp.append(m6.smooth_series(cmp_["mag_i"]))
    series_moc.append(cmp_["mag_m"])
    valid_icp = [np.isfinite(cmp_[k]) for k in ("flex_iz", "lat_iz", "axi_iz", "mag_i")]

    # S from the neutral pose alone (camera anatomical frame vs mocap plate frame),
    # fitted to nothing. When it disagrees with 07's fitted S the axis assignment for
    # this subject is not trustworthy, whichever one is right.
    S_geo = R_neu @ cmp_["Rz_m"].T
    S_diff = float(np.degrees(np.linalg.norm(m6.Rotation.from_matrix(
        cmp_["S"] @ S_geo.T).as_rotvec())))
    print(f"  S fitted vs geometric: {S_diff:.0f} deg apart"
          + ("  <-- axis split unreliable" if S_diff > 30 else ""))

    stats = {}
    for k, nm in enumerate(names):
        a, b = series_icp[k], series_moc[k]
        # only frames ICP actually solved: smooth_series interpolates across gaps,
        # and an interpolated value is not a measurement
        m = valid_icp[k] & np.isfinite(b) & in_win
        if m.sum() < 50:
            stats[nm] = dict(n=int(m.sum()))
            continue
        d = a[m] - b[m]
        stats[nm] = dict(
            n=int(m.sum()),
            r=float(np.corrcoef(a[m], b[m])[0, 1]),
            rmse=float(np.sqrt((d ** 2).mean())),
            bias=float(d.mean()), sd=float(d.std(ddof=1)),
            rom_icp=float(np.percentile(a[m], 97.5) - np.percentile(a[m], 2.5)),
            rom_moc=float(np.percentile(b[m], 97.5) - np.percentile(b[m], 2.5)),
            slope=float(np.polyfit(b[m], a[m], 1)[0]))
        s = stats[nm]
        print(f"  {nm:8s} r={s['r']:+.2f} RMSE={s['rmse']:4.1f}d "
              f"bias={s['bias']:+5.1f}d LoA=[{s['bias'] - 1.96 * s['sd']:+.1f}, "
              f"{s['bias'] + 1.96 * s['sd']:+.1f}]d  ROM {s['rom_icp']:.0f}/"
              f"{s['rom_moc']:.0f}d  slope={s['slope']:.2f}  n={s['n']}")

    npz = rec_dir / "icp_vs_mocap.npz"
    np.savez_compressed(
        npz, subject=subject, t_s=t_s, in_win=in_win,
        **{f"icp_{nm}": series_icp[k] for k, nm in enumerate(names)},
        **{f"moc_{nm}": series_moc[k] for k, nm in enumerate(names)},
        **{f"valid_{nm}": valid_icp[k] for k, nm in enumerate(names)},
        n_frames=n, n_icp_valid=int(np.isfinite(icp["flex"]).sum()),
        n_gated=int(icp["gated"].sum()), S_diff_deg=S_diff,
        S_deg=float(np.degrees(np.linalg.norm(
            m6.Rotation.from_matrix(cmp_["S"]).as_rotvec()))),
        **{f"{nm}_{k}": v for nm, d in stats.items() for k, v in d.items()})
    print(f"[{subject}] -> {npz}")

    # ---- per-subject trace, same style as the poster figure ----
    m12.poster_style()
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    for k, ax in enumerate(axes):
        ax.plot(t_s, series_moc[k], color=m12.C_MOCAP, lw=1.6, label="mocap")
        # gaps stay gaps: drop the frames ICP never solved
        ax.plot(t_s, np.where(valid_icp[k], series_icp[k], np.nan),
                color=m12.C_ICP[names[k]], lw=1.8, label="camera ICP")
        ax.set_ylabel(f"{names[k]}\n(deg)")
        ax.grid(alpha=0.25)
        s = stats[names[k]]
        if "r" in s:
            ax.text(0.995, 0.04, f"r = {s['r']:+.2f}   RMSE = {s['rmse']:.1f}$\\degree$",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=12)
        if k == 0:
            ax.legend(loc="upper right", ncol=2, frameon=False)
    axes[-1].set_xlabel("time (s)")
    lo, hi = t_s[in_win].min(), t_s[in_win].max()
    axes[-1].set_xlim(lo, hi)
    fig.suptitle(f"Trunk angles, ICP rigid registration vs mocap - {subject}", y=0.98)
    fig.tight_layout()
    png = out_dir / f"{subject}_icp_vs_mocap.png"
    fig.savefig(png)
    print(f"[{subject}] -> {png}")
    return 0


# %% Main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default=str(DEFAULT_SESSION),
                    help="session directory holding the subject recordings")
    ap.add_argument("--subject", help="run one subject (recording folder name)")
    ap.add_argument("--all", action="store_true", help="run every _wc subject")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="explicit subject list for --all")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("TRUNK_WORKERS", "6")),
                    help="geometry-pass processes per subject")
    ap.add_argument("--game", action="store_true",
                    help="run the NOARK game recordings under <session>/noark_game")
    ap.add_argument("--stride", type=int, default=None,
                    help="use every Nth camera frame (default 3 with --game, 1 otherwise)")
    a = ap.parse_args()
    session_dir = Path(a.session).resolve()
    # 30 Hz over six minutes is far more than the trunk motion needs, and the geometry
    # cache is ~170 kB a frame; 10 Hz keeps both the run and the cache in proportion.
    stride = a.stride if a.stride else (3 if a.game else None)
    if a.subject:
        return run_subject(session_dir, a.subject, a.game, stride)
    if a.all or a.subjects is not None:
        return run_all(session_dir, a.subjects, a.workers, {}, a.game, stride)
    ap.error("give --subject NAME or --all")


if __name__ == "__main__":
    sys.exit(main())
