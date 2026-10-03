# %% 17_subject_icp_videos.py
# Render 08_icp_video.py for every subject of a multi-subject session, writing each
# video into that subject's own recording folder.
#
#   <subject>/icp_trunk_video_icp_only.mp4
#
# The videos exist to make the failure visible rather than statistical: the top-right
# panel shows the neutral cloud (grey) against the live cloud pushed through the ICP
# transform (cyan), and prints how much of the neutral shell is still visible. When the
# trunk turns away from the camera that number collapses, ICP keeps reporting a healthy
# rms on the surface it can still see, and the angle under-reads -- which is what the
# per-axis slopes below 1 in data/sep23_26/output are made of.
#
# Angle traces are ICP vs mocap only (TRUNK_ANGLE_PLOT_MODE=icp_only, 08's default);
# the plane+shoulder baseline is left out.
#
#   uv run python trunkpose/dual_notebooks/17_subject_icp_videos.py --all
#   uv run python trunkpose/dual_notebooks/17_subject_icp_videos.py --subjects kabil_wc
#
# One subprocess per subject: 06 resolves its configuration at import time from the
# recording directory, so a single interpreter cannot switch subjects (same reason as
# 15_subject_icp.py).

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SESSION = PROJECT_ROOT / "data" / "sep23_26"
VIDEO_SCRIPT = SCRIPT_DIR / "08_icp_video.py"


def subjects_of(session_dir, explicit):
    if explicit:
        return explicit
    import toml

    cfg = toml.load(session_dir / "session.toml")
    subs = cfg.get("subjects", {}).get("wc", [])
    if not subs:
        raise SystemExit(f"No subjects given and none listed in {session_dir}/session.toml")
    return subs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default=str(DEFAULT_SESSION))
    ap.add_argument("--all", action="store_true", help="every _wc subject in session.toml")
    ap.add_argument("--subjects", nargs="*", default=None, help="explicit subject list")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("TRUNK_WORKERS", "6")),
                    help="render processes per subject")
    ap.add_argument("--frames", type=int, default=0,
                    help="cap rendered frames per subject (0 = whole take)")
    ap.add_argument("--mode", default="icp_only", choices=("icp_only", "all", "mocap_only"),
                    help="which angle series the bottom-right panel draws")
    a = ap.parse_args()
    if not (a.all or a.subjects):
        ap.error("give --all or --subjects NAME [NAME ...]")

    session_dir = Path(a.session).resolve()
    subs = subjects_of(session_dir, a.subjects)
    failed = []
    for s in subs:
        rec_dir = session_dir / s
        if not rec_dir.exists():
            print(f"!! {s}: no such recording ({rec_dir}) -- skipping", flush=True)
            failed.append(s)
            continue
        print(f"\n{'=' * 70}\n== {s}\n{'=' * 70}", flush=True)
        env = {**os.environ,
               "TRUNK_RECORDING_NAME": s,
               "TRUNK_RECORDING_DIR": str(rec_dir),
               "TRUNK_ANGLE_PLOT_MODE": a.mode,
               "TRUNK_WORKERS": str(a.workers)}
        if a.frames:
            env["TRUNK_VIDEO_FRAMES"] = str(a.frames)
        r = subprocess.run([sys.executable, str(VIDEO_SCRIPT)], env=env, check=False)
        if r.returncode != 0:
            failed.append(s)
            print(f"!! {s} failed (exit {r.returncode}) -- continuing", flush=True)
    print(f"\nDone. {len(subs) - len(failed)}/{len(subs)} videos ok"
          + (f", failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
