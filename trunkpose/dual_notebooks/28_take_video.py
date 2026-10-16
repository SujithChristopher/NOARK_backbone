# %% 28_take_video.py
# Plain single-camera mp4 of one subject's four movement takes (wc, woc, game,
# game_wtc) straight from the recorded msgpack -- no tracking, no overlays.
#
# Protocol takes (std_movement/<subject>_wc|woc) are written whole. The two game takes
# share one rec_* recording; each is cut to its GPIO pulse (session.toml [game] takes).
# The camera is the session's reference camera (cam1 when [camera] swap = true) and the
# frame is rotated 180 when [camera] rot180 = true, so the subject is upright.
#
#   uv run python trunkpose/dual_notebooks/28_take_video.py \
#       --session data/trunk_take_oct/oct9_26 --subject s3_lee [--speed 2] [--fps 24]
#
# Writes <session>/videos/<subject>_<take>[_<N>x].mp4 (H.264 via ffmpeg, must be on PATH)

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import msgpack as mp
import msgpack_numpy as mpn
import numpy as np
import toml
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def stream_paths(rec, stem):
    """Single <stem>.msgpack, or the game recorder's <stem>_0000.msgpack ... shards."""
    one = rec / f"{stem}.msgpack"
    if one.exists():
        return [one]
    shards = sorted(rec.glob(f"{stem}_[0-9][0-9][0-9][0-9].msgpack"))
    if not shards:
        raise SystemExit(f"No {stem} stream in {rec}")
    return shards


def records(paths):
    for p in paths:
        with open(p, "rb") as f:
            yield from mp.Unpacker(f, object_hook=mpn.decode)


def sync_pulses(sync):
    """[(rise, fall)] of every GPIO-high run; fall is one past the last high frame."""
    d = np.diff(np.concatenate(([0], (np.asarray(sync) == 1).astype(int), [0])))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def write_video(frames, times, out, lo, hi, rot180, out_fps=30, speed=1.0, crf=26):
    """Frames [lo, hi) of the stream to an H.264 mp4 at out_fps, played speed x faster.

    The recorded rate is frames / elapsed time, not 1 / median dt: the game recorder's
    frame interval jitters (median ~30 ms, mean ~34 ms), and the median would play it
    ~13% fast. Frames go to ffmpeg at recorded rate x speed and ffmpeg drops down to
    out_fps. libx264 instead of OpenCV's mp4v: ~5-10x smaller at the same look.
    """
    span = (times[hi - 1] - times[lo]).astype("timedelta64[us]").astype(float) / 1e6
    rec_fps = (hi - lo - 1) / span
    proc = None
    for i, f in enumerate(tqdm(records(frames), total=hi, desc=out.name)):
        if i < lo:
            continue
        if i >= hi:
            break
        img = cv2.rotate(f, cv2.ROTATE_180) if rot180 else f
        if proc is None:
            h, w = img.shape[:2]
            pix = "gray" if img.ndim == 2 else "bgr24"
            proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{w}x{h}",
                 "-r", f"{rec_fps * speed:.4f}", "-i", "-",
                 "-r", str(out_fps), "-c:v", "libx264", "-preset", "medium",
                 "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 str(out)],
                stdin=subprocess.PIPE)
        proc.stdin.write(np.ascontiguousarray(img).tobytes())
    proc.stdin.close()
    if proc.wait():
        raise SystemExit(f"ffmpeg failed on {out}")
    print(f"Saved {out}  frames {lo}-{hi} ({span:.1f} s real @ {rec_fps:.1f} fps -> "
          f"{span / speed:.1f} s at {speed:g}x, {out_fps} fps, "
          f"{out.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True, help="e.g. data/trunk_take_oct/oct9_26")
    ap.add_argument("--subject", required=True, help="e.g. s3_lee")
    ap.add_argument("--cam", choices=["ref", "cam0", "cam1"], default="ref",
                    help="ref = session's reference camera (cam1 when swap = true)")
    ap.add_argument("--fps", type=float, default=30, help="output fps (default 30)")
    ap.add_argument("--speed", type=float, default=1, help="playback speed-up, e.g. 2 or 3")
    ap.add_argument("--crf", type=int, default=26, help="x264 quality, higher = smaller")
    ap.add_argument("--out", default=None, help="output dir (default <session>/videos)")
    args = ap.parse_args()

    sess = Path(args.session)
    sess = sess if sess.is_absolute() else PROJECT_ROOT / sess
    cfg = toml.load(sess / "session.toml")
    cam_cfg = cfg.get("camera", {})
    rot180 = bool(cam_cfg.get("rot180", False))
    cam = args.cam if args.cam != "ref" else ("cam1" if cam_cfg.get("swap") else "cam0")
    out_dir = Path(args.out) if args.out else sess / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = f"_{args.speed:g}x" if args.speed != 1 else ""

    def load(rec):
        frames = stream_paths(rec, f"{cam}_frame")
        ts = list(records(stream_paths(rec, f"{cam}_timestamp")))
        sync = np.array([int(r[0]) for r in ts])
        times = np.array([datetime.fromisoformat(r[1]) for r in ts], dtype="datetime64[us]")
        return frames, times, sync

    # Protocol takes: whole recording.
    for take in ("wc", "woc"):
        rec = sess / "std_movement" / f"{args.subject}_{take}"
        if not rec.exists():
            print(f"Skipping {take}: {rec} not found")
            continue
        frames, times, _ = load(rec)
        write_video(frames, times, out_dir / f"{args.subject}_{take}{sfx}.mp4",
                    0, len(times), rot180, args.fps, args.speed, args.crf)

    # Game takes: one recording, cut per GPIO pulse.
    recs = sorted((sess / "game" / args.subject).glob("Session-*/Video/rec_*"))
    if not recs:
        print(f"Skipping game takes: no rec_* under {sess / 'game' / args.subject}")
        return
    frames, times, sync = load(recs[-1])
    pulses = sync_pulses(sync)
    for take, pulse in cfg.get("game", {}).get("takes", {"game": 1}).items():
        if len(pulses) < pulse:
            print(f"Skipping {take}: needs GPIO pulse {pulse}, recording has {len(pulses)}")
            continue
        lo, hi = pulses[pulse - 1]
        write_video(frames, times, out_dir / f"{args.subject}_{take}{sfx}.mp4",
                    lo, hi, rot180, args.fps, args.speed, args.crf)


if __name__ == "__main__":
    main()
