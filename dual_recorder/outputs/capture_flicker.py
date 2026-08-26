"""Capture both OV9281s with the 50 Hz flicker compensation deliberately off.

dual_recorder_rcam.py normally picks ExposureTime as a whole number of mains
half-periods (10000 us at 50 Hz), which integrates the same amount of light no
matter where in the AC cycle the shutter lands. The OV9281 is global shutter, so
getting that wrong does *not* produce rolling-shutter bands - the whole frame
pulses in brightness from one frame to the next. Half a period (5000 us) is the
worst case and is the default here, so the effect is as visible as it gets.

Writes into this directory:
    flicker_50hz.mp4              both cameras side by side, with a per-frame
                                  mean-intensity readout burnt in
    flicker_50hz_timestamps.npz   sensor_ns + sequence + frame mean per camera,
                                  for plot_delay.py

The sensors are left free-running (no phase align, no mid-take resync) so the
timestamps carry the real crystal skew that plot_delay.py fits.

    uv run python dual_recorder/outputs/capture_flicker.py
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np

RCAM_SRC = os.environ.get("RCAM_PATH", "/home/radxa/Documents/rcam/src")
try:
    from rcam import Camera, list_cameras
except ImportError:
    sys.path.insert(0, RCAM_SRC)
    from rcam import Camera, list_cameras

HERE = os.path.dirname(os.path.abspath(__file__))


class Grabber(threading.Thread):
    """Pull frames from one camera into RAM for the length of the take.

    Ten seconds of 1280x800 mono at 30 fps is ~300 MB per camera, which fits,
    and buffering means the H.264 encode never competes with capture for CPU.
    """

    def __init__(self, cam, stop_evt):
        super().__init__(name=f"grab-{cam.label}", daemon=True)
        self.cam = cam
        self.stop_evt = stop_evt
        self.frames: list[np.ndarray] = []
        self.sensor_ns: list[int] = []
        self.seq: list[int] = []
        self.error: BaseException | None = None

    def run(self):
        try:
            while not self.stop_evt.is_set():
                frame, sensor_ns, seq = self.cam.capture_array_meta()
                self.frames.append(frame)
                self.sensor_ns.append(-1 if sensor_ns is None else sensor_ns)
                self.seq.append(-1 if seq is None else seq)
        except BaseException as exc:  # noqa: BLE001 - reported by the main thread
            self.error = exc
            self.stop_evt.set()


def open_camera(label, fps, exposure_us, gain):
    cam = Camera(label)
    cam.configure(size=(1280, 800), bit_depth=8)
    # FrameRate first: it sets vertical blanking, which bounds the exposure the
    # driver will accept.
    cam.set_controls({"FrameRate": fps}, settle=False)
    cam.set_controls({"ExposureTime": exposure_us, "AnalogueGain": gain}, settle=False)
    actual_us = cam.get_control("exposure") * cam._line_time_us()
    print(f"  {label}: exposure {actual_us:.0f} us (asked {exposure_us}), gain {gain}")
    return cam.start(), actual_us


def pair_by_timestamp(ts_a, ts_b):
    """Nearest-timestamp pairing, so a dropped frame shifts nothing downstream.

    Pairing by index would silently slide every later pair by a frame period the
    moment either sensor loses one.

    Returns (index_into_b, valid). Frames at either end of the take can outlive
    the other camera's stream; the nearest partner is then a frame periods away
    and the pair is meaningless, so `valid` marks anything more than half a frame
    period off as unpaired rather than letting it through as a huge delay.
    """
    idx = np.searchsorted(ts_b, ts_a)
    idx = np.clip(idx, 1, len(ts_b) - 1)
    left, right = ts_b[idx - 1], ts_b[idx]
    take_left = np.abs(ts_a - left) <= np.abs(ts_a - right)
    idx = np.where(take_left, idx - 1, idx)
    period_ns = np.median(np.diff(ts_a)) if len(ts_a) > 1 else np.inf
    valid = np.abs(ts_b[idx] - ts_a) <= period_ns / 2
    return idx, valid


def write_video(path, grabbers, labels, pairs, fps):
    h, w = grabbers[0].frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h))
    if not writer.isOpened():
        raise SystemExit(f"cannot open {path} for writing (mp4v codec missing?)")
    means = [[], []]
    for i, j in pairs:
        tiles = []
        for k, (grab, label, n) in enumerate(zip(grabbers, labels, (i, j))):
            frame = grab.frames[n]
            mean = float(frame.mean())
            means[k].append(mean)
            tile = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                tile, f"{label}  mean {mean:6.2f}", (16, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2, cv2.LINE_AA,
            )
            tiles.append(tile)
        writer.write(np.hstack(tiles))
    writer.release()
    return [np.asarray(m) for m in means]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--duration", type=float, default=10.0, help="seconds to record")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument(
        "--exposure", type=float, default=5000.0,
        help="exposure in us. The default is half a 50 Hz light period, i.e. the "
        "worst case for flicker; pass 10000 to get the compensated reference.",
    )
    ap.add_argument("--gain", type=float, default=6.0, help="analogue gain 1.0-16.0")
    ap.add_argument("--hz", type=int, default=50, help="mains frequency")
    ap.add_argument("--cams", nargs=2, metavar=("CAM_A", "CAM_B"))
    ap.add_argument("--out", default=os.path.join(HERE, "flicker_50hz.mp4"))
    args = ap.parse_args()

    half_period_us = 1_000_000 / (args.hz * 2)
    periods = args.exposure / half_period_us
    off_by = abs(periods - round(periods))
    print(
        f"{args.hz} Hz mains -> light period {half_period_us:.0f} us; "
        f"exposure {args.exposure:.0f} us = {periods:.2f} periods "
        + ("(FLICKER EXPECTED)" if off_by > 0.02 else "(compensated)")
    )

    detected = list_cameras()
    print(f"Detected {len(detected)} cameras: {detected or 'none'}")
    if len(detected) < 2:
        sys.exit("need 2 cameras; is ov9282 loaded? (sudo modprobe ov9282)")
    labels = list(args.cams) if args.cams else detected[:2]

    print("Opening cameras:")
    cams = [open_camera(lbl, args.fps, args.exposure, args.gain)[0] for lbl in labels]
    for cam in cams:
        cam.flush(4)  # drop the warm-up frames the driver already queued

    stop_evt = threading.Event()
    grabbers = [Grabber(cam, stop_evt) for cam in cams]
    print(f"Recording {args.duration:.0f} s, sensors free-running...")
    t0 = time.monotonic()
    for g in grabbers:
        g.start()
    try:
        while time.monotonic() - t0 < args.duration and not stop_evt.is_set():
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("interrupted")
    stop_evt.set()
    for g in grabbers:
        g.join(timeout=5)
    for cam in cams:
        cam.stop()

    for lbl, g in zip(labels, grabbers):
        if g.error is not None:
            sys.exit(f"{lbl} failed: {g.error!r}")
        print(f"  {lbl}: {len(g.frames)} frames")
    if min(len(g.frames) for g in grabbers) < 2:
        sys.exit("too few frames captured")

    ts = [np.asarray(g.sensor_ns, dtype=np.int64) for g in grabbers]
    if any((t < 0).any() for t in ts):
        sys.exit("driver reported no buffer timestamps - cannot pair or plot delay")
    j, valid = pair_by_timestamp(ts[0], ts[1])
    pairs = [(i, j[i]) for i in np.flatnonzero(valid)]
    unpaired = int((~valid).sum())
    if unpaired:
        print(f"  {unpaired} frame(s) had no partner within half a frame - dropped")

    print(f"Encoding {len(pairs)} paired frames -> {args.out}")
    means = write_video(args.out, grabbers, labels, pairs, args.fps)

    npz = os.path.splitext(args.out)[0] + "_timestamps.npz"
    np.savez(
        npz,
        cam0_sensor_ns=ts[0], cam1_sensor_ns=ts[1],
        cam0_seq=np.asarray(grabbers[0].seq, dtype=np.int64),
        cam1_seq=np.asarray(grabbers[1].seq, dtype=np.int64),
        pair_index=np.asarray(j, dtype=np.int64),
        pair_valid=valid,
        cam0_mean=means[0], cam1_mean=means[1],
        labels=np.array(labels), fps=args.fps, exposure_us=args.exposure,
        mains_hz=args.hz, gain=args.gain,
    )
    print(f"Timestamps -> {npz}")

    for lbl, m in zip(labels, means):
        pk = (m.max() - m.min()) / m.mean() * 100 if m.mean() else 0.0
        print(
            f"  {lbl}: frame mean {m.mean():6.2f}, "
            f"peak-to-peak {pk:5.2f}% of mean, sd {m.std():.3f}"
        )


if __name__ == "__main__":
    main()
